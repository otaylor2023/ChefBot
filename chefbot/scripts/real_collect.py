"""Drive around a counter and grab RGB-D from the robot's cameras at many base poses: training and localization data
that can be used off the robot afterwards. Every move is real_nav's (Robot(RealEnv).go_to_pose on the map, with the
robot layer's stops); grabs are watch-only and refused without a live mocap pose.

    python -m chefbot.scripts.real_collect plan --map maps/src_counter --mode ring          # list the viewpoints (no robot)
    python -m chefbot.scripts.real_collect run  --map maps/src_counter --mode ring --pass p1 --profile teleop
    python -m chefbot.scripts.real_collect run  ... --mode wall --side south                # one side only, sliding sideways

Modes (all around the map's counter box, world frame):
    ring    each side at several distances out, positions along it, the heading square to the side and turned
            +-30 deg; plus the corners, facing the counter's centre
    wall    one side only (the counter against walls elsewhere): square to it at 0.45 / 0.6 m out, every 0.25 m
            along it, so consecutive stops differ by a sideways slide
    odd     seeded random poses 0.45-1.3 m out on every side, heading up to +-50 deg off square

Each capture: runs/collect/<pass>/<NNN>_<tag>_<camera>.png (RGB), _depth.npz (uint16 mm, compressed), .json (mocap
pose before/after the grab, base odometry, arm joints as last read, intrinsics, the configured cam_to_world, timing,
viewpoint). index.jsonl lists every stop, captured or skipped. A viewpoint the planner refuses (no path / in
collision) is skipped; any other failed move stops the pass (resume with --start N).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from chefbot.environments.real.env import CONFIG, RealEnv, load_config
from chefbot.interface.types import Pose2D
from chefbot.scripts.real_nav import REPO, base_speeds, counter_box, drive, make_robot, robot_footprints

POLE_NOTE = ("pole camera re-angled by Jack on 2026-11-05 before these passes: the configured pole extrinsics (and so "
             "cam_to_world for the pole) predate it and are NOT valid; solve them offline")


def _jsonable(o):
    return o.tolist() if hasattr(o, "tolist") else float(o)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def sides(box) -> list[tuple[str, np.ndarray, np.ndarray, float, float]]:
    """(name, centre of the side's edge, unit outward normal, unit along, half length) for each side of the box."""
    (cx, cy), (sx, sy), yaw, _ = box
    c, s = math.cos(yaw), math.sin(yaw)
    ex, ey = np.array([c, s]), np.array([-s, c])
    ctr = np.array([cx, cy])
    return [("south", ctr - ey * sy / 2, -ey, ex, sx / 2), ("north", ctr + ey * sy / 2, ey, -ex, sx / 2),
            ("west", ctr - ex * sx / 2, -ex, -ey, sy / 2), ("east", ctr + ex * sx / 2, ex, ey, sy / 2)]  # fmt: skip


def _vp(tag: str, p: np.ndarray, facing: np.ndarray, turn_deg: float = 0.0) -> dict:
    yaw = math.atan2(facing[1], facing[0]) + math.radians(turn_deg)
    return {"tag": tag, "x": round(float(p[0]), 4), "y": round(float(p[1]), 4), "yaw_deg": round(math.degrees(_wrap(yaw)), 2)}


def viewpoints(box, mode: str, side: str | None = None, seed: int = 0, n_odd: int = 24) -> list[dict]:
    out = []
    for name, edge, normal, along, half in sides(box):
        if side and name != side:
            continue
        if mode == "ring":
            for d in (0.45, 0.7, 1.0):
                n = max(2, int(round(2 * half / 0.5)) + 1)
                for k, a in enumerate(np.linspace(-half + 0.1, half - 0.1, n)):
                    for turn in (0, 30, -30) if k % 2 == 0 else (0,):
                        out.append(_vp(f"{name}_d{int(d * 100)}_a{k}_t{turn:+d}", edge + normal * d + along * a, -normal, turn))
        elif mode == "wall":
            for d in (0.45, 0.6):
                n = max(2, int(round(2 * half / 0.25)) + 1)
                pts = list(np.linspace(-half + 0.05, half - 0.05, n))
                for k, a in enumerate(pts if d == 0.45 else pts[::-1]):  # snake: one way at 0.45, back at 0.6
                    out.append(_vp(f"wall_{name}_d{int(d * 100)}_a{k}", edge + normal * d + along * a, -normal))
        elif mode == "odd":
            rng = np.random.default_rng(seed + hash(name) % 1000)
            for k in range(max(1, n_odd // 4)):
                d, a, turn = rng.uniform(0.45, 1.3), rng.uniform(-half, half), rng.uniform(-50, 50)
                out.append(_vp(f"odd_{name}_{k}", edge + normal * d + along * a, -normal, turn))
        else:
            raise SystemExit(f"unknown mode {mode}")
    if mode == "ring" and not side:
        (cx, cy), (sx, sy), yaw, _ = box
        for i, (u, v) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
            corner = np.array([cx + u * sx / 2, cy + v * sy / 2])
            outward = np.array([u, v]) / math.sqrt(2)
            out.append(_vp(f"corner{i}", corner + outward * 0.65, -outward))
    return out


def tour(vps: list[dict], start: tuple[float, float] | None) -> list[dict]:
    """Greedy nearest-neighbour order from the start (keeps the wall mode's own order)."""
    if start is None or not vps or vps[0]["tag"].startswith("wall_"):
        return vps
    left, here, order = list(vps), np.array(start), []
    while left:
        i = min(range(len(left)), key=lambda j: np.hypot(left[j]["x"] - here[0], left[j]["y"] - here[1]))
        order.append(left.pop(i))
        here = np.array([order[-1]["x"], order[-1]["y"]])
    return order


def write_png(path: Path, rgb: np.ndarray) -> None:
    import struct
    import zlib

    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))  # fmt: skip


def _pose(loc) -> dict | None:
    p = loc.mocap_pose
    return None if p is None else {"x": p.x, "y": p.y, "yaw_deg": math.degrees(p.yaw)}


def capture(env: RealEnv, cams: list[str], stem: str, out: Path, still_s: float = 0.3) -> list[dict]:
    time.sleep(still_s)
    rows = []
    for cam in cams:
        before = env.localization()
        if not before.live:
            rows.append({"camera": cam, "ok": False, "reason": "mocap not live"})
            continue
        t0 = time.time()
        try:
            f = env.get_camera_rgbd(cam)
        except Exception as exc:  # a camera that won't grab: note it, keep the pass going
            rows.append({"camera": cam, "ok": False, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        t1 = time.time()
        after, raw = env.localization(), env.raw_state()
        base = out / f"{stem}_{cam}"
        write_png(base.with_suffix(".png"), f.rgb)
        depth = np.nan_to_num(np.asarray(f.depth, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        np.savez_compressed(f"{base}_depth.npz", depth_mm=np.clip(depth * 1000.0, 0, 65535).astype(np.uint16))
        valid = depth[depth > 0]
        moved = None if before.mocap_pose is None or after.mocap_pose is None else float(
            math.hypot(after.mocap_pose.x - before.mocap_pose.x, after.mocap_pose.y - before.mocap_pose.y))
        meta = {"camera": cam, "wall_request": t0, "wall_reply": t1, "mocap_before": _pose(before),
                "mocap_after": _pose(after), "moved_during_grab_m": moved,
                "odom_pose": raw.get("odom_pose"), "arm_q": raw.get("arm_q"), "arm_live": raw.get("arm_live"),
                "intrinsics": np.asarray(f.intrinsics).tolist(), "cam_to_world_configured": np.asarray(f.cam_to_world).tolist(),
                "shape": list(f.rgb.shape), "depth_valid_frac": float(valid.size / depth.size),
                "depth_median_m": float(np.median(valid)) if valid.size else None,
                "notes": [POLE_NOTE if cam == "pole" else "wrist: arm locked; cam_to_world from the last joint readings and the embodiment kinematics (arm mount z 0.335 there; the real plate is 0.44 m: approximate)",
                          "cam_to_world is computed at reply time from the latest state, not the capture instant"]}  # fmt: skip
        base.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=_jsonable))
        rows.append({"camera": cam, "ok": True, "file": base.name, "grab_s": round(t1 - t0, 2), "moved_m": moved,
                     "depth_valid": round(meta["depth_valid_frac"], 3)})
    return rows


class RemoteGrabber:
    """onboard/collect_grab.py on the robot over one ssh session: full-resolution snapshots of every camera stream and a
    Redis snapshot (all mocap bodies, the arm, base odometry), saved on the robot under ~/chefbot_collect/data/<pass>."""

    def __init__(self, pass_name: str, host: str = "chefbot", control_path: str = "~/.ssh/cm-chefbot",
                 python: str = "~/chefbot_calib/env/bin/python", remote_dir: str = "~/chefbot_collect"):
        cmd = f"cd {remote_dir} && exec {python} -u collect_grab.py {remote_dir}/data/{pass_name}"
        self.p = subprocess.Popen(["ssh", "-o", f"ControlPath={control_path}", host, cmd], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True)  # fmt: skip
        ready = json.loads(self.p.stdout.readline() or "{}")
        if not ready.get("ready"):
            raise RuntimeError(f"the robot-side grabber did not start: {ready}")
        self.remote_out = ready["out"]

    def grab(self, stem: str) -> dict:
        self.p.stdin.write(f"grab {stem}\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("the robot-side grabber exited")
        return json.loads(line)

    def close(self) -> None:
        try:
            self.p.stdin.write("quit\n")
            self.p.stdin.flush()
            self.p.wait(10)
        except Exception:
            self.p.kill()


def capture_remote(env: RealEnv, grabber: RemoteGrabber, stem: str, still_s: float = 0.3) -> dict:
    time.sleep(still_s)
    before = env.localization()
    if not before.live:
        return {"ok": False, "reason": "mocap not live"}
    t0 = time.time()
    res = grabber.grab(stem)
    after, raw = env.localization(), env.raw_state()
    moved = None if before.mocap_pose is None or after.mocap_pose is None else float(
        math.hypot(after.mocap_pose.x - before.mocap_pose.x, after.mocap_pose.y - before.mocap_pose.y))
    return {"ok": bool(res.get("ok")), "wall_request": t0, "wall_reply": time.time(), "mocap_before": _pose(before),
            "mocap_after": _pose(after), "moved_during_grab_m": moved, "odom_pose": raw.get("odom_pose"),
            "arm_q": raw.get("arm_q"), "grab": res}  # fmt: skip


def open_env(url: str | None = None) -> RealEnv:
    """RealEnv for base moves and grabs: no arm driver needed; the wrist camera's pose from the arm's last joint readings
    through the embodiment's kinematics (as environments/real/host.py), so the wrist camera can be grabbed too."""
    from chefbot import embodiments

    kin = embodiments.load("tidybot_fr3").kin
    mount = kin.root_in_world(Pose2D(0.0, 0.0, 0.0))
    env = RealEnv.from_config(np.eye(4), url=url or load_config(CONFIG)["url"])  # no pole mount tracker: raw frames
    env._flange_in_base = lambda q: mount @ kin.flange_pose(q)  # for the wrist camera's configured pose only
    env.require_arm = False
    return env


def _git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return "?"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("plan", "run"))
    ap.add_argument("--map", type=Path, default=REPO / "maps" / "src_counter")
    ap.add_argument("--counter", default=None)
    ap.add_argument("--mode", choices=("ring", "wall", "odd"), default="ring")
    ap.add_argument("--side", choices=("south", "north", "west", "east"), default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-odd", type=int, default=24)
    ap.add_argument("--pass", dest="pass_name", default=None, help="name of this pass (objects as they are now)")
    ap.add_argument("--note", default="", help="free text saved with the pass (what was moved, who is in view)")
    ap.add_argument("--cameras", nargs="+", default=["pole", "wrist"])
    ap.add_argument("--profile", default="teleop")
    ap.add_argument("--margin", type=float, default=0.09)
    ap.add_argument("--start", type=int, default=0, help="skip the first N viewpoints (resume)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--url", default=None)
    ap.add_argument("--grabber", choices=("robot", "server"), default="robot",
                    help="robot: full-resolution, every stream, on the robot (onboard/collect_grab.py); server: 640x480 RGB-D through the onboard server")
    args = ap.parse_args()

    box = counter_box(args.map, args.counter)
    vps = viewpoints(box, args.mode, args.side, args.seed, args.n_odd)
    if args.cmd == "plan":
        for i, v in enumerate(vps):
            print(f"{i:3d} {v['tag']:28} x {v['x']:7.3f}  y {v['y']:7.3f}  yaw {v['yaw_deg']:8.2f}")
        print(f"{len(vps)} viewpoints ({args.mode}{', ' + args.side if args.side else ''})")
        return

    from chefbot import embodiments
    from chefbot.scripts.real_map import load as load_map

    speed, yaw_speed, profile = base_speeds(embodiments.load("tidybot_fr3"), args.profile, None, None)
    grid, _ = load_map(args.map)
    env = open_env(args.url)
    try:
        t0 = time.monotonic()
        while not env.localization().live and time.monotonic() - t0 < 3.0:
            time.sleep(0.05)
        here = env.localization().mocap_pose
        name = args.pass_name or f"{datetime.now():%Y%m%d_%H%M%S}_{args.mode}"
        done_idx = REPO / "runs" / "collect" / name / "index.jsonl"
        done = set()
        if done_idx.exists():  # a pass run again picks up where it stopped: viewpoints already captured are skipped
            for line in done_idx.read_text().splitlines():
                r = json.loads(line)
                if r.get("ok") and (r.get("capture") or {}).get("ok", bool(r.get("captures"))):
                    done.add(r["tag"])
        order = tour([v for v in vps if v["tag"] not in done], None if here is None else (here.x, here.y))[args.start:]
        if done:
            print(f"{len(done)} viewpoints already captured in this pass: skipped")
        if args.limit:
            order = order[: args.limit]
        robot = make_robot(env, grid, speed=speed, yaw_speed=yaw_speed, margin=args.margin, settle_s=0.2, still_mps=0.01)
        _, fp = robot_footprints(env.raw_state(), "auto")
        out = REPO / "runs" / "collect" / name
        out.mkdir(parents=True, exist_ok=True)
        session = {"pass": name, "mode": args.mode, "side": args.side, "seed": args.seed, "profile": profile,
                   "speed": speed, "margin": args.margin, "cameras": args.cameras, "commit": _git_head(),
                   "started": time.time(), "note": args.note, "map": str(args.map), "counter_box": list(box[:3]),
                   "footprint": fp.get("note"), "pole_note": POLE_NOTE, "viewpoints": order}  # fmt: skip
        sess_path = out / f"session_{datetime.now():%H%M%S}.json"
        sess_path.write_text(json.dumps(session, indent=1, default=_jsonable))
        print(f"pass {name}: {len(order)} viewpoints, profile {profile} ({speed} m/s), cameras {args.cameras} -> {out}")
        grabber = None
        if args.grabber == "robot":
            env.keep_cameras_open([])  # the server lets its cameras idle out; the robot-side grabber opens them
            grabber = RemoteGrabber(name)
            session["robot_out"] = grabber.remote_out
            sess_path.write_text(json.dumps(session, indent=1, default=_jsonable))
            print(f"robot-side grabber ready: {grabber.remote_out}")
        index = (out / "index.jsonl").open("a")
        n_ok = 0
        try:
            i0 = sum(1 for _ in done_idx.open()) if done_idx.exists() else 0
            for i, v in enumerate(order, start=i0):
                goal = Pose2D(v["x"], v["y"], math.radians(v["yaw_deg"]))
                res = drive(robot, env, goal, confirm=lambda *_: None, quiet=True)
                for _retry in range(2):  # a brief mocap or link dropout: wait for a live pose, then the same stop again
                    if res["ok"] or str(res["reason"]).startswith("no path"):
                        break
                    print(f"[{i}] {v['tag']}: {res['reason'][:70]}; waiting for a live pose and retrying", flush=True)
                    t_wait = time.monotonic()
                    while not env.localization().live and time.monotonic() - t_wait < 10.0:
                        time.sleep(0.1)
                    time.sleep(1.0)
                    res = drive(robot, env, goal, confirm=lambda *_: None, quiet=True)
                row = {"i": i, **v, "ok": res["ok"], "reason": res["reason"], "seconds": round(res.get("seconds", 0.0), 1),
                       "final_error_m": res.get("final_error_m"), "final_yaw_error_deg": res.get("final_yaw_error_deg")}  # fmt: skip
                if res["ok"]:
                    if grabber is not None:
                        row["capture"] = capture_remote(env, grabber, f"{i:03d}_{v['tag']}")
                    else:
                        row["captures"] = capture(env, args.cameras, f"{i:03d}_{v['tag']}", out)
                    n_ok += 1
                index.write(json.dumps(row, default=_jsonable) + "\n")
                index.flush()
                caps = " ".join(f"{c['camera']}:{'ok' if c['ok'] else c['reason']}" for c in row.get("captures", []))
                if "capture" in row:
                    g = row["capture"].get("grab") or {}
                    caps = " ".join(f"{n}:{'ok' if c.get('ok') else c.get('reason')}" for n, c in (g.get("cameras") or {}).items()) \
                        or row["capture"].get("reason", "")
                print(f"[{i}] {v['tag']:28} {res['reason'][:60]:60} {row['seconds']:5.1f}s {caps}", flush=True)
                if not res["ok"] and res.get("moved", False) is not False and not str(res["reason"]).startswith("no path"):
                    print(f"stopping the pass at {i}: the move did not finish ({res['reason']}); resume with --start {i}")
                    break
        finally:
            index.close()
            if grabber is not None:
                grabber.close()
        print(f"done: {n_ok} stops captured -> {out}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
