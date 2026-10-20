"""Full action chunks: the model plans the gripper's whole pick, from its current
pose, as waypoints marked on the plots (a follow-up to v1's T3, which marked a
fixed pre-grasp / grasp / lift).

    python -m experiments.ortho_grasp.chunk --model claude --scenes 0-9 [--repeats 1] [--e3]

Inputs: the partial cloud (v1's best single-call render) with the gripper's
start pose (the arm at ready) drawn in gray as waypoint 0. Output: 1-8
waypoints (A, B, F per view, open/closed); the first "closed" one is the grasp.

Scored: chunk_valid = the grasp passes E1 (with the 5 mm contact tolerance)
and the whole path is feasible: the gripper swept along the straight segments
from the start clear of the ground-truth cup and the counter until the grasp
(the fingers may touch only in the last 1 cm), clear of the counter after it,
IK continuous along every segment, and the last waypoint lifts the cup >= 5 cm.
`--e3` executes the waypoints in RoboCasa as straight lines (no planner).
"""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from experiments.ortho_grasp import run as R
from experiments.ortho_grasp import scenes as scene_set
from experiments.ortho_grasp.evaluate import e1_geometric
from experiments.ortho_grasp.frames import FLANGE, MAX_OPEN, box_corners, box_distance, gripper_boxes, pose_to_robot
from experiments.ortho_grasp.query import models as M
from experiments.ortho_grasp.query.schemas import ParseError, extract_json
from experiments.ortho_grasp.render import plots
from experiments.ortho_grasp.resolve import CLEARANCE, FINGER_TOLERANCE, lift, resolve_p1

PROMPT = R.HERE / "prompts" / "chunk_v1" / "chunk.md"
MARGIN = 0.06  # m around the cup and the start gripper
TOL = R.CONTACT_TOLERANCE
STEP = 0.005  # m between path samples
MAX_WAYPOINTS = 8
LIFT_MIN = 0.05


# -- inputs ---------------------------------------------------------------------------


def start_points(w) -> dict[str, np.ndarray]:
    """The gripper at the arm's ready pose (robot frame), fully open: A0, B0, F0."""
    tcp = pose_to_robot(w.arm.frames.pose_to_world(w.base, w.arm.ik.fk(w.arm.ready)), w.base)
    m, x, z = tcp[:3, 3], tcp[:3, 0], tcp[:3, 2]
    return {"A0": m - x * MAX_OPEN / 2, "B0": m + x * MAX_OPEN / 2, "F0": m - z * FLANGE}


def inputs(cfg: dict, scene, w, g) -> Path:
    d = cfg["results"] / "inputs" / "chunk" / scene.name
    if (d / "done").exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    start = start_points(w)
    pts = np.vstack([g.full.min(axis=0), g.full.max(axis=0), *start.values()])
    specs = plots.specs_for(np.array([pts.min(axis=0), pts.max(axis=0)]), MARGIN, cfg["image_size"])
    for view, spec in specs.items():
        plots.draw(spec, d / f"{view}.png", points=g.partial, counter_z=g.counter_z, marks="ABF",
                   color_range=(g.bottom_z, g.rim_z), gripper_at=start)  # fmt: skip
    (d / "start.json").write_text(json.dumps({k: v.tolist() for k, v in start.items()}))
    (d / "done").write_text("")
    return d


def request(cfg: dict, model: dict, d: Path, scene, repeat: int) -> M.Request:
    appearance = R.schemas._appearance({**cfg, "render": "v1", "geometry": "partial"})
    text = PROMPT.read_text().format(appearance_text=appearance)
    meta = {"config": f"chunk_{cfg['model']}", "scene": scene.name, "repeat": repeat, "call": 1}
    temperature = model["temperature"] if "temperature" in model else cfg["temperature"]
    return M.Request(model["provider"], model["model"], text, [str(d / "top.png"), str(d / "side.png")], temperature,
                     int(cfg["max_tokens"]), repeat, meta=meta)  # fmt: skip


# -- parsing and resolving ------------------------------------------------------------


def parse(text: str, specs: dict) -> list[dict]:
    d = extract_json(text)
    wps = d.get("waypoints")
    if not isinstance(wps, list) or not 1 <= len(wps) <= MAX_WAYPOINTS:
        raise ParseError("bad_waypoints")
    out = []
    for i, wp in enumerate(wps):
        if not isinstance(wp, dict) or wp.get("gripper") not in ("open", "closed"):
            raise ParseError(f"bad_value:wp{i + 1}.gripper")
        cfg = {"output": "o1"}
        out.append({"gripper": wp["gripper"],
                    "top": R.schemas._view_marks(cfg, wp.get("top"), specs["top"], ["A", "B", "F"], f"wp{i + 1}.top"),
                    "side": R.schemas._view_marks(cfg, wp.get("side"), specs["side"], ["A", "B", "F"], f"wp{i + 1}.side")})  # fmt: skip
    return out


def _slerp_poses(p0: np.ndarray, p1: np.ndarray, n: int) -> list[np.ndarray]:
    from scipy.spatial.transform import Rotation, Slerp

    rots = Rotation.from_matrix([p0[:3, :3], p1[:3, :3]])
    slerp = Slerp([0.0, 1.0], rots)
    out = []
    for t in np.linspace(0.0, 1.0, n):
        T = np.eye(4)
        T[:3, :3] = slerp([t]).as_matrix()[0]
        T[:3, 3] = (1 - t) * p0[:3, 3] + t * p1[:3, 3]
        out.append(T)
    return out


def _clear(pose: np.ndarray, width: float, cloud: np.ndarray | None, counter_z: float, fingers_on_cup: bool) -> bool:
    for i, box in enumerate(gripper_boxes(pose, width)):
        fingers = i >= 2
        if box_corners(box)[:, 2].min() < counter_z - 1e-4:
            return False
        if cloud is None or (fingers and fingers_on_cup):
            continue
        if fingers:
            box = (box[0], box[1], np.maximum(box[2] - FINGER_TOLERANCE, 1e-4))
            if np.any(box_distance(cloud, box) <= 0.0):
                return False
        elif np.any(box_distance(cloud, box) <= CLEARANCE):
            return False
    return True


def score(poses: list[np.ndarray], widths: list[float], grasp: int, g, ik) -> dict:
    """poses[0] is the start; poses[grasp] the grasp. Returns the path checks."""
    out = {"path_clear": True, "ik_path": True, "collision_at": None, "ik_break_at": None}
    q = None
    for i in range(len(poses) - 1):
        p0, p1 = poses[i], poses[i + 1]
        n = max(2, int(np.ceil(np.linalg.norm(p1[:3, 3] - p0[:3, 3]) / STEP)) + 1)
        n = max(n, int(np.ceil(np.degrees(np.arccos(np.clip((np.trace(p0[:3, :3].T @ p1[:3, :3]) - 1) / 2, -1, 1))) / 2)) + 1)
        samples = _slerp_poses(p0, p1, n)
        to_grasp = i + 1 <= grasp
        seg_len = np.linalg.norm(p1[:3, 3] - p0[:3, 3])
        for k, pose in enumerate(samples):
            left = seg_len * (1 - k / (n - 1))
            if to_grasp:
                ok = _clear(pose, widths[i + 1], g.full, g.counter_z, fingers_on_cup=(i + 1 == grasp and left <= 0.01))
            else:  # holding the cup: only the counter
                ok = _clear(pose, widths[grasp], None, g.counter_z, False)
            if not ok and out["path_clear"]:
                out["path_clear"], out["collision_at"] = False, f"segment {i + 1}"
            q_new = ik(pose, q)
            if q_new is None or (q is not None and np.abs(q_new - q).max() > 0.25):
                if out["ik_path"]:
                    out["ik_path"], out["ik_break_at"] = False, f"segment {i + 1}"
                q_new = None
            q = q_new
    return out


def evaluate_one(text: str, specs: dict, start: dict, g, ik) -> tuple[dict, list[dict], dict]:
    """(metrics, waypoints [{pose, width, gripper}], marks per view)."""
    try:
        wps = parse(text, specs)
    except ParseError as e:
        return {"e0_parse": False, "resolved": False, "reason": f"parse:{e}", "e2_valid": False}, [], {}
    marks = {"top": {}, "side": {}}
    resolved, grippers = [], []
    for i, wp in enumerate(wps):
        pts, dis = lift(wp["top"], wp["side"], ["A", "B", "F"])
        r = resolve_p1(pts, dis)
        if not r.ok:
            return {"e0_parse": True, "resolved": False, "reason": f"wp{i + 1}:{r.reason}", "e2_valid": False,
                    "n_waypoints": len(wps)}, [], marks  # fmt: skip
        resolved.append(r)
        grippers.append(wp["gripper"])
        for v in ("top", "side"):
            marks[v].update({f"{k}{i + 1}": c for k, c in wp[v].items()})
    if "closed" not in grippers:
        return {"e0_parse": True, "resolved": False, "reason": "no_grasp", "e2_valid": False, "n_waypoints": len(wps)}, [], marks
    grasp = grippers.index("closed")
    s = {k: np.asarray(v) for k, v in start.items()}
    p_start = resolve_p1({"A": s["A0"], "B": s["B0"], "F": s["F0"]}, {})
    poses = [p_start.pose] + [r.pose for r in resolved]
    widths = [MAX_OPEN] + [min(MAX_OPEN, r.width + 2 * TOL) for r in resolved]
    gr = resolved[grasp]
    A, B = gr.fingertips
    c = gr.pose[:3, 0]
    e1 = e1_geometric.evaluate(A - c * TOL, B + c * TOL, widths[grasp + 1], g)
    path = score(poses, widths, grasp + 1, g, ik)
    lifted = resolved[-1].pose[2, 3] - gr.pose[2, 3] >= LIFT_MIN and grippers[-1] == "closed"
    valid = e1["e1_valid"] and path["path_clear"] and path["ik_path"] and lifted
    metrics = {"e0_parse": True, "resolved": True, "reason": None, "n_waypoints": len(wps), "grasp_waypoint": grasp + 1,
               **e1, **path, "lift_ok": bool(lifted), "e2_valid": bool(valid), "chunk_valid": bool(valid)}  # fmt: skip
    waypoints = [{"pose": p, "width": w_, "gripper": gp} for p, w_, gp in zip(poses[1:], widths[1:], grippers)]
    return metrics, waypoints, marks


# -- execution -----------------------------------------------------------------------


def execute(w, scene, waypoints: list[dict], out: Path, g) -> dict:
    from experiments.ortho_grasp.evaluate.e3_sim import CLIP_CAMERA, CLIP_EVERY, HOLD_S, _arm_pose, _bottom

    env, arm = w.env, w.arm.arm
    preempt = threading.Event()
    w.apply(scene)
    env.set_arm_joints(w.arm.ready)
    env.wait_for_next_tick(0.2)
    cup = w.cup_root(scene)
    frames, tick, original = [], [0], env.wait_for_next_tick
    track: list[dict] = []  # per video frame: where the gripper really is (robot frame), for the synced plot video

    def recording(dt):
        original(dt)
        tick[0] += 1
        if tick[0] % CLIP_EVERY == 0:
            frames.append(w.render(CLIP_CAMERA, 480, 360))
            state = env.get_state()
            tcp = pose_to_robot(w.arm.frames.pose_to_world(w.base, w.arm.ik.fk(state.arm_q)), w.base)
            track.append({"pose": tcp.tolist(), "width": float(state.gripper_width), "t": float(state.time)})

    env.wait_for_next_tick = recording
    result = {"enclosed": False, "lifted": False, "e3_reason": None}
    try:
        grasp = next(i for i, wp in enumerate(waypoints) if wp["gripper"] == "closed")
        opening = waypoints[grasp]["width"]
        if opening < MAX_OPEN - 0.002:
            env.set_gripper(True)
            for _ in range(100):
                env.wait_for_next_tick(0.05)
                if env.get_state().gripper_width <= opening:
                    break
            env.gripper_hold = True
        for i, wp in enumerate(waypoints):
            line = arm.plan_line(arm.current_target(), _arm_pose(w, wp["pose"]))
            if line is None:
                result["e3_reason"] = f"no_line_to_wp{i + 1}"
                break
            arm.follow(line, preempt, expect_contact=True)
            if i == grasp:
                env.gripper_hold = False
                w.arm.gripper.close(preempt)
                result["enclosed"] = bool(cup in env._grasped_roots())
        for _ in range(int(HOLD_S / 0.05)):
            env.wait_for_next_tick(0.05)
        result["lifted"] = bool(_bottom(env, cup) >= g.counter_z + LIFT_MIN and cup in env._grasped_roots())
        if not result["lifted"] and result["e3_reason"] is None:
            result["e3_reason"] = "not_enclosed" if not result["enclosed"] else "dropped_or_not_lifted"
    finally:
        env.wait_for_next_tick = original
        env.gripper_hold = False
        if frames:
            iio.imwrite(out / "exec.mp4", np.stack(frames), fps=20 // CLIP_EVERY, codec="libx264")
            (out / "exec_track.json").write_text(json.dumps(track))
    return result


# -- main -----------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Full action chunks marked on the plots.")
    ap.add_argument("--model", default="claude")
    ap.add_argument("--scenes", default="0-9")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--e3", action="store_true", help="also execute the chunks in the simulator")
    ap.add_argument("--max-tokens", type=int, help="output limit (a thinking model's reasoning counts toward it)")
    args = ap.parse_args()
    M.load_env(R.REPO / ".env")
    cfg = R.load_config(R.HERE / "configs" / "v1_partial.yaml")
    cfg["model"] = args.model
    if args.max_tokens:
        cfg["max_tokens"] = args.max_tokens
    model = cfg["model_spec"] = __import__("yaml").safe_load((R.HERE / "configs" / "models.yaml").read_text())[args.model]
    name = f"chunk_{args.model}"
    world = R.Lazy(cfg["experiment"])
    cache = M.Cache(cfg["results"] / "cache")
    try:
        all_scenes = scene_set.load(cfg["results"] / "scenes.json")
        scenes = [all_scenes[i] for i in R.parse_range(args.scenes, len(all_scenes))]
        dirs = {}
        for s in scenes:
            g = R.load_geometry(cfg, s)
            if not (cfg["results"] / "inputs" / "chunk" / s.name / "done").exists():
                world().apply(s)
            dirs[s.name] = inputs(cfg, s, world() if not (cfg["results"] / "inputs" / "chunk" / s.name / "done").exists() else None, g)
        reqs = [request(cfg, model, dirs[s.name], s, r) for s in scenes for r in range(args.repeats)]
        done = M.run_sync(reqs, cache)
        print(f"{name}: {len(done)} responses, {sum(1 for d in done if d.get('error'))} errors", flush=True)
        rows = []
        for s in scenes:
            g = R.load_geometry(cfg, s)
            d = dirs[s.name]
            specs = {v: plots.PlotSpec.load(d / f"{v}.json") for v in ("top", "side")}
            start = json.loads((d / "start.json").read_text())
            for r in range(args.repeats):
                out = cfg["results"] / name / f"{s.name}_{r}"
                out.mkdir(parents=True, exist_ok=True)
                resp = cache.get(request(cfg, model, d, s, r).key) or {}
                metrics, wps, marks = evaluate_one(resp.get("text") or "", specs, start, g, lambda pose, seed: R.ik_robot(world(), pose, seed))
                metrics.update({"config": name, "scene": s.name, "repeat": r, "cup_model": s.cup_model,
                                "distractors": len(s.distractors), "call1_cost": resp.get("cost"),
                                "call1_output_tokens": resp.get("output_tokens")})  # fmt: skip
                if args.e3 and wps:
                    metrics.update(execute(world(), s, wps, out, g))
                (out / "response.json").write_text(json.dumps({"call1": resp}, indent=1, default=str))
                (out / "waypoints.json").write_text(json.dumps([{**w_, "pose": w_["pose"].tolist()} for w_ in wps], indent=1))
                (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str))
                note = f"{name} {s.name}: " + ("chunk valid" if metrics.get("chunk_valid") else f"fail ({metrics.get('reason') or _why(metrics)})")
                if "lifted" in metrics:
                    note += f" | sim: {'lifted' if metrics['lifted'] else metrics.get('e3_reason')}"
                for v in ("top", "side"):
                    plots.overlay(specs[v], d / f"{v}.png", out / f"overlay_{v}.png", {k: np.asarray(c) for k, c in marks.get(v, {}).items()},
                                  bool(metrics.get("chunk_valid")), note)  # fmt: skip
                rows.append(metrics)
                print(f"  {s.name}_{r}: {note.split(': ', 1)[1]}  ({metrics.get('n_waypoints', '-')} waypoints)", flush=True)
        print(f"{name}: chunk_valid {sum(bool(m.get('chunk_valid')) for m in rows)}/{len(rows)}"
              + (f", lifted in sim {sum(bool(m.get('lifted')) for m in rows)}/{len(rows)}" if args.e3 else ""), flush=True)  # fmt: skip
    finally:
        world.close()
    return 0


def _why(m: dict) -> str:
    for k in ("straddle", "clear_tips", "width_ok", "height_ok"):
        if m.get(k) is False:
            return f"grasp {k}"
    if not m.get("path_clear"):
        return f"collision on {m.get('collision_at')}"
    if not m.get("ik_path"):
        return f"IK breaks on {m.get('ik_break_at')}"
    if not m.get("lift_ok"):
        return "no lift"
    return "?"


if __name__ == "__main__":
    raise SystemExit(main())
