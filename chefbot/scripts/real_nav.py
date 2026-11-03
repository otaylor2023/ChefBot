"""Point A to point B on the real base, by mocap, through the onboard server: the first autonomous moves.

    ssh -L 8765:localhost:8765 tidybot01               # with chefbot/onboard/bringup/base.sh running there
    python -m chefbot.scripts.real_nav mark A          # the base where you put it, standing still: saved as A
    python -m chefbot.scripts.real_nav mark B
    python -m chefbot.scripts.real_nav list
    python -m chefbot.scripts.real_nav go B [--speed 0.1] [--yaw-speed 0.3]     # Enter to start, Ctrl-C stops

Points are base poses in the mocap world (x, y in m, yaw in deg), in runs/real_nav/points.json.

`go` drives straight at the target: the base driver plans the motion to the pose it is sent (RealEnv.set_base_target),
and the target is re-sent every 0.1 s through the latest mocap tie, so mocap closes the loop. No obstacle avoidance:
the floor between the points must be clear. Before starting it checks that mocap and the base driver are live and
that B is within --max-dist (bringup/base.sh, which it needs, refuses to start while the arm driver runs). While driving it stops the
base (base_hold) and exits on: lost mocap or driver, a halt or error from the server, more than --corridor off the
straight line A-B, more than 0.10 m further from B than it has been, --timeout, or Ctrl-C. Reached: within --tol and
--yaw-tol for 0.5 s. The onboard server and the driver have their own stops behind these (0.3 s and 0.5 s without
commands). Every sample goes to runs/real_nav/<time>_<point>.jsonl.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from chefbot.envs.real.env import CONFIG, RealEnv, load_config
from chefbot.interface.types import Pose2D

REPO = Path(__file__).resolve().parents[2]
POINTS = REPO / "runs" / "real_nav" / "points.json"


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def load_points(path: Path = POINTS) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def mark(env: RealEnv, name: str, path: Path = POINTS, seconds: float = 1.0) -> dict:
    """Average the live mocap pose over `seconds` with the base still; refuse if not live or not still."""
    poses, t0 = [], time.monotonic()
    while time.monotonic() - t0 < seconds:
        loc = env.localization()
        if not loc.live:
            raise SystemExit(f"mark {name}: no live mocap pose (stale or not tracked)")
        poses.append([loc.mocap_pose.x, loc.mocap_pose.y, loc.mocap_pose.yaw])
        time.sleep(0.02)
    P = np.array(poses)
    spread = float(np.ptp(P[:, :2], axis=0).max())
    if spread > 0.005:
        raise SystemExit(f"mark {name}: the base moved {spread * 1000:.1f} mm while marking; hold it still")
    yaw = math.atan2(np.sin(P[:, 2]).mean(), np.cos(P[:, 2]).mean())
    point = {"x": float(P[:, 0].mean()), "y": float(P[:, 1].mean()), "yaw_deg": math.degrees(yaw),
             "marked": datetime.now().isoformat(timespec="seconds"), "samples": len(P)}  # fmt: skip
    points = load_points(path)
    points[name] = point
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(points, indent=2))
    print(f"{name}: x {point['x']:.4f} m, y {point['y']:.4f} m, yaw {point['yaw_deg']:.2f} deg ({len(P)} samples) -> {path}")
    return point


def go(env: RealEnv, goal: Pose2D, *, speed: float = 0.1, yaw_speed: float = 0.3, tol: float = 0.03,
       yaw_tol_deg: float = 3.0, corridor: float = 0.15, max_dist: float = 1.5, timeout_s: float | None = None,
       confirm=input, log_path: Path | None = None, rate_hz: float = 10.0) -> dict:  # fmt: skip
    """Drive to `goal` (world). Returns a result dict; the base is held on every exit."""
    loc, raw = env.localization(), env.raw_state()
    problems = []
    if not loc.live:
        problems.append("no live mocap pose")
    if not raw.get("base_driver_live"):
        problems.append("the base driver is not running (bringup/base.sh)")
    if problems:
        return {"ok": False, "reason": "; ".join(problems), "moved": False}
    start = loc.mocap_pose
    d0 = math.hypot(goal.x - start.x, goal.y - start.y)
    turn = math.degrees(wrap(goal.yaw - start.yaw))
    if d0 > max_dist:
        return {"ok": False, "reason": f"the target is {d0:.2f} m away (> --max-dist {max_dist} m)", "moved": False}
    timeout_s = timeout_s or 10.0 + 3.0 * max(d0 / speed, abs(math.radians(turn)) / yaw_speed)
    print(f"from  x {start.x:.3f}  y {start.y:.3f}  yaw {math.degrees(start.yaw):7.2f}")
    print(f"to    x {goal.x:.3f}  y {goal.y:.3f}  yaw {math.degrees(goal.yaw):7.2f}   ({d0:.3f} m, turn {turn:+.1f} deg)")
    print(f"at most {speed} m/s, {yaw_speed} rad/s; stops if > {corridor} m off the line, timeout {timeout_s:.0f} s")
    try:
        confirm("Floor clear and e-stop in hand? Enter to drive, Ctrl-C to cancel: ")
    except (KeyboardInterrupt, EOFError):
        return {"ok": False, "reason": "cancelled", "moved": False}

    line = np.array([goal.x - start.x, goal.y - start.y])
    line_len = float(np.linalg.norm(line))
    u = line / line_len if line_len > 1e-6 else np.zeros(2)
    best, settled_since, errors_seen = float("inf"), None, len(env.errors)
    log = log_path.open("w") if log_path else None
    t0 = time.monotonic()
    result = {"ok": False, "reason": "?", "moved": True}
    try:
        while True:
            t = time.monotonic() - t0
            loc, raw = env.localization(), env.raw_state()
            if env.halted:
                result["reason"] = f"the server held the robot: {env.halted}"
                break
            if len(env.errors) > errors_seen:
                result["reason"] = f"server error: {env.errors[-1]}"
                break
            if not loc.live or not raw.get("base_driver_live"):
                result["reason"] = "lost the live mocap pose" if not loc.live else "the base driver stopped"
                break
            p = loc.mocap_pose
            rel = np.array([p.x - start.x, p.y - start.y])
            off_line = float(abs(rel[0] * u[1] - rel[1] * u[0])) if line_len > 1e-6 else float(np.linalg.norm(rel))
            dist = math.hypot(goal.x - p.x, goal.y - p.y)
            yaw_err = math.degrees(wrap(goal.yaw - p.yaw))
            if log:
                log.write(json.dumps({"t": t, "pose": [p.x, p.y, p.yaw], "dist": dist, "yaw_err_deg": yaw_err,
                                      "off_line": off_line, "mocap_age_s": loc.mocap_age_s}) + "\n")  # fmt: skip
            print(f"\r{t:5.1f}s  to go {dist:6.3f} m  yaw {yaw_err:+6.1f} deg  off line {off_line:5.3f} m", end="", flush=True)
            best = min(best, dist)
            if off_line > corridor:
                result["reason"] = f"{off_line:.2f} m off the line A-B (> {corridor} m)"
                break
            if dist > best + 0.10:
                result["reason"] = f"moving away from the target ({dist:.2f} m, best {best:.2f} m)"
                break
            if dist <= tol and abs(yaw_err) <= yaw_tol_deg:
                settled_since = settled_since if settled_since is not None else t
                if t - settled_since >= 0.5:
                    result.update(ok=True, reason="reached")
                    break
            else:
                settled_since = None
            if t > timeout_s:
                result["reason"] = f"timed out after {timeout_s:.0f} s"
                break
            env.set_base_target(goal, (speed, speed, yaw_speed))
            time.sleep(1.0 / rate_hz)
    except KeyboardInterrupt:
        result["reason"] = "stopped (Ctrl-C)"
    finally:
        env.set_base_velocity(0.0, 0.0, 0.0)  # base_hold: decelerate and hold
        if log:
            log.close()
        print()
    end = env.localization().mocap_pose
    if end is not None:
        result.update(final_error_m=math.hypot(goal.x - end.x, goal.y - end.y),
                      final_yaw_error_deg=math.degrees(wrap(goal.yaw - end.yaw)))  # fmt: skip
    result.update(seconds=time.monotonic() - t0, distance_m=d0)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mark")
    m.add_argument("name")
    sub.add_parser("list")
    g = sub.add_parser("go")
    g.add_argument("name")
    g.add_argument("--speed", type=float, default=0.1, help="m/s (the driver caps it at its own limit)")
    g.add_argument("--yaw-speed", type=float, default=0.3, help="rad/s")
    g.add_argument("--tol", type=float, default=0.03, help="m")
    g.add_argument("--yaw-tol", type=float, default=3.0, help="deg")
    g.add_argument("--corridor", type=float, default=0.15, help="m off the straight line A-B before stopping")
    g.add_argument("--max-dist", type=float, default=1.5, help="m: refuse targets further away")
    g.add_argument("--timeout", type=float, default=None, help="s (default from the distance and speed)")
    args = ap.parse_args()

    if args.cmd == "list":
        for name, p in load_points().items():
            print(f"{name:10} x {p['x']:.4f}  y {p['y']:.4f}  yaw {p['yaw_deg']:7.2f}  ({p['marked']})")
        return
    if args.cmd == "go" and not (0 < args.speed <= 0.2 and 0 < args.yaw_speed <= 0.5):
        raise SystemExit("refusing: --speed <= 0.2 m/s and --yaw-speed <= 0.5 rad/s for now")
    env = RealEnv.from_config(np.eye(4), url=args.url or load_config(CONFIG)["url"])  # arm poses are never sent
    try:
        if args.cmd == "mark":
            mark(env, args.name)
            return
        points = load_points()
        if args.name not in points:
            raise SystemExit(f"no point {args.name!r}: mark it first ({', '.join(points) or 'none yet'})")
        p = points[args.name]
        log_path = REPO / "runs" / "real_nav" / f"{datetime.now():%Y%m%d_%H%M%S}_{args.name}.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        result = go(env, Pose2D(p["x"], p["y"], math.radians(p["yaw_deg"])), speed=args.speed, yaw_speed=args.yaw_speed,
                    tol=args.tol, yaw_tol_deg=args.yaw_tol, corridor=args.corridor, max_dist=args.max_dist,
                    timeout_s=args.timeout, log_path=log_path)  # fmt: skip
        print(json.dumps(result, indent=2))
        print(f"log: {log_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
