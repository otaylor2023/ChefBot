"""Point A to point B on the real base, by mocap, through the onboard server: the first autonomous moves.

    ssh -L 8765:localhost:8765 tidybot01               # with chefbot/onboard/bringup/base.sh running there
    python -m chefbot.scripts.real_nav mark A          # the base where you put it, standing still: saved as A
    python -m chefbot.scripts.real_nav mark B
    python -m chefbot.scripts.real_nav point S1 --from A --forward 0.3 [--left 0] [--turn 0]   # relative to A's heading
    python -m chefbot.scripts.real_nav list
    python -m chefbot.scripts.real_nav go B [--speed 0.1] [--yaw-speed 0.3]     # Enter to start, Ctrl-C stops
    python -m chefbot.scripts.real_nav go stove --map maps/src_kitchen     # A* around the map's obstacles
    python -m chefbot.scripts.real_nav arm                                 # the arm's outline over the floor, now

With --map (scripts/real_map.py), the target may also be one of the map's places, and the base goes there by A* on
the map grown by --footprint + --margin, one straight segment at a time (each a `go` as below, stopping at each
corner), and stops if its footprint ever overlaps an obstacle. --footprint 0.33 m: the base reaches at most ~10 in
(0.254 m) beyond the mocap body, whose origin is 0.072 m ahead of the base centre; --margin 0.10 m on top, for the
arm at the front and for the map's measuring error. The footprint is a circle, so the margin protects the front too.

With the arm driver running (or its last joint readings in Redis), --arm auto (the default) grows the footprint to
cover the arm too: robot/kinematics/arm_outline.py projects the arm's links onto the floor from its joint angles.
And every `go` stops if the arm moves more than 0.05 rad on any joint while the base drives.

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


def relative_point(origin: dict, forward: float, left: float, turn_deg: float) -> dict:
    """A point `forward` / `left` (m) of `origin` in its own heading, turned by `turn_deg`."""
    yaw = math.radians(origin["yaw_deg"])
    c, s = math.cos(yaw), math.sin(yaw)
    return {"x": origin["x"] + c * forward - s * left, "y": origin["y"] + s * forward + c * left,
            "yaw_deg": math.degrees(wrap(yaw + math.radians(turn_deg))), "marked": datetime.now().isoformat(timespec="seconds"),
            "relative_to": {"forward": forward, "left": left, "turn_deg": turn_deg}}  # fmt: skip


ARM_MOVED_RAD = 0.05  # a joint moving more than this while the base drives stops it


def low_footprint(raw: dict, low_height: float, base_radius: float = 0.33, link_radius: float = 0.08,
                  clearance: float = 0.05) -> float:  # fmt: skip
    """The footprint against LOW obstacles (counter tops): the base, plus only the arm's points lower than
    low_height + clearance (higher parts pass over). Unknown arm: the full arm counts."""
    q = raw.get("arm_q")
    if q is None or len(q) != 7:
        return arm_footprint(raw, base_radius, link_radius)["radius"]
    from chefbot.robot.kinematics.arm_outline import arm_points

    pts = arm_points(q)
    # Thickness about each point: the links' link_radius; the flange and hand narrower; the fingertips thin.
    radii = [link_radius] * (len(pts) - 2) + [0.06, 0.03]
    under = [(p, r) for p, r in zip(pts, radii) if p[2] - r < low_height + clearance]
    return max([base_radius] + [math.hypot(p[0], p[1]) + r for p, r in under])


def layered_grids(tall, low, low_height: float, raw: dict, base_radius: float = 0.33, margin: float = 0.10):
    """Plan and body grids: tall obstacles grown by the full footprint (arm included), low ones by low_footprint."""
    from chefbot.robot.nav.grid import OccupancyGrid

    full = arm_footprint(raw, base_radius)["radius"]
    lowfp = low_footprint(raw, low_height, base_radius)
    def both(extra):
        t, l = tall.inflate(full + extra), low.inflate(lowfp + extra)
        return OccupancyGrid(t.occupied | l.occupied, tall.resolution, tall.origin)
    return both(margin), both(0.0), {"full_m": full, "low_m": lowfp, "low_height_m": low_height}


def arm_footprint(raw: dict, base_radius: float = 0.33, link_radius: float = 0.08) -> dict:
    """The footprint circle (base centre) covering the base and, if its joints are known, the arm seen from above."""
    q = raw.get("arm_q")
    out = {"base_radius": base_radius, "radius": base_radius, "arm": None, "arm_live": bool(raw.get("arm_live"))}
    if q is None or len(q) != 7:
        out["note"] = "arm pose unknown (no joint readings): base footprint only"
        return out
    from chefbot.robot.kinematics.arm_outline import arm_box, reach_ahead

    box = arm_box(q, link_radius=link_radius)
    (cx, cy), (L, W) = box["center"], box["size"]
    corners = [math.hypot(cx + sx * L / 2, cy + sy * W / 2) for sx in (-1, 1) for sy in (-1, 1)]
    out.update(arm=box, reach_ahead_m=reach_ahead(q, link_radius=link_radius), radius=max(base_radius, max(corners)))
    out["note"] = "arm from live joints" if out["arm_live"] else "arm from the last joint readings in Redis (driver not running)"
    return out


class Fence:
    """A square the base centre must stay in: centred at (x, y), sides along `yaw_deg`, half-width `half` (m)."""

    def __init__(self, x: float, y: float, yaw_deg: float, half: float):
        self.x, self.y, self.yaw, self.half = float(x), float(y), math.radians(yaw_deg), float(half)

    def local(self, x: float, y: float) -> tuple[float, float]:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        dx, dy = x - self.x, y - self.y
        return c * dx + s * dy, -s * dx + c * dy

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        u, v = self.local(x, y)
        return abs(u) <= self.half + margin and abs(v) <= self.half + margin


def summarize_move(samples: list[dict], start: Pose2D, goal: Pose2D) -> dict:
    """From a move's samples: overshoot past the goal along the line, max off-line, peak speed, time to 5 cm."""
    if len(samples) < 2:
        return {}
    P = np.array([s["pose"] for s in samples])
    T = np.array([s["t"] for s in samples])
    # Rates from the robot server's send times (a laptop's receive times jitter, which made a spike out of 0.17 m/s).
    TS = np.array([s.get("server_t") or np.nan for s in samples], dtype=float)
    T_rate = TS if np.all(np.isfinite(TS)) else T
    line = np.array([goal.x - start.x, goal.y - start.y])
    L = float(np.linalg.norm(line))
    out = {"samples": len(samples), "max_off_line_m": float(max(s["off_line"] for s in samples))}
    if L > 0.02:
        along = (P[:, :2] - [start.x, start.y]) @ (line / L)
        out["overshoot_m"] = float(max(0.0, along.max() - L))
        within = np.where(np.hypot(P[:, 0] - goal.x, P[:, 1] - goal.y) <= 0.05)[0]
        out["time_to_5cm_s"] = float(T[within[0]]) if len(within) else None
    turned = np.degrees(np.unwrap(P[:, 2]) - P[0, 2])
    target_turn = math.degrees(wrap(goal.yaw - start.yaw))
    if abs(target_turn) > 2:
        out["yaw_overshoot_deg"] = float(max(0.0, (turned * np.sign(target_turn)).max() - abs(target_turn)))
    # Over >= 0.1 s windows of distinct states (each state sample can repeat at 20 Hz polling of a 50 Hz stream).
    keep = np.r_[True, np.diff(T_rate) > 1e-4]
    Pk, Tk = P[keep], T_rate[keep]
    v, w = [], []
    j = 0
    for i in range(len(Tk)):
        while j < i and Tk[i] - Tk[j] > 0.1:
            j += 1
        if j > 0 and Tk[i] - Tk[j - 1] >= 0.1:
            dt = Tk[i] - Tk[j - 1]
            v.append(math.hypot(Pk[i, 0] - Pk[j - 1, 0], Pk[i, 1] - Pk[j - 1, 1]) / dt)
            w.append(abs(wrap(Pk[i, 2] - Pk[j - 1, 2])) / dt)
    if v:
        out["peak_speed_mps"] = float(max(v))
        out["peak_yaw_rate_dps"] = float(math.degrees(max(w)))
    # No odometry-vs-mocap ratio here: odom_world is odometry re-tied to mocap on every read, so it equals mocap.
    # The real ratio needs the driver's raw odometry: calibration/base.py record on the robot logs it.
    return out


def go(env: RealEnv, goal: Pose2D, *, speed: float = 0.1, yaw_speed: float = 0.3, tol: float = 0.03,
       yaw_tol_deg: float = 3.0, corridor: float = 0.15, max_dist: float = 1.5, timeout_s: float | None = None,
       confirm=input, log_path: Path | None = None, rate_hz: float = 20.0, fence: Fence | None = None,
       tail_s: float = 1.0, quiet: bool = False, keep_free=None) -> dict:  # fmt: skip
    """Drive to `goal` (world). Returns a result dict; the base is held on every exit. With a `fence`, a goal outside
    it is refused and leaving it (plus 5 cm) stops the move. `tail_s`: keep logging this long after the hold."""
    loc, raw = env.localization(), env.raw_state()
    arm_q0 = np.asarray(raw["arm_q"], dtype=float) if raw.get("arm_live") and raw.get("arm_q") is not None else None
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
    if fence is not None and not (fence.contains(goal.x, goal.y) and fence.contains(start.x, start.y, 0.05)):
        return {"ok": False, "reason": "the start or the target is outside the fence", "moved": False}
    timeout_s = timeout_s or 10.0 + 3.0 * max(d0 / speed, abs(math.radians(turn)) / yaw_speed)
    if not quiet:
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
    samples: list[dict] = []
    t0 = time.monotonic()
    result = {"ok": False, "reason": "?", "moved": True, "wall_start": time.time(),
              "start": [start.x, start.y, start.yaw], "goal": [goal.x, goal.y, goal.yaw], "speed": speed, "yaw_speed": yaw_speed}

    def sample(t, loc, raw, phase):
        p = loc.mocap_pose
        rel = np.array([p.x - start.x, p.y - start.y])
        row = {"t": t, "wall": time.time(), "server_t": raw.get("t"), "phase": phase, "pose": [p.x, p.y, p.yaw],
               "off_line": float(abs(rel[0] * u[1] - rel[1] * u[0])) if line_len > 1e-6 else float(np.linalg.norm(rel)),
               "dist": math.hypot(goal.x - p.x, goal.y - p.y), "yaw_err_deg": math.degrees(wrap(goal.yaw - p.yaw)),
               "odom_world": None if loc.base_pose is None else [loc.base_pose.x, loc.base_pose.y, loc.base_pose.yaw],
               "base_velocity": None if raw.get("base_velocity") is None else [float(v) for v in raw["base_velocity"]],
               "mocap_age_s": loc.mocap_age_s, "state_age_s": loc.state_age_s, "seq": loc.seq}  # fmt: skip
        samples.append(row)
        if log:
            log.write(json.dumps(row) + "\n")
        return row
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
            if not loc.state_current:
                result["reason"] = f"no state from the onboard server for {loc.state_age_s:.2f} s (server or link down)"
                break
            if not loc.live or not raw.get("base_driver_live"):
                result["reason"] = "lost the live mocap pose" if not loc.live else "the base driver stopped"
                break
            row = sample(t, loc, raw, "drive")
            p, off_line, dist, yaw_err = loc.mocap_pose, row["off_line"], row["dist"], row["yaw_err_deg"]
            if not quiet:
                print(f"\r{t:5.1f}s  to go {dist:6.3f} m  yaw {yaw_err:+6.1f} deg  off line {off_line:5.3f} m", end="", flush=True)
            best = min(best, dist)
            if fence is not None and not fence.contains(p.x, p.y, 0.05):
                result["reason"] = "left the fence"
                break
            if keep_free is not None and not keep_free.is_free(p.x, p.y):
                result["reason"] = "the base's footprint reached an obstacle on the map"
                break
            if arm_q0 is not None and raw.get("arm_q") is not None:
                moved = float(np.max(np.abs(np.asarray(raw["arm_q"], dtype=float) - arm_q0)))
                if moved > ARM_MOVED_RAD:
                    result["reason"] = f"the arm moved {moved:.3f} rad while the base was driving"
                    break
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
        t_hold = time.monotonic()
        try:
            while time.monotonic() - t_hold < tail_s:  # how it comes to rest after the hold
                loc = env.localization()
                if not loc.live:
                    break
                sample(time.monotonic() - t0, loc, env.raw_state(), "hold")
                time.sleep(1.0 / rate_hz)
        except Exception:  # the server may be gone (a stop test): the tail is best-effort
            pass
        if log:
            log.close()
        if not quiet:
            print()
    end = env.localization().mocap_pose
    if end is not None:
        result.update(final_error_m=math.hypot(goal.x - end.x, goal.y - end.y),
                      final_yaw_error_deg=math.degrees(wrap(goal.yaw - end.yaw)))  # fmt: skip
    result.update(seconds=time.monotonic() - t0, distance_m=d0, turn_deg=turn)
    result.update(summarize_move([s for s in samples], start, goal))
    if samples and samples[-1]["phase"] == "hold":
        drive = [s for s in samples if s["phase"] == "drive"]
        if drive:
            last, rest = np.array(drive[-1]["pose"][:2]), np.array(samples[-1]["pose"][:2])
            result["moved_after_hold_m"] = float(np.linalg.norm(rest - last))  # how far it rolled once held
    return result


def subdivide(points, max_segment: float) -> list:
    """Split segments longer than `max_segment` into equal straight pieces (the same path, more waypoints)."""
    out = [points[0]]
    for a, b in zip(points, points[1:]):
        n = max(1, math.ceil(math.dist(a, b) / max_segment))
        out += [(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) for k in range(1, n + 1)]
    return out


def go_planned(env: RealEnv, goal: Pose2D, grid, *, footprint: float = 0.33, margin: float = 0.10, confirm=input,
               log_dir: Path | None = None, plan_grid=None, body_grid=None, max_segment: float = 1.0,
               **go_args) -> dict:  # fmt: skip
    """A* from where the base is to `goal` on `grid` (the furniture itself) grown by footprint + margin, then each
    segment with go(). The base centre must stay where its footprint clears the furniture (grid grown by footprint)."""
    from chefbot.robot.nav.astar import AStarPlanner, NoPath, path_length, with_yaw

    loc = env.localization()
    if not loc.live:
        return {"ok": False, "reason": "no live mocap pose", "moved": False}
    start = loc.mocap_pose
    # Callers with a centre-only limit (a fence) pass their own grids; by default everything on `grid` is furniture.
    plan_grid = plan_grid if plan_grid is not None else grid.inflate(footprint + margin)
    body_grid = body_grid if body_grid is not None else grid.inflate(footprint)
    if not body_grid.is_free(start.x, start.y):
        return {"ok": False, "reason": "the base is touching an obstacle on the map where it stands", "moved": False}
    try:
        points = AStarPlanner(plan_grid).plan_xy((start.x, start.y), (goal.x, goal.y))
    except NoPath as exc:
        return {"ok": False, "reason": f"no path: {exc}", "moved": False}
    poses = with_yaw(subdivide(points, max_segment), start.yaw, goal.yaw)[1:]  # each step within go's --max-dist
    print(f"A* path {path_length(points):.2f} m in {len(poses)} segment(s): "
          + " -> ".join(f"({p.x:.2f}, {p.y:.2f}, {math.degrees(p.yaw):.0f})" for p in poses))  # fmt: skip
    try:
        confirm("Floor clear along the path and e-stop in hand? Enter to drive, Ctrl-C to cancel: ")
    except (KeyboardInterrupt, EOFError):
        return {"ok": False, "reason": "cancelled", "moved": False}
    legs = []
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    for i, waypoint in enumerate(poses):
        log_path = None if log_dir is None else log_dir / f"leg{i:02d}.jsonl"
        r = go(env, waypoint, confirm=lambda _: None, log_path=log_path, keep_free=body_grid, quiet=True, **go_args)
        legs.append(r)
        print(f"  segment {i + 1}/{len(poses)}: {r['reason']}" + (f", {1000 * r['final_error_m']:.1f} mm" if r.get("final_error_m") is not None else ""))
        if not r.get("ok"):
            return {"ok": False, "reason": f"segment {i + 1}: {r['reason']}", "moved": True, "legs": legs, "path": points}
    last = legs[-1]
    return {"ok": True, "reason": "reached", "moved": True, "legs": legs, "path": points,
            "final_error_m": last.get("final_error_m"), "final_yaw_error_deg": last.get("final_yaw_error_deg"),
            "path_m": path_length(points)}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mark")
    m.add_argument("name")
    sub.add_parser("list")
    q = sub.add_parser("point")
    q.add_argument("name")
    q.add_argument("--from", dest="origin", required=True)
    q.add_argument("--forward", type=float, default=0.0, help="m along the origin's heading (the way the arm faces)")
    q.add_argument("--left", type=float, default=0.0, help="m to the origin's left")
    q.add_argument("--turn", type=float, default=0.0, help="deg, counter-clockwise")
    g = sub.add_parser("go")
    g.add_argument("name")
    g.add_argument("--speed", type=float, default=0.1, help="m/s (the driver caps it at its own limit)")
    g.add_argument("--yaw-speed", type=float, default=0.3, help="rad/s")
    g.add_argument("--tol", type=float, default=0.03, help="m")
    g.add_argument("--yaw-tol", type=float, default=3.0, help="deg")
    g.add_argument("--corridor", type=float, default=0.15, help="m off the straight line A-B before stopping")
    g.add_argument("--max-dist", type=float, default=1.5, help="m: refuse targets further away")
    g.add_argument("--timeout", type=float, default=None, help="s (default from the distance and speed)")
    g.add_argument("--map", type=Path, default=None, help="a map from scripts/real_map.py: plan around its obstacles")
    g.add_argument("--footprint", type=float, default=0.33, help="m, base centre to its farthest point (with --map)")
    g.add_argument("--margin", type=float, default=0.10, help="m added to the footprint for planning (with --map)")
    g.add_argument("--arm", choices=("auto", "off"), default="auto", help="grow the footprint to cover the arm (with --map)")
    sub.add_parser("arm")
    args = ap.parse_args()

    if args.cmd == "point":
        points = load_points()
        if args.origin not in points:
            raise SystemExit(f"no point {args.origin!r}")
        points[args.name] = relative_point(points[args.origin], args.forward, args.left, args.turn) | {"from": args.origin}
        POINTS.write_text(json.dumps(points, indent=2))
        p = points[args.name]
        print(f"{args.name}: x {p['x']:.4f}  y {p['y']:.4f}  yaw {p['yaw_deg']:.2f}  ({args.forward} m forward, {args.left} m left, "
              f"{args.turn} deg of {args.origin})")  # fmt: skip
        return
    if args.cmd == "list":
        for name, p in load_points().items():
            print(f"{name:10} x {p['x']:.4f}  y {p['y']:.4f}  yaw {p['yaw_deg']:7.2f}  ({p['marked']})")
        return
    if args.cmd == "go" and not (0 < args.speed <= 0.2 and 0 < args.yaw_speed <= 0.5):
        raise SystemExit("refusing: --speed <= 0.2 m/s and --yaw-speed <= 0.5 rad/s for now")
    env = RealEnv.from_config(np.eye(4), url=args.url or load_config(CONFIG)["url"])  # arm poses are never sent
    t0 = time.monotonic()  # the server needs two mocap frames to call the pose live
    while not env.localization().live and time.monotonic() - t0 < 3.0:
        time.sleep(0.05)
    try:
        if args.cmd == "mark":
            mark(env, args.name)
            return
        if args.cmd == "arm":
            fp = arm_footprint(env.raw_state())
            print(json.dumps(fp, indent=2))
            return
        points = load_points()
        grid = None
        if args.map is not None:
            from chefbot.scripts.real_map import load as load_map

            grid, map_places = load_map(args.map)
            points = map_places | points
        if args.name not in points:
            raise SystemExit(f"no point {args.name!r}: mark it first ({', '.join(points) or 'none yet'})")
        p = points[args.name]
        stamp = f"{datetime.now():%Y%m%d_%H%M%S}_{args.name}"
        if grid is not None:
            log_dir = REPO / "runs" / "real_nav" / stamp
            log_dir.mkdir(parents=True, exist_ok=True)
            footprint = args.footprint
            grids = {}
            if args.arm == "auto":
                fp = arm_footprint(env.raw_state(), base_radius=args.footprint)
                footprint = fp["radius"]
                print(f"footprint {footprint:.3f} m ({fp['note']}" + (f"; arm reaches {fp['reach_ahead_m']:.2f} m ahead)" if fp["arm"] else ")"))
                from chefbot.scripts.real_map import load_layers

                layers = load_layers(args.map)
                if layers is not None:
                    plan_g, body_g, info = layered_grids(*layers, env.raw_state(), base_radius=args.footprint, margin=args.margin)
                    grids = {"plan_grid": plan_g, "body_grid": body_g}
                    print(f"  against walls {info['full_m']:.3f} m; against low obstacles (<= {info['low_height_m']:.2f} m) "
                          f"{info['low_m']:.3f} m: the arm passes over them")  # fmt: skip
            result = go_planned(env, Pose2D(p["x"], p["y"], math.radians(p["yaw_deg"])), grid, footprint=footprint, **grids,
                                margin=args.margin, log_dir=log_dir, speed=args.speed, yaw_speed=args.yaw_speed, tol=args.tol,
                                yaw_tol_deg=args.yaw_tol, corridor=args.corridor, max_dist=args.max_dist)  # fmt: skip
            (log_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
            print(json.dumps({k: v for k, v in result.items() if k != "legs"}, indent=2, default=str))
            print(f"logs: {log_dir}")
            return
        log_path = REPO / "runs" / "real_nav" / f"{stamp}.jsonl"
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
