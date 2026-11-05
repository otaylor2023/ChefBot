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

from chefbot.environments.real.env import CONFIG, RealEnv, load_config
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
       tail_s: float = 1.0, quiet: bool = False, keep_free=None, follow=None, stop_model=None, accel: float = 0.1,
       settle_s: float = 0.5, still_mps: float | None = None, stop_buffer_m: float = 0.03) -> dict:  # fmt: skip
    """Drive to `goal` (world). Returns a result dict; the base is held on every exit. With a `fence`, a goal outside
    it is refused and leaving it (plus 5 cm) stops the move. `tail_s`: keep logging this long after the hold.

    `follow`: a robot/nav BaseController over a whole path ending at `goal`: each tick the target sent is its carrot
    (0.3 m ahead on the path), "off line" is the distance from the path, and there is no "moving away" stop (a path
    around a counter moves away from its goal at times); --max-dist applies to each carrot, not to the goal.

    `stop_model` (robot/nav/speed.py, the robot's measured stop) with `keep_free`: each tick the speed is capped so the
    base can stop within the free space ahead of it, along the way it is moving, on `keep_free` (the footprint-grown
    grid, arm included); a `follow` carrot is sized for that speed at `accel` and shortened until the straight line to
    it is clear. Arrival: within tolerance (and slower than `still_mps`, when given) for `settle_s`."""
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
    path_len = None
    if follow is not None:
        path_len = float(follow._cum[-1])
    elif d0 > max_dist:
        return {"ok": False, "reason": f"the target is {d0:.2f} m away (> --max-dist {max_dist} m)", "moved": False}
    if fence is not None and not (fence.contains(goal.x, goal.y) and fence.contains(start.x, start.y, 0.05)):
        return {"ok": False, "reason": "the start or the target is outside the fence", "moved": False}
    timeout_s = timeout_s or 10.0 + 3.0 * max((path_len or d0) / speed, abs(math.radians(turn)) / yaw_speed)
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

    def sample(t, loc, raw, phase, extra=None):
        p = loc.mocap_pose
        rel = np.array([p.x - start.x, p.y - start.y])
        row = {"t": t, "wall": time.time(), "server_t": raw.get("t"), "phase": phase, "pose": [p.x, p.y, p.yaw],
               "off_line": float(abs(rel[0] * u[1] - rel[1] * u[0])) if line_len > 1e-6 else float(np.linalg.norm(rel)),
               "dist": math.hypot(goal.x - p.x, goal.y - p.y), "yaw_err_deg": math.degrees(wrap(goal.yaw - p.yaw)),
               "odom_world": None if loc.base_pose is None else [loc.base_pose.x, loc.base_pose.y, loc.base_pose.yaw],
               "base_velocity": None if raw.get("base_velocity") is None else [float(v) for v in raw["base_velocity"]],
               "mocap_age_s": loc.mocap_age_s, "state_age_s": loc.state_age_s, "seq": loc.seq}  # fmt: skip
        row.update(extra or {})
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
            target, extra = goal, {}
            v_cap = speed
            p_now = loc.mocap_pose
            if stop_model is not None and keep_free is not None:
                from chefbot.robot.nav.speed import free_distance

                bv = raw.get("base_velocity")
                way = (bv[0], bv[1]) if bv is not None and math.hypot(bv[0], bv[1]) > 0.02 else None
                if follow is not None:
                    from chefbot.robot.nav.speed import lookahead_for

                    look = lookahead_for(speed, accel, 0.1)
                    target = follow.target_at(p_now, look)
                    while look > 0.12 and not keep_free.segment_free((p_now.x, p_now.y), (target.x, target.y)):
                        look = max(0.12, look * 0.7)  # no corner cut into an obstacle
                        target = follow.target_at(p_now, look)
                    extra["lookahead"] = look
                way = way or (target.x - p_now.x, target.y - p_now.y)
                free = free_distance(keep_free, (p_now.x, p_now.y), way)
                v_cap = min(speed, stop_model.speed_for(free - stop_buffer_m))
                extra.update(free_m=free, v_cap=v_cap)
            elif follow is not None:
                target = follow.target(p_now)
            if follow is not None:
                extra.update(carrot=[target.x, target.y, target.yaw], off_line=follow.deviation)
            row = sample(t, loc, raw, "drive", extra or None)
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
            if follow is None and dist > best + 0.10:
                result["reason"] = f"moving away from the target ({dist:.2f} m, best {best:.2f} m)"
                break
            bv = raw.get("base_velocity")
            still = still_mps is None or (bv is not None and math.hypot(bv[0], bv[1]) < still_mps)
            if dist <= tol and abs(yaw_err) <= yaw_tol_deg and still:
                settled_since = settled_since if settled_since is not None else t
                if t - settled_since >= settle_s:
                    result.update(ok=True, reason="reached")
                    break
            else:
                settled_since = None
            if t > timeout_s:
                result["reason"] = f"timed out after {timeout_s:.0f} s"
                break
            if math.hypot(target.x - p.x, target.y - p.y) > max_dist:
                result["reason"] = f"the next target is more than {max_dist} m away"
                break
            env.set_base_target(target, (max(v_cap, 0.02), max(v_cap, 0.02), yaw_speed))  # 0.02: creep, never 0
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
    result.update(seconds=time.monotonic() - t0, distance_m=d0, turn_deg=turn, path_m=path_len)
    result.update(summarize_move([s for s in samples], start, goal))
    if samples and samples[-1]["phase"] == "hold":
        drive = [s for s in samples if s["phase"] == "drive"]
        if drive:
            last, rest = np.array(drive[-1]["pose"][:2]), np.array(samples[-1]["pose"][:2])
            result["moved_after_hold_m"] = float(np.linalg.norm(rest - last))  # how far it rolled once held
    return result


def escape_start(plan_grid, body_grid, start, max_m: float = 0.3):
    """Where planning starts: `start` itself if it is clear with the margin, else the nearest cell that is (within
    `max_m`) when the straight move there is clear for the body itself; None if neither."""
    if plan_grid.is_free(start[0], start[1]):
        return tuple(start)
    cell = plan_grid.nearest_free_cell(start[0], start[1], max_m)
    if cell is None:
        return None
    p = plan_grid.cell_to_world(cell)
    return tuple(p) if body_grid.segment_free(tuple(start), tuple(p)) else None


def subdivide(points, max_segment: float) -> list:
    """Split segments longer than `max_segment` into equal straight pieces (the same path, more waypoints)."""
    out = [points[0]]
    for a, b in zip(points, points[1:]):
        n = max(1, math.ceil(math.dist(a, b) / max_segment))
        out += [(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) for k in range(1, n + 1)]
    return out


def go_planned(env: RealEnv, goal: Pose2D, grid, *, footprint: float = 0.33, margin: float = 0.10, confirm=input,
               log_dir: Path | None = None, plan_grid=None, body_grid=None, max_segment: float = 1.0,
               smooth: bool = True, approach_m: float = 0.0, lookahead: float = 0.12, **go_args) -> dict:  # fmt: skip
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
    # A square finish: A* to a point `approach_m` straight back from the goal along its heading, turning to the goal's
    # heading on the way, then straight in at that heading (for a station facing a counter: along its normal).
    pre = Pose2D(goal.x - approach_m * math.cos(goal.yaw), goal.y - approach_m * math.sin(goal.yaw), goal.yaw)
    if approach_m > 0 and not body_grid.segment_free((pre.x, pre.y), (goal.x, goal.y)):
        return {"ok": False, "reason": "the straight approach to the goal is not clear", "moved": False}
    begin = escape_start(plan_grid, body_grid, (start.x, start.y))
    if begin is None:
        return {"ok": False, "reason": "no path: the start is inside the margin with no clear way out", "moved": False}
    try:
        points = AStarPlanner(plan_grid).plan_xy(begin, (pre.x, pre.y))
    except NoPath as exc:
        return {"ok": False, "reason": f"no path: {exc}", "moved": False}
    if begin != (start.x, start.y):  # out of the margin first, straight (clear for the body)
        points = [(start.x, start.y)] + points
    if approach_m > 0:
        poses = with_yaw(subdivide(points, max_segment), start.yaw, goal.yaw) + [goal]
        points = points + [(goal.x, goal.y)]
    else:
        poses = with_yaw(subdivide(points, max_segment), start.yaw, goal.yaw)
    if smooth:
        from chefbot.robot.config import RobotConfig
        from chefbot.robot.nav.base_controller import BaseController

        print(f"A* path {path_length(points):.2f} m, followed smoothly (carrot {lookahead:.2f} m ahead)"
              + (f", straight in for the last {approach_m:.2f} m" if approach_m > 0 else "") + ": "
              + " -> ".join(f"({p.x:.2f}, {p.y:.2f}, {math.degrees(p.yaw):.0f})" for p in poses[1:]))  # fmt: skip
        try:
            confirm("Floor clear along the path and e-stop in hand? Enter to drive, Ctrl-C to cancel: ")
        except (KeyboardInterrupt, EOFError):
            return {"ok": False, "reason": "cancelled", "moved": False}
        if log_dir is not None:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
        follow_poses = poses[:-1] if approach_m > 0 else poses  # up to `pre` (already at the goal's heading)
        legs = []
        if len(follow_poses) >= 2 and math.dist((start.x, start.y), (follow_poses[-1].x, follow_poses[-1].y)) > 1e-3:
            # A short carrot: commands at 20 Hz keep the base moving, and a corner is cut by at most ~lookahead/2 x
            # sin(turn) (0.3 m cut A* corners by up to ~10 cm, more than the margin, in a dry run).
            import dataclasses

            follow = BaseController(follow_poses, dataclasses.replace(RobotConfig(), lookahead=lookahead))
            r = go(env, follow_poses[-1], confirm=lambda _: None, keep_free=body_grid, quiet=True, follow=follow,
                   log_path=None if log_dir is None else Path(log_dir) / "leg00.jsonl", **go_args)  # fmt: skip
            legs.append(r)
            print(f"  followed: {r['reason']}" + (f", {1000 * r['final_error_m']:.1f} mm" if r.get("final_error_m") is not None else ""))
            if not r.get("ok"):
                return r | {"legs": legs, "path": points, "path_m": path_length(points)}
        if approach_m > 0:  # straight in along the goal's heading: no carrot to cut the corner
            r = go(env, goal, confirm=lambda _: None, keep_free=body_grid, quiet=True,
                   log_path=None if log_dir is None else Path(log_dir) / f"leg{len(legs):02d}.jsonl", **go_args)  # fmt: skip
            legs.append(r)
            print(f"  straight in: {r['reason']}" + (f", {1000 * r['final_error_m']:.1f} mm" if r.get("final_error_m") is not None else ""))
        r = legs[-1]
        return r | {"legs": legs, "path": points, "path_m": path_length(points)}
    poses = poses[1:]  # each step within go's --max-dist
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


def counter_box(map_dir: Path, name: str | None = None) -> tuple:
    """(centre, size, yaw rad, name) of a box obstacle in the map's spec (the first one unless named)."""
    import yaml

    spec = yaml.safe_load((Path(map_dir) / "spec.yaml").read_text())
    for o in spec.get("obstacles") or []:
        if "box" in o and (name is None or o["name"] == name):
            b = o["box"]
            return tuple(b["center"]), tuple(b["size"]), math.radians(b.get("yaw_deg", 0.0)), o["name"]
    raise SystemExit(f"no box obstacle {name or ''} in {map_dir}/spec.yaml")


def random_targets(box, n: int, seed: int) -> list[tuple[float, float]]:
    """Targets drawn as the pick task draws them: a side, a spot along it >= 0.35 m from its ends, 0.10-0.16 m in."""
    from chefbot.robot.nav.goal import PARK_END_MARGIN_M, TRAINED_INSET_M, counter_faces

    rng = np.random.default_rng(seed)
    faces = counter_faces(*box[:3])
    out = []
    for _ in range(n):
        f = faces[int(rng.integers(len(faces)))]
        along = rng.uniform(-1, 1) * (f["half_along"] - PARK_END_MARGIN_M)
        inset = rng.uniform(*TRAINED_INSET_M)
        p = np.asarray(box[0], dtype=float) + f["t"] * along + f["n"] * (f["half_n"] - inset)
        out.append((round(float(p[0]), 3), round(float(p[1]), 3)))
    return out


class _ByRule:
    """Reachability for stances built by the start rule: the pick skills were trained from exactly these."""

    def is_reachable(self, base_pose, target_world) -> bool:
        return True


def choose_stance(env: RealEnv, box, target, grid, grids: dict, footprint: float, margin: float, out_m, approach: float):
    """The stance for `target` by the start rule that is clear for the whole footprint (arm included) and cheapest to
    reach by A* from where the base is: (pose, info) or (None, reason)."""
    from chefbot.robot.nav.astar import AStarPlanner
    from chefbot.robot.nav.goal import counter_park_candidates, select_standoff

    cands = counter_park_candidates(*box[:3], target, out_m=tuple(out_m or (0.45, 0.48)))
    if not cands:
        return None, "the target is not 0-0.25 m in from a side and >= 0.35 m from its ends: no stance by the rule"
    plan_g = grids.get("plan_grid") or grid.inflate(footprint + margin)
    body_g = grids.get("body_grid") or grid.inflate(footprint)
    clear = []
    for pose, info in cands:
        pre = (pose.x - approach * math.cos(pose.yaw), pose.y - approach * math.sin(pose.yaw))
        if plan_g.is_free(pose.x, pose.y) and plan_g.is_free(*pre) and body_g.segment_free(pre, (pose.x, pose.y)):
            clear.append((pose, info))
    if not clear:
        return None, f"no stance clear for the footprint ({len(cands)} by the rule, all blocked)"
    start = env.localization().mocap_pose
    begin = escape_start(plan_g, body_g, (start.x, start.y))
    if begin is None:
        return None, "the base is inside the margin with no clear way out"
    pick = select_standoff(AStarPlanner(plan_g), Pose2D(begin[0], begin[1], start.yaw), np.array([*target, 0.93]),
                           [p for p, _ in clear], _ByRule())  # fmt: skip
    if pick is None:
        return None, "no stance reachable by A* from here"
    return pick, next(i for p, i in clear if p is pick)


def turn_taken_deg(log_dir: Path) -> float:
    """The signed rotation the base actually made over a move's logs (unwrapped mocap yaw)."""
    yaws = [json.loads(line)["pose"][2] for f in sorted(Path(log_dir).glob("leg*.jsonl")) for line in f.open()]
    return float(np.degrees(np.unwrap(yaws)[-1] - yaws[0])) if yaws else 0.0


def park(env: RealEnv, args) -> list[dict]:
    """Each target in turn: choose the stance, drive there (one smooth path, or the stations' straight-in), report
    where the base ended against the start rule. Stops the session at the first move that does not end `reached`,
    and on a rotation more than 15 deg beyond the short way."""
    from chefbot.robot.nav.goal import park_report

    box = counter_box(args.map, args.counter)
    targets = [tuple(t) for t in args.target or []] + random_targets(box, args.random, args.seed)
    if not targets:
        raise SystemExit("no targets: --target X Y and/or --random N")
    from chefbot.scripts.real_map import load as load_map

    grid, _ = load_map(args.map)
    footprint, grids = map_grids(env, args)
    from chefbot.robot.nav.speed import StopModel

    stop_model = StopModel.from_limits(env.server_limits)  # refuses when the robot reports no measured stop
    accel = float(env.server_limits.get("base_max_accel_xy", 0.1))
    print(f"stop model: tau {stop_model.tau_s:.3f} s, brake {stop_model.brake_mps2:.2f} m/s^2 "
          f"({1000 * stop_model.distance(args.speed):.0f} mm from {args.speed:.3f} m/s); driver accel {accel} m/s^2")
    session = REPO / "runs" / "real_park" / f"{datetime.now():%Y%m%d_%H%M%S}"
    session.mkdir(parents=True, exist_ok=True)
    (session / "session.json").write_text(json.dumps({"targets": targets, "args": {k: (str(v) if isinstance(v, Path) else v)
                                                      for k, v in vars(args).items()}}, indent=2))  # fmt: skip
    results = []
    for i, target in enumerate(targets):
        pose, info = choose_stance(env, box, target, grid, grids, footprint, args.margin, args.out, args.approach)
        head = f"[{i + 1}/{len(targets)}] target ({target[0]:.3f}, {target[1]:.3f})"
        if pose is None:
            print(f"{head}: skipped, {info}")  # nothing moved: go on to the next target
            results.append({"target": target, "ok": False, "reason": info, "moved": False, "skipped": True})
            continue
        print(f"{head}: {info['side']} side, {info['inset_m']:.3f} m in, stance ({pose.x:.3f}, {pose.y:.3f}, "
              f"{math.degrees(pose.yaw):.1f} deg) {info['out_m']:.2f} m out" + ("" if info["trained"] else "  [inset outside 0.10-0.16]"))
        if args.plan_only:
            results.append({"target": target, "stance": [pose.x, pose.y, pose.yaw], **info, "ok": True, "reason": "planned"})
            continue
        start = env.localization().mocap_pose
        log_dir = session / f"{i:02d}_{info['side']}"
        t0 = time.monotonic()
        r = go_planned(env, pose, grid, footprint=footprint, **grids, margin=args.margin, log_dir=log_dir,
                       confirm=lambda _: None, speed=args.speed, yaw_speed=args.yaw_speed, tol=args.tol,
                       yaw_tol_deg=args.yaw_tol, corridor=args.corridor, max_dist=args.max_dist, smooth=args.smooth,
                       approach_m=args.approach, stop_model=stop_model, accel=accel, settle_s=0.2, still_mps=0.01,
                       tail_s=0.3)  # fmt: skip
        elapsed = time.monotonic() - t0
        end = env.localization().mocap_pose
        asked = math.degrees(wrap(pose.yaw - start.yaw))
        took = turn_taken_deg(log_dir)
        row = {"target": target, "stance": [pose.x, pose.y, pose.yaw], **info, "ok": bool(r.get("ok")), "reason": r["reason"],
               "seconds": elapsed, "path_m": r.get("path_m"), "final_error_m": r.get("final_error_m"),
               "final_yaw_error_deg": r.get("final_yaw_error_deg"), "turn_asked_deg": asked, "turn_taken_deg": took,
               "stops": [leg["reason"] for leg in r.get("legs", [])]}  # fmt: skip
        if end is not None:
            row["rule"] = park_report(*box[:3], info["side"], end, target)
        (log_dir / "park.json").write_text(json.dumps(row | {"result": r}, indent=2, default=str))
        results.append(row)
        rule = row.get("rule", {})
        nan = float("nan")
        err_m, err_deg = r.get("final_error_m"), r.get("final_yaw_error_deg")
        print(f"    {r['reason']} in {elapsed:.1f} s, {1000 * (nan if err_m is None else err_m):.1f} mm, "
              f"{nan if err_deg is None else err_deg:+.2f} deg; turned {took:+.0f} (asked {asked:+.0f}); "
              f"out {rule.get('out_m', float('nan')):.3f} m, target {rule.get('target_lateral_m', float('nan')) * 1000:+.0f} mm "
              f"off centre, {rule.get('yaw_off_square_deg', float('nan')):+.2f} deg off square")  # fmt: skip
        if abs(took) > abs(asked) + 15.0:
            print(f"    STOP: the base turned {took:+.0f} deg for {asked:+.0f}: not the short way")
            break
        if not r.get("ok"):
            break
    (session / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"session: {session}")
    return results


def map_grids(env: RealEnv, args) -> tuple[float, dict]:
    """The footprint and the planning / body grids for --map (the arm's footprint from its joints with --arm auto)."""
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
    return footprint, grids


def add_go_args(g) -> None:
    g.add_argument("--profile", default=None,
                   help="speed profile (embodiments/speed_profiles.yaml; default: the robot's control.speed)")
    g.add_argument("--speed", type=float, default=None, help="m/s, at most the profile's (default: the profile's)")
    g.add_argument("--yaw-speed", type=float, default=None, help="rad/s, at most the profile's (default: the profile's)")
    g.add_argument("--tol", type=float, default=0.03, help="m")
    g.add_argument("--yaw-tol", type=float, default=3.0, help="deg")
    g.add_argument("--corridor", type=float, default=0.15, help="m off the straight line A-B before stopping")
    g.add_argument("--max-dist", type=float, default=1.5, help="m: refuse targets further away")
    g.add_argument("--timeout", type=float, default=None, help="s (default from the distance and speed)")
    g.add_argument("--map", type=Path, default=None, help="a map from scripts/real_map.py: plan around its obstacles")
    g.add_argument("--footprint", type=float, default=0.33, help="m, base centre to its farthest point (with --map)")
    g.add_argument("--margin", type=float, default=0.10, help="m added to the footprint for planning (with --map)")
    g.add_argument("--arm", choices=("auto", "off"), default="auto", help="grow the footprint to cover the arm (with --map)")
    g.add_argument("--segments", dest="smooth", action="store_false",
                   help="stop at each corner of the A* path (default: follow it smoothly)")  # fmt: skip


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
    add_go_args(g)
    k = sub.add_parser("park", help="park for counter work targets by the pick task's start rule (--map needed)")
    k.add_argument("--target", type=float, nargs=2, action="append", metavar=("X", "Y"), help="a target on the counter (world)")
    k.add_argument("--random", type=int, default=0, help="this many targets drawn by the start rule, one after another")
    k.add_argument("--seed", type=int, default=0)
    k.add_argument("--counter", default=None, help="the counter's obstacle name in the map spec (default: the first box)")
    k.add_argument("--plan-only", action="store_true", help="choose and print the stances; nothing moves")
    k.add_argument("--out", type=float, nargs="+", default=None,
                   help="m, base centre out from the edge, first preferred (default 0.45 0.48: the stations' closeness)")
    k.add_argument("--approach", type=float, default=0.0,
                   help="m straight in at the end (0: one smooth path straight to the stance; 0.25: the stations' way)")
    add_go_args(k)
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
    if args.cmd in ("go", "park"):
        from chefbot import embodiments

        prof = embodiments.load("tidybot_fr3").speed(args.profile)
        vx, _, vyaw = prof.base_max_vel
        args.profile = prof.name
        args.speed = vx if args.speed is None else min(args.speed, vx)
        args.yaw_speed = vyaw if args.yaw_speed is None else min(args.yaw_speed, vyaw)
        if not (0 < args.speed and 0 < args.yaw_speed):
            raise SystemExit("refusing: speeds must be positive")
        print(f"speed profile {prof.name}: {args.speed:.3f} m/s, {args.yaw_speed:.3f} rad/s (the driver clips to its ceiling)")
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
        if args.cmd == "park":
            if args.map is None:
                raise SystemExit("park needs --map")
            if not args.plan_only:
                input(f"Park for {len(args.target or []) + args.random} target(s), one after another, by A* on {args.map}. "
                      "Floor clear and e-stop in hand? Enter to drive, Ctrl-C to cancel: ")  # fmt: skip
            park(env, args)
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
            footprint, grids = map_grids(env, args)
            result = go_planned(env, Pose2D(p["x"], p["y"], math.radians(p["yaw_deg"])), grid, footprint=footprint, **grids,
                                margin=args.margin, log_dir=log_dir, speed=args.speed, yaw_speed=args.yaw_speed, tol=args.tol,
                                yaw_tol_deg=args.yaw_tol, corridor=args.corridor, max_dist=args.max_dist, smooth=args.smooth,
                                approach_m=float(p.get("approach_m", 0.0)))  # fmt: skip
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
