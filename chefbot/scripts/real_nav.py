"""Point A to point B on the real base, by mocap, through the onboard server.

    chefbot/environments/real/up.sh                    # the tunnel; launch.sh (or base.sh) running on the robot
    python -m chefbot.scripts.real_nav mark A --map maps/src_kitchen     # the base where you put it, still: place A
    python -m chefbot.scripts.real_nav point S1 --from A --forward 0.3 [--left 0] [--turn 0] --map maps/src_kitchen
    python -m chefbot.scripts.real_nav list --map maps/src_kitchen
    python -m chefbot.scripts.real_nav go stove --map maps/src_kitchen   # A* around the map's obstacles; Enter, Ctrl-C stops
    python -m chefbot.scripts.real_nav go A --map maps/src_kitchen --straight   # straight on open floor, no A*
    python -m chefbot.scripts.real_nav arm                                     # the arm's outline over the floor, now
    python -m chefbot.scripts.real_nav where --map maps/src_kitchen            # where the base is on the map, and clear?
    (Parking for counter targets, and its trials: scripts/real_park.py.)

Places are the map's (maps/README.md): built from its spec (places.yaml) or marked here (marked.yaml), base poses in
the mocap world (x, y in m, yaw in deg). The base goes there by A*, keeping its footprint --margin (0.10 m, for the
map's measuring error) clear of the map's obstacles at the heading it will have at each point of the path, holding
the map's heading zones, and stops if its footprint ever reaches one. The footprint is the robot's measured body
(embodiments/tidybot_fr3/robot.yaml `body:`, interface/workspace.py Body.footprint): the plate's outline (0.56 x 0.48 m
about the base centre; corners 0.37 m out), turned with the base (robot/nav/footprint.py).

With the arm driver running (or its last joint readings in Redis), --arm auto (the default) adds the arm, from its
joint angles (interface/workspace.py arm_points, 18 cm ahead of the base centre on its plate); against a map's low
obstacles only the arm's points low enough to hit them count.
And every `go` stops if the arm moves more than 0.05 rad on any joint while the base drives.

This script is thin: every move is the robot layer's Robot(RealEnv).go_to_pose (make_robot), the same planning,
drive loop and stops as the agent's go_to and the sims (robot/robot.py, robot/safety.py BaseGuard). The robot layer
ties the base's odometry to mocap (robot/localization.py) and sends targets in odometry each control tick (20 Hz), so
mocap closes the loop. With --straight, `go` drives straight on a map with nothing on it: the floor between the points
must be clear, and B within --max-dist. Before starting it checks that mocap and the base driver are live
(bringup/base.sh). While driving the base stops (held) on: lost mocap, the base driver gone or a halt from the server,
the footprint reaching the map, more than --corridor off the path, a straight move more than 0.10 m further from B
than it has been, the arm moving, --timeout, or Ctrl-C; its speed is held to what it can stop in within the free
space ahead (the robot's measured stop, robot.yaml control.limits). Reached: within --tol and --yaw-tol for 0.5 s. The
onboard server and the driver have their own stops behind these (0.3 s and 0.5 s without commands). Every control
tick goes to runs/real_nav/<time>_<point>.jsonl (planned: a folder, one file per leg).
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


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def load_points(map_dir: Path) -> dict:
    """The map's places: built from its spec, and marked here (environments/real/build_map.py load_places)."""
    from chefbot.environments.real.build_map import load_places

    return load_places(map_dir)


def mark(env: RealEnv, name: str, map_dir: Path, seconds: float = 1.0) -> dict:
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
    from chefbot.environments.real.build_map import save_mark

    path = save_mark(map_dir, name, point)
    print(f"{name}: x {point['x']:.4f} m, y {point['y']:.4f} m, yaw {point['yaw_deg']:.2f} deg ({len(P)} samples) -> {path}")
    return point


def relative_point(origin: dict, forward: float, left: float, turn_deg: float) -> dict:
    """A point `forward` / `left` (m) of `origin` in its own heading, turned by `turn_deg`."""
    yaw = math.radians(origin["yaw_deg"])
    c, s = math.cos(yaw), math.sin(yaw)
    return {"x": origin["x"] + c * forward - s * left, "y": origin["y"] + s * forward + c * left,
            "yaw_deg": math.degrees(wrap(yaw + math.radians(turn_deg))), "marked": datetime.now().isoformat(timespec="seconds"),
            "relative_to": {"forward": forward, "left": left, "turn_deg": turn_deg}}  # fmt: skip




EMBODIMENT = "tidybot_fr3"


def embodiment():
    from chefbot import embodiments

    return embodiments.load(EMBODIMENT)


def robot_stop():
    """The robot's measured stop and its acceleration (robot.yaml control.limits; refuses without a measured stop)."""
    e = embodiment()
    stop = e.stop_model()
    if stop is None:
        raise SystemExit(f"refusing: {EMBODIMENT} has no measured stop (robot.yaml control.limits base_stop_*)")
    return stop, e.base_accel


def robot_footprints(raw: dict, arm: str = "auto"):
    """The robot's footprint from its measured body (embodiments/tidybot_fr3/robot.yaml `body:`, interface/workspace.py),
    as a function of an obstacle's top: the plate's outline, plus the arm's points (from its joints, with --arm auto and
    readings to go on) whose underside is lower than the top + LOW_CLEARANCE_M (all of them for inf, a wall): the
    embodiment's footprint_below (embodiments/__init__.py, what the robot layer drives with). Returns (footprint_below,
    info)."""
    q = raw.get("arm_q") if arm == "auto" else None
    known = q is not None and len(q) == 7
    q = q if known else None
    footprint_below = embodiment().footprint_below(q)

    if arm != "auto":
        note = "arm off (--arm off): the plate only"
    elif not known:
        note = "arm pose unknown (no joint readings): the plate only"
    else:
        note = "arm from live joints" if raw.get("arm_live") else "arm from the last joint readings in Redis (driver not running)"
    return footprint_below, footprint_below().summary() | {"note": note, "arm_live": bool(raw.get("arm_live"))}


def nav_space(grid, raw: dict, arm: str = "auto"):
    """Where the base's centre may be at each heading on `grid` (robot/nav/footprint.py), the body itself (no margin):
    each obstacle grown by the footprint of what the robot has lower than its top (the map's heights)."""
    from chefbot.robot.nav.footprint import FootprintSpace

    footprint_below, info = robot_footprints(raw, arm)
    space = FootprintSpace.by_height(grid, footprint_below)
    info["by_top"] = {("wall" if math.isinf(t) else f"{t:.2f} m"): round(footprint_below(t).radius, 3) for t in grid.tops()}
    return space, info


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


def base_speeds(robot, profile: str | None, speed: float | None, yaw_speed: float | None) -> tuple[float, float, str]:
    """The base's speeds for a run: the speed profile's (embodiments/speed_profiles.yaml, through Robot.speed, which
    refuses a profile over the robot's hard cap), or lower ones from --speed / --yaw-speed. Asking for more than the
    profile, or for nothing, is refused (SystemExit), not clipped."""
    try:
        prof = robot.speed(base=profile, use_arm=False, use_base=True)
    except ValueError as exc:
        raise SystemExit(f"refusing: {exc}") from None
    vx, _, vyaw = prof.base_max_vel
    for what, asked, allowed in (("--speed", speed, vx), ("--yaw-speed", yaw_speed, vyaw)):
        if asked is not None and asked > allowed + 1e-9:
            raise SystemExit(f"refusing: {what} {asked} is over speed profile {prof.name}'s {allowed} (pick a faster profile)")
    speed = vx if speed is None else speed
    yaw_speed = vyaw if yaw_speed is None else yaw_speed
    if not (0 < speed and 0 < yaw_speed):
        raise SystemExit("refusing: speeds must be positive")
    return float(speed), float(yaw_speed), prof.name


def check_server_cap(limits: dict, speed: float, yaw_speed: float) -> None:
    """The robot computer reports its base cap in hello. A run's speed over it means the laptop and the robot disagree
    on the cap (an old deploy): refused (ValueError), not clipped."""
    for key, asked in (("base_max_vel_xy", speed), ("base_max_vel_yaw", yaw_speed)):
        if key not in limits:
            raise ValueError(f"the robot reports no {key} (an onboard server older than the base cap?)")
        if asked > float(limits[key]) + 1e-9:
            raise ValueError(f"{asked} is over the robot's reported {key} {limits[key]}: deploy the same commit on both sides")


def open_grid(points, pad: float = 1.0, res: float = 0.05):
    """A map with nothing on it, covering `points` (x, y) and `pad` m around them: for moves without --map."""
    from chefbot.robot.nav.grid import OccupancyGrid

    xy = np.asarray(points, dtype=float)
    lo, hi = xy.min(axis=0) - pad, xy.max(axis=0) + pad
    shape = (int(math.ceil((hi[1] - lo[1]) / res)), int(math.ceil((hi[0] - lo[0]) / res)))
    return OccupancyGrid(np.zeros(shape, dtype=bool), res, (float(lo[0]), float(lo[1])))


def fence_cells(fence: Fence, grid, margin: float = 0.05):
    """The grid's cells outside the fence (plus `margin`): where the base's centre may not go."""
    from chefbot.robot.nav.grid import OccupancyGrid

    rows, cols = grid.shape
    cc, rr = np.meshgrid(np.arange(cols), np.arange(rows))
    x, y = grid.origin[0] + (cc + 0.5) * grid.resolution, grid.origin[1] + (rr + 0.5) * grid.resolution
    c, s_ = math.cos(fence.yaw), math.sin(fence.yaw)
    u, v = c * (x - fence.x) + s_ * (y - fence.y), -s_ * (x - fence.x) + c * (y - fence.y)
    return OccupancyGrid((np.abs(u) > fence.half + margin) | (np.abs(v) > fence.half + margin), grid.resolution, grid.origin)


def where(env: RealEnv, map_dir: Path, margin: float = 0.10, arm: str = "auto") -> dict:
    """Where the base is (mocap) on the map: the nearest place, whether its footprint (the plate and the arm as it is)
    is clear of the map there, with and without the planning margin, and if not, the nearest clear spot at this heading."""
    from chefbot.scripts.real_map import load as load_map

    loc = env.localization()
    p = loc.mocap_pose
    if p is None:
        return {"live": False, "reason": "no mocap pose"}
    grid, places = load_map(map_dir)
    space, info = nav_space(grid, env.raw_state(), arm)
    plan = space.with_margin(margin)
    out = {"live": bool(loc.live), "x": round(p.x, 3), "y": round(p.y, 3), "yaw_deg": round(math.degrees(p.yaw), 1),
           "footprint": info["note"], "reach_m": round(info["radius_m"], 3), "clear": space.is_free(p.x, p.y, p.yaw),
           f"clear_with_{margin:.2f}_margin": plan.is_free(p.x, p.y, p.yaw)}  # fmt: skip
    if places:
        near = sorted(places, key=lambda n: math.hypot(places[n]["x"] - p.x, places[n]["y"] - p.y))[:3]
        out["nearest_places"] = {n: round(math.hypot(places[n]["x"] - p.x, places[n]["y"] - p.y), 2) for n in near}
    if not out[f"clear_with_{margin:.2f}_margin"]:
        free = ~plan.at(p.yaw).occupied
        cells = np.argwhere(free)
        if len(cells):
            xy = np.array([plan.at(p.yaw).cell_to_world(tuple(c)) for c in cells])
            k = int(np.argmin(np.hypot(xy[:, 0] - p.x, xy[:, 1] - p.y)))
            out["nearest_clear"] = {"x": round(float(xy[k, 0]), 3), "y": round(float(xy[k, 1]), 3),
                                    "away_m": round(float(np.hypot(xy[k, 0] - p.x, xy[k, 1] - p.y)), 3)}  # fmt: skip
    return out


def make_robot(env: RealEnv, grid, *, speed: float = 0.1, yaw_speed: float = 0.3, tol: float = 0.03,
               yaw_tol_deg: float = 3.0, corridor: float = 0.15, margin: float = 0.10, arm: str = "auto",
               settle_s: float = 0.5, still_mps: float | None = None, lookahead: float = 0.12,
               timeout_s: float | None = None, centre_limit=None, footprint_below=None, stop_model="robot", accel=None,
               arrive_hold: bool = False, path_clearance: float = 0.0, turn_in_room: bool = False,
               allow_arm_env: bool = False):  # fmt: skip
    """The robot layer on the real robot: the embodiment's navigator (embodiments/__init__.py: Robot(RealEnv) on `grid`
    with the robot's footprint, its measured stop and acceleration from robot.yaml), the run's speeds and tolerances.
    The base drives with the arm as it is (real_nav never moves the arm; the footprint covers it and the arm moving
    stops the drive). Every stop is the robot layer's (robot/safety.py BaseGuard and Robot's drive): lost mocap, the
    footprint on the map, the fence, the arm moving, off the path (corridor), moving away, the timeout, a halt from the
    server. footprint_below / stop_model=None / accel: a trial's own (real_trials). allow_arm_env: an env that wants
    the arm's joints (the real host's, scripts/real_kitchen.py: it holds the arm at its joints while the base drives)."""
    from chefbot.embodiments import Speed

    if env.require_arm and not allow_arm_env:
        raise ValueError("base moves need no arm driver: build the env with RealEnv(..., require_arm=False)")
    limits = env.server_limits
    if "base_max_vel_xy" in limits:
        check_server_cap(limits, float(speed), float(yaw_speed))  # over the robot's cap: refused (ValueError), not clipped
    raw = env.raw_state()
    q = raw.get("arm_q") if arm == "auto" else None
    q = q if q is not None and len(q) == 7 else None
    kw = {"centre_limit": centre_limit, "reachability": _ByRule(), "tuck_arm": False}
    if footprint_below is not None:
        kw["footprint_below"] = footprint_below
    if stop_model != "robot":
        kw["stop_model"] = stop_model
    if accel is not None:
        kw["base_accel"] = float(accel)
    config = {"pos_tolerance": float(tol), "yaw_tolerance": math.radians(yaw_tol_deg), "deviation_threshold": float(corridor),
              "safety_margin": float(margin), "snap_radius": 0.0, "arrive_settle_s": float(settle_s),
              "arrive_still_mps": still_mps, "lookahead": float(lookahead), "control_rate": 20.0,
              "motion_timeout_s": float(timeout_s or 3600.0), "arrive_hold": bool(arrive_hold),
              "path_clearance_m": float(path_clearance), "turn_in_room": bool(turn_in_room)}  # fmt: skip
    speeds = Speed("real_nav", (float(speed), float(speed), float(yaw_speed)), 0.0, 0.0, 0.0)
    return embodiment().navigator(env, grid, speeds, arm_q=q, config=config, **kw)


class _ByRule:
    """Reachability for stances built by the start rule: the pick skills were trained from exactly these."""

    def is_reachable(self, base_pose, target_world) -> bool:
        return True


def drive(robot, env: RealEnv, goal: Pose2D, *, approach_m: float = 0.0, confirm=input, log_dir: Path | None = None,
          log_path: Path | None = None, quiet: bool = False, preflight=None) -> dict:  # fmt: skip
    """robot.go_to_pose(goal) on the real robot, with the checks before it, Enter first (`confirm`), Ctrl-C to stop,
    every tick logged (log_dir/leg00.jsonl.. or `log_path`), and a result dict: ok, reason, moved, final errors (by
    mocap), path, path_m, the legs' reasons, and the move's summary (summarize_move). `preflight`: a reason not to move,
    or None."""
    from chefbot.interface.types import MotionStatus

    loc, raw = env.localization(), env.raw_state()
    problems = [] if loc.live else ["no live mocap pose"]
    if not raw.get("base_driver_live"):
        problems.append("the base driver is not running (bringup/base.sh)")
    if not problems and preflight is not None:
        problems += [p for p in [preflight(loc.mocap_pose)] if p]
    if problems:
        return {"ok": False, "reason": "; ".join(problems), "moved": False}
    start = loc.mocap_pose
    d0, turn = math.hypot(goal.x - start.x, goal.y - start.y), math.degrees(wrap(goal.yaw - start.yaw))
    if not quiet:
        print(f"from  x {start.x:.3f}  y {start.y:.3f}  yaw {math.degrees(start.yaw):7.2f}")
        print(f"to    x {goal.x:.3f}  y {goal.y:.3f}  yaw {math.degrees(goal.yaw):7.2f}   ({d0:.3f} m, turn {turn:+.1f} deg)")
    try:
        confirm("Floor clear and e-stop in hand? Enter to drive, Ctrl-C to cancel: ")
    except (KeyboardInterrupt, EOFError):
        return {"ok": False, "reason": "cancelled", "moved": False}
    rows: list[dict] = []
    t0, wall0 = time.monotonic(), time.time()

    def log(row):
        row = dict(row, t=time.monotonic() - t0, wall=time.time(), server_t=env.raw_state().get("t"))
        if row.get("off_line") is None:  # coming to rest: the path deviation as it last was (as the old logs had)
            row["off_line"] = next((r["off_line"] for r in reversed(rows) if r.get("off_line") is not None), 0.0)
        rows.append(row)
        if not quiet and row["phase"] == "drive":
            print(f"\r{row['t']:5.1f}s  to go {row['dist']:6.3f} m  yaw {row['yaw_err_deg']:+6.1f} deg  "
                  f"off path {row.get('off_line') or 0.0:5.3f} m", end="", flush=True)  # fmt: skip

    robot.drive_log = log
    robot.go_to_pose(goal, approach_m)
    try:
        res = robot.wait()
    except KeyboardInterrupt:
        robot.stop()
        res = robot.wait()
        res.message = "stopped (Ctrl-C)"
    finally:
        robot.drive_log = None
        if not quiet:
            print()
    ok = res.status == MotionStatus.SUCCESS
    reason = "reached" if ok else res.message
    if res.status == MotionStatus.NOT_LOCALIZED:
        reason = f"lost the live mocap pose ({res.message})"
    elif res.status == MotionStatus.NO_PATH and not res.message.startswith("no path"):
        reason = f"no path: {res.message}"
    legs = sorted({r["leg"] for r in rows})
    for leg in legs:
        path = (Path(log_dir) / f"leg{leg:02d}.jsonl") if log_dir is not None else (log_path if leg == 0 else None)
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("".join(json.dumps(r, default=float) + "\n" for r in rows if r["leg"] == leg))
    result = {"ok": ok, "reason": reason, "status": res.status, "moved": bool(rows), "wall_start": wall0,
              "start": [start.x, start.y, start.yaw], "goal": [goal.x, goal.y, goal.yaw],
              "legs": [{"leg": leg, "reason": reason if leg == legs[-1] else "reached"} for leg in legs],
              "path": res.data.get("path") or [], "path_m": res.data.get("path_m") or 0.0, "seconds": time.monotonic() - t0,
              "distance_m": d0, "turn_deg": turn}  # fmt: skip
    end = env.localization().mocap_pose
    if end is not None:
        result.update(final_error_m=math.hypot(goal.x - end.x, goal.y - end.y),
                      final_yaw_error_deg=math.degrees(wrap(goal.yaw - end.yaw)))  # fmt: skip
    result.update(summarize_move([dict(r, off_line=r.get("off_line") or 0.0) for r in rows], start, goal))
    drive_rows, hold = [r for r in rows if r["phase"] == "drive"], [r for r in rows if r["phase"] == "hold"]
    if drive_rows and hold:
        result["moved_after_hold_m"] = float(np.linalg.norm(np.array(hold[-1]["pose"][:2]) - drive_rows[-1]["pose"][:2]))
    return result


def go(env: RealEnv, goal: Pose2D, *, speed: float = 0.1, yaw_speed: float = 0.3, tol: float = 0.03,
       yaw_tol_deg: float = 3.0, corridor: float = 0.15, max_dist: float = 1.5, timeout_s: float | None = None,
       confirm=input, log_path: Path | None = None, fence: Fence | None = None, quiet: bool = False,
       settle_s: float = 0.5, **_) -> dict:  # fmt: skip
    """Straight to `goal` (world) on open floor (no map): Robot(RealEnv).go_to_pose on a map with nothing on it, the
    target within `max_dist`; with a `fence`, a target or start outside it is refused and the base's centre may not
    leave it (plus 5 cm)."""
    loc, raw = env.localization(), env.raw_state()
    problems = [] if loc.live else ["no live mocap pose"]
    if not raw.get("base_driver_live"):
        problems.append("the base driver is not running (bringup/base.sh)")
    if problems:
        return {"ok": False, "reason": "; ".join(problems), "moved": False}
    start = loc.mocap_pose
    d0 = math.hypot(goal.x - start.x, goal.y - start.y)
    if d0 > max_dist:
        return {"ok": False, "reason": f"the target is {d0:.2f} m away (> --max-dist {max_dist} m)", "moved": False}
    if fence is not None and not (fence.contains(goal.x, goal.y) and fence.contains(start.x, start.y, 0.05)):
        return {"ok": False, "reason": "the start or the target is outside the fence", "moved": False}
    grid = open_grid([(start.x, start.y), (goal.x, goal.y)])
    robot = make_robot(env, grid, speed=speed, yaw_speed=yaw_speed, tol=tol, yaw_tol_deg=yaw_tol_deg, corridor=corridor,
                       margin=0.0, settle_s=settle_s, timeout_s=timeout_s,
                       centre_limit=None if fence is None else fence_cells(fence, grid))  # fmt: skip
    return drive(robot, env, goal, confirm=confirm, log_path=log_path, quiet=quiet)


def go_planned(env: RealEnv, goal: Pose2D, grid, *, robot=None, margin: float = 0.10, confirm=input,
               log_dir: Path | None = None, approach_m: float = 0.0, arm: str = "auto", **robot_args) -> dict:  # fmt: skip
    """A* on `grid` to `goal` and the path driven: Robot(RealEnv).go_to_pose, its footprint kept
    `margin` clear of the map at the heading the base will have along the path (the robot layer plans, drives and
    stops)."""
    robot = robot or make_robot(env, grid, margin=margin, arm=arm, **robot_args)
    return drive(robot, env, goal, approach_m=approach_m, confirm=confirm, log_dir=log_dir, quiet=True)


def map_robot(env: RealEnv, args, grid, **kw):
    """make_robot for --map, with the run's speeds and tolerances, its footprint and the map's heading zones said."""
    _, info = robot_footprints(env.raw_state(), args.arm)
    print(f"footprint: plate {info['plate']['x']} x {info['plate']['y']} m about the base centre ({info['note']}); "
          f"reach {info['radius_m']:.2f} m, {info['reach_ahead_m']:.2f} m ahead")  # fmt: skip
    robot = make_robot(env, grid, speed=args.speed, yaw_speed=args.yaw_speed, tol=args.tol, yaw_tol_deg=args.yaw_tol,
                       corridor=args.corridor, margin=args.margin, arm=args.arm, timeout_s=args.timeout, **kw)  # fmt: skip
    for _, allowed in grid.heading_zones:  # the map's (OccupancyGrid.load), held by the robot layer
        print(f"heading zone: facing {[round(math.degrees(y)) for y, _ in allowed]} deg only")
    return robot


def add_go_args(g) -> None:
    g.add_argument("--profile", default=None,
                   help="speed profile (embodiments/speed_profiles.yaml; default: the robot's control.speed); "
                        "one over the robot's base cap (robot.yaml control.limits) is refused")
    g.add_argument("--speed", type=float, default=None, help="m/s, at most the profile's, more is refused (default: the profile's)")
    g.add_argument("--yaw-speed", type=float, default=None, help="rad/s, at most the profile's, more is refused (default: the profile's)")
    g.add_argument("--tol", type=float, default=0.03, help="m")
    g.add_argument("--yaw-tol", type=float, default=3.0, help="deg")
    g.add_argument("--corridor", type=float, default=0.15, help="m off the straight line A-B before stopping")
    g.add_argument("--max-dist", type=float, default=1.5, help="m: refuse targets further away")
    g.add_argument("--timeout", type=float, default=None, help="s (default from the distance and speed)")
    g.add_argument("--map", type=Path, default=None, help="a map folder (maps/README.md): its places, and A* around its obstacles")
    g.add_argument("--margin", type=float, default=0.10, help="m added to the footprint for planning (with --map)")
    g.add_argument("--arm", choices=("auto", "off"), default="auto", help="grow the footprint to cover the arm (with --map)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mark")
    m.add_argument("name")
    m.add_argument("--map", type=Path, required=True, help="the map folder the place goes in (its marked.yaml)")
    ls = sub.add_parser("list")
    ls.add_argument("--map", type=Path, required=True)
    q = sub.add_parser("point")
    q.add_argument("name")
    q.add_argument("--map", type=Path, required=True)
    q.add_argument("--from", dest="origin", required=True)
    q.add_argument("--forward", type=float, default=0.0, help="m along the origin's heading (the way the arm faces)")
    q.add_argument("--left", type=float, default=0.0, help="m to the origin's left")
    q.add_argument("--turn", type=float, default=0.0, help="deg, counter-clockwise")
    g = sub.add_parser("go")
    g.add_argument("name")
    g.add_argument("--no-turn-in-room", action="store_true",
                   help="with --map: turn evenly along the path (the default turns where it has room: a short path from "
                        "beside a counter can't turn round as it starts)")
    g.add_argument("--no-hold", action="store_true",
                   help="with --map: keep re-aiming at the goal by mocap once inside the tolerances (the default holds "
                        "it in odometry there: re-aiming on mocap's millimetres oscillated a few mm and 1 deg)")
    g.add_argument("--straight", action="store_true",
                   help="straight there on open floor (no A*, the map's obstacles not used): within --max-dist only")
    add_go_args(g)
    sub.add_parser("arm")
    w = sub.add_parser("where")
    w.add_argument("--map", type=Path, required=True)
    w.add_argument("--margin", type=float, default=0.10)
    w.add_argument("--arm", choices=("auto", "off"), default="auto")
    args = ap.parse_args()

    if args.cmd == "point":
        from chefbot.environments.real.build_map import save_mark

        points = load_points(args.map)
        if args.origin not in points:
            raise SystemExit(f"no place {args.origin!r} in {args.map}")
        p = relative_point(points[args.origin], args.forward, args.left, args.turn) | {"from": args.origin}
        save_mark(args.map, args.name, p)
        print(f"{args.name}: x {p['x']:.4f}  y {p['y']:.4f}  yaw {p['yaw_deg']:.2f}  ({args.forward} m forward, {args.left} m left, "
              f"{args.turn} deg of {args.origin})")  # fmt: skip
        return
    if args.cmd == "list":
        for name, p in load_points(args.map).items():
            print(f"{name:10} x {p['x']:.4f}  y {p['y']:.4f}  yaw {p['yaw_deg']:7.2f}  ({p.get('marked', 'built')})")
        return
    if args.cmd == "go":
        args.speed, args.yaw_speed, args.profile = base_speeds(embodiment(), args.profile, args.speed, args.yaw_speed)
        print(f"speed profile {args.profile}: {args.speed:.3f} m/s, {args.yaw_speed:.3f} rad/s")
    env = RealEnv.from_config(np.eye(4), url=args.url or load_config(CONFIG)["url"], require_arm=False)  # arm poses are never sent
    if args.cmd == "go":
        try:
            check_server_cap(env.server_limits, args.speed, args.yaw_speed)
        except ValueError as exc:
            env.close()
            raise SystemExit(f"refusing: {exc}") from None
    t0 = time.monotonic()  # the server needs two mocap frames to call the pose live
    while not env.localization().live and time.monotonic() - t0 < 3.0:
        time.sleep(0.05)
    try:
        if args.cmd == "mark":
            mark(env, args.name, args.map)
            return
        if args.cmd == "arm":
            print(json.dumps(robot_footprints(env.raw_state())[1], indent=2))
            return
        if args.cmd == "where":
            print(json.dumps(where(env, args.map, args.margin, args.arm), indent=2))
            return
        if args.map is None:
            raise SystemExit("go needs --map: its places are where it goes (real_nav mark NAME --map DIR adds one)")
        from chefbot.scripts.real_map import load as load_map

        grid, points = load_map(args.map)
        if args.name not in points:
            raise SystemExit(f"no place {args.name!r} in {args.map}: mark it first ({', '.join(points) or 'none yet'})")
        p = points[args.name]
        stamp = f"{datetime.now():%Y%m%d_%H%M%S}_{args.name}"
        if not args.straight:
            log_dir = REPO / "runs" / "real_nav" / stamp
            log_dir.mkdir(parents=True, exist_ok=True)
            robot = map_robot(env, args, grid, turn_in_room=not args.no_turn_in_room, arrive_hold=not args.no_hold)
            result = go_planned(env, Pose2D(p["x"], p["y"], math.radians(p["yaw_deg"])), grid, robot=robot, log_dir=log_dir,
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
