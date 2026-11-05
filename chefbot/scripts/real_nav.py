"""Point A to point B on the real base, by mocap, through the onboard server: the first autonomous moves.

    ssh -L 8765:localhost:8765 tidybot01               # with chefbot/onboard/bringup/base.sh running there
    python -m chefbot.scripts.real_nav mark A          # the base where you put it, standing still: saved as A
    python -m chefbot.scripts.real_nav mark B
    python -m chefbot.scripts.real_nav point S1 --from A --forward 0.3 [--left 0] [--turn 0]   # relative to A's heading
    python -m chefbot.scripts.real_nav list
    python -m chefbot.scripts.real_nav go B [--speed 0.1] [--yaw-speed 0.3]     # Enter to start, Ctrl-C stops
    python -m chefbot.scripts.real_nav go stove --map maps/src_kitchen     # A* around the map's obstacles
    python -m chefbot.scripts.real_nav arm                                 # the arm's outline over the floor, now

With --map (a map folder, maps/README.md), the target may also be one of the map's places, and the base goes there by A*,
keeping its footprint --margin (0.10 m, for the map's measuring error) clear of the map's obstacles at the heading it
will have at each point of the path, and stops if its footprint ever reaches one. The footprint is the robot's
measured body (embodiments/tidybot_fr3/robot.yaml `body:`, interface/workspace.py Body.footprint): the plate's
outline (0.56 x 0.48 m about the base centre; corners 0.37 m out), turned with the base (robot/nav/footprint.py).

With the arm driver running (or its last joint readings in Redis), --arm auto (the default) adds the arm, from its
joint angles (interface/workspace.py arm_points, 18 cm ahead of the base centre on its plate); against a map's low
obstacles only the arm's points low enough to hit them count.
And every `go` stops if the arm moves more than 0.05 rad on any joint while the base drives.

Points are base poses in the mocap world (x, y in m, yaw in deg), in runs/real_nav/points.json.

This script is thin: every move is the robot layer's Robot(RealEnv).go_to_pose (make_robot), the same planning,
drive loop and stops as the agent's go_to and the sims (robot/robot.py, robot/safety.py BaseGuard). The robot layer
ties the base's odometry to mocap (robot/localization.py) and sends targets in odometry each control tick (20 Hz), so
mocap closes the loop. Without --map, `go` drives straight on a map with nothing on it: the floor between the points
must be clear, and B within --max-dist. Before starting it checks that mocap and the base driver are live
(bringup/base.sh). While driving the base stops (held) on: lost mocap, the base driver gone or a halt from the server,
the footprint reaching the map, more than --corridor off the path, a straight move more than 0.10 m further from B
than it has been, the arm moving, --timeout, or Ctrl-C; its speed is held to what it can stop in within the free
space ahead (the robot's measured stop, from the onboard server). Reached: within --tol and --yaw-tol for 0.5 s. The
onboard server and the driver have their own stops behind these (0.3 s and 0.5 s without commands). Every control
tick goes to runs/real_nav/<time>_<point>.jsonl (with --map: a folder, one file per leg).
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




LOW_CLEARANCE_M = 0.05  # an arm point passes over a low obstacle only with its underside this much above its top


def robot_footprints(raw: dict, arm: str = "auto"):
    """The robot's footprint from its measured body (embodiments/tidybot_fr3/robot.yaml `body:`, interface/workspace.py),
    as a function of an obstacle's top: the plate's outline, plus the arm's points (from its joints, with --arm auto and
    readings to go on) whose underside is lower than the top + LOW_CLEARANCE_M (all of them for inf, a wall). Returns
    (footprint_below, info)."""
    from chefbot.interface.workspace import load_body

    body = load_body()
    q = raw.get("arm_q") if arm == "auto" else None
    known = q is not None and len(q) == 7
    q = q if known else None

    def footprint_below(top: float = math.inf):
        return body.footprint(q, below=None if math.isinf(top) else top + LOW_CLEARANCE_M)

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


def make_robot(env: RealEnv, grid, *, speed: float = 0.1, yaw_speed: float = 0.3, tol: float = 0.03,
               yaw_tol_deg: float = 3.0, corridor: float = 0.15, margin: float = 0.10, arm: str = "auto",
               settle_s: float = 0.5, still_mps: float | None = None, lookahead: float = 0.12,
               timeout_s: float | None = None, centre_limit=None, footprint_below=None, stop_model="robot", accel=None,
               arrive_hold: bool = False):  # fmt: skip
    """The robot layer on the real robot: Robot(RealEnv) on `grid` with the robot's footprint (robot_footprints), its
    measured stop and acceleration as the onboard server reports them (stop_model="robot"; None: none), the run's
    speeds and tolerances. The base drives with the arm as it is (tuck_arm=False: real_nav never moves the arm; the
    footprint covers it and the arm moving stops the drive). Every stop is the robot layer's (robot/safety.py BaseGuard
    and Robot's drive): lost mocap, the footprint on the map, the fence, the arm moving, off the path (corridor),
    moving away, the timeout, a halt from the server."""
    from dataclasses import replace

    from chefbot import embodiments
    from chefbot.robot.nav.speed import StopModel
    from chefbot.robot.robot import Robot

    env.require_arm = False  # base moves need no arm driver (its last readings, if any, still give its footprint)
    cfg = replace(embodiments.load("tidybot_fr3").config, max_lin_vel=float(speed), max_ang_vel=float(yaw_speed),
                  pos_tolerance=float(tol), yaw_tolerance=math.radians(yaw_tol_deg), deviation_threshold=float(corridor),
                  safety_margin=float(margin), snap_radius=0.0, arrive_settle_s=float(settle_s), arrive_still_mps=still_mps,
                  lookahead=float(lookahead), control_rate=20.0, motion_timeout_s=float(timeout_s or 3600.0),
                  arrive_hold=bool(arrive_hold))  # fmt: skip
    if footprint_below is None:
        footprint_below = robot_footprints(env.raw_state(), arm)[0]
    limits = env.server_limits
    if "base_max_vel_xy" in limits:
        check_server_cap(limits, float(speed), float(yaw_speed))  # over the robot's cap: refused (ValueError), not clipped
    if stop_model == "robot":
        stop_model = StopModel.from_limits(limits) if "base_stop_tau_s" in limits else None
    accel = float(limits.get("base_max_accel_xy", 0.1)) if accel is None else float(accel)
    return Robot(env, grid, cfg, _ByRule(), footprint_below=footprint_below, stop_model=stop_model, base_accel=accel,
                 centre_limit=centre_limit, tuck_arm=False)  # fmt: skip


def drive(robot, env: RealEnv, goal: Pose2D, *, approach_m: float = 0.0, confirm=input, log_dir: Path | None = None,
          log_path: Path | None = None, quiet: bool = False, preflight=None, via=None) -> dict:  # fmt: skip
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
    if via:
        robot.go_through(via, goal)  # one smooth drive through the via poses
    else:
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
               log_dir: Path | None = None, approach_m: float = 0.0, arm: str = "auto", via=None, **robot_args) -> dict:  # fmt: skip
    """A* on `grid` to `goal` and the path driven: Robot(RealEnv).go_to_pose (or go_through `via`), its footprint kept
    `margin` clear of the map at the heading the base will have along the path (the robot layer plans, drives and
    stops)."""
    robot = robot or make_robot(env, grid, margin=margin, arm=arm, **robot_args)
    return drive(robot, env, goal, approach_m=approach_m, confirm=confirm, log_dir=log_dir, quiet=True, via=via)


def counter_box(map_dir: Path, name: str | None = None) -> tuple:
    """(centre, size, yaw rad, name) of a box obstacle in the map's spec (the first one unless named)."""
    import yaml

    spec = yaml.safe_load((Path(map_dir) / "spec.yaml").read_text())
    for o in spec.get("obstacles") or []:
        if "box" in o and (name is None or o["name"] == name):
            b = o["box"]
            return tuple(b["center"]), tuple(b["size"]), math.radians(b.get("yaw_deg", 0.0)), o["name"]
    raise SystemExit(f"no box obstacle {name or ''} in {map_dir}/spec.yaml")


def random_targets(box, n: int, seed: int, sides=None, x_range=None) -> list[tuple[float, float]]:
    """Targets drawn as the pick task draws them: a side, a spot along it >= 0.35 m from its ends, 0.10-0.16 m in.
    `sides`: only these (south / north / east / west), e.g. a counter against a wall reached from one side."""
    from chefbot.robot.nav.goal import PARK_END_MARGIN_M, TRAINED_INSET_M, counter_faces

    rng = np.random.default_rng(seed)
    faces = [f for f in counter_faces(*box[:3]) if not sides or f["name"] in sides]
    if not faces:
        raise SystemExit(f"no counter side among {sides}")
    out, tries = [], 0
    while len(out) < n:  # `x_range` (world x min, max): only targets in this stretch, e.g. one section of a kitchen
        tries += 1
        if tries > 1000 * max(n, 1):
            raise SystemExit(f"no targets on {sides or 'any side'} within x {x_range}")
        f = faces[int(rng.integers(len(faces)))]
        along = rng.uniform(-1, 1) * (f["half_along"] - PARK_END_MARGIN_M)
        inset = rng.uniform(*TRAINED_INSET_M)
        p = np.asarray(box[0], dtype=float) + f["t"] * along + f["n"] * (f["half_n"] - inset)
        if x_range and not (x_range[0] <= p[0] <= x_range[1]):
            continue
        out.append((round(float(p[0]), 3), round(float(p[1]), 3)))
    return out


class _ByRule:
    """Reachability for stances built by the start rule: the pick skills were trained from exactly these."""

    def is_reachable(self, base_pose, target_world) -> bool:
        return True


def choose_stance(robot, box, target, out_m, approach: float):
    """The stance for `target` by the start rule (robot/nav/goal.py counter_park_candidates) that the robot layer finds
    clear for the whole footprint at its own heading, with its straight approach clear, and cheapest to reach from
    where the base is (Robot.choose_stance): (pose, info) or (None, reason)."""
    from chefbot.robot.nav.goal import counter_park_candidates

    cands = counter_park_candidates(*box[:3], target, out_m=tuple(out_m or (0.45, 0.48)))
    if not cands:
        return None, "the target is not 0-0.25 m in from a side and >= 0.35 m from its ends: no stance by the rule"
    body = robot._space
    ok = [(p, i) for p, i in cands
          if body.at(p.yaw).segment_free((p.x - approach * math.cos(p.yaw), p.y - approach * math.sin(p.yaw)), (p.x, p.y))]  # fmt: skip
    pick, why = robot.choose_stance([p for p, _ in ok])
    if pick is None:
        return None, why
    return pick, next(i for p, i in ok if p is pick)


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
    targets = [tuple(t) for t in args.target or []] + random_targets(box, args.random, args.seed, args.sides, args.x_range)
    boxes = [box] * len(targets)
    if args.alternate:  # back and forth between two counters: A, B, A, B ... (B's targets from its own side(s))
        box_b = counter_box(args.map, args.alternate[0])
        tb = random_targets(box_b, args.random, args.seed + 1000, args.alternate[1:], args.x_range)
        ta = targets[len(targets) - args.random:]
        mixed, mixed_boxes = [tuple(t) for t in args.target or []], [box] * len(args.target or [])
        for a, b in zip(ta, tb):
            mixed += [a, b]
            mixed_boxes += [box, box_b]
        targets, boxes = mixed, mixed_boxes
    if not targets:
        raise SystemExit("no targets: --target X Y and/or --random N")
    from chefbot.scripts.real_map import load as load_map

    grid, _ = load_map(args.map)
    from chefbot.robot.nav.speed import StopModel

    stop_model = StopModel.from_limits(env.server_limits)  # refuses when the robot reports no measured stop
    accel = float(env.server_limits.get("base_max_accel_xy", 0.1))
    robot = map_robot(env, args, grid, settle_s=0.2, still_mps=0.01, arrive_hold=args.hold)
    print(f"stop model: tau {stop_model.tau_s:.3f} s, brake {stop_model.brake_mps2:.2f} m/s^2 "
          f"({1000 * stop_model.distance(args.speed):.0f} mm from {args.speed:.3f} m/s); driver accel {accel} m/s^2")
    session = REPO / "runs" / "real_park" / f"{datetime.now():%Y%m%d_%H%M%S}"
    session.mkdir(parents=True, exist_ok=True)
    (session / "session.json").write_text(json.dumps({"targets": targets, "args": {k: (str(v) if isinstance(v, Path) else v)
                                                      for k, v in vars(args).items()}}, indent=2))  # fmt: skip
    results = []
    route = None
    if args.around:
        import yaml

        route = yaml.safe_load((Path(args.map) / "routes.yaml").read_text())["island_north_south"]
    for i, target in enumerate(targets):
        box = boxes[i]
        via = None
        if route is not None:
            here = env.localization().mocap_pose
            side_now = "north" if here.y > box[0][1] else "south"
            side_to = "north" if target[1] > box[0][1] else "south"
            if side_now != side_to:  # one smooth drive through the gap's middle, the heading kept until after it
                wps = route["north"] + route["gap"] + route["south"]
                wps = wps if side_now == "north" else wps[::-1]
                via = [Pose2D(wx, wy, here.yaw) for wx, wy in wps]
        if via is not None:
            from chefbot.robot.nav.goal import counter_park_candidates

            plan_space = robot._space.with_margin(args.margin)
            clear = [(p, inf) for p, inf in counter_park_candidates(*box[:3], target, out_m=tuple(args.out or (0.45, 0.48)))
                     if plan_space.is_free(p.x, p.y, p.yaw)]  # fmt: skip
            pose, info = clear[0] if clear else (None, "no stance clear for the footprint")
            if pose is not None:
                print(f"[{i + 1}/{len(targets)}] around the island, {side_now} to {side_to}, through {len(via)} points without stopping")
        else:
            pose, info = choose_stance(robot, box, target, args.out, args.approach)
        head = f"[{i + 1}/{len(targets)}] target ({target[0]:.3f}, {target[1]:.3f})"
        if pose is None and info == "no stance reachable from here" and not args.plan_only:
            # A path that needs a turn where the base may not turn (a heading zone, a tight aisle): turn in place to
            # the stance's heading first, where the base is (if it may), then plan again.
            from chefbot.robot.nav.goal import counter_park_candidates

            cands = counter_park_candidates(*box[:3], target, out_m=tuple(args.out or (0.45, 0.48)))
            here = env.localization().mocap_pose
            if cands and here is not None:
                yaw = cands[0][0].yaw
                # Back away from what the base faces (into the aisle) and turn there; else turn where it is.
                for back in (0.35, 0.0):
                    turn = Pose2D(here.x - back * math.cos(here.yaw), here.y - back * math.sin(here.yaw), yaw)
                    print(f"{head}: no path with the turn on the way; {'backing ' + format(back, '.2f') + ' m and ' if back else ''}"
                          f"turning to {math.degrees(yaw):.0f} deg first")
                    tlog = session / f"{i:02d}_turn_first_{int(back * 100)}"
                    tlog.mkdir(parents=True, exist_ok=True)
                    rt = go_planned(env, turn, grid, robot=robot, log_dir=tlog, confirm=lambda _: None)
                    print(f"    turn: {rt['reason']}")
                    if rt.get("ok"):
                        pose, info = choose_stance(robot, box, target, args.out, args.approach)
                        break
                    if rt.get("moved"):
                        break  # it moved and stopped: don't try another
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
        log_dir.mkdir(parents=True, exist_ok=True)  # go_planned may refuse before it makes it
        t0 = time.monotonic()
        r = go_planned(env, pose, grid, robot=robot, log_dir=log_dir, confirm=lambda _: None, approach_m=args.approach, via=via)
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


def heading_zones(map_dir: Path, grid) -> list:
    """The map's heading zones (map_dir/heading_zones.yaml, optional): [(cells grid, [(yaw rad, tolerance rad)])],
    the cells the base centre's positions inside each zone's box on `grid`."""
    import yaml

    from chefbot.robot.nav.grid import OccupancyGrid

    path = Path(map_dir) / "heading_zones.yaml"
    if not path.exists():
        return []
    out = []
    h, w = grid.shape
    ys, xs = np.mgrid[0:h, 0:w]
    ox, oy = grid.origin[0], grid.origin[1]
    wx, wy = ox + (xs + 0.5) * grid.resolution, oy + (ys + 0.5) * grid.resolution
    for z in yaml.safe_load(path.read_text()).get("zones") or []:
        (cx, cy), (sx, sy), t = z["box"]["center"], z["box"]["size"], math.radians(z["box"].get("yaw_deg", 0.0))
        a = (wx - cx) * math.cos(t) + (wy - cy) * math.sin(t)
        b = -(wx - cx) * math.sin(t) + (wy - cy) * math.cos(t)
        cells = OccupancyGrid((np.abs(a) <= sx / 2) & (np.abs(b) <= sy / 2), grid.resolution, grid.origin)
        tol = math.radians(float(z.get("tolerance_deg", 20.0)))
        out.append((cells, [(math.radians(float(y)), tol) for y in z["allowed_yaw_deg"]]))
        print(f"heading zone {z.get('name', '?')}: facing {z['allowed_yaw_deg']} deg (+-{z.get('tolerance_deg', 20.0)}) only")
    return out


def map_robot(env: RealEnv, args, grid, **kw):
    """make_robot for --map, with the run's speeds and tolerances, its footprint said, and the map's heading zones."""
    _, info = robot_footprints(env.raw_state(), args.arm)
    print(f"footprint: plate {info['plate']['x']} x {info['plate']['y']} m about the base centre ({info['note']}); "
          f"reach {info['radius_m']:.2f} m, {info['reach_ahead_m']:.2f} m ahead")  # fmt: skip
    robot = make_robot(env, grid, speed=args.speed, yaw_speed=args.yaw_speed, tol=args.tol, yaw_tol_deg=args.yaw_tol,
                       corridor=args.corridor, margin=args.margin, arm=args.arm, timeout_s=args.timeout, **kw)  # fmt: skip
    zones = heading_zones(args.map, grid) if getattr(args, "map", None) else []
    if zones:
        robot.set_heading_zones(zones)
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
    g.add_argument("--map", type=Path, default=None, help="a map folder (maps/README.md): plan around its obstacles")
    g.add_argument("--margin", type=float, default=0.10, help="m added to the footprint for planning (with --map)")
    g.add_argument("--arm", choices=("auto", "off"), default="auto", help="grow the footprint to cover the arm (with --map)")


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
    k.add_argument("--sides", nargs="+", choices=("south", "north", "east", "west"), default=None,
                   help="draw --random targets on these counter sides only (a counter against a wall)")
    k.add_argument("--around", action="store_true",
                   help="between the island's sides, go round by the map's routes.yaml waypoints (keeping the heading)")
    k.add_argument("--hold", action="store_true",
                   help="once inside --tol / --yaw-tol, hold the base where it is (no re-aiming for millimetres)")
    k.add_argument("--alternate", nargs="+", default=None, metavar=("COUNTER", "SIDE"),
                   help="alternate --random targets with as many on this second counter (its name, then its side(s))")
    k.add_argument("--x-range", nargs=2, type=float, default=None, metavar=("XMIN", "XMAX"),
                   help="draw --random targets only within this world-x stretch (one section of a kitchen)")
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

        args.speed, args.yaw_speed, args.profile = base_speeds(embodiments.load("tidybot_fr3"), args.profile, args.speed, args.yaw_speed)
        print(f"speed profile {args.profile}: {args.speed:.3f} m/s, {args.yaw_speed:.3f} rad/s")
    env = RealEnv.from_config(np.eye(4), url=args.url or load_config(CONFIG)["url"])  # arm poses are never sent
    if args.cmd in ("go", "park"):
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
            mark(env, args.name)
            return
        if args.cmd == "arm":
            print(json.dumps(robot_footprints(env.raw_state())[2], indent=2))
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
            robot = map_robot(env, args, grid)
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
