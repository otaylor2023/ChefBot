"""Random A* trials inside a fenced square, around imaginary obstacles, with every logged pose checked for violations.

    python -m chefbot.scripts.real_trials --sim --half 1.2 --trials 30 --seed 1          # dry run: onboard/sim.py
    python -m chefbot.scripts.real_trials --half 0.2 --scale 0.25 --trials 8 --seed 1    # the robot (base.sh + tunnel)

Each session: the fence is a square of half-width --half about where the base starts (C), along its heading (the
base CENTRE stays inside; set --half for the floor you have). --obstacles imaginary boxes are placed at random
inside it (nothing is really there). Then --trials random goals (free, at least --min-move away, a random heading
within --max-turn), each reached with A* (scripts/real_nav.go_planned: Robot(RealEnv).go_to_pose) from wherever the base is.

The footprint is the robot's measured body (embodiments/tidybot_fr3/robot.yaml `body:`, interface/workspace.py
Body.footprint): the plate's outline about the base centre, plus with --arm-from-robot the arm from its current joints.
--scale shrinks the footprint and the obstacles together, so a trial fits a small real square: the obstacles are
imaginary anyway, and the fence (not scaled) is what keeps the real base on its floor.

Planning keeps the footprint, at the heading the base will have, --margin away from obstacles (real_nav.go_planned);
the check then measures the footprint, turned to the logged heading, against the obstacles at every logged pose, and
the centre against the fence.
Any violation, or a goal not reached, is reported; on the robot the session stops at the first one.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from chefbot.environments.real.env import CONFIG, RealEnv, load_config
from chefbot.interface.types import Pose2D
from chefbot.interface.workspace import Footprint
from chefbot.robot.nav.grid import OccupancyGrid
from chefbot.environments.real.build_map import box_polygon, inside
from chefbot.scripts.real_map import render
from chefbot.scripts.real_nav import go_planned

REPO = Path(__file__).resolve().parents[2]


# -- geometry ---------------------------------------------------------------------------


def scaled(fp: Footprint, k: float) -> Footprint:
    """The footprint shrunk (or grown) by k about the base centre."""
    return Footprint(tuple(v * k for v in fp.rect), np.asarray(fp.discs, dtype=float) * k)


def outline(fp: Footprint, pose) -> list[np.ndarray]:
    """Points that must stay off the obstacles with the base at `pose` (x, y, yaw), in the world: the plate's corners
    and the arm discs' centres."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    x0, x1, y0, y1 = fp.rect
    local = np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)] + [tuple(d[:2]) for d in fp.discs])
    return local @ np.array([[c, s], [-s, c]]) + np.array([x, y])


def clearance_at(fp: Footprint, pose, polygon, step: float = 0.005) -> float:
    """How far the footprint at `pose` is from an obstacle polygon (m; 0: touching or overlapping)."""
    pts = np.asarray(polygon, dtype=float)
    edge = [a + (b - a) * t for a, b in zip(pts, np.roll(pts, -1, axis=0))
            for t in np.linspace(0, 1, max(2, int(np.linalg.norm(b - a) / step) + 1))]  # fmt: skip
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    local = (np.asarray(edge) - np.array([x, y])) @ np.array([[c, -s], [s, c]])
    if inside(outline(fp, pose), pts).any():  # a corner or an arm point within the obstacle
        return 0.0
    return float(fp.distance(local).min())


# -- session ------------------------------------------------------------------------------


def make_world(c: Pose2D, half: float, n_obstacles: int, box: float, keep_clear: float, rng, res: float = 0.02):
    """The fence polygon, the obstacles (polygons), the obstacle grid and the outside-the-fence grid (separately:
    obstacles are grown by the footprint, the fence is not, as it limits the base centre)."""
    fence = box_polygon((c.x, c.y), (2 * half, 2 * half), math.degrees(c.yaw))
    obstacles = []
    tries = 0
    while len(obstacles) < n_obstacles and tries < 1000:
        tries += 1
        u, v = rng.uniform(-half + box, half - box, 2)
        cx, cy = c.x + math.cos(c.yaw) * u - math.sin(c.yaw) * v, c.y + math.sin(c.yaw) * u + math.cos(c.yaw) * v
        if math.hypot(cx - c.x, cy - c.y) < keep_clear + box:  # not on top of the start
            continue
        obstacles.append(box_polygon((cx, cy), (box, box), math.degrees(c.yaw) + rng.uniform(0, 90)))
    xs, ys = [p[0] for p in fence], [p[1] for p in fence]
    origin = (min(xs) - 2 * res, min(ys) - 2 * res)
    cols, rows = int(math.ceil((max(xs) - min(xs)) / res)) + 4, int(math.ceil((max(ys) - min(ys)) / res)) + 4
    cc, rr = np.meshgrid(np.arange(cols), np.arange(rows))
    centres = np.column_stack([origin[0] + (cc.ravel() + 0.5) * res, origin[1] + (rr.ravel() + 0.5) * res])
    outside = ~inside(centres, fence)
    occupied = np.zeros(len(centres), dtype=bool)
    for o in obstacles:
        occupied |= inside(centres, o)
    shape = (rows, cols)
    return fence, obstacles, OccupancyGrid(occupied.reshape(shape), res, origin), OccupancyGrid(outside.reshape(shape), res, origin)


def combine(a: OccupancyGrid, b: OccupancyGrid) -> OccupancyGrid:
    return OccupancyGrid(a.occupied | b.occupied, a.resolution, a.origin)


def check_poses(poses, footprint: Footprint, obstacles, fence) -> dict:
    """Violations along logged poses: the footprint (turned with the base) reaching an obstacle; the centre outside
    the fence. Also the least clearance between the footprint and any obstacle."""
    hits, outside_fence, clearance = 0, 0, math.inf
    for pose in poses:
        if not inside(np.asarray([pose[:2]]), np.asarray(fence))[0]:
            outside_fence += 1
        gaps = [clearance_at(footprint, pose, o) for o in obstacles]
        clearance = min([clearance] + gaps)
        if any(g <= 0.0 for g in gaps):
            hits += 1
    return {"footprint_hits": hits, "outside_fence": outside_fence, "min_clearance_m": clearance, "poses": len(poses)}


def sample_goal(plan, here: Pose2D, c: Pose2D, half: float, min_move: float, max_turn_deg: float, rng,
                detour: float = 0.5, max_move: float = 1.2):  # fmt: skip
    """A free goal min_move..max_move away. With probability `detour` it must be one the straight line cannot reach
    (an obstacle in the way), so A* has to plan around; falls back to any free goal if none is found."""
    want_blocked = rng.random() < detour
    fallback = None
    for _ in range(2000):
        u, v = rng.uniform(-half, half, 2)
        x, y = c.x + math.cos(c.yaw) * u - math.sin(c.yaw) * v, c.y + math.sin(c.yaw) * u + math.cos(c.yaw) * v
        d = math.hypot(x - here.x, y - here.y)
        if not min_move <= d <= max_move:
            continue
        goal = Pose2D(x, y, here.yaw + math.radians(rng.uniform(-max_turn_deg, max_turn_deg)))
        if not plan.is_free(x, y, goal.yaw):
            continue
        blocked = not plan.path_clear([(here.x, here.y, here.yaw), (x, y, goal.yaw)])  # at the headings along it
        if blocked == want_blocked:
            return goal
        fallback = fallback or goal
    return fallback


def run_session(env: RealEnv, *, half: float, trials: int, seed: int, obstacles: int, box: float, footprint: Footprint, margin: float,
                min_move: float, max_turn_deg: float, speed: float, yaw_speed: float, out: Path, confirm=input,
                stop_on_violation: bool = True, detour: float = 0.5) -> dict:  # fmt: skip
    rng = np.random.default_rng(seed)
    loc = env.localization()
    if not loc.live:
        raise SystemExit("no live mocap pose")
    c = loc.mocap_pose
    from chefbot.robot.nav.footprint import FootprintSpace

    radius = footprint.radius
    fence, obs, obstacle_grid, outside = make_world(c, half, obstacles, box, radius + margin, rng)
    space = FootprintSpace(obstacle_grid, footprint, centre_limit=outside)  # the fence limits the centre only
    plan = space.with_margin(margin)
    plan_grid = plan.at(c.yaw)  # for the map drawing
    grid = combine(obstacle_grid, outside)
    out.mkdir(parents=True, exist_ok=True)
    setup = {"seed": seed, "start": [c.x, c.y, c.yaw], "half": half, "fence": fence, "obstacles": obs, "footprint": footprint.summary(),
             "enclosing_radius": radius, "margin": margin, "started": datetime.now().isoformat()}  # fmt: skip
    (out / "setup.json").write_text(json.dumps(setup, indent=2))
    print(f"C ({c.x:.3f}, {c.y:.3f}, {math.degrees(c.yaw):.1f}); fence +-{half} m; {len(obs)} imaginary obstacle(s) of {box:.2f} m; "
          f"footprint: plate and arm, reaching {radius:.3f} m, + margin {margin:.2f} m at each heading")  # fmt: skip
    try:
        confirm(f"{trials} random A* trials. Floor clear inside the fence and e-stop in hand? Enter to start, Ctrl-C to stop: ")
    except (KeyboardInterrupt, EOFError):
        return {"trials": [], "cancelled": True}
    results, paths = [], []
    for i in range(trials):
        here = env.localization().mocap_pose
        goal = sample_goal(plan, here, c, half, min_move, max_turn_deg, rng, detour=detour)
        if goal is None:
            print(f"[{i + 1}/{trials}] no reachable goal sampled; stopping")
            break
        tdir = out / f"trial{i:02d}"
        try:
            r = go_planned(env, goal, obstacle_grid, margin=margin, confirm=lambda _: None, log_dir=tdir, speed=speed,
                           yaw_speed=yaw_speed, footprint_below=lambda top=math.inf: footprint, centre_limit=outside,
                           stop_model=None)  # fmt: skip  (as before: on the robot <= 0.12 m/s, a short carrot)
        except KeyboardInterrupt:
            r = {"ok": False, "reason": "stopped (Ctrl-C)", "moved": True}
        poses = [json.loads(line)["pose"] for leg in sorted(tdir.glob("leg*.jsonl")) for line in leg.read_text().splitlines()]
        v = check_poses(poses, footprint, obs, fence)
        r = {k: val for k, val in r.items() if k != "legs"} | v | {"trial": i, "goal": [goal.x, goal.y, goal.yaw],
             "straight_line_blocked": not plan.path_clear([(here.x, here.y, here.yaw), (goal.x, goal.y, goal.yaw)])}  # fmt: skip
        results.append(r)
        if r.get("path"):
            paths.append(r["path"])
        bad = v["footprint_hits"] or v["outside_fence"]
        print(f"[{i + 1}/{trials}] {r['reason']:9} " + (f"{1000 * (r.get('final_error_m') or 0):5.1f} mm  " if r.get("final_error_m") is not None else "")
              + f"path {r.get('path_m', 0):.2f} m  clearance {v['min_clearance_m'] * 1000:6.0f} mm  "
              + ("VIOLATION" if bad else "ok"))  # fmt: skip
        (out / "results.jsonl").open("a").write(json.dumps(r, default=str) + "\n")
        if stop_on_violation and (bad or not r.get("ok")):
            print("session stopped")
            break
    summary = summarize(results)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    marks = {"C": (c.x, c.y)}
    all_path = [p for path in paths for p in path]
    (out / "map.txt").write_text(render(grid, plan_grid, all_path if all_path else None, marks))
    return {"trials": results, "summary": summary}


def summarize(results) -> dict:
    ok = [r for r in results if r.get("ok")]
    errs = [r["final_error_m"] for r in ok if r.get("final_error_m") is not None]
    return {
        "trials": len(results), "reached": len(ok),
        "violations": sum(1 for r in results if r["footprint_hits"] or r["outside_fence"]),
        "footprint_hits": sum(r["footprint_hits"] for r in results), "outside_fence": sum(r["outside_fence"] for r in results),
        "min_clearance_m": min((r["min_clearance_m"] for r in results), default=None),
        "final_error_mm_mean": 1000 * float(np.mean(errs)) if errs else None,
        "final_error_mm_max": 1000 * float(np.max(errs)) if errs else None,
        "segments_mean": float(np.mean([len(r["path"]) - 1 for r in results if r.get("path")])) if results else None,
        "detours": sum(1 for r in results if r.get("straight_line_blocked")),
        "detours_reached": sum(1 for r in results if r.get("straight_line_blocked") and r.get("ok")),
    }  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true", help="against onboard/sim.py's simulated base (no robot)")
    ap.add_argument("--url", default=None)
    ap.add_argument("--half", type=float, required=True, help="m the base centre may go from C, each way")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--obstacles", type=int, default=1)
    ap.add_argument("--box", type=float, default=None, help="m, side of an imaginary obstacle (default 0.3 x --scale)")
    ap.add_argument("--scale", type=float, default=1.0, help="shrink the footprint and the obstacles (small real squares)")
    ap.add_argument("--arm-from-robot", action="store_true", help="add the arm from its current joints (Body.footprint)")
    ap.add_argument("--margin", type=float, default=None, help="m kept beyond the footprint (default 0.10 x --scale)")
    ap.add_argument("--min-move", type=float, default=0.08)
    ap.add_argument("--max-turn", type=float, default=90.0, help="deg")
    ap.add_argument("--detour", type=float, default=0.5, help="share of goals with an obstacle in the straight way")
    ap.add_argument("--speed", type=float, default=0.1)
    ap.add_argument("--yaw-speed", type=float, default=0.3)
    args = ap.parse_args()
    if not args.sim and not (0 < args.half <= 0.5 and args.speed <= 0.12 and args.yaw_speed <= 0.4):
        raise SystemExit("on the robot: --half <= 0.5 m, --speed <= 0.12 m/s, --yaw-speed <= 0.4 rad/s")
    box = args.box if args.box is not None else 0.3 * args.scale
    margin = args.margin if args.margin is not None else 0.10 * args.scale
    out = REPO / "runs" / "trials" / (datetime.now().strftime("%Y%m%d_%H%M%S") + ("_sim" if args.sim else ""))
    sim = None
    if args.sim:
        # The simulated robot runs as its own process (laptop code does not import chefbot.onboard: it is deployed alone).
        import subprocess
        import sys

        sim = subprocess.Popen([sys.executable, "-m", "chefbot.onboard.sim", "--start", "0.285", "0.276", "179.6",
                                "--seed", str(args.seed)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)  # fmt: skip
        url = sim.stdout.readline().strip()
    else:
        url = args.url or load_config(CONFIG)["url"]
    env = RealEnv.from_config(np.eye(4), url=url, require_arm=False)
    from chefbot.interface.workspace import load_body

    q = None
    if args.arm_from_robot:
        q = env.raw_state().get("arm_q")
        if q is None:
            raise SystemExit("--arm-from-robot: no arm joint readings from the robot")
    footprint = scaled(load_body().footprint(q), args.scale)
    print(f"footprint (x {args.scale}): {footprint.summary()}")
    try:
        import time

        t0 = time.monotonic()
        while not env.localization().live and time.monotonic() - t0 < 5:
            time.sleep(0.05)
        res = run_session(env, half=args.half, trials=args.trials, seed=args.seed, obstacles=args.obstacles, box=box,
                          footprint=footprint, margin=margin, min_move=args.min_move, max_turn_deg=args.max_turn, speed=args.speed,
                          yaw_speed=args.yaw_speed, out=out, confirm=(lambda _: None) if args.sim else input,
                          stop_on_violation=not args.sim, detour=args.detour)  # fmt: skip
    finally:
        env.close()
        if sim:
            sim.stdin.close()
            sim.wait(timeout=10)
    print(json.dumps(res.get("summary", {}), indent=2))
    print(f"logs: {out}   (map with every path: {out / 'map.txt'})")


if __name__ == "__main__":
    main()
