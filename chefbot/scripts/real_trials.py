"""Random A* trials inside a fenced square, around imaginary obstacles, with every logged pose checked for violations.

    python -m chefbot.scripts.real_trials --sim --half 1.2 --trials 30 --seed 1          # dry run: onboard/sim.py
    python -m chefbot.scripts.real_trials --half 0.2 --scale 0.25 --trials 8 --seed 1    # the robot (base.sh + tunnel)

Each session: the fence is a square of half-width --half about where the base starts (C), along its heading (the
base CENTRE stays inside; set --half for the floor you have). --obstacles imaginary boxes are placed at random
inside it (nothing is really there). Then --trials random goals (free, at least --min-move away, a random heading
within --max-turn), each reached with A* (scripts/real_nav.go_planned) from wherever the base is.

The footprint is the base box plus any arm boxes, in the base frame (x forward, the way the arm faces):
    --base-box 0.56 0.50          length, width of the base, centred on the base centre (m)
    --arm-box 0.30 0.20 0.25      length, width, and how far ahead of the base centre the box's centre is (repeatable)
--scale shrinks the footprint and the obstacles together, so a trial fits a small real square: the obstacles are
imaginary anyway, and the fence (not scaled) is what keeps the real base on its floor.

Planning keeps the circle enclosing the footprint, plus --margin, away from obstacles; the check then tests the
real (rotated) footprint boxes against the obstacles at every logged pose, and the centre against the fence.
Any violation, or a goal not reached, is reported; on the robot the session stops at the first one.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from chefbot.envs.real.env import CONFIG, RealEnv, load_config
from chefbot.interface.types import Pose2D
from chefbot.robot.nav.grid import OccupancyGrid
from chefbot.scripts.real_map import box_polygon, inside, render
from chefbot.scripts.real_nav import go_planned

REPO = Path(__file__).resolve().parents[2]


# -- geometry ---------------------------------------------------------------------------


def footprint_boxes(base_box, arm_boxes=(), scale: float = 1.0) -> list[dict]:
    """Rectangles in the base frame: {center: (x, y), size: (length, width)}."""
    boxes = [{"center": (0.0, 0.0), "size": (base_box[0] * scale, base_box[1] * scale)}]
    boxes += [{"center": (a[2] * scale, 0.0), "size": (a[0] * scale, a[1] * scale)} for a in arm_boxes]
    return boxes


def enclosing_radius(boxes) -> float:
    return max(math.hypot(abs(b["center"][0]) + b["size"][0] / 2, abs(b["center"][1]) + b["size"][1] / 2) for b in boxes)


def placed(boxes, pose) -> list[list[tuple[float, float]]]:
    """The footprint boxes as world polygons with the base at `pose` (x, y, yaw)."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for b in boxes:
        bx, by = b["center"]
        out.append(box_polygon((x + c * bx - s * by, y + s * bx + c * by), b["size"], math.degrees(yaw)))
    return out


def convex_overlap(p, q) -> bool:
    """Separating-axis test for two convex polygons."""
    for poly in (p, q):
        pts = np.asarray(poly)
        for a, b in zip(pts, np.roll(pts, -1, axis=0)):
            n = np.array([a[1] - b[1], b[0] - a[0]])
            pp, qq = np.asarray(p) @ n, np.asarray(q) @ n
            if pp.max() < qq.min() or qq.max() < pp.min():
                return False
    return True


def point_polygon_distance(pt, poly) -> float:
    pts = np.asarray(poly)
    if inside(np.asarray([pt]), pts)[0]:
        return 0.0
    best = math.inf
    for a, b in zip(pts, np.roll(pts, -1, axis=0)):
        ab, ap = b - a, np.asarray(pt) - a
        t = np.clip(ap @ ab / max(ab @ ab, 1e-12), 0.0, 1.0)
        best = min(best, float(np.linalg.norm(ap - t * ab)))
    return best


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


def check_poses(poses, boxes, obstacles, fence) -> dict:
    """Violations along logged poses: footprint boxes overlapping an obstacle; the centre outside the fence."""
    hits, outside_fence, clearance = 0, 0, math.inf
    r = enclosing_radius(boxes)
    for pose in poses:
        if not inside(np.asarray([pose[:2]]), np.asarray(fence))[0]:
            outside_fence += 1
        for o in obstacles:
            clearance = min(clearance, point_polygon_distance(pose[:2], o) - r)
            if any(convex_overlap(fp, o) for fp in placed(boxes, pose)):
                hits += 1
                break
    return {"footprint_hits": hits, "outside_fence": outside_fence, "min_circle_clearance_m": clearance, "poses": len(poses)}


def sample_goal(grid_plan: OccupancyGrid, here: Pose2D, c: Pose2D, half: float, min_move: float, max_turn_deg: float, rng,
                detour: float = 0.5, max_move: float = 1.2):  # fmt: skip
    """A free goal min_move..max_move away. With probability `detour` it must be one the straight line cannot reach
    (an obstacle in the way), so A* has to plan around; falls back to any free goal if none is found."""
    want_blocked = rng.random() < detour
    fallback = None
    for _ in range(2000):
        u, v = rng.uniform(-half, half, 2)
        x, y = c.x + math.cos(c.yaw) * u - math.sin(c.yaw) * v, c.y + math.sin(c.yaw) * u + math.cos(c.yaw) * v
        d = math.hypot(x - here.x, y - here.y)
        if not grid_plan.is_free(x, y) or not min_move <= d <= max_move:
            continue
        goal = Pose2D(x, y, here.yaw + math.radians(rng.uniform(-max_turn_deg, max_turn_deg)))
        blocked = not grid_plan.segment_free((here.x, here.y), (x, y))
        if blocked == want_blocked:
            return goal
        fallback = fallback or goal
    return fallback


def run_session(env: RealEnv, *, half: float, trials: int, seed: int, obstacles: int, box: float, boxes, margin: float,
                min_move: float, max_turn_deg: float, speed: float, yaw_speed: float, out: Path, confirm=input,
                stop_on_violation: bool = True, detour: float = 0.5) -> dict:  # fmt: skip
    rng = np.random.default_rng(seed)
    loc = env.localization()
    if not loc.live:
        raise SystemExit("no live mocap pose")
    c = loc.mocap_pose
    radius = enclosing_radius(boxes)
    fence, obs, obstacle_grid, outside = make_world(c, half, obstacles, box, radius + margin, rng)
    plan_grid = combine(obstacle_grid.inflate(radius + margin), outside)
    body_grid = combine(obstacle_grid.inflate(radius), outside)
    grid = combine(obstacle_grid, outside)
    out.mkdir(parents=True, exist_ok=True)
    setup = {"seed": seed, "start": [c.x, c.y, c.yaw], "half": half, "fence": fence, "obstacles": obs, "footprint": boxes,
             "enclosing_radius": radius, "margin": margin, "started": datetime.now().isoformat()}  # fmt: skip
    (out / "setup.json").write_text(json.dumps(setup, indent=2))
    print(f"C ({c.x:.3f}, {c.y:.3f}, {math.degrees(c.yaw):.1f}); fence +-{half} m; {len(obs)} imaginary obstacle(s) of {box:.2f} m; "
          f"footprint circle {radius:.3f} m + margin {margin:.2f} m")  # fmt: skip
    try:
        confirm(f"{trials} random A* trials. Floor clear inside the fence and e-stop in hand? Enter to start, Ctrl-C to stop: ")
    except (KeyboardInterrupt, EOFError):
        return {"trials": [], "cancelled": True}
    results, paths = [], []
    for i in range(trials):
        here = env.localization().mocap_pose
        goal = sample_goal(plan_grid, here, c, half, min_move, max_turn_deg, rng, detour=detour)
        if goal is None:
            print(f"[{i + 1}/{trials}] no reachable goal sampled; stopping")
            break
        tdir = out / f"trial{i:02d}"
        try:
            r = go_planned(env, goal, grid, footprint=radius, margin=margin, confirm=lambda _: None, log_dir=tdir,
                           plan_grid=plan_grid, body_grid=body_grid, speed=speed, yaw_speed=yaw_speed)  # fmt: skip
        except KeyboardInterrupt:
            r = {"ok": False, "reason": "stopped (Ctrl-C)", "moved": True}
        poses = [json.loads(line)["pose"] for leg in sorted(tdir.glob("leg*.jsonl")) for line in leg.read_text().splitlines()]
        v = check_poses(poses, boxes, obs, fence)
        r = {k: val for k, val in r.items() if k != "legs"} | v | {"trial": i, "goal": [goal.x, goal.y, goal.yaw],
             "straight_line_blocked": not plan_grid.segment_free((here.x, here.y), (goal.x, goal.y))}  # fmt: skip
        results.append(r)
        if r.get("path"):
            paths.append(r["path"])
        bad = v["footprint_hits"] or v["outside_fence"]
        print(f"[{i + 1}/{trials}] {r['reason']:9} " + (f"{1000 * (r.get('final_error_m') or 0):5.1f} mm  " if r.get("final_error_m") is not None else "")
              + f"path {r.get('path_m', 0):.2f} m  clearance {v['min_circle_clearance_m'] * 1000:6.0f} mm  "
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
        "min_circle_clearance_m": min((r["min_circle_clearance_m"] for r in results), default=None),
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
    ap.add_argument("--base-box", type=float, nargs=2, default=[0.56, 0.50], metavar=("LENGTH", "WIDTH"))
    ap.add_argument("--arm-box", type=float, nargs=3, action="append", default=[], metavar=("LENGTH", "WIDTH", "AHEAD"))
    ap.add_argument("--scale", type=float, default=1.0, help="shrink the footprint and the obstacles (small real squares)")
    ap.add_argument("--margin", type=float, default=None, help="m added to the footprint circle (default 0.10 x --scale)")
    ap.add_argument("--min-move", type=float, default=0.08)
    ap.add_argument("--max-turn", type=float, default=90.0, help="deg")
    ap.add_argument("--detour", type=float, default=0.5, help="share of goals with an obstacle in the straight way")
    ap.add_argument("--speed", type=float, default=0.1)
    ap.add_argument("--yaw-speed", type=float, default=0.3)
    args = ap.parse_args()
    if not args.sim and not (0 < args.half <= 0.5 and args.speed <= 0.12 and args.yaw_speed <= 0.4):
        raise SystemExit("on the robot: --half <= 0.5 m, --speed <= 0.12 m/s, --yaw-speed <= 0.4 rad/s")
    boxes = footprint_boxes(args.base_box, args.arm_box, args.scale)
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
    env = RealEnv.from_config(np.eye(4), url=url)
    try:
        import time

        t0 = time.monotonic()
        while not env.localization().live and time.monotonic() - t0 < 5:
            time.sleep(0.05)
        res = run_session(env, half=args.half, trials=args.trials, seed=args.seed, obstacles=args.obstacles, box=box,
                          boxes=boxes, margin=margin, min_move=args.min_move, max_turn_deg=args.max_turn, speed=args.speed,
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
