"""Show a map (maps/README.md) and where the base's centre may be on it, with an A* path if asked.

    python -m chefbot.scripts.real_map show maps/src_counter [--yaw 90 --margin 0.10] [--from X Y --to X Y]

The map itself is built env-side: environments/real/build_map.py (the real kitchen, from a spec), or
environments/robocasa/build_map.py. '+' is where the base's centre may not be at heading --yaw: its footprint (the
plate's outline, and the arm's points low enough to hit each obstacle, from the map's heights) within --margin of an
obstacle (real_nav.nav_space).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from chefbot.environments.real.build_map import load_places
from chefbot.robot.nav.astar import AStarPlanner, NoPath, path_length
from chefbot.robot.nav.grid import OccupancyGrid


def build(spec: dict) -> tuple[OccupancyGrid, dict]:
    """A spec's grid (with heights) and places in memory, as environments/real/build_map.py would write them."""
    from chefbot.environments.real.build_map import build_grid, build_places

    occupied, heights, meta = build_grid(spec)
    return OccupancyGrid(occupied, meta["resolution"], tuple(meta["origin"]), heights), build_places(spec)


def load(map_dir: Path) -> tuple[OccupancyGrid, dict]:
    """A map folder's grid (with its heights) and its places."""
    map_dir = Path(map_dir)
    return OccupancyGrid.load(map_dir / "grid.npy"), load_places(map_dir)


def render(grid: OccupancyGrid, inflated: OccupancyGrid | None = None, path=None, marks: dict | None = None) -> str:
    """ASCII, +y up: '#' obstacle, '+' too close for the base centre, '.' free, '*' path, letters for marks."""
    chars = np.where(grid.occupied, "#", ".").astype("<U1")
    if inflated is not None:
        chars[(inflated.occupied) & (~grid.occupied)] = "+"
    if path:
        for (ax, ay), (bx, by) in zip(path, path[1:]):
            n = max(1, int(math.hypot(bx - ax, by - ay) / (grid.resolution / 2)))
            for i in range(n + 1):
                r, c = grid.world_to_cell(ax + (bx - ax) * i / n, ay + (by - ay) * i / n)
                if grid.in_bounds((r, c)):
                    chars[r, c] = "*"
    for label, (x, y) in (marks or {}).items():
        r, c = grid.world_to_cell(x, y)
        if grid.in_bounds((r, c)):
            chars[r, c] = label[0]
    return "\n".join("".join(row) for row in chars[::-1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show")
    s.add_argument("map_dir", type=Path)
    s.add_argument("--yaw", type=float, default=0.0, help="deg: the base's heading (its footprint turns with it)")
    s.add_argument("--margin", type=float, default=0.10)
    s.add_argument("--from", dest="start", type=float, nargs=2)
    s.add_argument("--to", dest="goal", type=float, nargs=2)
    args = ap.parse_args()
    grid, places = load(args.map_dir)
    from chefbot.scripts.real_nav import nav_space

    inflated = nav_space(grid, {}, arm="off")[0].with_margin(args.margin).at(math.radians(args.yaw))
    path, marks = None, {name: (p["x"], p["y"]) for name, p in places.items()}
    if args.start and args.goal:
        try:
            path = AStarPlanner(inflated).plan_xy(tuple(args.start), tuple(args.goal))
            print(f"path {path_length(path):.2f} m, {len(path) - 1} segments: " + " -> ".join(f"({x:.2f}, {y:.2f})" for x, y in path))
        except NoPath as exc:
            print(f"no path: {exc}")
        marks |= {"S": tuple(args.start), "G": tuple(args.goal)}
    print(render(grid, inflated, path, marks))
    print(f"'#' obstacle, '+' where the base centre at {args.yaw:.0f} deg would put its plate within {args.margin:.2f} m of one, "
          "'*' path; +y up")


if __name__ == "__main__":
    main()
