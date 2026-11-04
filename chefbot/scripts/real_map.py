"""A map of the real kitchen, in the mocap (Motive) world frame, from a short hand-written description.

    python -m chefbot.scripts.real_map build maps/src_kitchen/spec.yaml      # -> grid.npy, grid.yaml, places.yaml beside it
    python -m chefbot.scripts.real_map show maps/src_kitchen [--footprint 0.33 --margin 0.10] [--from X Y --to X Y]
    python -m chefbot.scripts.real_map body probe.log Chef-botcounter-top      # an obstacle from a tracked rigid body

`body` reads a log of bringup/mocap.sh run without --quiet (it prints Motive's rigid-body descriptions and every
frame): the body's marker offsets and its last tracked pose give the obstacle, as a box covering the markers. Put
markers on the corners of a counter, make them a rigid body in Motive, and the counter maps itself; when the
markers are off, the saved spec keeps the last measurement.

The spec (all in metres, world frame, as real_pose / real_nav print poses):

    resolution: 0.05
    bounds: [[xmin, ymin], [xmax, ymax]]          # the floor the base may use; everything outside is occupied
    obstacles:                                    # counters, cabinets, tables, walls, a box on the floor
      - {name: counter_north, box: {center: [x, y], size: [length, depth], yaw_deg: 0}}
      - {name: island, polygon: [[x, y], [x, y], [x, y], ...]}
      - {name: counter, box: {...}, height_m: 0.93}  # with a height: a LOW obstacle the arm may reach over
    places:                                       # base poses to go to by name (real_nav go <name> --map ...)
      stove: {x: 1.5, y: -3.9, yaw_deg: -90}
    stations:                                     # places generated square to a box obstacle's sides:
      - {obstacle: counter, prefix: counter, standoff: 0.45, approach: 0.25}
        # -> counter_south / _north / _east / _west: the base centre `standoff` out from the middle of each edge,
        #    facing it squarely; real_nav goes the last `approach` m straight in along the edge's normal

The output is the repo's map format (robot/nav OccupancyGrid: grid.npy + grid.yaml, row along +y, col along +x),
so robot/nav's A* plans on it unchanged. Obstacles are the furniture itself; the planner grows them by the robot's
footprint (real_nav: --footprint and --margin). Measure corners with a tape from a mocap-tracked point, or later
with a tracked wand; a few cm is fine, the margin covers it.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import yaml

from chefbot.robot.nav.astar import AStarPlanner, NoPath, path_length
from chefbot.robot.nav.grid import OccupancyGrid


def box_polygon(center, size, yaw_deg: float = 0.0) -> list[tuple[float, float]]:
    cx, cy = center
    hl, hw = size[0] / 2, size[1] / 2
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return [(cx + c * x - s * y, cy + s * x + c * y) for x, y in ((-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw))]


def inside(points: np.ndarray, polygon) -> np.ndarray:
    """Even-odd rule, vectorized over (N, 2) points."""
    poly = np.asarray(polygon, dtype=float)
    x, y = points[:, 0], points[:, 1]
    result = np.zeros(len(points), dtype=bool)
    for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, axis=0)):
        crosses = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        result ^= crosses & (x < xi)
    return result


def stations(spec: dict) -> dict:
    """Base poses square to the sides of box obstacles (spec `stations`), named by the side's outward direction in the
    world (+x east, +y north): the base centre `standoff` m out from the middle of the edge, facing it."""
    out = {}
    boxes = {o["name"]: o["box"] for o in spec.get("obstacles") or [] if "box" in o}
    for st in spec.get("stations") or []:
        b = boxes[st["obstacle"]]
        (cx, cy), (L, W), yaw = b["center"], b["size"], math.radians(b.get("yaw_deg", 0.0))
        c, s_ = math.cos(yaw), math.sin(yaw)
        for along, across, half in ((1, 0, L / 2), (-1, 0, L / 2), (0, 1, W / 2), (0, -1, W / 2)):
            nx, ny = c * along - s_ * across, s_ * along + c * across  # the edge's outward normal, world
            d = half + float(st["standoff"])
            heading = math.degrees(math.atan2(-ny, -nx))  # facing the edge
            side = ("east" if nx > 0 else "west") if abs(nx) >= abs(ny) else ("north" if ny > 0 else "south")
            out[f"{st.get('prefix', st['obstacle'])}_{side}"] = {
                "x": cx + nx * d, "y": cy + ny * d, "yaw_deg": round(heading, 6), "approach_m": float(st.get("approach", 0.0))}
    return out


def obstacle_polygon(o: dict) -> list:
    if "box" in o:
        b = o["box"]
        return box_polygon(b["center"], b["size"], b.get("yaw_deg", 0.0))
    return [tuple(p) for p in o["polygon"]]


def build(spec: dict) -> tuple[OccupancyGrid, dict]:
    """The occupancy grid (bounds' outside and every obstacle occupied) and the places."""
    tall, low, low_height, places = build_layers(spec)
    return OccupancyGrid(tall.occupied | low.occupied, tall.resolution, tall.origin), places


def build_layers(spec: dict):
    """Two grids: TALL (outside the bounds = walls, and obstacles without a height) and LOW (obstacles with a
    `height_m`, e.g. a counter top), the low obstacles' greatest height, and the places. The arm may reach over a low
    obstacle when every part of it there is higher (scripts/real_nav.py), never over a tall one."""
    res = float(spec.get("resolution", 0.05))
    (xmin, ymin), (xmax, ymax) = spec["bounds"]
    pad = 2 * res  # a ring of occupied cells around the bounds
    origin = (xmin - pad, ymin - pad)
    cols = int(math.ceil((xmax - xmin + 2 * pad) / res))
    rows = int(math.ceil((ymax - ymin + 2 * pad) / res))
    cc, rr = np.meshgrid(np.arange(cols), np.arange(rows))
    centres = np.column_stack([origin[0] + (cc.ravel() + 0.5) * res, origin[1] + (rr.ravel() + 0.5) * res])
    tall = ~((centres[:, 0] >= xmin) & (centres[:, 0] <= xmax) & (centres[:, 1] >= ymin) & (centres[:, 1] <= ymax))
    low = np.zeros(len(centres), dtype=bool)
    low_height = 0.0
    for o in spec.get("obstacles") or []:
        poly = np.asarray(obstacle_polygon(o))
        layer = low if o.get("height_m") is not None else tall
        if o.get("height_m") is not None:
            low_height = max(low_height, float(o["height_m"]))
        layer |= inside(centres, poly)
        # Thin obstacles (a wall drawn as a line-thin box) still block: mark the cells their corners fall in.
        for x, y in poly:
            r, c = int((y - origin[1]) // res), int((x - origin[0]) // res)
            if 0 <= r < rows and 0 <= c < cols:
                layer[r * cols + c] = True
    places = {name: {"x": float(p["x"]), "y": float(p["y"]), "yaw_deg": float(p.get("yaw_deg", 0.0))}
              for name, p in (spec.get("places") or {}).items()}  # fmt: skip
    places |= stations(spec)
    shape = (rows, cols)
    return OccupancyGrid(tall.reshape(shape), res, origin), OccupancyGrid(low.reshape(shape), res, origin), low_height, places


def load(map_dir: Path) -> tuple[OccupancyGrid, dict]:
    map_dir = Path(map_dir)
    places = yaml.safe_load((map_dir / "places.yaml").read_text()) if (map_dir / "places.yaml").exists() else {}
    return OccupancyGrid.load(map_dir / "grid.npy"), places or {}


def load_layers(map_dir: Path):
    """(tall, low, low_height) if the map has layers (built from a spec with heights), else None."""
    map_dir = Path(map_dir)
    if not (map_dir / "grid_tall.npy").exists():
        return None
    meta = yaml.safe_load((map_dir / "grid_low.yaml").read_text())
    return OccupancyGrid.load(map_dir / "grid_tall.npy"), OccupancyGrid.load(map_dir / "grid_low.npy"), float(meta.get("height_m", 0.0))


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


def body_from_probe(text: str, name: str) -> dict:
    """The rigid body `name` in a mocap.sh probe log: its marker offsets (body frame), last tracked pose, and the
    floor box covering its markers in the world.

    Motive's descriptions give the offsets in its internal y-up convention (here: x and z spread, y ~ 0), while the
    stream is z up. Converting, the floor axes are x and +-z: which sign is not settled from this data (a mirror),
    so the box covers the markers both ways (for near-symmetric corner markers, a few cm more)."""
    lines = text.splitlines()
    try:
        i = next(k for k, line in enumerate(lines) if line.strip() == f"RigidBody Name : {name}")
    except StopIteration:
        raise SystemExit(f"no rigid body {name!r} in the log (names: "
                         + ", ".join(sorted({l.split(':', 1)[1].strip() for l in lines if l.startswith('RigidBody Name')})) + ")")
    offsets = []
    for line in lines[i + 1 :]:
        t = line.strip()
        if t.startswith("Position:"):
            offsets.append([float(v) for v in t.split(":", 1)[1].split(",")])
        elif t.startswith("Data Description") or t.startswith("RigidBody Name"):
            break
    pose = None
    for k, line in enumerate(lines):
        if line.startswith(f"{name} [") and "Tracked=1" in line and k + 2 < len(lines):
            pose = [float(v) for v in lines[k + 2].split()]
    if pose is None:
        raise SystemExit(f"{name} was never tracked in the log")
    x, y, z, qx, qy, qz, qw = pose
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    O = np.asarray(offsets)
    along = float(np.abs(O[:, 0]).max())
    across = float(np.abs(O[:, 2]).max())  # +-z: covers both readings of the mirror
    return {"name": name, "markers_body": offsets, "pose": pose, "yaw_deg": math.degrees(yaw),
            "box": {"center": [x, y], "size": [2 * along, 2 * across], "yaw_deg": math.degrees(yaw)},
            "height_m": z}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("spec", type=Path)
    o = sub.add_parser("body")
    o.add_argument("probe", type=Path)
    o.add_argument("name")
    s = sub.add_parser("show")
    s.add_argument("map_dir", type=Path)
    s.add_argument("--footprint", type=float, default=0.33)
    s.add_argument("--margin", type=float, default=0.10)
    s.add_argument("--from", dest="start", type=float, nargs=2)
    s.add_argument("--to", dest="goal", type=float, nargs=2)
    args = ap.parse_args()
    if args.cmd == "body":
        b = body_from_probe(args.probe.read_text(), args.name)
        print(yaml.safe_dump({"obstacles": [{"name": b["name"], "box": {k: [round(v, 3) for v in val] if isinstance(val, list) else round(val, 2)
                                                                          for k, val in b["box"].items()}}]}, sort_keys=False))  # fmt: skip
        print(f"# from {len(b['markers_body'])} markers; pose x {b['pose'][0]:.3f} y {b['pose'][1]:.3f} z {b['height_m']:.3f}, yaw {b['yaw_deg']:.2f} deg")
        return
    if args.cmd == "build":
        spec = yaml.safe_load(args.spec.read_text())
        grid, places = build(spec)
        tall, low, low_height, _ = build_layers(spec)
        out = args.spec.parent
        grid.save(out / "grid.npy", out / "grid.yaml")
        tall.save(out / "grid_tall.npy", out / "grid_tall.yaml")
        low.save(out / "grid_low.npy", out / "grid_low.yaml")
        meta = yaml.safe_load((out / "grid_low.yaml").read_text()) | {"height_m": low_height}
        (out / "grid_low.yaml").write_text(yaml.safe_dump(meta))
        (out / "places.yaml").write_text(yaml.safe_dump(places, sort_keys=True))
        free = 100 * (1 - grid.occupied.mean())
        print(f"{grid.shape[0]} x {grid.shape[1]} cells at {grid.resolution} m, {free:.0f}% free; {len(places)} places -> {out}")
        return
    grid, places = load(args.map_dir)
    inflated = grid.inflate(args.footprint + args.margin)
    path, marks = None, {name: (p["x"], p["y"]) for name, p in places.items()}
    if args.start and args.goal:
        try:
            path = AStarPlanner(inflated).plan_xy(tuple(args.start), tuple(args.goal))
            print(f"path {path_length(path):.2f} m, {len(path) - 1} segments: " + " -> ".join(f"({x:.2f}, {y:.2f})" for x, y in path))
        except NoPath as exc:
            print(f"no path: {exc}")
        marks |= {"S": tuple(args.start), "G": tuple(args.goal)}
    print(render(grid, inflated, path, marks))
    print(f"'#' obstacle, '+' within {args.footprint + args.margin:.2f} m (footprint + margin) of one, '*' path; +y up")


if __name__ == "__main__":
    main()
