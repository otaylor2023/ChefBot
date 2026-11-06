"""Build the map files for the real kitchen from a short hand-written spec, in the mocap (Motive) world frame: the real
world's twin of environments/robocasa/build_map.py (NAV_SPEC §11). Writes the repo's map format (maps/README.md):

  grid.npy + grid.yaml   occupancy grid for the robot layer (row along +y, col along +x)
  heights.npy            each occupied cell's top (m; inf: walls, and obstacles given no height)
  semantic_map.yaml      fixtures / objects with rough locations for the agent (blank when the spec names none)
  places.yaml            named base poses (optional part of the format: the spec's places and stations)

    python -m chefbot.environments.real.build_map build maps/src_counter/spec.yaml   # -> the files beside it
    python -m chefbot.environments.real.build_map body probe.log Chef-botcounter-top # an obstacle from a tracked body

`body` reads a log of bringup/mocap.sh run without --quiet (it prints Motive's rigid-body descriptions and every
frame): the body's marker offsets and its last tracked pose give the obstacle, as a box covering the markers. Put
markers on the corners of a counter, make them a rigid body in Motive, and the counter maps itself; when the markers
are off, the saved spec keeps the last measurement.

The spec (all in metres, world frame, as real_pose / real_nav print poses):

    resolution: 0.05
    bounds: [[xmin, ymin], [xmax, ymax]]          # the floor the base may use; everything outside is a wall
    obstacles:                                    # counters, cabinets, tables, walls, a box on the floor
      - {name: counter_north, box: {center: [x, y], size: [length, depth], yaw_deg: 0}}   # no height: a wall
      - {name: island, polygon: [[x, y], [x, y], [x, y], ...]}
      - {name: counter, box: {...}, height_m: 0.93, fixture: counter}   # its top; `fixture`: on the semantic map
    places:                                       # base poses to go to by name (real_nav go <name> --map ...)
      stove: {x: 1.5, y: -3.9, yaw_deg: -90}
    stations:                                     # places generated square to a box obstacle's sides:
      - {obstacle: counter, prefix: counter, standoff: 0.45, approach: 0.25}
        # -> counter_south / _north / _east / _west: the base centre `standoff` out from the middle of each edge,
        #    facing it squarely; real_nav goes the last `approach` m straight in along the edge's normal

Obstacles are the furniture itself; the robot layer grows them by the robot's footprint at each heading, and decides
from the heights which parts of the robot pass over what (robot/nav/footprint.py). Measure corners with a tape from a
mocap-tracked point, or later with a tracked wand; a few cm is fine, the margin covers it.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import yaml

GRID_FILE, GRID_META_FILE, HEIGHTS_FILE = "grid.npy", "grid.yaml", "heights.npy"
SEMANTIC_MAP_FILE, PLACES_FILE, MARKED_FILE = "semantic_map.yaml", "places.yaml", "marked.yaml"
BLANK_SEMANTIC_MAP = {"frame": "world", "fixtures": {}, "objects": {}}


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


def build_grid(spec: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """(occupied[row=y, col=x], heights, metadata): outside the bounds a wall (inf), each obstacle at its height_m
    (none given: inf). Where obstacles overlap, the taller top."""
    res = float(spec.get("resolution", 0.05))
    (xmin, ymin), (xmax, ymax) = spec["bounds"]
    pad = 2 * res  # a ring of occupied cells around the bounds
    origin = (xmin - pad, ymin - pad)
    cols = int(math.ceil((xmax - xmin + 2 * pad) / res))
    rows = int(math.ceil((ymax - ymin + 2 * pad) / res))
    cc, rr = np.meshgrid(np.arange(cols), np.arange(rows))
    centres = np.column_stack([origin[0] + (cc.ravel() + 0.5) * res, origin[1] + (rr.ravel() + 0.5) * res])
    outside = ~((centres[:, 0] >= xmin) & (centres[:, 0] <= xmax) & (centres[:, 1] >= ymin) & (centres[:, 1] <= ymax))
    heights = np.where(outside, np.inf, 0.0)
    for o in spec.get("obstacles") or []:
        poly = np.asarray(obstacle_polygon(o))
        top = math.inf if o.get("height_m") is None else float(o["height_m"])
        hit = inside(centres, poly)
        # Thin obstacles (a wall drawn as a line-thin box) still block: mark the cells their corners fall in.
        for x, y in poly:
            r, c = int((y - origin[1]) // res), int((x - origin[0]) // res)
            if 0 <= r < rows and 0 <= c < cols:
                hit[r * cols + c] = True
        heights[hit] = np.maximum(heights[hit], top)
    heights = heights.reshape(rows, cols)
    meta = {"resolution": res, "origin": [float(origin[0]), float(origin[1])], "frame": "world"}
    return heights > 0.0, heights, meta


def build_places(spec: dict) -> dict:
    places = {name: {"x": float(p["x"]), "y": float(p["y"]), "yaw_deg": float(p.get("yaw_deg", 0.0))}
              for name, p in (spec.get("places") or {}).items()}  # fmt: skip
    return places | stations(spec)


def build_semantic_map(spec: dict) -> dict:
    """Obstacles the spec names a `fixture` for, at their centre (NAV_SPEC §7.1); blank when there are none."""
    out = {"frame": "world", "fixtures": {}, "objects": {}}
    for o in spec.get("obstacles") or []:
        if o.get("fixture"):
            xy = np.asarray(obstacle_polygon(o), dtype=float).mean(axis=0)
            out["fixtures"][o["fixture"]] = {"location": [round(float(xy[0]), 3), round(float(xy[1]), 3)]}
    return out


def build_map(spec: dict, out_dir: str | Path) -> Path:
    """Write grid.npy, grid.yaml, heights.npy, semantic_map.yaml and places.yaml; returns the folder."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    occupied, heights, meta = build_grid(spec)
    np.save(out_dir / GRID_FILE, occupied)
    np.save(out_dir / HEIGHTS_FILE, heights)
    (out_dir / GRID_META_FILE).write_text(yaml.safe_dump(meta))
    (out_dir / SEMANTIC_MAP_FILE).write_text(yaml.safe_dump(build_semantic_map(spec), sort_keys=False, default_flow_style=None))
    (out_dir / PLACES_FILE).write_text(yaml.safe_dump(build_places(spec), sort_keys=True))
    return out_dir


def load_places(map_dir: str | Path) -> dict:
    """A map's named base poses: places.yaml (built from the spec) and marked.yaml (marked on the robot, real_nav mark;
    a mark wins over a built place of the same name), {} when it has none."""
    out = {}
    for name in (PLACES_FILE, MARKED_FILE):
        path = Path(map_dir) / name
        if path.exists():
            out |= yaml.safe_load(path.read_text()) or {}
    return out


def save_mark(map_dir: str | Path, name: str, place: dict) -> Path:
    """A marked base pose ({x, y, yaw_deg, ...}) into the map's marked.yaml (kept when the map is rebuilt from its
    spec, which rewrites places.yaml)."""
    path = Path(map_dir) / MARKED_FILE
    marks = (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}
    marks[name] = place
    path.write_text(yaml.safe_dump(marks, sort_keys=False))
    return path


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
    args = ap.parse_args()
    if args.cmd == "body":
        b = body_from_probe(args.probe.read_text(), args.name)
        print(yaml.safe_dump({"obstacles": [{"name": b["name"], "box": {k: [round(v, 3) for v in val] if isinstance(val, list) else round(val, 2)
                                                                          for k, val in b["box"].items()}}]}, sort_keys=False))  # fmt: skip
        print(f"# from {len(b['markers_body'])} markers; pose x {b['pose'][0]:.3f} y {b['pose'][1]:.3f} z {b['height_m']:.3f}, yaw {b['yaw_deg']:.2f} deg")
        return
    spec = yaml.safe_load(args.spec.read_text())
    out = build_map(spec, args.spec.parent)
    occupied = np.load(out / GRID_FILE)
    tops = sorted({float(t) for t in np.unique(np.load(out / HEIGHTS_FILE)[occupied])})
    print(f"{occupied.shape[0]} x {occupied.shape[1]} cells at {spec.get('resolution', 0.05)} m, {100 * (1 - occupied.mean()):.0f}% free; "
          f"obstacle tops {tops}; {len(load_places(out))} places -> {out}")


if __name__ == "__main__":
    main()
