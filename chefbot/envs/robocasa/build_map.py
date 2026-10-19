"""Build the map files for a RoboCasa scene from simulator ground truth.

Writes, into one folder per layout/style:
  grid.npy + grid.yaml   occupancy grid for the robot layer
  semantic_map.yaml      fixtures / objects with rough locations for the agent

    python -m chefbot.envs.robocasa.build_map --layout 1 --style 1

With --start, writes semantic_map_start.yaml instead: the fixtures alone, each
with its footprint (`size`), and no objects. That is the map an exploring
agent starts from (chefbot/explore); the grid is written too.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import yaml

from chefbot.envs.robocasa.appliances import burner_position, burners
from chefbot.envs.robocasa.config import RoboCasaConfig
from chefbot.envs.robocasa.detector import FixtureBox, named_fixtures, numbered, scene_objects
from chefbot.envs.robocasa.env import RoboCasaEnv

GRID_FILE, GRID_META_FILE, SEMANTIC_MAP_FILE = "grid.npy", "grid.yaml", "semantic_map.yaml"
START_MAP_FILE = "semantic_map_start.yaml"
# How far in front of a fixture there must be open floor for a facing to count.
_FACING_CLEARANCE = 0.6


def _geom_box_points(env: RoboCasaEnv, geom: int, spacing: float) -> np.ndarray:
    """World-frame lattice of points filling a geom's bounding box."""
    center, half = env.model.geom_aabb[geom, :3], env.model.geom_aabb[geom, 3:]
    axes = [
        center[i] + np.linspace(-half[i], half[i], max(2, math.ceil(2 * half[i] / spacing) + 1))
        for i in range(3)
    ]
    local = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return local @ env.data.geom_xmat[geom].reshape(3, 3).T + env.data.geom_xpos[geom]


def build_grid(env: RoboCasaEnv) -> tuple[np.ndarray, dict]:
    """Project collision geometry in the height band onto the floor.

    Returns (occupied[row=y, col=x], metadata). Cells that are not over the
    floor are occupied.
    """
    cfg, model = env.config, env.model
    resolution = cfg.grid_resolution
    z_min, z_max = cfg.grid_height_band
    with env.lock:
        floor, obstacles = [], []
        for geom in range(model.ngeom):
            if not (model.geom_contype[geom] or model.geom_conaffinity[geom]) or geom in env.robot_geoms:
                continue
            points = _geom_box_points(env, geom, resolution / 2.0)
            if env.root_body_name(int(model.geom_bodyid[geom])).startswith("floor"):
                floor.append(points[:, :2])
            else:
                in_band = points[(points[:, 2] >= z_min) & (points[:, 2] <= z_max)]
                if len(in_band):
                    obstacles.append(in_band[:, :2])
    if not floor:
        raise RuntimeError("scene has no floor geoms to bound the map")
    floor_xy = np.concatenate(floor)
    origin = floor_xy.min(axis=0) - cfg.grid_margin
    extent = floor_xy.max(axis=0) + cfg.grid_margin - origin
    cols, rows = (int(math.ceil(e / resolution)) for e in extent)

    def cells(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        col = np.floor((xy[:, 0] - origin[0]) / resolution).astype(int)
        row = np.floor((xy[:, 1] - origin[1]) / resolution).astype(int)
        keep = (row >= 0) & (row < rows) & (col >= 0) & (col < cols)
        return row[keep], col[keep]

    occupied = np.ones((rows, cols), dtype=bool)
    occupied[cells(floor_xy)] = False
    if obstacles:
        occupied[cells(np.concatenate(obstacles))] = True
    meta = {"resolution": resolution, "origin": [float(origin[0]), float(origin[1])], "frame": "world"}
    return occupied, meta


def _is_free(occupied: np.ndarray, meta: dict, x: float, y: float) -> bool:
    col = math.floor((x - meta["origin"][0]) / meta["resolution"])
    row = math.floor((y - meta["origin"][1]) / meta["resolution"])
    return 0 <= row < occupied.shape[0] and 0 <= col < occupied.shape[1] and not occupied[row, col]


def _facing(box: FixtureBox, occupied: np.ndarray, meta: dict) -> list[float] | None:
    """Direction from the fixture toward open floor: its front if that is
    clear, else whichever other side is, else None."""
    front = box.facing
    left = np.array([-front[1], front[0]])
    half_depth = np.linalg.norm(box.depth[:2]) / 2.0
    half_width = np.linalg.norm(box.width[:2]) / 2.0
    for direction, reach in ((front, half_depth), (left, half_width), (-left, half_width), (-front, half_depth)):
        probe = box.center[:2] + direction * (reach + _FACING_CLEARANCE)
        if _is_free(occupied, meta, probe[0], probe[1]):
            return [round(float(direction[0]), 3) + 0.0, round(float(direction[1]), 3) + 0.0]  # no -0.0
    return None


def build_semantic_map(env: RoboCasaEnv, occupied: np.ndarray, meta: dict) -> dict:
    with env.lock:
        fixtures = named_fixtures(env)
        objects = [(label, env.data.xpos[body].copy()) for _, label, body in scene_objects(env)]

    fixture_names = [f.name for f in fixtures]
    out: dict = {"frame": "world", "fixtures": {}, "objects": {}}
    for name, box in ((f.name, f.box) for f in fixtures):
        entry: dict = {"location": [round(float(box.center[0]), 3), round(float(box.center[1]), 3)]}
        facing = _facing(box, occupied, meta)
        if facing is not None:
            entry["facing"] = facing
        out["fixtures"][name] = entry

    # Burners are fixed parts: on the map so they keep their names (burner_1, ...) between detects.
    stove_facing = {f.sim_name: out["fixtures"].get(f.name, {}).get("facing") for f in fixtures}
    for burner in burners(env):
        xy = burner_position(env, burner)[:2]
        entry = {"location": [round(float(xy[0]), 3), round(float(xy[1]), 3)]}
        if stove_facing.get(burner.stove):
            entry["facing"] = stove_facing[burner.stove]
        out["fixtures"][burner.name] = entry

    objects.sort(key=lambda o: (o[0], round(float(o[1][0]), 2), round(float(o[1][1]), 2)))
    taken = set(fixture_names)
    object_names = numbered([label for label, _ in objects])
    for name, (_, position) in zip(object_names, objects):
        while name in taken:  # an object sharing a fixture's name
            name += "_object"
        taken.add(name)
        entry = {"location": [round(float(position[0]), 3), round(float(position[1]), 3)]}
        holder = next(
            (f.name for f in fixtures if f.box.contains_xy(position[0], position[1])),
            None,
        )
        if holder is not None:
            entry["in"] = holder
        out["objects"][name] = entry
    return out


def _footprint(box: FixtureBox, facing: list[float]) -> list[float]:
    """[width across `facing`, depth along it] of the box's footprint."""
    f = np.array(facing, dtype=float)
    a = np.array([-f[1], f[0]])
    edges = (box.width[:2], box.depth[:2])
    return [round(float(sum(abs(e @ a) for e in edges)), 3), round(float(sum(abs(e @ f) for e in edges)), 3)]


def build_start_map(env: RoboCasaEnv, occupied: np.ndarray, meta: dict) -> dict:
    """The semantic map without its objects, fixtures with their footprints."""
    full = build_semantic_map(env, occupied, meta)
    with env.lock:
        boxes = {f.name: f.box for f in named_fixtures(env)}
    for name, entry in full["fixtures"].items():
        if name in boxes and "facing" in entry:
            entry["size"] = _footprint(boxes[name], entry["facing"])
    full["objects"] = {}
    return full


def write_grid(env: RoboCasaEnv, out_dir: str | Path) -> Path:
    """Write grid.npy and grid.yaml only (geometry, no semantics); returns the folder."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    occupied, meta = build_grid(env)
    np.save(out_dir / GRID_FILE, occupied)
    (out_dir / GRID_META_FILE).write_text(yaml.safe_dump(meta))
    return out_dir


def build_map(env: RoboCasaEnv, out_dir: str | Path | None = None) -> Path:
    """Write grid.npy, grid.yaml and semantic_map.yaml; returns the folder."""
    out_dir = Path(out_dir) if out_dir else env.config.map_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    occupied, meta = build_grid(env)
    np.save(out_dir / GRID_FILE, occupied)
    (out_dir / GRID_META_FILE).write_text(yaml.safe_dump(meta))
    semantic = build_semantic_map(env, occupied, meta)
    (out_dir / SEMANTIC_MAP_FILE).write_text(yaml.safe_dump(semantic, sort_keys=False, default_flow_style=None))
    return out_dir


def ensure_map(env: RoboCasaEnv, out_dir: str | Path | None = None) -> Path:
    """The scene's map folder, built first if any file is missing."""
    out_dir = Path(out_dir) if out_dir else env.config.map_dir
    if all((out_dir / f).exists() for f in (GRID_FILE, GRID_META_FILE, SEMANTIC_MAP_FILE)):
        return out_dir
    return build_map(env, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layout", type=int)
    parser.add_argument("--style", type=int)
    parser.add_argument("--out", type=Path, help="output folder (default: maps/robocasa_layoutNNN_styleNNN)")
    parser.add_argument("--start", action="store_true", help=f"write {START_MAP_FILE} (fixtures only) and the grid")
    args = parser.parse_args()
    env = RoboCasaEnv(RoboCasaConfig.load(layout=args.layout, style=args.style))
    if args.start:
        out_dir = write_grid(env, args.out or env.config.map_dir)
        occupied, meta = np.load(out_dir / GRID_FILE), yaml.safe_load((out_dir / GRID_META_FILE).read_text())
        start = build_start_map(env, occupied, meta)
        (out_dir / START_MAP_FILE).write_text(yaml.safe_dump(start, sort_keys=False, default_flow_style=None))
        print(f"wrote {out_dir / START_MAP_FILE}: fixtures {', '.join(start['fixtures'])}")
        env.close()
        return
    out_dir = build_map(env, args.out)
    occupied = np.load(out_dir / GRID_FILE)
    semantic = yaml.safe_load((out_dir / SEMANTIC_MAP_FILE).read_text())
    print(f"wrote {out_dir}")
    print(f"  grid {occupied.shape[1]}x{occupied.shape[0]} cells, {100 * occupied.mean():.0f}% occupied")
    print(f"  fixtures: {', '.join(semantic['fixtures'])}")
    print(f"  objects: {', '.join(semantic['objects']) or '(none)'}")
    env.close()


if __name__ == "__main__":
    main()
