"""Object spawner: a Kitchen task that adds a configured set of loose objects
and sets them down on fixtures, so the plain kitchen layouts gain things to
pick up (MANIP_SPEC §6.4).

RoboCasa's own placement sampler is random and fails often with several
objects in one region; here every object gets a deterministic spot found by
raycasting the fixture it belongs on.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import robocasa
from robocasa.environments.kitchen.kitchen import Kitchen

if TYPE_CHECKING:
    from chefbot.envs.robocasa.env import RoboCasaEnv

TASK_NAME = "ChefKitchen"
# Rays are cast against collision geometry only, as in the detector.
_RAY_GROUPS = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
_LATTICE = 0.04  # spacing of candidate spots on a surface
_FLATNESS = 0.006  # max height spread under an object's footprint
_DROP = 0.004  # objects are released this far above their spot
_SETTLE_STEPS = 60
_SHELF_HEADROOM = 0.14  # free height needed above an object on a shelf (room for the gripper)
_SHELF_BEST_HEIGHT = 1.35  # shelves near this height are the easiest to reach into with a level gripper


class ChefKitchen(Kitchen):
    """The plain Kitchen task plus `chef_objects` (dicts with a `category`).
    Subclassing registers it with robosuite under its class name."""

    def __init__(self, *args, chef_objects=(), **kwargs):
        self.chef_objects = tuple(chef_objects)
        super().__init__(*args, **kwargs)

    def _get_obj_cfgs(self):
        # Created without a location; place_objects() positions them after reset.
        nowhere = dict(size=(0.0, 0.0), ensure_valid_placement=False, ensure_object_boundary_in_range=False)
        return [
            dict(name=name, obj_groups=_model_path(spec), placement=dict(nowhere))
            for name, spec in zip(object_names(self.chef_objects), self.chef_objects)
        ]


def _model_path(spec: dict) -> str:
    """What RoboCasa samples the object from: a category (it picks a model at
    random), or the path of one model when the spec pins it with `model`."""
    if "model" not in spec:
        return spec["category"]
    # "objaverse/..." is the default set; "aigen_objs/..." and "lightwheel/..." name theirs.
    model = spec["model"]
    head = model.split("/", 1)[0]
    root = os.path.join(robocasa.models.assets_root, "objects")
    path = os.path.join(root, model) if head in _ASSET_SETS else os.path.join(root, "objaverse", model)
    return os.path.join(path, "model.xml")


_ASSET_SETS = {"objaverse", "aigen_objs", "lightwheel"}


def object_names(specs) -> list[str]:
    """Sim names for the spawned objects: the category, numbered when repeated."""
    categories = [spec["category"] for spec in specs]
    seen: dict[str, int] = {}
    names = []
    for category in categories:
        seen[category] = seen.get(category, 0) + 1
        names.append(category if categories.count(category) == 1 else f"{category}_{seen[category]}")
    return names


def place_objects(env: "RoboCasaEnv") -> None:
    """Set every configured object down on (`surface`) or inside (`inside`) its fixture."""
    from chefbot.envs.robocasa.detector import named_fixtures
    from chefbot.envs.robocasa.objects import object_points

    specs = env.config.objects
    if not specs:
        return
    fixtures = named_fixtures(env)
    placed: list[tuple[np.ndarray, float]] = []  # (xy, radius) of objects already down
    with env.lock:
        for name, spec in zip(object_names(specs), specs):
            obj = env.suite.objects[name]
            inside = "inside" in spec
            where = spec["inside"] if inside else spec["surface"]
            # The part that rests on the surface: a pan's handle may overhang.
            radius = float(min(obj.size[0], obj.size[1])) / 2.0
            height = float(obj.size[2])
            spot = None
            candidates = [f for f in fixtures if where in (f.name, f.label)]
            if not candidates and inside and where in _STORAGE:
                candidates = _storage(env, where)  # cabinets and drawers are not map fixtures
            for fixture in candidates:
                find = _shelf_spot if inside else _surface_spot
                spot = find(env, fixture, radius, height, placed)
                if spot is not None:
                    break
            if spot is None:
                raise RuntimeError(f"no room to spawn {name} {'in' if inside else 'on'} {where}")
            start, end = env.suite.sim.model.get_joint_qpos_addr(obj.joints[0])
            env.data.qpos[start : start + 3] = [spot[0], spot[1], spot[2] - obj.bottom_offset[2] + _DROP]
            env.data.qpos[start + 3 : end] = [1.0, 0.0, 0.0, 0.0]
            mujoco.mj_forward(env.model, env.data)
            # Turn the object so the side that sticks out (a pan's or mug's handle)
            # points out of the fixture, where the robot can get at it.
            body = env.suite.obj_body_id[name]
            points = object_points(env, body)[:, :2]
            lopsided = (points.min(axis=0) + points.max(axis=0)) / 2 - env.data.xpos[body][:2]
            spans = np.ptp(points, axis=0)
            yaw = None
            if np.linalg.norm(lopsided) > 0.01:
                facing = fixture.box.facing
                yaw = math.atan2(facing[1], facing[0]) - math.atan2(lopsided[1], lopsided[0])
            elif max(spans) > 1.2 * min(spans):
                # Long but even (a cutting board): lay it along the fixture's front edge.
                along = fixture.box.width[:2]
                yaw = math.atan2(along[1], along[0]) - (0.0 if spans[0] >= spans[1] else math.pi / 2)
            if yaw is not None:
                env.data.qpos[start + 3 : end] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
                mujoco.mj_forward(env.model, env.data)
            placed.append((spot[:2], radius))
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
    # Let the objects come to rest.
    for _ in range(_SETTLE_STEPS):
        env.wait_for_next_tick(1.0 / env.config.control_freq)


_STORAGE = {"cabinet": ("HingeCabinet", "SingleCabinet"), "drawer": ("Drawer",)}


def _storage(env: "RoboCasaEnv", label: str) -> list:
    """Cabinets or drawers to put things inside, by RoboCasa class (they are not on the map)."""
    from chefbot.envs.robocasa.detector import NamedFixture, fixture_box

    out = []
    for sim_name, fixture in sorted(env.suite.fixtures.items()):
        if type(fixture).__name__ in _STORAGE[label] and (box := fixture_box(fixture)) is not None:
            out.append(NamedFixture(sim_name, sim_name, label, box))
    return out


def _ray_down(env: "RoboCasaEnv", x: float, y: float, z: float) -> tuple[int, float]:
    """(geom, height) of the first collision surface below (x, y, z); geom -1 if none."""
    geom = np.array([-1], dtype=np.int32)
    distance = mujoco.mj_ray(
        env.model, env.data, np.array([x, y, z]), np.array([0.0, 0.0, -1.0]), _RAY_GROUPS, 1, -1, geom
    )
    return (int(geom[0]), z - distance) if distance >= 0 else (-1, -math.inf)


def _ray_up(env: "RoboCasaEnv", x: float, y: float, z: float) -> float:
    """Free height above (x, y, z)."""
    geom = np.array([-1], dtype=np.int32)
    distance = mujoco.mj_ray(
        env.model, env.data, np.array([x, y, z]), np.array([0.0, 0.0, 1.0]), _RAY_GROUPS, 1, -1, geom
    )
    return distance if distance >= 0 else math.inf


def _footprint(x: float, y: float, radius: float) -> list[tuple[float, float]]:
    ring = [(x + radius * math.cos(a), y + radius * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)]
    return [(x, y)] + ring


def _flat_height(env: "RoboCasaEnv", fixture, x: float, y: float, radius: float, from_z: float) -> float | None:
    """Height of the fixture's surface under a disc at (x, y), if it is flat and all the fixture's own."""
    heights = []
    for px, py in _footprint(x, y, radius):
        geom, z = _ray_down(env, px, py, from_z)
        if geom < 0 or env.root_body_name(int(env.model.geom_bodyid[geom])) != fixture.root:
            return None
        heights.append(z)
    return float(np.mean(heights)) if max(heights) - min(heights) <= _FLATNESS else None


def _clear_of(placed, xy: np.ndarray, radius: float, spacing: float) -> bool:
    return all(np.linalg.norm(xy - other) >= radius + other_radius + spacing for other, other_radius in placed)


def _lattice(length: float, margin: float) -> np.ndarray:
    """Fractions along an edge of `length`, `margin` in from both ends."""
    if length <= 2 * margin:
        return np.array([])
    count = max(1, int((length - 2 * margin) / _LATTICE) + 1)
    return np.linspace(margin, length - margin, count) / length


def _surface_spot(env, fixture, radius: float, height: float, placed) -> np.ndarray | None:
    """A flat, free spot on the fixture's top, near its front edge."""
    cfg, box = env.config, fixture.box
    width, depth = float(np.linalg.norm(box.width)), float(np.linalg.norm(box.depth))
    # Near the front edge: further back, a wall cabinet is in the way of an arm coming down from above.
    inset = max(cfg.spawn_front_inset, radius + 0.04)
    best, best_cost = None, math.inf
    for u in _lattice(width, radius + 0.03):
        for v in _lattice(depth, radius + 0.03):
            cost = (abs(v * depth - inset), u)
            if cost >= (best_cost if best is not None else (math.inf, math.inf)):
                continue
            point = box.point(u, v, 1.0)
            if not _clear_of(placed, point[:2], radius, cfg.spawn_spacing):
                continue
            z = _flat_height(env, fixture, point[0], point[1], radius, box.top + 0.25)
            if z is None or abs(z - box.top) > 0.03:
                continue
            best, best_cost = np.array([point[0], point[1], z]), cost
    return best


def _shelf_spot(env, fixture, radius: float, height: float, placed) -> np.ndarray | None:
    """A flat, free spot on a shelf inside the fixture, near its front."""
    cfg, box = env.config, fixture.box
    width, depth = float(np.linalg.norm(box.width)), float(np.linalg.norm(box.depth))
    best, best_cost = None, math.inf
    u_low, u_high = _compartment(env, fixture)
    for u in _lattice(width, radius + 0.05):
        if not u_low + 0.05 <= u <= u_high - 0.05:
            continue
        for v in _lattice(depth, radius + 0.08):
            point = box.point(u, v, 1.0)
            if not _clear_of(placed, point[:2], radius, cfg.spawn_spacing):
                continue
            # Walk down through the fixture, collecting every surface on this vertical line.
            z = box.top - 0.02
            while z > 0.3:
                geom, surface = _ray_down(env, point[0], point[1], z)
                if geom < 0 or surface < 0.3:
                    break
                z = surface - 0.02
                if _ray_up(env, point[0], point[1], surface + 0.005) < height + _SHELF_HEADROOM:
                    continue
                flat = _flat_height(env, fixture, point[0], point[1], radius, surface + 0.05)
                if flat is None or abs(flat - surface) > _FLATNESS:
                    continue
                cost = abs(surface - _SHELF_BEST_HEIGHT) + v * depth + 0.2 * abs(u - (u_low + u_high) / 2)
                if cost < best_cost:
                    best, best_cost = np.array([point[0], point[1], flat]), cost
    return best


def _compartment(env, fixture) -> tuple[float, float]:
    """Range along the fixture's width (0..1) that its main door covers. For a
    fridge this is the refrigerator side, not the freezer."""
    from chefbot.envs.robocasa.fixture_actuator import door_joints, moved_geoms

    sim_fixture = env.suite.fixtures[fixture.sim_name]
    geoms = moved_geoms(env, door_joints(sim_fixture, "fridge"))
    if not geoms:
        return 0.0, 1.0
    box = fixture.box
    centers = env.data.geom_xpos[geoms]
    reach = np.linalg.norm(env.model.geom_aabb[geoms, 3:5], axis=1)
    along = (centers - box.origin) @ box.width / (box.width @ box.width)
    margin = reach / np.linalg.norm(box.width)
    return float(max(0.0, (along - margin).min())), float(min(1.0, (along + margin).max()))
