"""GroundTruthDetector and the RoboCasa <-> ChefBot name table.

The detector answers in ChefBot labels with ground-truth world positions, but
only for things the shoulder camera can actually see from the robot's current
pose: inside the camera frustum and not occluded (raycast).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import mujoco
import numpy as np

from chefbot.envs.robocasa.env import RoboCasaEnv
from chefbot.interface.detector import Detector
from chefbot.interface.types import Detection

# -- Name table (shared with build_map.py) ------------------------------------

# RoboCasa fixture class (or any base class) -> ChefBot label. These are the
# fixtures that go on the semantic map.
FIXTURE_LABELS: dict[str, str] = {
    "Stove": "stove",
    "Fridge": "fridge",
    "Microwave": "microwave",
    "Sink": "sink",
    "Counter": "counter",
    "Dishwasher": "dishwasher",
    "Oven": "oven",
    "CoffeeMachine": "coffee machine",
    "Toaster": "toaster",
    "ToasterOven": "toaster oven",
    "Blender": "blender",
    "StandMixer": "stand mixer",
    "ElectricKettle": "kettle",
}
# Fixtures that own detectable parts but are not on the map themselves.
_PART_OWNER_ONLY: dict[str, str] = {"Drawer": "drawer", "Cabinet": "cabinet"}

# (owner label, "geom" | "site", pattern on the name after the fixture prefix) -> part label.
PART_RULES: list[tuple[str, str, re.Pattern, str]] = [
    ("stove", "site", re.compile(r"^burner_.+_place_site$"), "burner"),
    ("stove", "geom", re.compile(r"^door_handle_main$"), "oven handle"),
    ("fridge", "geom", re.compile(r"^fridge\w*_door_handle_main$"), "fridge handle"),
    ("fridge", "geom", re.compile(r"^freezer\w*_door_handle_main$"), "freezer handle"),
    ("microwave", "geom", re.compile(r"^door_handle_main$"), "microwave handle"),
    ("dishwasher", "geom", re.compile(r"^door_handle_main$"), "dishwasher handle"),
    ("oven", "geom", re.compile(r"^door_handle_main$"), "oven handle"),
    ("sink", "geom", re.compile(r"^handle_main$"), "faucet"),
    ("cabinet", "geom", re.compile(r"door_handle_(g1|handle)$"), "cabinet handle"),
    ("drawer", "geom", re.compile(r"door_handle_(g1|handle)$"), "drawer handle"),
]


def _class_label(fixture, table: dict[str, str]) -> str | None:
    for cls in type(fixture).__mro__:
        if cls.__name__ in table:
            return table[cls.__name__]
    return None


def fixture_label(fixture) -> str | None:
    """ChefBot label of a map-worthy RoboCasa fixture, else None."""
    return _class_label(fixture, FIXTURE_LABELS)


def object_label(category: str) -> str:
    """ChefBot label of a RoboCasa object category."""
    return category.replace("_", " ")


@dataclass(frozen=True)
class FixtureBox:
    """A fixture's oriented bounding box in the world frame."""

    origin: np.ndarray  # front-bottom corner
    width: np.ndarray  # edge vectors: along the front,
    depth: np.ndarray  # front to back,
    height: np.ndarray  # and up

    def point(self, u: float, v: float, w: float) -> np.ndarray:
        return self.origin + u * self.width + v * self.depth + w * self.height

    @property
    def center(self) -> np.ndarray:
        return self.point(0.5, 0.5, 0.5)

    @property
    def top(self) -> float:
        return float(self.point(0.5, 0.5, 1.0)[2])

    @property
    def facing(self) -> np.ndarray:
        """Unit xy direction out of the front face."""
        out = -self.depth[:2]
        return out / np.linalg.norm(out)

    def contains_xy(self, x: float, y: float) -> bool:
        rel = np.array([x, y]) - self.origin[:2]
        u = rel @ self.width[:2] / (self.width[:2] @ self.width[:2])
        v = rel @ self.depth[:2] / (self.depth[:2] @ self.depth[:2])
        return 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0


def fixture_box(fixture) -> FixtureBox | None:
    try:
        p0, px, py, pz = (np.asarray(p, dtype=float) for p in fixture.get_ext_sites(relative=False))
    except Exception:
        return None
    box = FixtureBox(p0, px - p0, py - p0, pz - p0)
    if min(np.linalg.norm(box.width), np.linalg.norm(box.depth), np.linalg.norm(box.height)) < 1e-6:
        return None
    return box


def map_fixtures(env: RoboCasaEnv) -> list[tuple[str, str, FixtureBox]]:
    """(RoboCasa name, ChefBot label, box) for every fixture that belongs on the map."""
    out = []
    for name, fixture in env.suite.fixtures.items():
        label = fixture_label(fixture)
        box = fixture_box(fixture) if label else None
        if box is not None:
            out.append((name, label, box))
    return out


def scene_objects(env: RoboCasaEnv) -> list[tuple[str, str, int]]:
    """(RoboCasa name, ChefBot label, body id) for every loose object in the scene."""
    out = []
    body_ids = getattr(env.suite, "obj_body_id", {})
    for cfg in getattr(env.suite, "object_cfgs", []):
        category = cfg.get("info", {}).get("cat")
        if category and cfg["name"] in body_ids:
            out.append((cfg["name"], object_label(category), int(body_ids[cfg["name"]])))
    return out


# -- Detector -------------------------------------------------------------------


@dataclass(frozen=True)
class _Item:
    label: str
    kind: str  # "geom" | "site" | "body" | "fixture"
    id: int = -1
    root: str = ""  # top-level body the item belongs to
    box: FixtureBox | None = None


# Rays are cast against collision geometry only (group 0): visual group 1 also
# holds invisible region boxes that would block everything behind them.
_RAY_GROUPS = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
_OCCLUSION_TOLERANCE = 0.05  # a hit this close to the target point is the target itself
_NEAR_PLANE = 0.05
_FACE_SAMPLES = (0.1, 0.5, 0.9)


class GroundTruthDetector(Detector):
    def __init__(self, env: RoboCasaEnv, camera: str | None = None, max_range: float = 8.0):
        self._env = env
        self._camera = env.model.camera(camera or env.config.shoulder_camera).id
        self._aspect = env.config.camera_width / env.config.camera_height
        self._max_range = max_range
        self._items = self._index_scene()

    def labels(self) -> set[str]:
        """Every label this scene can produce."""
        return {item.label for item in self._items}

    def detect(self, classes: list[str]) -> list[Detection]:
        wanted = set(classes)
        with self._env.lock:
            return [
                Detection(item.label, self._position(item), 1.0)
                for item in self._items
                if item.label in wanted and self._visible(item)
            ]

    # -- scene index ----------------------------------------------------------

    def _index_scene(self) -> list[_Item]:
        env, model = self._env, self._env.model
        items: list[_Item] = []
        owners: dict[str, tuple[str, str]] = {}  # root body -> (fixture name, owner label)
        for name, fixture in env.suite.fixtures.items():
            label = _class_label(fixture, {**_PART_OWNER_ONLY, **FIXTURE_LABELS})
            if label:
                owners[f"{name}_main"] = (name, label)
        for name, label, box in map_fixtures(env):
            items.append(_Item(label, "fixture", root=f"{name}_main", box=box))

        def add_parts(kind: str, count: int, name_of, body_of) -> None:
            for i in range(count):
                root = env.root_body_name(int(body_of(i)))
                if root not in owners:
                    continue
                fixture_name, owner_label = owners[root]
                local = name_of(i).removeprefix(fixture_name + "_")
                for rule_owner, rule_kind, pattern, label in PART_RULES:
                    if rule_owner == owner_label and rule_kind == kind and pattern.search(local):
                        items.append(_Item(label, kind, i, root))
                        break

        add_parts("geom", model.ngeom, lambda g: model.geom(g).name, lambda g: model.geom_bodyid[g])
        add_parts("site", model.nsite, lambda s: model.site(s).name, lambda s: model.site_bodyid[s])
        for _, label, body in scene_objects(env):
            items.append(_Item(label, "body", body, env.root_body_name(body)))
        return items

    # -- geometry ---------------------------------------------------------------

    def _position(self, item: _Item) -> np.ndarray:
        data = self._env.data
        if item.kind == "geom":
            return data.geom_xpos[item.id].copy()
        if item.kind == "site":
            return data.site_xpos[item.id].copy()
        if item.kind == "body":
            return data.xpos[item.id].copy()
        # Fixtures: the middle of the work surface if there is one within reach,
        # otherwise the middle of the front face.
        box = item.box
        return box.point(0.5, 0.5, 1.0) if box.top <= 1.2 else box.point(0.5, 0.0, 0.5)

    def _visible(self, item: _Item) -> bool:
        if item.kind == "fixture":
            box = item.box
            samples = [box.point(u, 0.0, w) for u in _FACE_SAMPLES for w in _FACE_SAMPLES]
            samples += [box.point(u, v, 1.0) for u in _FACE_SAMPLES for v in _FACE_SAMPLES]
            return any(self._point_visible(p, item.root) for p in samples)
        own_root = item.root if item.kind == "body" else None
        return self._point_visible(self._position(item), own_root)

    def _point_visible(self, point: np.ndarray, own_root: str | None) -> bool:
        """In the camera frustum and not occluded. A ray that ends on `own_root`'s
        geometry counts as reaching the target."""
        data = self._env.data
        origin = data.cam_xpos[self._camera]
        # MuJoCo cameras look along -z with x right and y up.
        in_camera = data.cam_xmat[self._camera].reshape(3, 3).T @ (point - origin)
        forward = -in_camera[2]
        if not _NEAR_PLANE < forward <= self._max_range:
            return False
        half_height = math.tan(math.radians(self._env.model.cam_fovy[self._camera]) / 2.0)
        if abs(in_camera[1]) > forward * half_height or abs(in_camera[0]) > forward * half_height * self._aspect:
            return False
        distance = float(np.linalg.norm(point - origin))
        hit_geom, hit_distance = self._first_hit(origin, (point - origin) / distance)
        if hit_geom < 0 or hit_distance >= distance - _OCCLUSION_TOLERANCE:
            return True
        return own_root is not None and self._env.root_body_name(int(self._env.model.geom_bodyid[hit_geom])) == own_root

    def _first_hit(self, origin: np.ndarray, direction: np.ndarray) -> tuple[int, float]:
        """First non-robot geom along the ray, and its distance from `origin`."""
        env = self._env
        geom = np.array([-1], dtype=np.int32)
        travelled = 0.0
        start = np.array(origin, dtype=float)
        for _ in range(32):
            step = mujoco.mj_ray(env.model, env.data, start, direction, _RAY_GROUPS, 1, -1, geom)
            if step < 0:
                return -1, math.inf
            travelled += step
            if int(geom[0]) not in env.robot_geoms:
                return int(geom[0]), travelled
            # The camera looks past the robot's own body: keep going from just behind the hit.
            travelled += 1e-4
            start = origin + travelled * direction
        return -1, math.inf
