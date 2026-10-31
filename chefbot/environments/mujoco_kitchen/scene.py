"""A MuJoCo kitchen's objects and fixtures as the task types use them (harness side only), the same surface as the
RoboCasa scene (environments/robocasa/scene.py) where they overlap.

    scene.objects()                     the loose objects' names
    scene.object("egg").place(x, y, z, yaw)        its bottom on height z
    scene.extra("tongs")                the extra objects (generated ones), the same way
    scene.counters(), scene.fixture(name)          detector-like NamedFixtures from the export's meta
    scene.flat_height(fixture, x, y, r) the fixture's top under a disc at (x, y), if flat and free
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from chefbot.environments.mujoco_kitchen.env import MujocoKitchenEnv

DROP = 0.003  # objects are set down this far above their spot
PARKED = np.array([60.0, 60.0, 5.0])  # out of the kitchen


@dataclass(frozen=True)
class Box:
    origin: np.ndarray
    width: np.ndarray
    depth: np.ndarray
    height: np.ndarray

    def point(self, u: float, v: float, w: float) -> np.ndarray:
        return self.origin + u * self.width + v * self.depth + w * self.height

    @property
    def top(self) -> float:
        return float(self.point(0.5, 0.5, 1.0)[2])

    @property
    def facing(self) -> np.ndarray:
        out = -self.depth[:2]
        return out / np.linalg.norm(out)


@dataclass(frozen=True)
class Fixture:
    name: str
    label: str
    root: str
    box: Box


class Obj:
    """A free body: placed by its collision geometry's bottom, whatever its frame."""

    def __init__(self, env: MujocoKitchenEnv, name: str, body: str, category: str):
        self.env, self.name, self.category = env, name, category
        self.body = env.body_root(body)
        self.qpos = env.free_qpos(self.body)
        self.geoms = env.geoms_of(self.body)
        self.size, self.bottom = self._rest_extent()

    def _rest_extent(self) -> tuple[np.ndarray, float]:
        """Its collision box's size, and its bottom below its origin, standing as it was made."""
        m = self.env.model
        d = mujoco.MjData(m)
        d.qpos[self.qpos : self.qpos + 7] = [0, 0, 0, 1, 0, 0, 0]
        mujoco.mj_kinematics(m, d)
        lo, hi = _box(m, d, self.geoms)
        return hi - lo, float(-lo[2])

    @property
    def radius(self) -> float:
        return float(min(self.size[0], self.size[1])) / 2.0

    @property
    def reach(self) -> float:
        return float(np.hypot(self.size[0], self.size[1])) / 2.0

    def place(self, x: float, y: float, z: float, yaw: float, quat=None) -> None:
        """Its bottom on height z at (x, y), turned `yaw` (about its own vertical), at rest."""
        d = self.env.data
        d.qpos[self.qpos : self.qpos + 3] = [x, y, z + self.bottom + DROP]
        d.qpos[self.qpos + 3 : self.qpos + 7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        d.qvel[self.env.model.jnt_dofadr[self.env.model.body_jntadr[self.body]] :][:6] = 0.0

    def park(self, slot: int) -> None:
        self.place(*(PARKED + [2.0 * slot, 0.0, 0.0]), 0.0)

    def box(self) -> tuple[np.ndarray, np.ndarray]:
        return _box(self.env.model, self.env.data, self.geoms)

    def center(self) -> np.ndarray:
        lo, hi = self.box()
        return (lo + hi) / 2


def _box(m, d, geoms) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in geoms:
        R = d.geom_xmat[g].reshape(3, 3)
        c = d.geom_xpos[g] + R @ m.geom_aabb[g][:3]
        h = np.abs(R) @ m.geom_aabb[g][3:]
        lo, hi = np.minimum(lo, c - h), np.maximum(hi, c + h)
    return lo, hi


class Scene:
    def __init__(self, env: MujocoKitchenEnv):
        self.env = env
        self._objects = {o["name"]: Obj(env, o["name"], o["body"], o["category"]) for o in env.meta["objects"]}
        self._extras = {n: Obj(env, n, body, n) for n, body in env.extras.items()}
        self._fixtures = {}
        for f in env.meta["fixtures"]:
            b = f["box"]
            box = Box(*(np.asarray(b[k], float) for k in ("origin", "width", "depth", "height")))
            self._fixtures[f["name"]] = Fixture(f["name"], f["label"], f["root"], box)

    def objects(self) -> list[str]:
        return list(self._objects)

    def object(self, name: str) -> Obj:
        return self._objects[name]

    def extra(self, name: str) -> Obj:
        return self._extras[name]

    def fixture(self, name: str) -> Fixture:
        return self._fixtures[name]

    def counters(self) -> list[Fixture]:
        return [f for f in self._fixtures.values() if f.label == "counter"]

    def ray_down(self, x: float, y: float, z: float, exclude: int = -1) -> tuple[int, float]:
        """(the body root it hits, the height) straight down from (x, y, z), seeing only the kitchen's collision geoms
        (RoboCasa's group 0): not the robot or the generated extras (group 3), not visual geoms."""
        m, d = self.env.model, self.env.data
        gid = np.zeros(1, np.int32)
        groups = np.array([1, 0, 0, 0, 0, 0], np.uint8)
        dist = mujoco.mj_ray(m, d, np.array([x, y, z]), np.array([0, 0, -1.0]), groups, 1, exclude, gid)
        if dist < 0:
            return -1, -np.inf
        return int(m.body_rootid[m.geom_bodyid[gid[0]]]), float(z - dist)

    def flat_height(self, fixture: Fixture, x: float, y: float, radius: float, below: float = 0.4) -> float | None:
        """The fixture's top under a disc at (x, y): every ray of the disc hits the fixture within 3 mm of the same
        height, near its top (nothing in the way)."""
        root = self.env.body_root(fixture.root)
        hs = []
        for dx, dy in [(0, 0)] + [(radius * math.cos(a), radius * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)]:
            hit, z = self.ray_down(x + dx, y + dy, fixture.box.top + below)
            if hit != root:
                return None
            hs.append(z)
        if max(hs) - min(hs) > 0.003 or abs(hs[0] - fixture.box.top) > 0.03:
            return None
        return float(np.median(hs))

    def penetrating(self, ours: set[int], allowed_roots: set[int]) -> list[tuple[str, str]]:
        """(our geom, other) for contacts of `ours` sunk more than 2 mm into anything not ours, not the floor, not
        under `allowed_roots` (body ids)."""
        env = self.env
        m, d = env.model, env.data
        out = []
        for i in range(d.ncon):
            a, b = int(d.contact[i].geom1), int(d.contact[i].geom2)
            if (a in ours) == (b in ours) or d.contact[i].dist >= -0.002:
                continue
            own, other = (a, b) if a in ours else (b, a)
            if other in env.floor_geoms or int(m.body_rootid[m.geom_bodyid[other]]) in allowed_roots:
                continue
            out.append((m.geom(own).name, m.body(int(m.body_rootid[m.geom_bodyid[other]])).name))
        return out

    def forward(self) -> None:
        mujoco.mj_forward(self.env.model, self.env.data)
