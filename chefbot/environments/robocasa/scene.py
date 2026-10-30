"""A RoboCasa kitchen's objects and fixtures as the task types use them (harness side only).

    scene.objects()                     the loose objects' names (their categories)
    scene.object("banana").place(x, y, z, yaw)
    scene.counters()                    the counters (detector.NamedFixture: name, box, root)
    scene.flat_height(fixture, x, y, r) the fixture's top under a disc at (x, y), if flat and free
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from chefbot.environments.robocasa.detector import NamedFixture, named_fixtures
from chefbot.environments.robocasa.env import RoboCasaEnv
from chefbot.environments.robocasa.spawn import _flat_height, object_names

DROP = 0.004  # objects are set down this far above their spot
PARKED = np.array([60.0, 60.0, 5.0])  # out of the kitchen: where objects wait while spots are raycast


class Obj:
    def __init__(self, env: RoboCasaEnv, name: str, category: str):
        self.env, self.name, self.category = env, name, category
        self.model = env.suite.objects[name]
        self.body = int(env.suite.obj_body_id[name])
        self.qpos = env.suite.sim.model.get_joint_qpos_addr(self.model.joints[0])[0]
        self.geoms = frozenset(
            int(g) for g in np.flatnonzero(env.model.body_rootid[env.model.geom_bodyid] == env.model.body_rootid[self.body])
        )

    @property
    def radius(self) -> float:
        """Half the narrower side of its footprint: the part that rests on the surface."""
        return float(min(self.model.size[0], self.model.size[1])) / 2.0

    @property
    def reach(self) -> float:
        """Half the diagonal of its footprint: how far it can stick out from its center, turned any way."""
        return float(np.hypot(self.model.size[0], self.model.size[1])) / 2.0

    def place(self, x: float, y: float, z: float, yaw: float) -> None:
        """Stand it upright with its bottom on height z at (x, y), turned `yaw`, at rest. Call with env.lock held."""
        d = self.env.data
        d.qpos[self.qpos : self.qpos + 3] = [x, y, z - self.model.bottom_offset[2] + DROP]
        d.qpos[self.qpos + 3 : self.qpos + 7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        d.qvel[:] = 0.0

    def park(self, slot: int) -> None:
        self.place(*(PARKED + [2.0 * slot, 0.0, 0.0]), 0.0)

    def box(self) -> tuple[np.ndarray, np.ndarray]:
        """World min / max corners of its collision geometry."""
        return self.env._world_box(self.body)


class Scene:
    def __init__(self, env: RoboCasaEnv, categories: list[str]):
        self.env = env
        self._objects = {n: Obj(env, n, c) for n, c in zip(object_names([{"category": c} for c in categories]), categories)}
        self._fixtures = {f.name: f for f in named_fixtures(env)}

    def objects(self) -> list[str]:
        return list(self._objects)

    def object(self, name: str) -> Obj:
        return self._objects[name]

    def fixture(self, name: str) -> NamedFixture:
        return self._fixtures[name]

    def counters(self) -> list[NamedFixture]:
        return [f for f in self._fixtures.values() if f.label == "counter"]

    def flat_height(self, fixture: NamedFixture, x: float, y: float, radius: float) -> float | None:
        """Height of the fixture's top under a disc at (x, y), if it is flat, the fixture's own and near its top."""
        z = _flat_height(self.env, fixture, x, y, radius, fixture.box.top + 0.25)
        return z if z is not None and abs(z - fixture.box.top) <= 0.03 else None

    def penetrating(self, ours: set[int], allowed_roots: set[str]) -> list[tuple[str, str]]:
        """(our geom, other geom) for contacts of `ours` with anything else that is neither theirs, the floor nor
        under `allowed_roots` (body names), sunk in more than 2 mm."""
        env = self.env
        m, d = env.model, env.data
        out = []
        for i in range(d.ncon):
            a, b = int(d.contact[i].geom1), int(d.contact[i].geom2)
            if (a in ours) == (b in ours) or d.contact[i].dist >= -0.002:
                continue
            own, other = (a, b) if a in ours else (b, a)
            if other in env._floor_geoms or env.root_body_name(int(m.geom_bodyid[other])) in allowed_roots:
                continue
            out.append((m.geom(own).name, m.geom(other).name))
        return out

    def forward(self) -> None:
        mujoco.mj_forward(self.env.model, self.env.data)
