"""ScriptedFixtureActuator: open and close RoboCasa fixtures by setting their
door / drawer joints directly (MANIP_SPEC §6.3). No arm involved yet."""

from __future__ import annotations

import mujoco
import numpy as np

from chefbot.envs.robocasa.detector import GroundTruthDetector
from chefbot.envs.robocasa.env import RoboCasaEnv
from chefbot.envs.robocasa.appliances import burner_on, nearest_burner, set_burner
from chefbot.interface.fixtures import AJAR, CLOSED, OFF, ON, OPEN, UNKNOWN, FixtureActuator
from chefbot.interface.types import ActuationResult, Detection

# Which compartment of a fridge a label refers to.
_COMPARTMENT = {"freezer handle": "freezer"}
_OPEN_FRACTION = 0.9  # a joint this far along its range counts as open
_CLOSED_FRACTION = 0.05  # and this little as closed; in between it is ajar (in the way, but not open)
_SWEEP_STEPS = 8
_LOCATE_RADIUS = 1.5
_SAME_SPOT = 0.05
_POINT_SPACING = 0.04
_HEIGHT_BAND = (0.05, 1.8)  # as the occupancy grid: what the base and arm can run into


def door_joints(fixture, compartment: str = "fridge") -> list[str]:
    """Names of the joints that open `fixture`: its doors, or a drawer's slide."""
    infos = getattr(fixture, "_joint_infos", None) or {}
    for attribute in (f"_{compartment}_door_joint_names", "door_joint_names"):
        names = getattr(fixture, attribute, None)
        if names:
            return [n for n in names if n in infos]
    return [n for n in infos if "slide" in n.lower() or "door" in n.lower()]


def moved_geoms(env: RoboCasaEnv, joint_names: list[str]) -> list[int]:
    """Collision geoms on the bodies that `joint_names` move."""
    model = env.model
    moving = {int(model.jnt_bodyid[model.joint(name).id]) for name in joint_names}
    geoms = []
    for geom in range(model.ngeom):
        if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        body = int(model.geom_bodyid[geom])
        while body != 0 and body not in moving:
            body = int(model.body_parentid[body])
        if body != 0:
            geoms.append(geom)
    return geoms


class ScriptedFixtureActuator(FixtureActuator):
    def __init__(self, env: RoboCasaEnv, detector: GroundTruthDetector):
        self._env = env
        self._detector = detector
        # Where each handle was detected when its fixture was last actuated: a
        # handle moves with its door, so a later call may still name the old spot.
        self._actuated_from: list[tuple[str, np.ndarray, str]] = []  # (label, position, sim fixture)
        self._rng = np.random.RandomState(0)  # RoboCasa picks a knob angle within "on"

    def get_power(self, appliance: Detection) -> str:
        if appliance.label != "burner":
            return UNKNOWN
        with self._env.lock:
            burner = nearest_burner(self._env, appliance.position_world)
            return UNKNOWN if burner is None else (ON if burner_on(self._env, burner) else OFF)

    def set_power(self, appliance: Detection, on: bool) -> ActuationResult:
        env = self._env
        with env.lock:
            burner = nearest_burner(env, appliance.position_world) if appliance.label == "burner" else None
            if burner is None:
                return ActuationResult(False, True, f"the {appliance.label} cannot be turned on or off here")
            set_burner(env, burner, on, self._rng)
            mujoco.mj_forward(env.model, env.data)
        return ActuationResult(True, True, f"{burner.name} is now {'on' if on else 'off'} (scripted)", fixture=burner.name)

    def get_state(self, fixture: Detection) -> str:
        with self._env.lock:
            found = self._resolve(fixture)
            if found is None:
                return UNKNOWN
            fraction = self._fraction(found[2])
            return OPEN if fraction >= _OPEN_FRACTION else CLOSED if fraction <= _CLOSED_FRACTION else AJAR

    def set_state(self, fixture: Detection, open: bool) -> ActuationResult:
        env = self._env
        with env.lock:
            found = self._resolve(fixture)
            if found is None:
                return ActuationResult(False, True, f"{fixture.label} is not an openable fixture")
            name, label, joints = found
            self._actuated_from = [entry for entry in self._actuated_from if entry[2] != name]
            if open:
                position = np.asarray(fixture.position_world, dtype=float).copy()
                self._actuated_from.append((fixture.label, position, name))
            self._set_fraction(env.data, joints, 1.0 if open else 0.0)
            for joint in joints:
                env.data.qvel[env.model.jnt_dofadr[env.model.joint(joint).id]] = 0.0
            mujoco.mj_forward(env.model, env.data)
            obstacle = self._floor_points(env.data, moved_geoms(env, joints)) if open else None
        state = "open" if open else "closed"
        return ActuationResult(True, True, f"the {label} is now {state} (scripted)", name, obstacle)

    def handle_now(self, fixture: Detection) -> np.ndarray | None:
        """Where the handle `fixture` names is now: it moves with its door (a drawer pulled out
        brings it toward you). None if it is not a handle of an openable fixture."""
        with self._env.lock:
            found = self._resolve(fixture)
            if found is None:
                return None
            items = [i for i in self._detector._items if i.fixture == found[0] and i.label == fixture.label]
            if not items:
                return None
            position = np.asarray(fixture.position_world, dtype=float)
            nows = [self._detector._position(i) for i in items]
        return min(nows, key=lambda p: float(np.linalg.norm(p - position)))

    def is_closed(self, sim_name: str) -> bool:
        """Whether the fixture `sim_name` (as an ActuationResult names it) is closed now."""
        fixture = self._env.suite.fixtures.get(sim_name)
        joints = door_joints(fixture) if fixture is not None else []
        if not joints:
            return False
        with self._env.lock:
            return self._fraction(joints) <= _CLOSED_FRACTION

    def swept_xy(self, fixture: Detection, open: bool) -> np.ndarray | None:
        env = self._env
        with env.lock:
            found = self._resolve(fixture)
            if found is None:
                return None
            _, _, joints = found
            geoms = moved_geoms(env, joints)
            scratch = mujoco.MjData(env.model)
            scratch.qpos[:] = env.data.qpos
            start, goal = self._fraction(joints), 1.0 if open else 0.0
            points = []
            for fraction in np.linspace(start, goal, _SWEEP_STEPS + 1):
                self._set_fraction(scratch, joints, float(fraction))
                mujoco.mj_kinematics(env.model, scratch)
                points.append(self._floor_points(scratch, geoms))
        return np.concatenate(points) if points else None

    # -- internals --------------------------------------------------------------

    def _resolve(self, detection: Detection) -> tuple[str, str, list[str]] | None:
        """(sim fixture name, fixture label, joints) for an openable fixture or handle."""
        position = np.asarray(detection.position_world, dtype=float)
        name = label = None
        # The handle of a fixture that was opened from exactly here: that fixture.
        for seen_label, seen_at, fixture_name in self._actuated_from:
            if seen_label == detection.label and np.linalg.norm(seen_at - position) <= _SAME_SPOT:
                name = fixture_name
        if name is None:
            # Otherwise the nearest thing with that label (a handle may have moved with its door).
            item = self._detector.locate(detection, radius=_LOCATE_RADIUS)
            if item is None or not item.fixture:
                return None
            name, label = item.fixture, item.owner_label
        fixture = self._env.suite.fixtures[name]
        joints = door_joints(fixture, _COMPARTMENT.get(detection.label, "fridge"))
        if not joints:
            return None
        return name, label or self._detector.fixture_label(name), joints

    def _range(self, joint: str) -> tuple[float, float]:
        """(closed, open) joint values."""
        low, high = self._env.model.joint(joint).range
        return (float(low), float(high)) if abs(low) <= abs(high) else (float(high), float(low))

    def _fraction(self, joints: list[str]) -> float:
        """How open the fixture is: 0 closed .. 1 open (the least open of its joints)."""
        model, data = self._env.model, self._env.data
        fractions = []
        for joint in joints:
            closed, opened = self._range(joint)
            value = float(data.qpos[model.jnt_qposadr[model.joint(joint).id]])
            fractions.append((value - closed) / (opened - closed) if opened != closed else 0.0)
        return min(fractions)

    def _set_fraction(self, data: mujoco.MjData, joints: list[str], fraction: float) -> None:
        model = self._env.model
        for joint in joints:
            closed, opened = self._range(joint)
            data.qpos[model.jnt_qposadr[model.joint(joint).id]] = closed + fraction * (opened - closed)

    def _floor_points(self, data: mujoco.MjData, geoms: list[int]) -> np.ndarray:
        """xy of the given geoms' bounding boxes, for the part within the height band."""
        model = self._env.model
        clouds = []
        for geom in geoms:
            center, half = model.geom_aabb[geom, :3], model.geom_aabb[geom, 3:]
            axes = [
                center[i] + np.linspace(-half[i], half[i], max(2, int(2 * half[i] / _POINT_SPACING) + 2))
                for i in range(3)
            ]
            local = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
            world = local @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
            clouds.append(world[(world[:, 2] >= _HEIGHT_BAND[0]) & (world[:, 2] <= _HEIGHT_BAND[1]), :2])
        return np.concatenate(clouds) if clouds else np.zeros((0, 2))
