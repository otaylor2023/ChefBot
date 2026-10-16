"""Detector and Observer for the symbolic kitchen.

The detector sees what is in front of the robot (within range and field of
view), not what is inside a closed fridge or cabinet, not what the gripper
holds, and misses objects at the task's `detect_miss` rate. The observer is
ground truth as relation text, for the StateChecker and the run log only.
"""

from __future__ import annotations

import math

import numpy as np

from chefbot.envs.symbolic.robot import SymbolicRobot
from chefbot.envs.symbolic.world import OPENABLE, World
from chefbot.interface.detector import Detector
from chefbot.interface.observer import Observation, Observer
from chefbot.interface.types import Detection

RANGE = 3.0  # m
HALF_FOV = math.radians(60)
_SURFACES = {"counter", "table"}  # big: detections carry the part nearest the robot


class SymbolicDetector(Detector):
    def __init__(self, robot: SymbolicRobot):
        self._robot = robot
        self._world: World = robot.world

    def detect(self, classes: list[str]) -> list[Detection]:
        # Looking takes a moment (timing.yaml: robot.detect); the kitchen goes on meanwhile.
        if self._robot._motion is None:
            self._world.advance(float(self._world.timing.robot.get("detect", 0.0)))
        world, pose = self._world, self._robot.get_base_pose()
        here = np.array([pose.x, pose.y])
        wanted = set(classes)
        found: list[Detection] = []

        def seen(position) -> bool:
            d = np.asarray(position[:2]) - here
            if float(np.linalg.norm(d)) > RANGE:
                return False
            off = math.atan2(d[1], d[0]) - pose.yaw
            return abs(math.atan2(math.sin(off), math.cos(off))) <= HALF_FOV

        for fixture in world.fixtures.values():
            position = world.position(fixture.name)
            if fixture.label in wanted and seen(position):
                near = None
                if fixture.kind in _SURFACES:
                    along = float(np.clip((here - fixture.location) @ fixture.across, -fixture.width / 2 + 0.15,
                                          fixture.width / 2 - 0.15))  # fmt: skip
                    near = np.array([*(fixture.location + along * fixture.across), fixture.height])
                found.append(Detection(fixture.label, position, 1.0, near, _state(fixture)))
            handle = f"{fixture.kind} handle"
            if fixture.kind in OPENABLE and handle in wanted:
                spot = np.array([*(fixture.location + 0.3 * fixture.facing), 1.0])
                if seen(spot):
                    found.append(Detection(handle, spot, 1.0, None, _state(fixture)))
        if "burner" in wanted:
            for burner in world.burners.values():
                if seen(burner.position):
                    found.append(Detection("burner", burner.position.copy(), 1.0, None, "on" if burner.on else "off"))
        for thing in world.things.values():
            if thing.kind not in wanted or thing.where is None or world.enclosed(thing.name):
                continue
            in_gripper = world.held is not None and world.fixture_of(thing.name) is None
            position = world.position(thing.name, here)
            if in_gripper:  # inside what the gripper holds (the egg in the held cracker): plainly visible
                found.append(Detection(thing.kind, position, 1.0, in_gripper=True))
            elif seen(position) and not self._robot._roll(self._robot.rates.detect_miss):
                found.append(Detection(thing.kind, position, 1.0))
        return found


def _state(fixture) -> str | None:
    """What a door- / appliance-state classifier would say."""
    if fixture.open is not None:
        return "open" if fixture.open else "closed"
    if fixture.powered is not None:
        return "on" if fixture.powered else "off"
    return None


class SymbolicObserver(Observer):
    def __init__(self, world: World):
        self._world = world

    def observe(self) -> Observation:
        return Observation(text=self._world.describe())
