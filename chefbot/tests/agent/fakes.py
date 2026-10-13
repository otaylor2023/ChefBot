"""FakeRobot and FakeDetector: stand-ins for the robot layer in agent tests."""

from __future__ import annotations

import math

import numpy as np

from chefbot.interface.detector import Detector
from chefbot.interface.robot_api import RobotAPI
from chefbot.interface.types import Detection, HeldObject, MotionResult, MotionStatus, Pose2D, wrap_angle


class FakeRobot(RobotAPI):
    """Teleports to its goal. `outcome` forces the result of the next motions;
    `hold` keeps a motion "moving" until `finish()` is called."""

    def __init__(self, pose: Pose2D = Pose2D(0.0, 0.0, 0.0)):
        self.pose = pose
        self.outcome = MotionStatus.SUCCESS
        self.hold = False
        self.goals: list[Pose2D] = []
        self.align_targets: list[np.ndarray] = []
        self.stops = 0
        self.held: HeldObject | None = None
        self.picks: list[Detection] = []
        self.places: list[Detection] = []
        self.actuations: list[tuple[Detection, bool]] = []
        self.openable = {"fridge", "fridge handle"}  # labels actuate_fixture works on
        self._result: MotionResult | None = None
        self._pending: Pose2D | None = None

    def go_to_pose(self, goal: Pose2D) -> None:
        self.goals.append(goal)
        self._start(goal)

    def align_to_point(self, target: np.ndarray) -> None:
        target = np.asarray(target, dtype=float)
        self.align_targets.append(target)
        # Stand 0.6 m from the target, on the side the robot is already on, facing it.
        away = self.pose.xy - target[:2]
        norm = float(np.linalg.norm(away))
        away = away / norm if norm > 1e-9 else np.array([1.0, 0.0])
        xy = target[:2] + 0.6 * away
        self._start(Pose2D(float(xy[0]), float(xy[1]), math.atan2(-away[1], -away[0])))

    def pick(self, target: Detection) -> None:
        self.picks.append(target)
        self._start(self.pose, lambda: setattr(self, "held", HeldObject(target.label, 0.05, target.position_world)))

    def place(self, surface: Detection) -> None:
        self.places.append(surface)
        self._start(self.pose, lambda: setattr(self, "held", None))

    def actuate_fixture(self, fixture: Detection, open: bool) -> None:
        self.actuations.append((fixture, open))
        if fixture.label not in self.openable:
            self._pending = None
            self._result = MotionResult(
                MotionStatus.UNSUPPORTED, self.pose, f"{fixture.label} is not an openable fixture"
            )
            return
        self._start(self.pose, data={"scripted": True})

    def get_held(self) -> HeldObject | None:
        return self.held

    def _start(self, goal: Pose2D, effect=None, data: dict | None = None) -> None:
        self._pending, self._result = goal, None
        self._effect, self._data = effect, data or {}
        if not self.hold:
            self.finish()

    def finish(self) -> None:
        if self._pending is None:
            return
        if self.outcome == MotionStatus.SUCCESS:
            self.pose = self._pending
            if self._effect is not None:
                self._effect()
        self._result = MotionResult(self.outcome, self.pose, "", dict(self._data))
        self._pending = None

    def wait(self, timeout_s: float | None = None) -> MotionResult:
        return self.get_status()

    def get_status(self) -> MotionResult:
        if self._pending is not None:
            return MotionResult(MotionStatus.MOVING, self.pose, "")
        return self._result or MotionResult(MotionStatus.IDLE, self.pose, "")

    def stop(self) -> None:
        self.stops += 1
        if self._pending is not None:
            self._pending = None
            self._result = MotionResult(MotionStatus.STOPPED, self.pose, "")

    def get_base_pose(self) -> Pose2D:
        return self.pose


class FakeDetector(Detector):
    """Sees the listed objects whose label was asked for and that fall inside a
    horizontal field of view around the robot's heading."""

    def __init__(self, robot: FakeRobot, objects: list[tuple[str, tuple[float, float, float]]], fov_deg: float = 360.0):
        self.robot = robot
        self.objects = list(objects)
        self.fov = math.radians(fov_deg)
        self.calls: list[list[str]] = []

    def detect(self, classes: list[str]) -> list[Detection]:
        self.calls.append(list(classes))
        pose = self.robot.pose
        out = []
        for label, position in self.objects:
            bearing = math.atan2(position[1] - pose.y, position[0] - pose.x)
            if label in classes and abs(wrap_angle(bearing - pose.yaw)) <= self.fov / 2:
                out.append(Detection(label, np.array(position, dtype=float), 0.9))
        return out
