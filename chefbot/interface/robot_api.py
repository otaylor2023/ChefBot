"""RobotAPI: the only thing the agent knows about the robot."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from chefbot.interface.types import Detection, HeldObject, MotionResult, Pose2D


class RobotAPI(ABC):
    """Every motion call is non-blocking and shares one motion thread: a new
    call preempts the motion in progress, and `wait` / `get_status` report on
    whichever was started last."""

    @abstractmethod
    def go_to_pose(self, goal: Pose2D) -> None:
        """Non-blocking. The robot plans a path to `goal` and drives it."""

    @abstractmethod
    def align_to_point(self, target: np.ndarray) -> None:
        """Non-blocking. The robot picks a reachable standoff pose around
        `target` (world xyz), plans to it and drives it."""

    @abstractmethod
    def wait(self, timeout_s: float | None = None) -> MotionResult:
        """Block until the current motion ends (None) or `timeout_s` elapses.
        An expired timeout returns status "moving"; the motion continues."""

    @abstractmethod
    def get_status(self) -> MotionResult:
        """Non-blocking snapshot of the current / last motion."""

    @abstractmethod
    def stop(self) -> None:
        """Preempt any motion; halts the base and the arm. Callable at any time,
        from any thread."""

    @abstractmethod
    def pick(self, target: Detection) -> None:
        """Non-blocking. Grasp the object at `target` and lift it. Does not
        move the base."""

    @abstractmethod
    def place(self, surface: Detection) -> None:
        """Non-blocking. Set the held object down on `surface`. Does not move the base."""

    @abstractmethod
    def actuate_fixture(self, fixture: Detection, open: bool) -> None:
        """Non-blocking. Open or close `fixture` (the fixture or its handle)."""

    @abstractmethod
    def get_held(self) -> HeldObject | None:
        """What the gripper is holding, if anything."""

    @abstractmethod
    def get_base_pose(self) -> Pose2D:
        """World frame; x, y in meters, yaw in radians."""
