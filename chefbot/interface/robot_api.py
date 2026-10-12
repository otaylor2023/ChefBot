"""RobotAPI: the only thing the agent knows about the robot."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from chefbot.interface.types import MotionResult, Pose2D


class RobotAPI(ABC):
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
    def stop_base(self) -> None:
        """Preempt any motion. Callable at any time, from any thread."""

    @abstractmethod
    def get_base_pose(self) -> Pose2D:
        """World frame; x, y in meters, yaw in radians."""
