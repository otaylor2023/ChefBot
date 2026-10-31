"""Env: the low-level surface the robot layer drives. No controllers, no planning."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from chefbot.interface.types import RGBD, EnvState, Pose2D


class Env(ABC):
    @abstractmethod
    def get_state(self) -> EnvState:
        """Base pose and velocity, arm joints, gripper width, contacts, time."""

    @abstractmethod
    def set_base_velocity(self, vx: float, vy: float, vyaw: float, frame: str = "world") -> None:
        """Command a base velocity (m/s, m/s, rad/s). `frame` is "world" or "base".
        The command holds until the next call."""

    def set_base_target(self, pose: Pose2D, max_vel: tuple[float, float, float]) -> bool:
        """Drive the base toward `pose` (world frame) within `max_vel` (m/s, m/s, rad/s), if the
        base does that itself (a real base with its own trajectory controller). Returns False
        when it does not: the robot layer then drives it with `set_base_velocity`."""
        return False

    @abstractmethod
    def set_arm_joint_targets(self, q: np.ndarray) -> None:
        """Joint position targets (rad), one per arm joint. They hold until the next call."""

    def set_arm_pose_target(self, flange_pose: np.ndarray) -> bool:
        """Move the flange toward the 4x4 pose `flange_pose` (arm base frame), if the arm has its
        own controller for that (a real arm's driver). Returns False when it does not: the robot
        layer then solves IK itself and sends `set_arm_joint_targets`. Holds until the next call."""
        return False

    @abstractmethod
    def set_gripper(self, closed: float) -> None:
        """Command the gripper: 0 open .. 1 closed (a bool is either end). A gripper that is only open or closed
        closes at >= 0.5. Holds until the next call."""

    @abstractmethod
    def get_camera_rgbd(self, cam: str) -> RGBD: ...

    @abstractmethod
    def wait_for_next_tick(self, dt: float) -> None:
        """The only notion of time on the interface. Sim: step physics by `dt`.
        Real: likely a no-op, provided something else (e.g. a blocking
        `get_state`) paces the control loop."""
