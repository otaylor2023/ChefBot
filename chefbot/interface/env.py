"""Env: the low-level surface the robot layer drives. No controllers, no planning."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from chefbot.interface.types import RGBD, EnvState


class Env(ABC):
    @abstractmethod
    def get_state(self) -> EnvState:
        """Base pose and velocity, arm joints, gripper width, contacts, time."""

    @abstractmethod
    def set_base_velocity(self, vx: float, vy: float, vyaw: float, frame: str = "world") -> None:
        """Command a base velocity (m/s, m/s, rad/s). `frame` is "world" or "base".
        The command holds until the next call."""

    @abstractmethod
    def set_arm_joint_targets(self, q: np.ndarray) -> None:
        """7-DoF joint position targets (rad). They hold until the next call."""

    @abstractmethod
    def set_gripper(self, closed: bool) -> None:
        """Command the gripper fully closed or fully open. Holds until the next call."""

    @abstractmethod
    def get_camera_rgbd(self, cam: str) -> RGBD: ...

    @abstractmethod
    def wait_for_next_tick(self, dt: float) -> None:
        """The only notion of time on the interface. Sim: step physics by `dt`.
        Real: likely a no-op, provided something else (e.g. a blocking
        `get_state`) paces the control loop."""
