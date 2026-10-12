"""Env: the low-level surface the robot layer drives. No controllers, no planning."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from chefbot.interface.types import RGBD, EnvState


class Env(ABC):
    @abstractmethod
    def get_state(self) -> EnvState:
        """Base pose, base velocity, base_in_contact, time."""

    @abstractmethod
    def set_base_velocity(self, vx: float, vy: float, vyaw: float, frame: str = "world") -> None:
        """Command a base velocity (m/s, m/s, rad/s). `frame` is "world" or "base".
        The command holds until the next call."""

    @abstractmethod
    def set_arm_joint_targets(self, q: np.ndarray) -> None:
        """Unused by nav v0.1; present for later."""

    @abstractmethod
    def get_camera_rgbd(self, cam: str) -> RGBD: ...

    @abstractmethod
    def wait_for_next_tick(self, dt: float) -> None:
        """The only notion of time on the interface. Sim: step physics by `dt`.
        Real: likely a no-op, provided something else (e.g. a blocking
        `get_state`) paces the control loop."""
