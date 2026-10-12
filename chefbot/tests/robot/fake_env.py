"""FakeEnv: a kinematic integrator standing in for a simulator."""

from __future__ import annotations

import math
import time

import numpy as np

from chefbot.interface.env import Env
from chefbot.interface.types import RGBD, EnvState, Pose2D, wrap_angle
from chefbot.robot.nav.grid import OccupancyGrid


class FakeEnv(Env):
    """Integrates the commanded base velocity exactly.

    `obstacles` (optional) are physical obstacles the base cannot enter; running
    into one holds the base in place and sets `base_in_contact`. `tick_sleep`
    slows ticks down in wall time so tests can interrupt a motion.
    """

    def __init__(
        self,
        pose: Pose2D = Pose2D(0.0, 0.0, 0.0),
        obstacles: OccupancyGrid | None = None,
        tick_sleep: float = 0.0,
    ):
        self.pose = pose
        self.obstacles = obstacles
        self.tick_sleep = tick_sleep
        self.velocity = np.zeros(3)  # world frame
        self.in_contact = False
        self.time = 0.0
        self.ticks = 0
        self.commands: list[tuple[float, np.ndarray]] = []  # (time, world-frame command)

    def get_state(self) -> EnvState:
        return EnvState(self.pose, self.velocity.copy(), self.in_contact, self.time)

    def set_base_velocity(self, vx: float, vy: float, vyaw: float, frame: str = "world") -> None:
        if frame == "base":
            c, s = math.cos(self.pose.yaw), math.sin(self.pose.yaw)
            vx, vy = c * vx - s * vy, s * vx + c * vy
        elif frame != "world":
            raise ValueError(f"unknown frame {frame!r}")
        self.velocity = np.array([vx, vy, vyaw], dtype=float)
        self.commands.append((self.time, self.velocity.copy()))

    def set_arm_joint_targets(self, q: np.ndarray) -> None:
        pass

    def get_camera_rgbd(self, cam: str) -> RGBD:
        raise NotImplementedError("FakeEnv has no cameras")

    def wait_for_next_tick(self, dt: float) -> None:
        if self.tick_sleep:
            time.sleep(self.tick_sleep)
        vx, vy, vyaw = self.velocity
        x, y = self.pose.x + vx * dt, self.pose.y + vy * dt
        if self.obstacles is not None and not self.obstacles.is_free(x, y):
            self.in_contact = True
            x, y = self.pose.x, self.pose.y
        else:
            self.in_contact = False
        self.pose = Pose2D(x, y, wrap_angle(self.pose.yaw + vyaw * dt))
        self.time += dt
        self.ticks += 1
