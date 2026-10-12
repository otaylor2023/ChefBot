"""ReachabilityChecker: can the arm reach a world point from a given base pose?"""

from __future__ import annotations

import math

import numpy as np

from chefbot.interface.types import Pose2D
from chefbot.robot.config import RobotConfig
from chefbot.robot.kinematics.fr3 import FR3PositionIK


class ReachabilityChecker:
    def __init__(self, config: RobotConfig):
        self._mount_xyz = np.asarray(config.arm_mount_xyz, dtype=float)
        self._mount_yaw = config.arm_mount_yaw
        self._ik = FR3PositionIK(
            tcp_offset=config.tcp_offset,
            pos_tolerance=config.ik_pos_tolerance,
            max_iters=config.ik_max_iters,
            dt=config.ik_dt,
            damping=config.ik_damping,
        )

    def to_arm_frame(self, base_pose: Pose2D, target_world: np.ndarray) -> np.ndarray:
        target_world = np.asarray(target_world, dtype=float)
        in_base = _rotate_z(target_world - [base_pose.x, base_pose.y, 0.0], -base_pose.yaw)
        return _rotate_z(in_base - self._mount_xyz, -self._mount_yaw)

    def solve(self, base_pose: Pose2D, target_world: np.ndarray) -> np.ndarray | None:
        """Arm joint angles that put the tcp on `target_world`, or None."""
        return self._ik.solve(self.to_arm_frame(base_pose, target_world))

    def is_reachable(self, base_pose: Pose2D, target_world: np.ndarray) -> bool:
        return self.solve(base_pose, target_world) is not None


def _rotate_z(v: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])
