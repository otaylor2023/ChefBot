"""ReachabilityChecker: can the arm work at a world point from a given base pose?

Used by `align_to_point` to pick a standoff pose, and by the skills to turn
world-frame targets into the arm base frame.
"""

from __future__ import annotations

import math

import numpy as np

from chefbot.interface.types import Pose2D
from chefbot.robot.arm.ik import ArmIK, arm_ik_from_config
from chefbot.robot.config import RobotConfig



def _approach(pitch_deg: float) -> np.ndarray:
    """TCP orientation in the arm base frame for a gripper pointing along +x,
    pitched down by `pitch_deg` (90: straight down). Fingers close sideways (y)."""
    pitch = math.radians(pitch_deg)
    z = np.array([math.cos(pitch), 0.0, -math.sin(pitch)])
    x = np.array([0.0, -1.0, 0.0])
    return np.stack([x, np.cross(z, x), z], axis=1)


# The ways a skill may come at a target: from above for things at counter
# height, level or angled down for things higher up (on a shelf, a door handle).
_FROM_ABOVE = (_approach(90.0),)
_FROM_THE_FRONT = (_approach(0.0), _approach(30.0))


class ReachabilityChecker:
    def __init__(self, config: RobotConfig, ik: ArmIK | None = None):
        self._mount_xyz = np.asarray(config.arm_mount_xyz, dtype=float)
        self._mount_yaw = config.arm_mount_yaw
        self._clearance = config.reach_clearance
        self._top_down_max_height = config.top_down_max_height
        self.ik = ik or arm_ik_from_config(config)

    # -- frames -----------------------------------------------------------------

    def arm_base_in_world(self, base_pose: Pose2D) -> np.ndarray:
        """World pose (4x4) of the arm base frame for a base pose."""
        yaw = base_pose.yaw + self._mount_yaw
        c, s = math.cos(yaw), math.sin(yaw)
        pose = np.eye(4)
        pose[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
        pose[:3, 3] = [base_pose.x, base_pose.y, 0.0] + _rotate_z(self._mount_xyz, base_pose.yaw)
        return pose

    def pose_to_arm_frame(self, base_pose: Pose2D, pose_world: np.ndarray) -> np.ndarray:
        return np.linalg.inv(self.arm_base_in_world(base_pose)) @ np.asarray(pose_world, dtype=float)

    def pose_to_world(self, base_pose: Pose2D, pose_arm: np.ndarray) -> np.ndarray:
        return self.arm_base_in_world(base_pose) @ np.asarray(pose_arm, dtype=float)

    def to_arm_frame(self, base_pose: Pose2D, target_world: np.ndarray) -> np.ndarray:
        point = np.append(np.asarray(target_world, dtype=float), 1.0)
        return (np.linalg.inv(self.arm_base_in_world(base_pose)) @ point)[:3]

    # -- reachability -------------------------------------------------------------

    def solve(self, base_pose: Pose2D, target_world: np.ndarray) -> np.ndarray | None:
        """Arm joint angles that put the tcp on `target_world` (any orientation), or None."""
        return self.ik.solve_position(self.to_arm_frame(base_pose, target_world))

    def can_touch(self, base_pose: Pose2D, target_world: np.ndarray) -> bool:
        """The gripper can get to `target_world` in some orientation (position-only IK)."""
        return self.solve(base_pose, target_world) is not None

    def is_reachable(self, base_pose: Pose2D, target_world: np.ndarray) -> bool:
        """True if the gripper can get to `target_world` with room to work.

        Position-only IK is the fast first test (NAV_SPEC §6.5). A point that
        passes is then required to be reachable the way a skill would approach
        it: gripper down from `reach_clearance` above for targets up to
        `top_down_max_height`, level (or angled down) from `reach_clearance` in
        front for higher ones. That keeps standoff poses off the very edge of
        the arm's reach.
        """
        target = self.to_arm_frame(base_pose, target_world)
        if self.ik.solve_position(target) is None:
            return False
        if self._clearance <= 0.0:
            return True
        heading = math.atan2(target[1], target[0])
        c, s = math.cos(heading), math.sin(heading)
        turn = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        approaches = _FROM_ABOVE if float(target_world[2]) <= self._top_down_max_height else _FROM_THE_FRONT
        for rotation in (turn @ approach for approach in approaches):
            pose = np.eye(4)
            pose[:3, :3] = rotation
            pose[:3, 3] = target
            at_target = self.ik.solve_pose(pose)
            if at_target is None:
                continue
            pose[:3, 3] = target - self._clearance * rotation[:, 2]
            if self.ik.solve_pose(pose, seed=at_target) is not None:
                return True
        return False


def _rotate_z(v: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])
