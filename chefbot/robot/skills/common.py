"""What the skills share: the arm stack they run on, and pose helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chefbot.interface.env import Env
from chefbot.interface.types import Pose2D
from chefbot.robot.arm.arm_controller import ArmController
from chefbot.robot.arm.gripper import Gripper
from chefbot.robot.arm.ik import ArmIK
from chefbot.robot.config import RobotConfig
from chefbot.robot.kinematics.reachability import ReachabilityChecker


@dataclass
class ArmStack:
    env: Env
    config: RobotConfig
    ik: ArmIK
    arm: ArmController
    gripper: Gripper
    frames: ReachabilityChecker

    @property
    def ready(self) -> np.ndarray:
        return np.asarray(self.config.arm_ready, dtype=float)

    def base_pose(self) -> Pose2D:
        return self.env.get_state().base_pose

    def to_arm(self, pose_world: np.ndarray) -> np.ndarray:
        """A world-frame TCP pose in the arm base frame, for the current base pose."""
        return self.frames.pose_to_arm_frame(self.base_pose(), pose_world)


@dataclass
class ApproachPlan:
    """Joint-space plan for one candidate: reach the approach pose from `ready`,
    then a straight line in to the action pose, then a straight line out."""

    q_approach: np.ndarray
    line_in: list[np.ndarray]
    line_out: list[np.ndarray]


def plan_approach(stack: ArmStack, approach: np.ndarray, action: np.ndarray, leave: np.ndarray) -> ApproachPlan | None:
    """IK for the whole approach / action / leave sequence (arm-frame poses), or None."""
    q_approach = stack.ik.solve_pose(approach, seed=stack.ready)
    if q_approach is None:
        return None
    line_in = stack.arm.plan_line(q_approach, action)
    if line_in is None:
        return None
    line_out = stack.arm.plan_line(line_in[-1], leave)
    if line_out is None:
        return None
    return ApproachPlan(q_approach, line_in, line_out)


def shifted(pose: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """`pose` translated by `offset` (same frame), orientation kept."""
    out = np.array(pose, dtype=float)
    out[:3, 3] += np.asarray(offset, dtype=float)
    return out
