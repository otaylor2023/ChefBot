"""What the skills share: the arm stack they run on, motion plans, and pose helpers."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from itertools import product

import numpy as np

from chefbot.interface.env import Env
from chefbot.interface.types import MotionStatus, Pose2D
from chefbot.robot.arm.arm_controller import OK, ArmController, ArmResult
from chefbot.robot.arm.gripper import Gripper
from chefbot.robot.arm.ik import ArmIK
from chefbot.robot.config import RobotConfig
from chefbot.robot.kinematics.collision import ArmCollisionModel, Workspace, snapshot
from chefbot.robot.kinematics.reachability import ReachabilityChecker

_BOX_CORNERS = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])


@dataclass
class ArmStack:
    env: Env
    config: RobotConfig
    ik: ArmIK
    arm: ArmController
    gripper: Gripper
    frames: ReachabilityChecker
    collision: ArmCollisionModel | None = None

    @property
    def ready(self) -> np.ndarray:
        return np.asarray(self.config.arm_ready, dtype=float)

    def base_pose(self) -> Pose2D:
        return self.env.get_state().base_pose

    def to_arm(self, pose_world: np.ndarray) -> np.ndarray:
        """A world-frame TCP pose in the arm base frame, for the current base pose."""
        return self.frames.pose_to_arm_frame(self.base_pose(), pose_world)

    def tcp_world(self) -> np.ndarray:
        """Where the TCP is now (world frame), from the measured joints."""
        return self.frames.pose_to_world(self.base_pose(), self.ik.fk(self.env.get_state().arm_q))

    def workspace(self, ignore_boxes: list[np.ndarray] = (), with_held: bool = False) -> Workspace:
        """The world as the planning camera sees it now, for one skill's collision
        checks. Unchecked (everything free) without a camera or a collision model."""
        cfg = self.config
        if self.collision is None or not cfg.planning_cameras:
            return Workspace.unchecked()
        try:
            images = [self.env.get_camera_rgbd(camera) for camera in cfg.planning_cameras]
        except (NotImplementedError, RuntimeError):
            return Workspace.unchecked()
        base, q = self.base_pose(), self.env.get_state().arm_q
        arm_base = self.frames.arm_base_in_world(base)
        boxes = [b for b in ignore_boxes if b is not None]
        points = np.concatenate([
            snapshot(image, base, arm_base, q, self.collision, cfg.self_filter_margin, cfg.base_box, boxes)
            for image in images
        ])  # fmt: skip
        return Workspace(self.collision, points, cfg.planning_voxel, self.ik.joint_limits, cfg.planning_timeout_s, with_held)


@dataclass
class ApproachPlan:
    """Joint-space plan for one candidate: `into` takes the arm from `ready` to the
    approach pose, then a straight line in to the action pose, a straight line
    out, and `back` from there to `ready`."""

    into: list[np.ndarray]
    line_in: list[np.ndarray]
    line_out: list[np.ndarray]
    back: list[np.ndarray]

    @property
    def q_approach(self) -> np.ndarray:
        return self.into[-1]


def plan_approach(
    stack: ArmStack, world: Workspace, approaches: list[np.ndarray], action: np.ndarray, leaves: list[np.ndarray]
) -> ApproachPlan | None:
    """The first collision-free approach / action / leave sequence (arm-frame
    poses), trying each approach pose and each leave pose in order (the full
    distances first, then shorter ones for tight spots), or None."""
    tried: dict[int, list[np.ndarray] | None] = {}
    for i, j in product(range(len(approaches)), range(len(leaves))):
        if i not in tried:
            tried[i] = _approach_line(stack, world, approaches[i], action)
        line_in = tried[i]
        if line_in is None:
            continue
        line_out = stack.arm.plan_line(line_in[-1], leaves[j])
        if line_out is None or not world.line_free(line_out):
            continue
        back = world.path(line_out[-1], stack.ready)
        if back is None:
            continue
        into = world.path(stack.ready, line_in[0])
        if into is None:
            tried[i] = None  # no way to get there: no leave pose will help
            continue
        return ApproachPlan(into, line_in, line_out, back)
    return None


def _approach_line(stack: ArmStack, world: Workspace, approach: np.ndarray, action: np.ndarray):
    q_approach = stack.ik.solve_pose(approach, seed=stack.ready)
    if q_approach is None or not world.free(q_approach):
        return None
    line_in = stack.arm.plan_line(q_approach, action)
    if line_in is None or not world.line_free(line_in):
        return None
    return line_in


def distances(full: float, shorter: tuple) -> list[float]:
    """`full`, then each of the shorter distances to fall back on."""
    return [full] + [d for d in shorter if d < full - 1e-9]


def follow_path(stack: ArmStack, path: list[np.ndarray], preempt: threading.Event) -> ArmResult:
    """Joint moves through a path's waypoints (the first is where the arm is)."""
    for q in path[1:]:
        result = stack.arm.move_joints(q, preempt)
        if not result.ok:
            return result
    return OK


def to_ready(stack: ArmStack, world: Workspace, preempt: threading.Event) -> ArmResult:
    """From wherever the arm is to `ready`, around what the camera sees."""
    path = world.path(stack.arm.current_target(), stack.ready)
    if path is None:
        return ArmResult(MotionStatus.UNREACHABLE, "no collision-free way to move the arm to its ready pose")
    return follow_path(stack, path, preempt)


def box_corners(box: np.ndarray) -> np.ndarray:
    """The 8 corners (8, 3) of a (2, 3) min / max box."""
    box = np.asarray(box, dtype=float)
    return box[0] + _BOX_CORNERS * (box[1] - box[0])


def shifted(pose: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """`pose` translated by `offset` (same frame), orientation kept."""
    out = np.array(pose, dtype=float)
    out[:3, 3] += np.asarray(offset, dtype=float)
    return out
