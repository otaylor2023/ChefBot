"""The arm's kinematics as the whole-body controller uses them (robot/control/wholebody.py): the IK on the flange,
the frames on the base, the arm step by the robot's arm solver around its constraints. Built from a robot's
embodiment (embodiments.load(name).kin).
"""

from __future__ import annotations

import math

import numpy as np

from chefbot.interface.types import Pose2D
from chefbot.robot.arm.ik import ArmIK
from chefbot.robot.config import RobotConfig

# The controller's IK steps are a few mm: tighter than the robot layer's execution tolerance, or a step would count as
# reached at its seed and never move.
IK_POS_TOLERANCE, IK_ROT_TOLERANCE = 0.0005, math.radians(0.3)


class Kinematics:
    """The arm's IK on the flange and the frames everything shares: the root on the base, the flange."""

    def __init__(self, config: RobotConfig, body=None, solver: str = "qp"):
        from chefbot.robot.control import constraints

        self.config = config
        self.body = body  # interface/workspace.py Body or None
        self.constraints = constraints.for_robot(body)  # where the arm may not go: every solver avoids them
        self.solver = solver  # robot/control/arm_solvers: the controller's arm solver (set before the first arm_step)
        self._solver = None
        self.log = None  # where the router says it's going around (or can't)
        self.ik = ArmIK(tcp_xyz=(0.0, 0.0, 0.0), joint_limits=config.joint_limits, posture=np.asarray(config.arm_ready),
                        pos_tolerance=config.ik_pos_tolerance, pose_pos_tolerance=IK_POS_TOLERANCE,
                        pose_rot_tolerance=IK_ROT_TOLERANCE, max_iters=config.ik_max_iters, dt=config.ik_dt,
                        damping=config.ik_damping, arm=config.arm)  # fmt: skip
        self.mount_xyz = np.asarray(config.arm_mount_xyz, dtype=float)
        self.mount_yaw = config.arm_mount_yaw

    def root_in_world(self, base: Pose2D) -> np.ndarray:
        """The arm base frame (4x4) for a base pose (the base center on the floor)."""
        yaw = base.yaw + self.mount_yaw
        c, s = math.cos(yaw), math.sin(yaw)
        cb, sb = math.cos(base.yaw), math.sin(base.yaw)
        m = self.mount_xyz
        T = np.eye(4)
        T[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
        T[:3, 3] = [base.x + cb * m[0] - sb * m[1], base.y + sb * m[0] + cb * m[1], m[2]]
        return T

    def base_for_root(self, root_world: np.ndarray) -> Pose2D:
        """The base pose that puts the robot root at `root_world`."""
        m = self.mount_xyz
        yaw = math.atan2(root_world[1, 0], root_world[0, 0]) - self.mount_yaw
        c, s = math.cos(yaw), math.sin(yaw)
        return Pose2D(float(root_world[0, 3] - (c * m[0] - s * m[1])), float(root_world[1, 3] - (s * m[0] + c * m[1])), float(yaw))

    def flange_pose(self, q) -> np.ndarray:
        """The flange (4x4) in the arm base frame, for joints q."""
        return self.ik.fk(np.asarray(q, dtype=float))

    def arm_step(self, target: np.ndarray, q, dt: float, max_lin: float, max_ang: float, max_joint: float) -> np.ndarray:
        """The controller's arm step: the joints after dt from q toward the flange pose `target`, within the joint
        limits and clear of the constraints, by the chosen solver (robot/control/arm_solvers). Raises mink's
        NoSolutionFound when it can't (the controller holds)."""
        if self._solver is None:
            from chefbot.robot.control import arm_solvers

            hand = self.body.hand_m if self.body is not None else 0.0
            self._solver = arm_solvers.make(self.solver, self.ik, self.constraints, hand, self.log)
        return self._solver.step(target, q, dt, max_lin, max_ang, max_joint)
