"""The robot layer on an env whose base and arm move to poses themselves (Env.set_base_target,
Env.set_arm_pose_target), as the real robot's drivers do."""

import math
import threading

import numpy as np

from chefbot.interface.types import MotionStatus, Pose2D, wrap_angle
from chefbot.robot.arm.arm_controller import ArmController
from chefbot.robot.arm.ik import ArmIK, arm_ik_from_config
from chefbot.robot.config import RobotConfig
from chefbot.robot.robot import Robot
from chefbot.tests.robot.fake_env import FakeEnv
from chefbot.tests.robot.test_robot import RingReach, room

CFG = RobotConfig(footprint_radius=0.2, safety_margin=0.05)
IK = arm_ik_from_config(CFG)
DOWN = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


class DriverEnv(FakeEnv):
    """FakeEnv with a base that drives to pose targets and an arm that moves to flange poses
    (solved here by its own IK on the flange, standing in for the arm's driver)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_targets: list[Pose2D] = []
        self.flange_targets: list[np.ndarray] = []
        self._driver_ik = ArmIK(tcp_xyz=(0.0, 0.0, 0.0), arm=CFG.arm)

    def set_base_target(self, pose, max_vel) -> bool:
        self.base_targets.append(pose)
        err = pose.xy - self.pose.xy
        v = 2.0 * err
        if np.linalg.norm(v) > max_vel[0]:
            v *= max_vel[0] / np.linalg.norm(v)
        w = float(np.clip(2.5 * wrap_angle(pose.yaw - self.pose.yaw), -max_vel[2], max_vel[2]))
        self.velocity = np.array([v[0], v[1], w])
        return True

    def set_arm_pose_target(self, flange_pose) -> bool:
        self.flange_targets.append(np.array(flange_pose))
        q = self._driver_ik.solve_pose(flange_pose, seed=self.arm_q)
        if q is not None:
            self.arm_target = q
        return True


def test_base_drives_by_pose_targets_and_never_by_velocity():
    env = DriverEnv(Pose2D(1.0, 1.0, 0.0))
    robot = Robot(env, room(), CFG, RingReach())
    goal = Pose2D(4.0, 2.0, math.pi / 2)
    robot.go_to_pose(goal)
    assert robot.wait().status == MotionStatus.SUCCESS
    assert env.pose.distance_to(goal) <= CFG.pos_tolerance
    assert env.base_targets and env.base_targets[-1] == goal
    # Only the zero velocities of stopping: the pose targets did the driving.
    assert all(np.all(v == 0.0) for _, v in env.commands)


def test_lines_go_to_the_arm_as_flange_poses_along_the_planned_path():
    env = DriverEnv(arm_q=np.array(CFG.arm_ready))
    arm = ArmController(env, IK, CFG)
    start = IK.fk(env.arm_q)
    goal = start.copy()
    goal[:3, 3] += [0.10, 0.05, -0.08]
    line = arm.plan_line(env.arm_q, goal)
    assert line is not None
    result = arm.follow(line, threading.Event())
    assert result.ok, result.message
    assert np.linalg.norm(IK.fk(env.arm_q)[:3, 3] - goal[:3, 3]) <= CFG.cart_goal_tolerance
    assert len(env.flange_targets) > 10
    # The flange targets are the TCP poses with the TCP offset taken off.
    tcp_offset = np.eye(4)
    c, s = math.cos(CFG.tcp_yaw), math.sin(CFG.tcp_yaw)
    tcp_offset[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    tcp_offset[:3, 3] = CFG.tcp_xyz
    np.testing.assert_allclose(env.flange_targets[-1] @ tcp_offset, goal, atol=1e-3)
    # No joint targets streamed during the line.
    assert not env.arm_targets


def test_a_blocked_arm_stops_the_line():
    env = DriverEnv(arm_q=np.array(CFG.arm_ready))
    env.set_arm_pose_target = lambda flange_pose: True  # takes targets, never moves
    arm = ArmController(env, IK, CFG)
    goal = IK.fk(env.arm_q)
    goal[:3, 3] += [0.0, 0.0, -0.15]
    result = arm.follow(arm.plan_line(env.arm_q, goal), threading.Event())
    assert result.status == MotionStatus.COLLISION
    np.testing.assert_allclose(env.arm_targets[-1][1], env.arm_q)  # held where it is


def test_an_env_without_pose_targets_gets_joints_as_before():
    env = FakeEnv(arm_q=np.array(CFG.arm_ready))
    arm = ArmController(env, IK, CFG)
    goal = IK.fk(env.arm_q)
    goal[:3, 3] += [0.05, 0.0, 0.0]
    assert arm.follow(arm.plan_line(env.arm_q, goal), threading.Event()).ok
    assert env.arm_targets
