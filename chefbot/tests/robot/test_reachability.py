import math

import numpy as np
import pytest

from chefbot.interface.types import Pose2D
from chefbot.robot.arm.ik import ArmIK, arm_ik_from_config
from chefbot.robot.config import RobotConfig
from chefbot.robot.kinematics.reachability import ReachabilityChecker

HOME_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
CONFIG = RobotConfig(arm_mount_xyz=(0.1, 0.0, 0.5), tcp_xyz=(0.0, 0.0, 0.16))


@pytest.fixture(scope="module")
def checker():
    return ReachabilityChecker(CONFIG)


def test_point_in_front_of_arm_is_reachable(checker):
    assert checker.is_reachable(Pose2D(0.0, 0.0, 0.0), np.array([0.6, 0.0, 0.9]))


def test_point_two_meters_away_is_not(checker):
    assert not checker.is_reachable(Pose2D(0.0, 0.0, 0.0), np.array([2.0, 0.0, 0.9]))


def test_reachability_follows_the_base_pose(checker):
    target = np.array([3.0, 4.6, 0.9])
    assert not checker.is_reachable(Pose2D(0.0, 0.0, 0.0), target)
    assert checker.is_reachable(Pose2D(3.0, 4.0, math.pi / 2), target)


def test_a_point_at_the_very_edge_of_reach_does_not_count(checker):
    # Position-only IK gets there, but there is no room to approach it from above or in front.
    base, edge = Pose2D(0.0, 0.0, 0.0), np.array([1.1, 0.0, 0.9])
    assert checker.solve(base, edge) is not None
    assert not checker.is_reachable(base, edge)
    lenient = ReachabilityChecker(RobotConfig(arm_mount_xyz=(0.1, 0.0, 0.5), reach_clearance=0.0), checker.ik)
    assert lenient.is_reachable(base, edge)


def test_solution_puts_tcp_on_target(checker):
    base, target = Pose2D(1.0, -2.0, 0.7), np.array([1.5, -1.5, 1.0])
    q = checker.solve(base, target)
    assert q is not None
    ik = arm_ik_from_config(CONFIG)
    assert np.linalg.norm(ik.tcp_position(q) - checker.to_arm_frame(base, target)) <= 0.01
    assert ik.within_limits(q, margin=1e-6)


def test_frames_round_trip(checker):
    base = Pose2D(1.0, -2.0, 0.7)
    pose = np.eye(4)
    pose[:3, 3] = [1.4, -1.6, 1.1]
    assert checker.pose_to_world(base, checker.pose_to_arm_frame(base, pose)) == pytest.approx(pose, abs=1e-9)
    assert checker.pose_to_arm_frame(base, pose)[:3, 3] == pytest.approx(checker.to_arm_frame(base, pose[:3, 3]))


def test_tcp_offset_extends_the_flange():
    short, long = ArmIK(tcp_xyz=(0.0, 0.0, 0.0)), ArmIK(tcp_xyz=(0.0, 0.0, 0.2))
    # At home the flange z-axis points straight down.
    assert long.tcp_position(HOME_Q) - short.tcp_position(HOME_Q) == pytest.approx([0.0, 0.0, -0.2], abs=1e-3)


def test_tcp_yaw_turns_the_closing_axis():
    plain, turned = ArmIK(tcp_xyz=(0.0, 0.0, 0.1)), ArmIK(tcp_xyz=(0.0, 0.0, 0.1), tcp_yaw=math.pi / 2)
    a, b = plain.fk(HOME_Q), turned.fk(HOME_Q)
    assert b[:3, 3] == pytest.approx(a[:3, 3], abs=1e-9)
    assert b[:3, 0] == pytest.approx(a[:3, 1], abs=1e-6)  # x of the turned frame is the old y


def test_joint_limit_override():
    limits = [[-1.0, 1.0]] * 3 + [[-3.0, -0.1]] + [[-1.0, 1.0]] + [[0.0, 3.7]] + [[-1.0, 1.0]]
    ik = ArmIK(joint_limits=limits)
    assert ik.joint_limits == pytest.approx(np.array(limits))
    assert not ik.within_limits(HOME_Q * 0 + 2.0)
