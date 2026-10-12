import math

import numpy as np
import pytest

from chefbot.interface.types import Pose2D
from chefbot.robot.config import RobotConfig
from chefbot.robot.kinematics.fr3 import HOME_Q, FR3PositionIK
from chefbot.robot.kinematics.reachability import ReachabilityChecker


@pytest.fixture(scope="module")
def checker():
    return ReachabilityChecker(RobotConfig(arm_mount_xyz=(0.1, 0.0, 0.5), tcp_offset=0.16))


def test_point_in_front_of_arm_is_reachable(checker):
    assert checker.is_reachable(Pose2D(0.0, 0.0, 0.0), np.array([0.6, 0.0, 0.9]))


def test_point_two_meters_away_is_not(checker):
    assert not checker.is_reachable(Pose2D(0.0, 0.0, 0.0), np.array([2.0, 0.0, 0.9]))


def test_reachability_follows_the_base_pose(checker):
    target = np.array([3.0, 4.6, 0.9])
    assert not checker.is_reachable(Pose2D(0.0, 0.0, 0.0), target)
    assert checker.is_reachable(Pose2D(3.0, 4.0, math.pi / 2), target)


def test_solution_puts_tcp_on_target(checker):
    base, target = Pose2D(1.0, -2.0, 0.7), np.array([1.5, -1.5, 1.0])
    q = checker.solve(base, target)
    assert q is not None
    ik = FR3PositionIK(tcp_offset=0.16)
    assert np.linalg.norm(ik.tcp_position(q) - checker.to_arm_frame(base, target)) <= 0.01
    assert np.all(q >= ik.model.jnt_range[:, 0] - 1e-6)
    assert np.all(q <= ik.model.jnt_range[:, 1] + 1e-6)


def test_tcp_offset_extends_the_flange():
    short, long = FR3PositionIK(tcp_offset=0.0), FR3PositionIK(tcp_offset=0.2)
    # At home the flange z-axis points straight down.
    assert long.tcp_position(HOME_Q) - short.tcp_position(HOME_Q) == pytest.approx(
        [0.0, 0.0, -0.2], abs=1e-3
    )
