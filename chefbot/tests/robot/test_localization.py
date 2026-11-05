"""robot/localization.py, and the robot layer driving on odometry tied to the world by an env's localization (#5):
base_pose stays odometry, mocap (or a sim's ground truth) is its own reading, and no env is driven without it."""

import math
import threading
import time

import numpy as np
import pytest

from chefbot.interface.types import EnvState, LocalizationReading, MotionStatus, Pose2D, wrap_angle
from chefbot.robot.localization import Localizer, compose, inverse
from chefbot.robot.robot import Robot
from chefbot.tests.robot.fake_env import FakeEnv
from chefbot.tests.robot.test_robot import CFG, RingReach, room


def _state(odom, reading=None, age=0.0, has=True):
    return EnvState(odom, np.zeros(3), False, 0.0, localization=LocalizationReading(reading, age) if has else None)


def test_compose_and_inverse():
    a, b = Pose2D(1.0, 2.0, 0.7), Pose2D(-0.3, 0.5, -2.0)
    back = compose(inverse(a), compose(a, b))
    assert (back.x, back.y, back.yaw) == pytest.approx((b.x, b.y, b.yaw))


def test_a_live_reading_ties_odometry_to_the_world():
    loc = Localizer()
    odom, world = Pose2D(1.0, 2.0, 0.5), Pose2D(3.0, 4.0, 1.0)
    assert loc.update(_state(odom, world))
    w = loc.world(odom)
    assert (w.x, w.y, w.yaw) == pytest.approx((3.0, 4.0, 1.0))
    o = loc.to_odom(Pose2D(3.0, 4.0, 1.0))
    assert (o.x, o.y, o.yaw) == pytest.approx((1.0, 2.0, 0.5))
    vx, vy = loc.velocity_to_odom(math.cos(0.5), math.sin(0.5))  # along the tie's yaw (0.5) in the world
    assert (vx, vy) == pytest.approx((1.0, 0.0))


def test_between_readings_the_base_moves_on_odometry():
    loc = Localizer()
    loc.update(_state(Pose2D(0.0, 0.0, 0.0), Pose2D(5.0, 5.0, math.pi / 2)))
    assert not loc.update(_state(Pose2D(1.0, 0.0, 0.0), None))  # lost: the last tie stands
    w = loc.world(Pose2D(1.0, 0.0, 0.0))
    assert (w.x, w.y) == pytest.approx((5.0, 6.0))


def test_why_not_live():
    loc = Localizer(max_age_s=0.5)
    assert loc.why_not_live(_state(Pose2D(0, 0, 0), has=False)) is None  # no localizer: odometry is the world
    assert loc.why_not_live(_state(Pose2D(0, 0, 0), Pose2D(1, 1, 0))) is None
    assert "no localization" in loc.why_not_live(_state(Pose2D(0, 0, 0), None))
    assert "old" in loc.why_not_live(_state(Pose2D(0, 0, 0), Pose2D(1, 1, 0), age=0.8))


# -- the robot layer --------------------------------------------------------------------------


def _robot(start=Pose2D(1.0, 1.0, 0.0), odom_origin=Pose2D(0.4, -0.3, 0.6), **kw):
    env = FakeEnv(start, localizer=True, odom_origin=odom_origin, **kw)
    return Robot(env, room(), CFG, RingReach()), env


def test_the_robot_reaches_a_world_goal_on_odometry_in_another_frame():
    robot, env = _robot()
    goal = Pose2D(4.0, 2.5, 1.2)
    p = robot.get_base_pose()
    assert (p.x, p.y, p.yaw) == pytest.approx((1.0, 1.0, 0.0))  # the world pose, not odometry
    assert env.odom().x != pytest.approx(1.0)
    robot.go_to_pose(goal)
    result = robot.wait()
    assert result.status == MotionStatus.SUCCESS, result.message
    assert env.pose.distance_to(goal) <= CFG.pos_tolerance + 1e-3  # truly there, in the world
    assert abs(wrap_angle(env.pose.yaw - goal.yaw)) <= CFG.yaw_tolerance + 1e-3


def test_drifting_odometry_is_corrected_by_each_reading():
    robot, env = _robot(tick_sleep=0.001)
    goal = Pose2D(4.0, 1.0, 0.0)
    robot.go_to_pose(goal)
    time.sleep(0.05)
    o = env.odom_origin
    env.odom_origin = Pose2D(o.x + 0.2, o.y - 0.1, o.yaw + 0.1)  # odometry jumps (drift) mid-move
    result = robot.wait()
    assert result.status == MotionStatus.SUCCESS, result.message
    assert env.pose.distance_to(goal) <= CFG.pos_tolerance + 1e-3


def test_no_drive_without_localization():
    robot, env = _robot()
    env.localization_available = False
    robot.go_to_pose(Pose2D(4.0, 1.0, 0.0))
    result = robot.wait()
    assert result.status == MotionStatus.NOT_LOCALIZED and "localization" in result.message
    assert env.pose.distance_to(Pose2D(1.0, 1.0, 0.0)) < 1e-9  # never moved


def test_losing_localization_mid_move_stops_the_base():
    robot, env = _robot(tick_sleep=0.002)
    robot.go_to_pose(Pose2D(5.0, 1.0, 0.0))
    deadline = time.monotonic() + 5
    while env.pose.x < 1.5 and time.monotonic() < deadline:
        time.sleep(0.005)
    env.localization_available = False
    result = robot.wait()
    assert result.status == MotionStatus.NOT_LOCALIZED
    assert np.all(env.commands[-1][1] == 0.0)  # velocity zeroed on exit
    assert env.pose.x < 4.0


def test_an_env_without_a_localizer_drives_on_odometry_as_before():
    env = FakeEnv(Pose2D(1.0, 1.0, 0.0))
    robot = Robot(env, room(), CFG, RingReach())
    robot.go_to_pose(Pose2D(3.0, 1.0, 0.0))
    assert robot.wait().status == MotionStatus.SUCCESS
