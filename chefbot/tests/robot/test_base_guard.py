"""The base's stops in the robot layer (#1: moved out of scripts/real_nav.py): robot/safety.py BaseGuard and Robot's
drive, on the fake env. Every env and caller (the agent's go_to, real_nav, the whole-body controller) shares them."""

import math
import threading
import time
from dataclasses import replace

import numpy as np
import pytest

from chefbot.interface.types import EnvState, LocalizationReading, MotionStatus, Pose2D
from chefbot.interface.workspace import BODY
from chefbot.robot.nav.footprint import FootprintSpace
from chefbot.robot.nav.grid import OccupancyGrid
from chefbot.robot.nav.speed import StopModel
from chefbot.robot.robot import Robot
from chefbot.robot.safety import BaseGuard
from chefbot.tests.robot.fake_env import FakeEnv
from chefbot.tests.robot.test_robot import CFG, RingReach, room

HT4D = StopModel(tau_s=0.333, brake_mps2=0.25)


def plate(top=math.inf):
    return BODY.footprint()


def test_the_guard_stops_on_the_footprint_at_its_heading():
    occ = np.zeros((40, 80), dtype=bool)
    occ[:, 40:] = True  # a wall at x >= 2.0
    space = FootprintSpace(OccupancyGrid(occ, 0.05, (0.0, 0.0)), BODY.footprint())
    guard = BaseGuard(space, arm_moved_rad=None)
    state = EnvState(Pose2D(0, 0, 0), np.zeros(3), False, 0.0)
    assert guard.stop_reason(state, Pose2D(2.0 - 0.33, 1.0, 0.0)) is None  # end on: the plate's front 0.28 m
    assert "footprint" in guard.stop_reason(state, Pose2D(2.0 - 0.33, 1.0, math.radians(45)))  # a corner leads


def test_the_guard_caps_the_speed_by_the_measured_stop_and_the_free_space_ahead():
    occ = np.zeros((40, 80), dtype=bool)
    occ[:, 40:] = True
    space = FootprintSpace(OccupancyGrid(occ, 0.05, (0.0, 0.0)), BODY.footprint())
    guard = BaseGuard(space, stop_model=HT4D, arm_moved_rad=None)
    far, _ = guard.speed_cap(Pose2D(0.5, 1.0, 0.0), (1.0, 0.0), 0.25)
    near, free = guard.speed_cap(Pose2D(1.55, 1.0, 0.0), (1.0, 0.0), 0.25)
    assert far == pytest.approx(0.25) and near < far and HT4D.distance(near) <= free - guard.stop_buffer_m + 1e-9


def _robot(**kw):
    env = FakeEnv(Pose2D(1.0, 1.0, 0.0), tick_sleep=kw.pop("tick_sleep", 0.0))
    return Robot(env, room(), kw.pop("cfg", CFG), RingReach(), footprint_below=plate, **kw), env


def test_the_arm_moving_while_the_base_drives_stops_it():
    robot, env = _robot(tuck_arm=False, tick_sleep=0.02)
    robot.go_to_pose(Pose2D(4.0, 1.0, 0.0))
    deadline = time.monotonic() + 5
    while env.pose.x < 1.4 and time.monotonic() < deadline:
        time.sleep(0.005)
    env.arm_target = env.arm_target + np.r_[0.1, np.zeros(6)]  # someone moves the arm (and it stays moved)
    env.arm_q = env.arm_q + np.r_[0.1, np.zeros(6)]
    result = robot.wait()
    assert result.status == MotionStatus.BLOCKED and "the arm moved" in result.message
    assert np.all(env.commands[-1][1] == 0.0)


def test_a_fence_keeps_the_base_centre_in():
    fence = np.ones((120, 120), dtype=bool)
    fence[10:30, 10:50] = False  # the centre may be in x 0.5..2.5, y 0.5..1.5
    robot, env = _robot(centre_limit=OccupancyGrid(fence, 0.05, (0.0, 0.0)))
    robot.go_to_pose(Pose2D(4.0, 1.0, 0.0))
    result = robot.wait()
    assert result.status in (MotionStatus.NO_PATH, MotionStatus.BLOCKED)
    assert env.pose.x <= 2.55


def test_the_speed_never_exceeds_what_the_base_can_stop_in():
    robot, env = _robot(stop_model=HT4D, base_accel=0.25, cfg=replace(CFG, max_lin_vel=0.25))
    robot.go_to_pose(Pose2D(4.5, 1.0, 0.0))  # toward the room's east wall (x 6.0)
    assert robot.wait().status == MotionStatus.SUCCESS
    speeds = np.array([np.hypot(*c[:2]) for _, c in env.commands])
    assert speeds.max() <= 0.25 + 1e-9


def test_a_station_is_reached_straight_in_along_its_heading():
    robot, env = _robot()
    rows = []
    robot.drive_log = rows.append
    goal = Pose2D(3.0, 3.0, math.pi / 2)
    robot.go_to_pose(goal, approach_m=0.4)
    result = robot.wait()
    assert result.status == MotionStatus.SUCCESS, result.message
    last = [r for r in rows if r["leg"] == 1 and r["phase"] == "drive"]
    assert last and max(abs(r["pose"][0] - goal.x) for r in last) < 0.02  # along the normal
    assert result.data["path_m"] > 0.4


def test_choose_stance_picks_a_clear_reachable_one():
    grid = room(blocks=[(2.0, 2.0, 3.0, 3.0)])
    env = FakeEnv(Pose2D(1.0, 1.0, 0.0))
    robot = Robot(env, grid, CFG, RingReach(), footprint_below=plate)
    blocked = Pose2D(2.5, 2.5, 0.0)  # inside the block
    near, far = Pose2D(1.5, 2.5, 0.0), Pose2D(4.5, 4.5, math.pi)
    pick, why = robot.choose_stance([blocked, far, near])
    assert pick is near, why
    assert robot.choose_stance([blocked])[0] is None
