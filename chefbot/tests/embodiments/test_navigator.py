"""The embodiment's navigator and base guard: every caller that drives the base gets the same footprint and stop."""

import math

import numpy as np
import pytest

from chefbot import embodiments
from chefbot.robot.nav.grid import OccupancyGrid
from chefbot.tests.robot.fake_env import FakeEnv

GRID = OccupancyGrid(np.zeros((40, 40), dtype=bool), 0.05, (-1.0, -1.0))


def test_the_fr3_drives_with_its_measured_plate_and_stop():
    robot = embodiments.load("tidybot_fr3")
    nav = robot.navigator(FakeEnv(), GRID, robot.speed(base="teleop", use_arm=False))
    plate = nav._footprint_below(math.inf)
    assert plate.rect == pytest.approx(robot.body.footprint().rect)
    assert len(plate.discs) == 0  # no arm given: the plate alone, and the arm is tucked before driving
    assert nav._tuck_arm
    assert nav._stop_model is not None and nav._stop_model == robot.stop_model()
    assert nav._base_accel == pytest.approx(0.25)
    assert nav._cfg.max_lin_vel == pytest.approx(0.25) and nav._cfg.max_ang_vel == pytest.approx(0.8)


def test_with_the_arm_as_it_stands_the_footprint_covers_it_and_it_stays():
    robot = embodiments.load("tidybot_fr3")
    q = robot.config.arm_ready
    nav = robot.navigator(FakeEnv(), GRID, robot.speed(base="teleop", use_arm=False), arm_q=q)
    assert len(nav._footprint_below(math.inf).discs) > 0 and not nav._tuck_arm


def test_a_robot_without_a_body_keeps_the_configs_circle():
    robot = embodiments.load("panda_franka")
    fp = robot.footprint_below()(math.inf)
    assert fp.radius == pytest.approx(robot.config.footprint_radius)
    assert robot.stop_model() is None


def test_a_speed_with_no_base_speed_is_refused():
    robot = embodiments.load("tidybot_fr3")
    with pytest.raises(ValueError, match="no speed"):
        robot.navigator(FakeEnv(), GRID, robot.speed(use_base=False))
