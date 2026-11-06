"""The agent's arm keeps out of the robot's own body: a planned move whose path would go into one of its constraints
(robot/control/constraints.py: the pole, the camera on its clamp, the base's plate) is refused before it starts, as
the real host's guard would stop it. Without a body (RoboCasa's PandaOmron) nothing is refused."""

import threading

import numpy as np
import pytest

from chefbot import embodiments
from chefbot.interface.types import MotionStatus
from chefbot.robot.arm.arm_controller import ArmController
from chefbot.robot.control.constraints import for_robot, path_violation
from chefbot.robot.robot import Robot
from chefbot.tests.robot.fake_env import FakeEnv
from chefbot.tests.robot.test_arm import CFG, IK, START, room

FR3 = embodiments.load("tidybot_fr3")
BODY = FR3.body
CONSTRAINTS = for_robot(BODY)
READY = np.array(FR3.config.arm_ready, dtype=float)  # clear of the body, a few mm from the camera clamp


def turned(q0: float) -> np.ndarray:
    """The ready pose with joint 1 turned to q0: 0.3 rad stays clear, 1.9 swings through the pole; from -0.5 (in the
    camera clamp's keep-out) to 0 it only comes out, toward -1.1 it goes further in."""
    q = READY.copy()
    q[0] = q0
    return q


def test_a_clear_path_passes():
    assert path_violation(CONSTRAINTS, BODY.hand_m, [READY, turned(0.3)]) is None


def test_a_path_through_the_pole_is_named():
    where = path_violation(CONSTRAINTS, BODY.hand_m, [READY, turned(1.9)])
    assert where is not None and "pole" in where and "cm in" in where


def test_out_of_the_body_is_let_be_deeper_is_not():
    assert path_violation(CONSTRAINTS, BODY.hand_m, [turned(-0.5), READY]) is None  # moving out of it
    assert path_violation(CONSTRAINTS, BODY.hand_m, [turned(-0.5), turned(-1.1)]) is not None  # further in


def test_without_a_body_nothing_is_refused():
    assert path_violation([], 0.0, [READY, turned(1.9)]) is None


def arm(q=READY, constraints=CONSTRAINTS):
    env = FakeEnv(START, arm_q=q)
    return ArmController(env, IK, CFG, constraints, BODY.hand_m), env


def test_a_joint_move_into_the_pole_is_refused_before_it_starts():
    controller, env = arm()
    result = controller.move_joints(turned(1.9), threading.Event())
    assert result.status == MotionStatus.UNREACHABLE and "the arm would go into the keep-out of the pole" in result.message
    assert env.arm_targets == []  # nothing was sent


def test_waypoints_into_the_pole_are_refused_before_they_start():
    controller, env = arm()
    result = controller.follow([READY, turned(0.9), turned(1.9)], threading.Event())
    assert result.status == MotionStatus.UNREACHABLE and env.arm_targets == []


def test_a_clear_joint_move_runs():
    controller, env = arm()
    assert controller.move_joints(turned(0.3), threading.Event()).ok and env.arm_targets


@pytest.mark.parametrize("body, refused", [(None, False), (BODY, True)])
def test_the_robot_gives_its_arm_its_body(body, refused):
    robot = Robot(FakeEnv(START, arm_q=READY), room(), CFG, body=body)
    result = robot._arm.arm.move_joints(turned(1.9), threading.Event())
    assert (result.status == MotionStatus.UNREACHABLE) == refused


def test_the_arm_and_the_real_hosts_guard_agree_on_every_path():
    # One rule, two places: the agent's planned move is refused exactly when the real host would refuse its path.
    from chefbot.environments.real.host import GuardedEnv
    from chefbot.tests.onboard.test_real_host import FakeRobot

    guard = GuardedEnv(FakeRobot(), FR3)
    angles = (-1.9, -1.1, -0.5, 0.0, 0.3, 1.1, 1.9, 2.5)
    for a in angles:
        for b in angles:
            ours = path_violation(CONSTRAINTS, BODY.hand_m, [turned(a), turned(b)]) is None
            assert ours == guard.path_clear(turned(a), turned(b)), (a, b)
