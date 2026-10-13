"""Interlocks between the base and the arm (MANIP_SPEC §5.3).

Enforced by the robot layer whatever the agent asks for: the base and the arm
never move at the same time, the arm is tucked before the base drives, and arm
skills only start with the base at rest.
"""

from __future__ import annotations

import math
import threading

import numpy as np

from chefbot.interface.env import Env
from chefbot.interface.types import HeldObject, MotionStatus
from chefbot.robot.arm.arm_controller import OK, ArmController, ArmResult
from chefbot.robot.config import RobotConfig

# The base counts as at rest below these speeds.
REST_LIN_VEL = 0.005  # m/s
REST_ANG_VEL = 0.01  # rad/s


def base_at_rest(env: Env) -> bool:
    vx, vy, vyaw = env.get_state().base_velocity
    return math.hypot(vx, vy) < REST_LIN_VEL and abs(vyaw) < REST_ANG_VEL


def driving_posture(config: RobotConfig, held: HeldObject | None) -> np.ndarray:
    """The arm configuration for base motion: `carry` when holding something, else `stow`."""
    return np.asarray(config.arm_carry if held is not None else config.arm_stow, dtype=float)


def tuck_arm_for_driving(
    arm: ArmController, config: RobotConfig, held: HeldObject | None, preempt: threading.Event
) -> ArmResult:
    """Move the arm to its driving posture unless it is already there."""
    posture = driving_posture(config, held)
    if arm.at(posture):
        return OK
    return arm.move_joints(posture, preempt)


def require_base_at_rest(env: Env) -> ArmResult:
    """The check every arm skill starts with."""
    if base_at_rest(env):
        return OK
    return ArmResult(MotionStatus.BLOCKED, "the base is moving; arm skills need it stopped")
