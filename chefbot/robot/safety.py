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


# -- the base's stops while it drives -------------------------------------------------------


class BaseStopped(RuntimeError):
    """A base stop (BaseGuard.stop_reason) in a loop that cannot return a MotionResult (the whole-body controller)."""

ARM_MOVED_RAD = 0.05  # a joint moving more than this while the base drives stops it


class BaseGuard:
    """The stops and the speed limit every loop that drives the base shares (Robot's drive, the whole-body controller
    under the real host with the base on), whatever asked for the move. Per tick, with the state and the base's world
    pose:

      stop_reason   the env's localization lost or stale (robot/localization.py); the footprint reaching an obstacle
                    on the map at the base's heading (robot/nav/footprint.py); the base's centre outside a fence
                    (`centre_limit`); the arm moving more than `arm_moved_rad` (None: not checked) since begin().
      speed_cap     the fastest the base may go along `way` and still stop, by the robot's measured stop
                    (robot/nav/speed.py StopModel), within the free space ahead on the map (the footprint at its
                    heading), less `stop_buffer_m`; the full speed without a stop model or a map.
      carrot        a path follower's target, sized for the speed allowed (lookahead_for, at `accel`), shortened until
                    the straight way to it is clear.
    """

    def __init__(self, space=None, localizer=None, stop_model=None, accel: float = 0.1, stop_buffer_m: float = 0.03,
                 arm_moved_rad: float | None = ARM_MOVED_RAD, centre_limit=None, min_lookahead: float = 0.12,
                 plan_space=None):  # fmt: skip
        from chefbot.robot.localization import Localizer

        self.space = space  # FootprintSpace: where the base's centre may be at each heading (the body itself)
        self.localizer = localizer if localizer is not None else Localizer()
        self.stop_model = stop_model
        self.accel = float(accel)
        self.stop_buffer_m = float(stop_buffer_m)
        self.arm_moved_rad = arm_moved_rad
        self.centre_limit = centre_limit  # OccupancyGrid: cells the base's centre may not enter
        self.min_lookahead = float(min_lookahead)
        self.plan_space = plan_space  # the same with the planning margin: a carrot keeps to it where the base is in it
        self._arm_q0 = None
        self.last_cap: float | None = None

    def begin(self, state) -> None:
        """At the start of a drive: the arm's joints as they are now (it must not move until the base stops)."""
        self._arm_q0 = None if state.arm_q is None else np.asarray(state.arm_q, dtype=float).copy()
        self.last_cap = None

    def stop_reason(self, state, pose) -> str | None:
        lost = self.localizer.why_not_live(state)
        if lost is not None:
            return lost
        if self.space is not None and not self.space.is_free(pose.x, pose.y, pose.yaw):
            return "the base's footprint reached an obstacle on the map"
        if self.centre_limit is not None and not self.centre_limit.is_free(pose.x, pose.y):
            return "the base left the area it may use (its fence)"
        if self.arm_moved_rad is not None and self._arm_q0 is not None and state.arm_q is not None:
            moved = float(np.max(np.abs(np.asarray(state.arm_q, dtype=float) - self._arm_q0)))
            if moved > self.arm_moved_rad:
                return f"the arm moved {moved:.3f} rad while the base was driving"
        return None

    def speed_cap(self, pose, way, v_max: float) -> tuple[float, float | None]:
        """(speed allowed, free distance ahead or None)."""
        if self.stop_model is None or self.space is None or way is None or math.hypot(way[0], way[1]) < 1e-9:
            return v_max, None
        from chefbot.robot.nav.speed import free_distance

        free = free_distance(self.space.at(pose.yaw), (pose.x, pose.y), way)
        cap = min(v_max, self.stop_model.speed_for(free - self.stop_buffer_m))
        self.last_cap = cap
        return cap, free

    def carrot(self, follow, pose, v_max: float, default_lookahead: float):
        """(target, lookahead) for a BaseController `follow`: sized for the speed with a stop model, else the default."""
        if self.stop_model is not None:
            from chefbot.robot.nav.speed import lookahead_for

            v = v_max if self.last_cap is None else min(v_max, self.last_cap)
            look = max(self.min_lookahead, lookahead_for(v, self.accel, 0.1))
        else:
            look = default_lookahead
        target = follow.target_at(pose, look)
        if self.space is None:
            return target, look
        clear = self.space.at(pose.yaw)
        if self.plan_space is not None and self.plan_space.is_free(pose.x, pose.y, pose.yaw):
            clear = self.plan_space.at(pose.yaw)  # corners cut no closer than the margin
        while look > self.min_lookahead and not clear.segment_free((pose.x, pose.y), (target.x, target.y)):
            look = max(self.min_lookahead, look * 0.7)  # no corner cut into an obstacle
            target = follow.target_at(pose, look)
        return target, look
