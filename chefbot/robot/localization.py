"""Localization: the base's odometry (EnvState.base_pose) tied to the world by an external localizer's readings
(EnvState.localization: mocap on the real robot, ground truth in a sim).

Each live reading re-ties odometry to the world (world_from_odom = reading * odometry^-1, from the same state); between
readings the base moves on odometry. The robot layer plans in the world and sends base targets in odometry through
the tie. An env that has a localizer and reports it lost is not driven: `require_live` says why (the real robot's
failure mode, shared by every env that models it). An env with no localizer (localization None): odometry is the world.

    loc = Localizer()
    loc.update(state)                 # every state read
    loc.world(state.base_pose)        # where the base is in the world
    loc.to_odom(goal_world)           # a world pose as an odometry target
    loc.why_not_live(state)           # None, or why the base may not drive now
"""

from __future__ import annotations

import math

from chefbot.interface.types import EnvState, Pose2D, wrap_angle


def compose(a: Pose2D, b: Pose2D) -> Pose2D:
    """a * b: b given in a's frame, into the frame a is in."""
    c, s = math.cos(a.yaw), math.sin(a.yaw)
    return Pose2D(a.x + c * b.x - s * b.y, a.y + s * b.x + c * b.y, wrap_angle(a.yaw + b.yaw))


def inverse(a: Pose2D) -> Pose2D:
    c, s = math.cos(a.yaw), math.sin(a.yaw)
    return Pose2D(-(c * a.x + s * a.y), -(-s * a.x + c * a.y), wrap_angle(-a.yaw))


IDENTITY = Pose2D(0.0, 0.0, 0.0)


class Localizer:
    def __init__(self, max_age_s: float = 0.5):
        self.max_age_s = float(max_age_s)  # a reading older than this does not count as live
        self.world_from_odom = IDENTITY
        self.tied = False  # a live reading has tied odometry to the world at least once
        self.has_localizer = False  # the env reports localization at all

    def update(self, state: EnvState) -> bool:
        """Re-tie from the state's reading if it is live; returns whether it was."""
        r = state.localization
        self.has_localizer = r is not None
        if r is None or not r.live or r.age_s > self.max_age_s:
            return False
        self.world_from_odom = compose(r.pose, inverse(state.base_pose))
        self.tied = True
        return True

    def world(self, odom_pose: Pose2D) -> Pose2D:
        return compose(self.world_from_odom, odom_pose)

    def to_odom(self, world_pose: Pose2D) -> Pose2D:
        return compose(inverse(self.world_from_odom), world_pose)

    def velocity_to_odom(self, vx: float, vy: float) -> tuple[float, float]:
        """A world-frame planar velocity in odometry's frame."""
        c, s = math.cos(self.world_from_odom.yaw), math.sin(self.world_from_odom.yaw)
        return c * vx + s * vy, -s * vx + c * vy

    def why_not_live(self, state: EnvState) -> str | None:
        """None if the base may drive by this state, else why not: the env has a localizer and it is lost or stale."""
        r = state.localization
        if r is None:
            return None
        if not r.live:
            return "no localization (mocap) pose: the base does not drive without one"
        if r.age_s > self.max_age_s:
            return f"the localization (mocap) pose is {r.age_s:.2f} s old: the base does not drive on a stale one"
        return None
