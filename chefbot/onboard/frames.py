"""World (mocap) <-> wheel odometry, for the base.

The hb1 base driver works in its wheel odometry, which drifts; the robot layer works in the mocap
frame. The two are tied together on every sample with a valid mocap pose (where the base is in
both), so base targets in the world go to the driver in odometry, and the world pose the robot
layer sees is odometry carried forward through the last tie while mocap is lost.

The only arm-related frame work happens on the other side: arm targets arrive already in the
frame the arm's controller uses, and are passed through.
"""

from __future__ import annotations

import math

import numpy as np


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def yaw_of_quat_xyzw(q) -> float:
    """Heading of the body +x axis projected on the floor (z up), from an x, y, z, w quaternion."""
    x, y, z, w = (float(v) for v in q[:4])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def compose(a, b) -> np.ndarray:
    """SE(2) a * b, each (x, y, yaw)."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return np.array([a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2])])


def inverse(a) -> np.ndarray:
    c, s = math.cos(a[2]), math.sin(a[2])
    return np.array([-c * a[0] - s * a[1], s * a[0] - c * a[1], wrap(-a[2])])


class BaseFrames:
    """Ties mocap to odometry. `marker_yaw_offset` is the marker's yaw on the base (rad, the base's
    heading is the marker's minus this); `marker_xy` is the marker's position in the base frame."""

    def __init__(self, marker_yaw_offset: float = 0.0, marker_xy=(0.0, 0.0)):
        self.marker_yaw_offset = float(marker_yaw_offset)
        self.marker_xy = np.asarray(marker_xy, dtype=float)
        self.world_from_odom: np.ndarray | None = None  # (x, y, yaw); None: never seen mocap

    def base_from_mocap(self, position, quat_xyzw) -> np.ndarray:
        """The base pose (x, y, yaw) in the world from the marker's pose."""
        yaw = wrap(yaw_of_quat_xyzw(quat_xyzw) - self.marker_yaw_offset)
        c, s = math.cos(yaw), math.sin(yaw)
        mx, my = self.marker_xy
        return np.array([position[0] - (c * mx - s * my), position[1] - (s * mx + c * my), yaw])

    def tie(self, base_world, base_odom) -> None:
        """The base is at `base_world` in the world and `base_odom` in odometry, now."""
        self.world_from_odom = compose(base_world, inverse(base_odom))

    @property
    def tied(self) -> bool:
        return self.world_from_odom is not None

    def to_world(self, pose_odom) -> np.ndarray:
        return compose(self.world_from_odom, pose_odom) if self.tied else np.asarray(pose_odom, dtype=float)

    def to_odom(self, pose_world) -> np.ndarray:
        return compose(inverse(self.world_from_odom), pose_world) if self.tied else np.asarray(pose_world, dtype=float)

    def velocity_to_world(self, velocity_odom) -> np.ndarray:
        """A velocity (vx, vy, vyaw) in odometry, in the world frame."""
        if not self.tied:
            return np.asarray(velocity_odom, dtype=float)
        c, s = math.cos(self.world_from_odom[2]), math.sin(self.world_from_odom[2])
        vx, vy, w = velocity_odom
        return np.array([c * vx - s * vy, s * vx + c * vy, w])
