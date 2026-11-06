"""The base's pose in the world (mocap) from its rigid body's marker pose.

The hb1 base driver works in its wheel odometry, which drifts; mocap gives the base in the world.
The onboard server reports both, unchanged; the robot layer on the other side ties them
(robot/localization.py, the one tie) and sends base targets in odometry.

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
    """`marker_yaw_offset` is the marker's yaw on the base (rad, the base's heading is the marker's minus
    this); `marker_xy` is the marker's position in the base frame (onboard/config.yaml base:)."""

    def __init__(self, marker_yaw_offset: float = 0.0, marker_xy=(0.0, 0.0)):
        self.marker_yaw_offset = float(marker_yaw_offset)
        self.marker_xy = np.asarray(marker_xy, dtype=float)

    def base_from_mocap(self, position, quat_xyzw) -> np.ndarray:
        """The base pose (x, y, yaw) in the world from the marker's pose."""
        yaw = wrap(yaw_of_quat_xyzw(quat_xyzw) - self.marker_yaw_offset)
        c, s = math.cos(yaw), math.sin(yaw)
        mx, my = self.marker_xy
        return np.array([position[0] - (c * mx - s * my), position[1] - (s * mx + c * my), yaw])
