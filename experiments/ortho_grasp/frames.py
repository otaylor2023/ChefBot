"""The robot frame and gripper geometry everything in the experiment shares.

Robot frame: the mobile base frame, origin on the floor under the base
center, x forward, y left, z up. The base never moves, so it is a fixed
transform of the world frame (`BASE_POSE`, set from the config).

TCP frame (the repo's convention): z along the approach direction (from the
flange toward the fingertips), x along the direction the fingers close, origin
between the finger pads.
"""

from __future__ import annotations

import math

import numpy as np

from chefbot.interface.types import Pose2D

# 2F-85, measured on the sim gripper (chefbot/robot/kinematics/collision.py):
# boxes in the TCP frame as (z from, z to, half-width along x, half-width along y).
GRIPPER_BODY = (-0.156, -0.10, 0.065, 0.04)
KNUCKLES = (-0.10, -0.045, 0.075, 0.02)
PAD = (-0.045, 0.02, 0.006, 0.011)  # one finger's pad (its x center is at +-width/2)
FINGER = (-0.045, 0.02, 0.008, 0.015)  # one finger, pad included, for collision checks
FLANGE = 0.156  # m from the TCP back to the flange, along -z
MAX_OPEN = 0.085


def base_transform(base: Pose2D) -> np.ndarray:
    """World pose (4x4) of the robot frame."""
    c, s = math.cos(base.yaw), math.sin(base.yaw)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    T[:3, 3] = [base.x, base.y, 0.0]
    return T


def to_robot(points_world: np.ndarray, base: Pose2D) -> np.ndarray:
    T = np.linalg.inv(base_transform(base))
    p = np.asarray(points_world, dtype=float)
    return p @ T[:3, :3].T + T[:3, 3]


def to_world(points_robot: np.ndarray, base: Pose2D) -> np.ndarray:
    T = base_transform(base)
    p = np.asarray(points_robot, dtype=float)
    return p @ T[:3, :3].T + T[:3, 3]


def pose_to_world(pose_robot: np.ndarray, base: Pose2D) -> np.ndarray:
    return base_transform(base) @ pose_robot


def pose_to_robot(pose_world: np.ndarray, base: Pose2D) -> np.ndarray:
    return np.linalg.inv(base_transform(base)) @ pose_world


def tcp_pose(position: np.ndarray, approach: np.ndarray, closing: np.ndarray) -> np.ndarray:
    """TCP pose: z along `approach`, x along `closing` made perpendicular to it."""
    z = np.asarray(approach, dtype=float) / np.linalg.norm(approach)
    x = np.asarray(closing, dtype=float) - (np.asarray(closing, dtype=float) @ z) * z
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, np.cross(z, x), z, position
    return T


def gripper_boxes(pose: np.ndarray, width: float, pads: bool = True) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """The open gripper at a TCP pose as boxes (center, rotation columns, half sizes):
    body, knuckles and, with `pads`, the two fingers at +-width/2 along x."""
    R, t = pose[:3, :3], pose[:3, 3]
    out = []
    for z0, z1, hx, hy in (GRIPPER_BODY, KNUCKLES):
        out.append((t + R[:, 2] * (z0 + z1) / 2, R, np.array([hx, hy, (z1 - z0) / 2])))
    if pads:
        z0, z1, hx, hy = FINGER
        for side in (-1.0, 1.0):
            center = t + R[:, 2] * (z0 + z1) / 2 + R[:, 0] * side * (width / 2 + hx)
            out.append((center, R, np.array([hx, hy, (z1 - z0) / 2])))
    return out


def box_distance(points: np.ndarray, box: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """Distance (N,) from each point to the box (0 inside)."""
    center, R, half = box
    local = np.abs((np.asarray(points) - center) @ R)
    return np.linalg.norm(np.maximum(local - half, 0.0), axis=1)


def box_corners(box: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    center, R, half = box
    signs = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], dtype=float)
    return center + (signs * half) @ R.T
