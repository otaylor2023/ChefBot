"""Where to hold a handle, given which points of the object are the handle.

Env-agnostic: the handle mask comes from a part detector (ground truth from the
mesh in sim, envs/robocasa/handles.py; a segmentation model on a real robot).
"""

from __future__ import annotations

import numpy as np

from chefbot.interface.types import Grasp

GRIP_FROM_BODY = 0.045  # grip a handle this far out from where it leaves the body
_WINDOW = (0.01, 0.02, 0.04)  # half-widths of the handle section the fingers close on, widened for sparse points
_DOWN = np.array([0.0, 0.0, -1.0])


def handle_grasps(
    points: np.ndarray,
    handle: np.ndarray,
    floor_clearance: float,
    grip_from_body: float = GRIP_FROM_BODY,
    score: float = 0.8,
) -> list[Grasp]:
    """Top-down grasps across a handle, near the body: there the object's weight
    has the least leverage to swing it down in the fingers (a short handle is
    held at its middle).

    points: (N, 3) world-frame points on the object. handle: (N,) bool mask of the
    handle's points. floor_clearance: how far above the object's lowest point the
    grip point must stay (fingertips clear of what it stands on).
    """
    grip = handle_grip_point(points, handle, grip_from_body)
    if grip is None:
        return []
    position, across, width = grip
    position[2] = max(position[2], float(points[:, 2].min()) + floor_clearance)
    closing = np.append(across, 0.0)
    return [
        Grasp(tcp_pose(position, _DOWN, closing), 0.0, width, score),
        Grasp(tcp_pose(position, _DOWN, -closing), 0.0, width, score - 0.01),  # turned half a turn
    ]


def handle_grip_point(
    points: np.ndarray, handle: np.ndarray, grip_from_body: float = GRIP_FROM_BODY
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """(grip position, horizontal closing direction across the handle, handle width there)."""
    handle_points, body_points = points[handle], points[~handle]
    if len(handle_points) < 2 or len(body_points) == 0:
        return None
    xy = handle_points[:, :2]
    centered = xy - xy.mean(axis=0)
    _, vectors = np.linalg.eigh(centered.T @ centered)
    axis = vectors[:, -1]  # along the handle
    if (xy.mean(axis=0) - body_points[:, :2].mean(axis=0)) @ axis < 0:
        axis = -axis  # pointing away from the body
    across = np.array([-axis[1], axis[0]])
    along = xy @ axis
    root, tip = float(along.min()), float(along.max())
    at = root + min(grip_from_body, (tip - root) / 2)
    for half in _WINDOW:
        piece = handle_points[np.abs(along - at) <= half]
        if len(piece) >= 2:
            break
    else:
        return None
    sideways = piece[:, :2] @ across
    low, high = float(sideways.min()), float(sideways.max())
    bottom, top = float(piece[:, 2].min()), float(piece[:, 2].max())
    center = at * axis + (low + high) / 2 * across
    # Close a little below the top of the handle, so the pads hold its full height.
    return np.array([center[0], center[1], top - 0.6 * (top - bottom)]), across, high - low


def tcp_pose(position: np.ndarray, approach: np.ndarray, closing: np.ndarray) -> np.ndarray:
    """TCP pose with z along `approach` and x along `closing` (made perpendicular to it)."""
    z = approach / np.linalg.norm(approach)
    x = closing - (closing @ z) * z
    x /= np.linalg.norm(x)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = x, np.cross(z, x), z, position
    return pose
