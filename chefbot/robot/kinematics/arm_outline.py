"""Where the FR3 is over the floor, from its joint angles: the box (in the base frame) that the arm's links cover,
seen from above, for keeping the arm, not just the base, away from obstacles while the base drives.

Numpy only. The kinematics are Franka's published modified-DH parameters, the same as onboard/calibration/fr3.py
(which this side may not import: onboard is deployed alone). Every joint origin, the flange and the hand's tip
are projected onto the floor and grown by `link_radius` (the links' thickness); the box covers all of them.
Height is ignored on purpose: an arm above a counter still counts as over it, which is conservative.

    box = arm_box(q)          # {"center": (x, y), "size": (length, width)} in the base frame, x forward
"""

from __future__ import annotations

import numpy as np

DH = [(0.0, 0.333, 0.0), (0.0, 0.0, -np.pi / 2), (0.0, 0.316, np.pi / 2), (0.0825, 0.0, np.pi / 2),
      (-0.0825, 0.384, -np.pi / 2), (0.0, 0.0, np.pi / 2), (0.088, 0.0, np.pi / 2)]  # fmt: skip
FLANGE_D = 0.107
HAND_TIP = 0.1034 + 0.06  # Franka Hand: flange to fingertip centre, plus the fingers' reach (m), along the flange z
ARM_MOUNT = (0.0, 0.0, 0.335)  # the arm base in the base frame (config/robot_tidybot_fr3.yaml; TODO(verify) there)


def _link(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    return np.array([[ct, -st, 0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca], [0, 0, 0, 1.0]])


def frames(q) -> list[np.ndarray]:
    """The 7 joint frames, the flange and the hand's tip, in the arm base frame."""
    T, out = np.eye(4), []
    for (a, d, alpha), th in zip(DH, q):
        T = T @ _link(a, d, alpha, th)
        out.append(T)
    flange = T @ _link(0.0, FLANGE_D, 0.0, 0.0)
    out.append(flange)
    out.append(flange @ _link(0.0, HAND_TIP, 0.0, 0.0))
    return out


def arm_points(q, mount=ARM_MOUNT, mount_yaw: float = 0.0) -> np.ndarray:
    """(N, 3) points along the arm, in the base frame: the arm base, each joint, the flange, the hand tip."""
    c, s = np.cos(mount_yaw), np.sin(mount_yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    pts = [np.zeros(3)] + [f[:3, 3] for f in frames(q)]
    return np.array([R @ p + np.asarray(mount, dtype=float) for p in pts])


def arm_box(q, mount=ARM_MOUNT, mount_yaw: float = 0.0, link_radius: float = 0.08) -> dict:
    """The base-frame rectangle covering the arm seen from above, grown by `link_radius`."""
    xy = arm_points(q, mount, mount_yaw)[:, :2]
    lo, hi = xy.min(axis=0) - link_radius, xy.max(axis=0) + link_radius
    return {"center": tuple(float(v) for v in (lo + hi) / 2), "size": tuple(float(v) for v in hi - lo)}


def reach_ahead(q, mount=ARM_MOUNT, mount_yaw: float = 0.0, link_radius: float = 0.08) -> float:
    """How far ahead of the base centre the arm reaches (m, along the base's forward axis)."""
    return float(arm_points(q, mount, mount_yaw)[:, 0].max() + link_radius)
