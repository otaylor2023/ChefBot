"""The FR3's kinematics in numpy (Franka's published modified-DH parameters, the same as the Panda's): the flange
pose and its geometric Jacobian in the arm base frame. For the robot computer, which has no simulator."""

from __future__ import annotations

import numpy as np

# Craig's convention, per joint: a_{i-1}, d_i, alpha_{i-1}; then the flange (link 8): d 0.107.
DH = [(0.0, 0.333, 0.0), (0.0, 0.0, -np.pi / 2), (0.0, 0.316, np.pi / 2), (0.0825, 0.0, np.pi / 2),
      (-0.0825, 0.384, -np.pi / 2), (0.0, 0.0, np.pi / 2), (0.088, 0.0, np.pi / 2)]  # fmt: skip
FLANGE_D = 0.107


def _link(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    return np.array([[ct, -st, 0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca], [0, 0, 0, 1.0]])


def frames(q) -> list[np.ndarray]:
    """The 7 joint frames and then the flange, in the arm base frame."""
    T, out = np.eye(4), []
    for (a, d, alpha), th in zip(DH, q):
        T = T @ _link(a, d, alpha, th)
        out.append(T)
    out.append(T @ _link(0.0, FLANGE_D, 0.0, 0.0))
    return out


def flange(q) -> np.ndarray:
    return frames(q)[-1]


def jacobian(q) -> np.ndarray:
    """The flange origin's geometric Jacobian (6 x 7: linear rows, then angular), in the arm base frame."""
    fs = frames(q)
    o = fs[-1][:3, 3]
    J = np.zeros((6, 7))
    for i in range(7):
        z, p = fs[i][:3, 2], fs[i][:3, 3]
        J[:3, i] = np.cross(z, o - p)
        J[3:, i] = z
    return J
