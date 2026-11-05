"""What the arm solvers share: points along the arm with their Jacobians (for the constraints, embodiments/
constraints.py), and scaling a step to the speed profile."""

from __future__ import annotations

import mujoco
import numpy as np

from chefbot.robot.arm.ik import TCP_SITE

MARGIN = 0.005  # m: the solvers keep this further out than the keep-out (the real host's guard stops at it)


class ArmPoints:
    """Points along the arm on the IK model (the joint origins, the flange, the fingertips, the midpoints between:
    interface/workspace.py arm_points' 17, the base column's first two) and their position Jacobians."""

    def __init__(self, model: mujoco.MjModel, hand_m: float):
        self.model, self.hand_m = model, hand_m
        self.bodies = [b for b in range(model.nbody) if model.body_jntnum[b] > 0]
        self.site = model.site(TCP_SITE).id
        self._jac = np.zeros((3, model.nv))

    def positions(self, data: mujoco.MjData) -> tuple[np.ndarray, list]:
        m, site = self.model, self.site
        flange = data.site_xpos[site].copy()
        tip = flange + data.site_xmat[site].reshape(3, 3)[:, 2] * self.hand_m
        owners = [(data.xpos[b].copy(), b) for b in self.bodies] + [(flange, m.site_bodyid[site]), (tip, m.site_bodyid[site])]
        P = [p for p, _ in owners]
        P += [(P[i] + P[i + 1]) / 2 for i in range(len(owners) - 1)]
        return np.array(P), owners

    def jacobians(self, data: mujoco.MjData, owners: list, which) -> np.ndarray:
        """The 3 x nv position Jacobians of points `which` (indices into positions')."""
        n = len(owners)
        base = {}

        def jac(i):
            if i not in base:
                p, b = owners[i]
                mujoco.mj_jac(self.model, data, self._jac, None, p, b)
                base[i] = self._jac.copy()
            return base[i]

        return np.array([jac(i) if i < n else (jac(i - n) + jac(i - n + 1)) / 2 for i in which])


def scale(dq: np.ndarray, Jp: np.ndarray, Jr: np.ndarray, dt: float, max_lin: float, max_ang: float, max_joint: float) -> float:
    """The factor (<= 1) that brings a step dq within the flange's linear / angular and the joints' speed limits."""
    s = 1.0
    for speed, limit in ((np.linalg.norm(Jp @ dq) / dt, max_lin), (np.linalg.norm(Jr @ dq) / dt, max_ang), (np.abs(dq).max() / dt, max_joint)):
        if speed > limit:
            s = min(s, limit / speed)
    return s


def mass_matrix(model: mujoco.MjModel, data: mujoco.MjData, out: np.ndarray) -> np.ndarray:
    """The joint-space mass matrix (nv x nv) into `out`, after mj_crb (or a forward pass). MjData.qM, its packed form,
    is gone in newer MuJoCo (3.17: the real-host tests failed on it), so without it the matrix is built a column at a
    time from mj_mulM (M times a unit vector), which every version has."""
    if hasattr(data, "qM"):
        mujoco.mj_fullM(model, out, data.qM)
        return out
    e, col = np.zeros(model.nv), np.zeros(model.nv)
    for i in range(model.nv):
        e[:] = 0.0
        e[i] = 1.0
        mujoco.mj_mulM(model, data, col, e)
        out[:, i] = col
    return out

