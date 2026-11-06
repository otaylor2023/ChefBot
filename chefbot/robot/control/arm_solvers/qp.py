"""qp: one QP per control tick for the joint motion that best closes the whole flange pose error, weighted by the
arm's inertia (its mass matrix): among the motions that do it, the one that moves the least inertia, so the light
wrist turns before the heavy shoulder and elbow swing (the choice an operational-space controller makes, osc.py, here
kinematic). Under hard constraints:
  - the joint limits (mink's ConfigurationLimit), so a limit in the way is gone around, not stalled on;
  - the arm's constraints (robot/control/constraints.py: the pole, the camera on its clamp, the plate, ...): no point
    along the arm closer to one, once within KEEP_ACTIVE, than a damper allows, nor past MARGIN outside it; it
    slides along them, never into them.
Then scaled, as a whole, to the speed profile's linear, angular and joint rates.

(2026-11-03, on the FR3, two low tilted grasps: INERTIA_WEIGHT 0.1 took the operational-space controller's tidy path
to the one it reached; 1e-3 swung joints 1 and 3 by 27 deg; 1.0 crawled.)
"""

from __future__ import annotations

import mink
import mujoco
import numpy as np
import qpsolvers
from scipy.spatial.transform import Rotation

from chefbot.robot.control import constraints as C
from chefbot.robot.control.arm_solvers.common import MARGIN, ArmPoints, scale
from chefbot.robot.arm.ik import TCP_SITE, ArmIK

INERTIA_WEIGHT = 0.1  # the motion's cost, dq' M dq, against the pose error's
ROT_WEIGHT = 0.3  # m per rad: a rotation error's weight against a position error's
KEEP_ACTIVE = 0.10  # m: a point this near the body (past the margin) is constrained
KEEP_GAIN = 0.3  # of the remaining clearance a point may close in one step


class ConstraintLimit(mink.Limit):
    """G dq <= h: each point along the arm within KEEP_ACTIVE of a constraint closes no more than KEEP_GAIN of its
    clearance in a step (already in: no closer; never forced out, so always feasible)."""

    def __init__(self, points: ArmPoints, constraints: list):
        self.points, self.constraints = points, constraints

    def compute_qp_inequalities(self, configuration: mink.Configuration, dt: float) -> mink.Constraint:
        data = configuration.data
        P, owners = self.points.positions(data)
        d, n, _ = C.nearest(self.constraints, P, MARGIN)
        near = np.where(d < KEEP_ACTIVE)[0]
        if not len(near):
            return mink.Constraint()
        J = self.points.jacobians(data, owners, near)
        G = -np.einsum("ij,ijk->ik", n[near], J)
        keep = np.linalg.norm(G, axis=1) > 1e-6
        if not keep.any():
            return mink.Constraint()
        return mink.Constraint(G=G[keep], h=KEEP_GAIN * np.maximum(d[near][keep], 0.0))


class QPSolver:
    def __init__(self, ik: ArmIK, constraints: list = (), hand_m: float = 0.0):
        self.ik, self.model = ik, ik.model
        self.lim = ik.joint_limits
        self.n = len(self.lim)
        self.config = mink.Configuration(self.model)
        self.site = self.model.site(TCP_SITE).id
        self.limits = [mink.ConfigurationLimit(self.model)]
        if constraints:
            self.limits.append(ConstraintLimit(ArmPoints(self.model, hand_m), list(constraints)))
        self._jp, self._jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        self._M = np.zeros((self.model.nv, self.model.nv))

    def step(self, target: np.ndarray, q, dt: float, max_lin: float, max_ang: float, max_joint: float) -> np.ndarray:
        m, d, n = self.model, self.config.data, self.n
        q = np.clip(np.asarray(q, dtype=float), self.lim[:, 0], self.lim[:, 1])
        self.config.update(q)
        mujoco.mj_crb(m, d)
        mujoco.mj_fullM(m, self._M, d.qM)
        mujoco.mj_jacSite(m, d, self._jp, self._jr, self.site)
        Jp, Jr = self._jp[:, :n], self._jr[:, :n]
        R, p = d.site_xmat[self.site].reshape(3, 3), d.site_xpos[self.site]
        e = np.r_[target[:3, 3] - p, ROT_WEIGHT * Rotation.from_matrix(target[:3, :3] @ R.T).as_rotvec()]
        # The error this step may close, at the profile's rates.
        lin, ang = np.linalg.norm(e[:3]), np.linalg.norm(e[3:]) / ROT_WEIGHT
        k = min(1.0, max_lin * dt / max(lin, 1e-9), max_ang * dt / max(ang, 1e-9))
        J = np.vstack([Jp, ROT_WEIGHT * Jr])
        H = J.T @ J + INERTIA_WEIGHT * self._M[:n, :n] + 1e-9 * np.eye(n)
        G, h = [], []
        for lim in self.limits:
            con = lim.compute_qp_inequalities(self.config, dt)
            if not con.inactive:
                G.append(np.atleast_2d(con.G)[:, :n])
                h.append(np.atleast_1d(con.h))
        dq = qpsolvers.solve_qp(H, -J.T @ (k * e), np.vstack(G) if G else None, np.concatenate(h) if h else None,
                                solver=self.ik._solver)  # fmt: skip
        if dq is None:
            raise mink.exceptions.NoSolutionFound(self.ik._solver)
        return np.clip(q + scale(dq, Jp, Jr, dt, max_lin, max_ang, max_joint) * dq, self.lim[:, 0], self.lim[:, 1])
