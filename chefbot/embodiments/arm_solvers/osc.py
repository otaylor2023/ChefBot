"""osc: OpenSai's operational-space motion task (sai-primitives' MotionForceTask, as onboard/bringup/chefbot_fr3.xml
configures it for the real FR3), ported. Each control tick a virtual arm on the IK model (its masses and inertias,
gravity compensated, the joint limits hard stops) runs under the task at 500 Hz for the tick's time; its joints are
the tick's targets. The task:
    F = Lambda (kp e - kv xdot)               e: toward a goal that moves to the target at the profile's rates (the OTG)
    tau = J' F + N' M (-kv_null qdot) + tau_constraints
with Lambda = (J M^-1 J')^-1 the task's inertia and N its dynamically consistent nullspace: among the motions that
make the task, the one moving the least inertia (the light wrist turns before the heavy shoulder swings). On the FR3
model it reached a low tilted grasp the old step stalled on, tidily.

The constraints (embodiments/constraints.py: the pole, the camera on its clamp, the plate, ...) push back on the points
along the arm near them (Khatib's potential field, within REPEL_M), so the arm bends away from them as it nears them;
a step that would still go further into one is cut short (an arm that starts inside may only move out).
"""

from __future__ import annotations

import copy

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from chefbot.embodiments import constraints as C
from chefbot.embodiments.arm_solvers.common import MARGIN, ArmPoints, mass_matrix
from chefbot.robot.arm.ik import TCP_SITE, ArmIK

KP, KV, KV_NULL = 100.0, 20.0, 10.0  # chefbot_fr3.xml's cartesian_task gains; the nullspace's joint damping
INNER_DT = 0.005  # s: the virtual arm's step (10 per 50 ms tick)
REPEL_M = 0.08  # m: a point nearer a constraint than this (past the margin) is pushed back
REPEL_GAIN = 0.02  # N m^2: Khatib's eta
REPEL_MAX_N = 60.0  # N per point


class OSCSolver:
    def __init__(self, ik: ArmIK, constraints: list = (), hand_m: float = 0.0):
        self.ik = ik
        self.model = copy.deepcopy(ik.model)
        m = self.model
        m.opt.timestep = INNER_DT
        m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION | mujoco.mjtDisableBit.mjDSBL_CONTACT
        self.n = len(ik.joint_limits)
        m.jnt_limited[: self.n] = 1
        self.lim = ik.joint_limits
        self.data = mujoco.MjData(m)
        self.site = m.site(TCP_SITE).id
        self.constraints = list(constraints)
        self.points = ArmPoints(m, hand_m)
        self._jp, self._jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        self._M = np.zeros((m.nv, m.nv))
        self._last = None  # the joints handed out last tick: the virtual arm carries on from them (velocity too)
        self._goal = None  # (p, R): where the task's goal is now, on its way to the target

    def _least(self, d) -> float:
        return float(C.nearest(self.constraints, self.points.positions(d)[0], 0.0)[0].min())

    def _repulsion(self) -> np.ndarray:
        d = self.data
        P, owners = self.points.positions(d)
        dist, n, _ = C.nearest(self.constraints, P, MARGIN)
        near = np.where(dist < REPEL_M)[0]
        tau = np.zeros(self.model.nv)
        if len(near):
            J = self.points.jacobians(d, owners, near)
            for k, i in enumerate(near):
                r = max(dist[i], 0.005)
                f = min(REPEL_GAIN * (1.0 / r - 1.0 / REPEL_M) / r**2, REPEL_MAX_N)
                tau += J[k].T @ (f * n[i])
        return tau

    def step(self, target: np.ndarray, q, dt: float, max_lin: float, max_ang: float, max_joint: float) -> np.ndarray:
        m, d, n = self.model, self.data, self.n
        q = np.clip(np.asarray(q, dtype=float), self.lim[:, 0], self.lim[:, 1])
        if self._last is None or np.abs(q - self._last).max() > 1e-6:  # a fresh start (or the arm didn't follow)
            d.qpos[:n], d.qvel[:n] = q, 0.0
            mujoco.mj_forward(m, d)
            self._goal = (d.site_xpos[self.site].copy(), d.site_xmat[self.site].reshape(3, 3).copy())
        gp, gR = self._goal
        # The goal moves toward the target at the profile's rates (OpenSai's OTG: here velocity-limited).
        dp = target[:3, 3] - gp
        if np.linalg.norm(dp) > max_lin * dt:
            dp *= max_lin * dt / np.linalg.norm(dp)
        rv = Rotation.from_matrix(target[:3, :3] @ gR.T).as_rotvec()
        if np.linalg.norm(rv) > max_ang * dt:
            rv *= max_ang * dt / np.linalg.norm(rv)
        gp, gR = gp + dp, Rotation.from_rotvec(rv).as_matrix() @ gR
        self._goal = (gp, gR)
        tau_rep = np.zeros(m.nv)
        for k in range(max(1, round(dt / INNER_DT))):
            if k % 2 == 0 and self.constraints:
                tau_rep = self._repulsion()
            mujoco.mj_jacSite(m, d, self._jp, self._jr, self.site)
            J = np.vstack([self._jp, self._jr])[:, :n]
            mass_matrix(m, d, self._M)
            M = self._M[:n, :n]
            MiJt = np.linalg.solve(M, J.T)
            Lam = np.linalg.pinv(J @ MiJt, rcond=1e-3)
            R = d.site_xmat[self.site].reshape(3, 3)
            e = np.r_[gp - d.site_xpos[self.site], Rotation.from_matrix(gR @ R.T).as_rotvec()]
            F = Lam @ (KP * e - KV * (J @ d.qvel[:n]))
            N = np.eye(n) - MiJt @ Lam @ J
            d.qfrc_applied[:n] = J.T @ F + N.T @ (M @ (-KV_NULL * d.qvel[:n])) + tau_rep[:n] + d.qfrc_bias[:n]
            q_prev = d.qpos[:n].copy()
            before = self._least(d) if self.constraints else 0.0
            mujoco.mj_step(m, d)
            d.qvel[:n] = np.clip(d.qvel[:n], -max_joint, max_joint)
            if self.constraints:
                after = self._least(d)
                if after < 0 and after < before - 1e-5:  # further into one: stop there
                    d.qpos[:n], d.qvel[:n] = q_prev, 0.0
                    mujoco.mj_forward(m, d)
                    break
        self._last = np.clip(d.qpos[:n].copy(), self.lim[:, 0], self.lim[:, 1])
        return self._last.copy()
