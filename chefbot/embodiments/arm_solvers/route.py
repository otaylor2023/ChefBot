"""The router every arm solver runs inside: it goes around the constraints when the solver can't.

A solver looks one tick ahead, so an obstacle (the pole) can trap it: every step that closes the error crosses the
constraint, and it stalls short. The router watches the solver's progress toward each target; when it has stalled
(STALL_S without closing STALL_M) short of the target, it plans:
  1. an IK solution for the target clear of every constraint (by MARGIN), the nearest the arm's joints of those found
     from a spread of seeds (the arm's joints, its posture, its pose seeds, joints around both);
  2. a joint-space path there that keeps every point along the arm clear of every constraint all the way (checked
     every PATH_STEP rad): straight, or through one waypoint (the posture, or clear joints between);
then follows the path at the profile's joint rate (and its flange rates), and hands back to the solver at the end
for the last of it. With no clear solution or path it says so (`blocked`) and the step raises NoSolutionFound once
per target: the controller holds, and the skill sees the move fall short.
"""

from __future__ import annotations

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from chefbot.embodiments import constraints as C
from chefbot.embodiments.arm_solvers.common import MARGIN, ArmPoints, scale
from chefbot.robot.arm.ik import TCP_SITE, ArmIK

STALL_S = 1.0  # s of control ticks
STALL_M = 0.003  # m (rotation counted at 0.1 m per rad): less progress than this over STALL_S is a stall
DONE_M = 0.004  # m: closer than this to the target, the solver finishes it
PATH_STEP = 0.03  # rad: a path's checks, along its largest joint
GOAL_SEEDS = 4  # random seeds, beside q, the posture and the pose seeds
GOAL_ITERS = 150


class Router:
    def __init__(self, solver, ik: ArmIK, constraints: list = (), hand_m: float = 0.0, log=None):
        self.solver, self.ik = solver, ik
        self.constraints = list(constraints)
        self.points = ArmPoints(ik.model, hand_m)
        self.data = mujoco.MjData(ik.model)
        self.lim = ik.joint_limits
        self.log = log or (lambda *a: None)
        self._target = None
        self._history = []  # (time along this target, its error)
        self._t = 0.0
        self._path = None  # joint waypoints left to follow
        self.blocked = False  # no clear way to this target
        self._rng = np.random.default_rng(0)
        self._ik_solver = None
        self._jp, self._jr = np.zeros((3, ik.model.nv)), np.zeros((3, ik.model.nv))

    # -- geometry --------------------------------------------------------------------

    def _error(self, q, target) -> float:
        F = self.ik.fk(q)
        return float(np.linalg.norm(target[:3, 3] - F[:3, 3]) + 0.1 * np.linalg.norm(
            Rotation.from_matrix(target[:3, :3] @ F[:3, :3].T).as_rotvec()))  # fmt: skip

    def clearance(self, q) -> float:
        """The arm's least clearance (m, past MARGIN) from the constraints at joints q."""
        if not self.constraints:
            return np.inf
        self.data.qpos[: len(q)] = q
        mujoco.mj_kinematics(self.ik.model, self.data)
        return float(C.nearest(self.constraints, self.points.positions(self.data)[0], MARGIN)[0].min())

    def _segment_clear(self, a, b) -> bool:
        """Clear all the way (from a start inside one: never further in, and clear by the end)."""
        k = max(1, int(np.ceil(np.abs(b - a).max() / PATH_STEP)))
        floor = min(0.0, self.clearance(a))
        for t in np.linspace(0, 1, k + 1)[1:]:
            c = self.clearance(a + (b - a) * t)
            if c < floor - 1e-5:
                return False
            floor = max(floor, min(0.0, c))
        return floor >= 0 or self.clearance(b) >= 0

    # -- planning --------------------------------------------------------------------

    def _goals(self, target, q) -> list:
        """Clear IK solutions for the target, nearest q first: the qp solver's constrained steps run to convergence
        (fast rates, no clock) from a few seeds (q, the posture, the pose seeds, joints between), so every one found
        is clear of the constraints by construction (it slides along them)."""
        from chefbot.embodiments.arm_solvers.qp import QPSolver

        if self._ik_solver is None:
            self._ik_solver = QPSolver(self.ik, self.constraints, self.points.hand_m)
        n = len(q)
        seeds = [q, self.ik._posture, *self.ik._pose_seeds]
        seeds += [np.clip((q + self.ik._posture) / 2 + self._rng.normal(0, 0.4, n), self.lim[:, 0], self.lim[:, 1]) for _ in range(GOAL_SEEDS)]
        out = []
        for s in seeds:
            g = np.asarray(s, dtype=float)
            for _ in range(GOAL_ITERS):
                try:
                    g = self._ik_solver.step(target, g, 0.05, 1.0, 5.0, 5.0)
                except mink.exceptions.NoSolutionFound:
                    break
                if self._error(g, target) < DONE_M / 2:
                    break
            if self._error(g, target) < DONE_M and self.clearance(g) >= 0:
                if not any(np.abs(g - o).max() < 0.05 for o in out):
                    out.append(g)
        return sorted(out, key=lambda g: np.abs(g - q).max())

    def _plan(self, target, q):
        goals = self._goals(target, q)
        if not goals:
            return None, "no IK solution for it clear of the constraints"
        vias = [self.ik._posture] + [np.clip((q + g) / 2 + self._rng.normal(0, 0.4, len(q)), self.lim[:, 0], self.lim[:, 1])
                                     for g in goals[:3] for _ in range(8)]  # fmt: skip
        for g in goals[:6]:
            if self._segment_clear(q, g):
                return [g], None
            for v in vias:
                if self.clearance(v) >= 0 and self._segment_clear(q, v) and self._segment_clear(v, g):
                    return [v, g], None
        return None, f"{len(goals)} clear IK solution(s), but no clear joint path to one"

    # -- the step --------------------------------------------------------------------

    def step(self, target: np.ndarray, q, dt: float, max_lin: float, max_ang: float, max_joint: float) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if self._target is None or not np.allclose(target, self._target, atol=1e-6):
            self._target, self._history, self._t, self._path, self.blocked = np.array(target), [], 0.0, None, False
        self._t += dt
        if self._path:
            return self._follow(q, dt, max_lin, max_ang, max_joint)
        if self.blocked:
            raise mink.exceptions.NoSolutionFound("router: no clear way to the target")
        try:
            q1, failed = self.solver.step(target, q, dt, max_lin, max_ang, max_joint), False
        except mink.exceptions.NoSolutionFound:  # the solver has no step at all: as good as stalled, plan now
            q1, failed = q, True
        err = self._error(q1, target)
        self._history.append((self._t, err))
        old = [e for t, e in self._history if t <= self._t - STALL_S]
        if (failed or (old and old[-1] - err < STALL_M)) and err > DONE_M:
            path, why = self._plan(target, q1)
            self._history = []
            if path is None:
                self.blocked = True
                self.log(f"[arm] can't reach the target clear of the constraints: {why}")
                if failed:
                    raise mink.exceptions.NoSolutionFound("router: no clear way to the target")
                return q1
            self.log(f"[arm] the solver stalled {err * 100:.1f} cm short: going around ({len(path)} joint waypoint(s))")
            self._path = path
        return q1

    def _follow(self, q, dt, max_lin, max_ang, max_joint) -> np.ndarray:
        w = self._path[0]
        dq = w - q
        mujoco.mj_kinematics(self.ik.model, self.data)
        self.data.qpos[: len(q)] = q
        mujoco.mj_kinematics(self.ik.model, self.data)
        mujoco.mj_comPos(self.ik.model, self.data)
        mujoco.mj_jacSite(self.ik.model, self.data, self._jp, self._jr, self.ik.model.site(TCP_SITE).id)
        n = len(q)
        s = min(1.0, scale(dq, self._jp[:, :n], self._jr[:, :n], dt, max_lin, max_ang, max_joint))
        q1 = q + s * dq
        if np.abs(w - q1).max() < 1e-4:
            self._path.pop(0)
            if not self._path:
                self._path, self._history = None, []
        return q1
