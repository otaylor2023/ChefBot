"""legacy: the controller's first arm step (to 2026-11-03), kept to compare against: a short Cartesian step toward
the target, clipped to the speed profile's rates, solved exactly by the arm's IK from the current joints, each joint's
change clipped too. At a joint limit with no solution the IK's fallback seeds pulled it back: it stalled 0.35 m short
of a low tilted grasp the FR3 reaches. The constraints (robot/control/constraints.py) only cut a step short here, at
the margin: going around them is the router's (route.py)."""

from __future__ import annotations

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from chefbot.robot.control import constraints as C
from chefbot.robot.control.arm_solvers.common import MARGIN, ArmPoints
from chefbot.robot.arm.ik import ArmIK


class LegacySolver:
    def __init__(self, ik: ArmIK, constraints: list = (), hand_m: float = 0.0):
        self.ik = ik
        self.constraints = list(constraints)
        self.points = ArmPoints(ik.model, hand_m)
        self.data = mujoco.MjData(ik.model)

    def _least(self, q) -> float:
        self.data.qpos[: len(q)] = q
        mujoco.mj_kinematics(self.ik.model, self.data)
        return float(C.nearest(self.constraints, self.points.positions(self.data)[0], MARGIN)[0].min())

    def step(self, target: np.ndarray, q, dt: float, max_lin: float, max_ang: float, max_joint: float) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        F = self.ik.fk(q)
        dp = target[:3, 3] - F[:3, 3]
        if np.linalg.norm(dp) > max_lin * dt:
            dp *= max_lin * dt / np.linalg.norm(dp)
        rot = Rotation.from_matrix(target[:3, :3] @ F[:3, :3].T).as_rotvec()
        if np.linalg.norm(rot) > max_ang * dt:
            rot *= max_ang * dt / np.linalg.norm(rot)
        Fd = np.eye(4)
        Fd[:3, :3] = Rotation.from_rotvec(rot).as_matrix() @ F[:3, :3]
        Fd[:3, 3] = F[:3, 3] + dp
        q_ik = self.ik.solve_pose(Fd, seed=q)
        if q_ik is None:
            raise mink.exceptions.NoSolutionFound("ArmIK.solve_pose")
        q1 = q + np.clip(q_ik - q, -max_joint * dt, max_joint * dt)
        if self.constraints:
            floor = min(0.0, self._least(q))  # an arm that starts inside may only move out
            if self._least(q1) >= floor - 1e-5:
                return q1
            for s in (0.5, 0.25, 0.1):  # cut short at the margin
                if self._least(q + s * (q1 - q)) >= floor - 1e-5:
                    return q + s * (q1 - q)
            raise mink.exceptions.NoSolutionFound("legacy: every step goes into a constraint")
        return q1
