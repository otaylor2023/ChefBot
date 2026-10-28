"""Arm kinematics and IK (mink on the arm's MJCF, config/arms/<arm>.yaml).

One model serves reachability (position-only, NAV_SPEC §6.5) and execution
(full pose), so "reachable" and "executable" agree. Everything here is in the
arm base frame; the TCP frame has z along the approach direction and x along
the direction the fingers close.
"""

from __future__ import annotations

from pathlib import Path

import mink
import mujoco
import numpy as np
import qpsolvers

from chefbot.robot.config import ArmSpec

ASSETS = Path(__file__).resolve().parent.parent / "kinematics" / "assets"
TCP_SITE = "tcp"


def load_arm(tcp_xyz, tcp_yaw: float, joint_limits=None, arm: ArmSpec = ArmSpec()) -> mujoco.MjModel:
    """The arm with a `tcp` site. `tcp_xyz` / `tcp_yaw` place it relative to the
    flange; `joint_limits` (n x [low, high]) replaces the arm's own limits."""
    spec = mujoco.MjSpec.from_file(str(ASSETS / arm.mjcf))
    flange = next(s for s in spec.sites if s.name == arm.flange_site)
    link = next(b for b in spec.bodies if b.name == arm.flange_body)
    link.add_site(
        name=TCP_SITE,
        pos=np.asarray(flange.pos) + np.asarray(tcp_xyz, dtype=float),
        quat=[np.cos(tcp_yaw / 2), 0.0, 0.0, np.sin(tcp_yaw / 2)],
    )
    model = spec.compile()
    if joint_limits is not None:
        model.jnt_range[: arm.n_joints] = np.asarray(joint_limits, dtype=float)
    return model


def _pick_solver() -> str:
    for name in ("daqp", "quadprog", "osqp", "proxqp"):
        if name in qpsolvers.available_solvers:
            return name
    raise RuntimeError("no QP solver available for IK; install qpsolvers[daqp]")


class ArmIK:
    """Owns its MjData; never touches sim state. Not thread-safe: one per thread."""

    def __init__(
        self,
        tcp_xyz=(0.0, 0.0, 0.16),
        tcp_yaw: float = 0.0,
        joint_limits=None,
        posture: np.ndarray | None = None,
        pos_tolerance: float = 0.01,
        pose_pos_tolerance: float = 0.002,
        pose_rot_tolerance: float = 0.02,
        max_iters: int = 150,
        dt: float = 0.1,
        damping: float = 1e-3,
        arm: ArmSpec = ArmSpec(),
    ):
        self.arm = arm
        self.model = load_arm(tcp_xyz, tcp_yaw, joint_limits, arm)
        self.joint_limits = self.model.jnt_range[: arm.n_joints].copy()
        self._shoulder = np.asarray(arm.shoulder, dtype=float)
        self._position_seeds = tuple(np.asarray(q, dtype=float) for q in arm.position_seeds)
        self._pose_seeds = tuple(np.asarray(q, dtype=float) for q in arm.pose_seeds)
        self._tcp_reach = float(np.linalg.norm(tcp_xyz))
        self._pos_tolerance = pos_tolerance
        self._pose_pos_tolerance = pose_pos_tolerance
        self._pose_rot_tolerance = pose_rot_tolerance
        self._max_iters = max_iters
        self._dt = dt
        self._damping = damping
        self._solver = _pick_solver()
        self._configuration = mink.Configuration(self.model)
        self._position_task = mink.FrameTask(
            frame_name=TCP_SITE, frame_type="site", position_cost=1.0, orientation_cost=0.0
        )
        self._pose_task = mink.FrameTask(
            frame_name=TCP_SITE, frame_type="site", position_cost=1.0, orientation_cost=1.0
        )
        # Nullspace pull: among the solutions, prefer the one nearest this posture.
        self._posture_task = mink.PostureTask(self.model, cost=1e-2)
        self._posture = self._position_seeds[0] if posture is None else np.asarray(posture, dtype=float)
        self._posture_task.set_target(self._posture)
        self._limits = [mink.ConfigurationLimit(self.model)]
        self._site_id = self.model.site(TCP_SITE).id

    # -- forward kinematics ---------------------------------------------------

    def fk(self, q: np.ndarray) -> np.ndarray:
        """TCP pose (4x4) for joint angles `q`."""
        self._configuration.update(np.asarray(q, dtype=float))
        return self._tcp_pose()

    def tcp_position(self, q: np.ndarray) -> np.ndarray:
        return self.fk(q)[:3, 3]

    def within_limits(self, q: np.ndarray, margin: float = 0.0) -> bool:
        q = np.asarray(q, dtype=float)
        return bool(np.all(q >= self.joint_limits[:, 0] - margin) and np.all(q <= self.joint_limits[:, 1] + margin))

    # -- inverse kinematics ---------------------------------------------------

    def solve_position(self, target: np.ndarray) -> np.ndarray | None:
        """Joint angles placing the tcp at `target` with any orientation, or None."""
        target = np.asarray(target, dtype=float)
        if self._too_far(target):
            return None
        self._position_task.set_target(mink.SE3.from_translation(target))
        for seed in self._position_seeds:
            q = seed.copy()
            # Turn joint 1 toward the target so each restart starts in the right half-space.
            q[0] = np.clip(np.arctan2(target[1], target[0]), *self.joint_limits[0])
            self._configuration.update(q)
            for _ in range(self._max_iters):
                if np.linalg.norm(self._tcp_pose()[:3, 3] - target) <= self._pos_tolerance:
                    return self._configuration.q.copy()
                self._step([self._position_task])
        return None

    def solve_pose(self, target: np.ndarray, seed: np.ndarray | None = None) -> np.ndarray | None:
        """Joint angles placing the tcp at the 4x4 pose `target`, or None.
        Starts from `seed` if given and stays near it; falls back to the posture
        and then to a few other starting configurations."""
        target = np.asarray(target, dtype=float)
        if self._too_far(target[:3, 3]):
            return None
        self._pose_task.set_target(mink.SE3.from_matrix(target))
        seeds = [] if seed is None else [np.asarray(seed, dtype=float)]
        seeds += [self._posture, *self._pose_seeds]
        heading = np.arctan2(target[1, 3], target[0, 3])
        for i, start in enumerate(seeds):
            start = np.array(start, dtype=float)
            from_seed = seed is not None and i == 0
            if not from_seed:
                # Restarts face the target, like the position-only seeds.
                start[0] = heading
            # Among the solutions, stay near the caller's seed (so neighboring
            # poses get neighboring joint angles); otherwise near the posture.
            self._posture_task.set_target(start if from_seed else self._posture)
            self._configuration.update(np.clip(start, self.joint_limits[:, 0], self.joint_limits[:, 1]))
            for _ in range(self._max_iters):
                if self._pose_reached(target):
                    return self._configuration.q.copy()
                self._step([self._pose_task, self._posture_task])
        return None

    # -- internals ------------------------------------------------------------

    def _too_far(self, position: np.ndarray) -> bool:
        return float(np.linalg.norm(position - self._shoulder)) > self.arm.max_shoulder_to_flange + self._tcp_reach

    def _tcp_pose(self) -> np.ndarray:
        data = self._configuration.data
        pose = np.eye(4)
        pose[:3, :3] = data.site_xmat[self._site_id].reshape(3, 3)
        pose[:3, 3] = data.site_xpos[self._site_id]
        return pose

    def _pose_reached(self, target: np.ndarray) -> bool:
        pose = self._tcp_pose()
        if np.linalg.norm(pose[:3, 3] - target[:3, 3]) > self._pose_pos_tolerance:
            return False
        return rotation_angle(pose[:3, :3].T @ target[:3, :3]) <= self._pose_rot_tolerance

    def _step(self, tasks) -> None:
        velocity = mink.solve_ik(
            self._configuration, tasks, self._dt, solver=self._solver, damping=self._damping, limits=self._limits
        )
        self._configuration.integrate_inplace(velocity, self._dt)


def arm_ik_from_config(config) -> ArmIK:
    """The ArmIK described by a RobotConfig."""
    return ArmIK(
        tcp_xyz=config.tcp_xyz,
        tcp_yaw=config.tcp_yaw,
        joint_limits=config.joint_limits,
        posture=np.asarray(config.arm_ready, dtype=float),
        pos_tolerance=config.ik_pos_tolerance,
        pose_pos_tolerance=config.ik_pose_pos_tolerance,
        pose_rot_tolerance=config.ik_pose_rot_tolerance,
        max_iters=config.ik_max_iters,
        dt=config.ik_dt,
        damping=config.ik_damping,
        arm=config.arm,
    )


def rotation_angle(rotation: np.ndarray) -> float:
    """Angle (rad) of a 3x3 rotation matrix."""
    return float(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))
