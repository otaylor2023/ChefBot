"""The whole-body controller under a skill: 11-D targets in odom -> base and joint commands on an Env.

A target is [x, y, z, qw, qx, qy, qz, gripper, base_x, base_y, base_yaw], all absolute in odom (the robot root
where the episode started, fixed): the flange pose, the gripper (0 open .. 1 closed; a binary gripper closes at
>= 0.5), and where the robot root should be. Each control step (1 / control.hz) runs ticks of the environment until the step's time is up. Every tick:
  - the base drives toward its target: by itself when the environment's base can (Env.set_base_target), else by
    velocity control (P on the error, within the speed profile's base limit);
  - the arm takes one clipped Cartesian step toward the flange target relative to where the base is now (so a held
    object stays put while the base moves), solved by IK, each joint's change clipped too.
Step sizes are the speed profile's rates times the tick.

Two partial modes (step(..., mode=)): "arm", an 8-D [x, y, z, qw, qx, qy, qz, gripper] flange target in the robot
root frame (it rides with the base), solved by IK alone while the base holds its last target; "base", a 3-D
[x, y, yaw] root target in odom, the arm holding its joints (no IK).

(agentic-skills-exploration's robots/wholebody.py, on ChefBot's Env and IK.)
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

from chefbot.embodiments import Robot, Speed
from chefbot.interface.env import Env

BASE_KP_LIN, BASE_KP_ANG = 4.0, 4.0  # 1/s: base velocity per unit error
TRACKING_SLACK = 0.15  # rad: further than this behind its command, the arm counts as blocked (step from where it is)


def R_of(q_wxyz) -> np.ndarray:
    w, x, y, z = np.asarray(q_wxyz, dtype=float) / np.linalg.norm(q_wxyz)
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def quat_wxyz(R: np.ndarray) -> list[float]:
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    q = np.array([w, x, y, z])
    return (q if q[0] >= 0 else -q).tolist()


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class WholeBodyController:
    def __init__(self, env: Env, robot: Robot, speed: Speed, tick_s: float = 0.05, on_tick=None):
        self.env, self.robot, self.kin, self.speed = env, robot, robot.kin, speed
        self.hz = float(robot.raw["control"]["hz"])
        self.tick_s = tick_s
        self.on_tick = on_tick  # called after every tick (e.g. to record a video frame)
        self.odom = np.eye(4)
        self._q_cmd = None
        self._t_next = 0.0
        self._base_hold = (0.0, 0.0, 0.0)  # the last base target (root in odom): what "arm" steps hold

    # -- frames ----------------------------------------------------------------------

    def reset(self) -> None:
        """Odom is where the robot root is now; the step clock starts now."""
        self.odom = self.root()
        self._q_cmd = None
        self._t_next = float(self.env.get_state().time)
        self._base_hold = (0.0, 0.0, 0.0)

    def root(self) -> np.ndarray:
        return self.kin.root_in_world(self.env.get_state().base_pose)

    def base_in_odom(self) -> tuple[float, float, float]:
        T = np.linalg.inv(self.odom) @ self.root()
        return float(T[0, 3]), float(T[1, 3]), float(math.atan2(T[1, 0], T[0, 0]))

    def flange_in_odom(self, q=None) -> np.ndarray:
        q = self.env.get_state().arm_q if q is None else q
        return np.linalg.inv(self.odom) @ self.root() @ self.kin.flange_pose(q)

    @property
    def q_cmd(self):
        return self._q_cmd

    # -- control ---------------------------------------------------------------------

    def step(self, target, mode: str = "whole") -> dict:
        """One control step toward an absolute 11-D target (or an "arm" / "base" one); returns the tracking errors."""
        if mode != "whole":
            return self._step_partial(np.asarray(target, dtype=float), mode)
        target = np.asarray(target, dtype=float)
        T_flange = np.eye(4)
        T_flange[:3, :3], T_flange[:3, 3] = R_of(target[3:7]), target[:3]
        T_flange_world = self.odom @ T_flange
        bx, by, byaw = target[8:11] if len(target) >= 11 else self.base_in_odom()
        self._base_hold = (float(bx), float(by), float(byaw))
        base_world = self.odom @ np.array([bx, by, 0.0, 1.0])
        yaw_world = math.atan2(self.odom[1, 0], self.odom[0, 0]) + byaw
        self.env.set_gripper(float(np.clip(target[7], 0.0, 1.0)))
        self._t_next += 1.0 / self.hz
        ik_ok = True
        while self.env.get_state().time < self._t_next - 1e-6:
            ik_ok &= self._tick(T_flange_world, base_world[:2], yaw_world)
        self.env.set_base_velocity(0.0, 0.0, 0.0)
        F = self.flange_in_odom()
        nbx, nby, nbyaw = self.base_in_odom()
        return {"target": target.tolist(), "ik_ok": bool(ik_ok), "pos_error_m": float(np.linalg.norm(target[:3] - F[:3, 3])),
                "rot_error_rad": float(np.linalg.norm(Rotation.from_matrix(R_of(target[3:7]) @ F[:3, :3].T).as_rotvec())),
                "base_error_m": float(math.hypot(bx - nbx, by - nby)), "base_yaw_error_rad": float(abs(wrap(byaw - nbyaw)))}  # fmt: skip

    def _step_partial(self, target: np.ndarray, mode: str) -> dict:
        if mode == "arm":
            arm_root = np.eye(4)
            arm_root[:3, :3], arm_root[:3, 3] = R_of(target[3:7]), target[:3]
            self.env.set_gripper(float(np.clip(target[7], 0.0, 1.0)))
            bx, by, byaw = self._base_hold
        elif mode == "base":
            arm_root = None
            bx, by, byaw = (float(v) for v in target[:3])
            self._base_hold = (bx, by, byaw)
        else:
            raise ValueError(f"mode must be whole, arm or base, not {mode!r}")
        base_world = self.odom @ np.array([bx, by, 0.0, 1.0])
        yaw_world = math.atan2(self.odom[1, 0], self.odom[0, 0]) + byaw
        self._t_next += 1.0 / self.hz
        ik_ok = True
        while self.env.get_state().time < self._t_next - 1e-6:
            ik_ok &= self._tick(None, base_world[:2], yaw_world, arm_root=arm_root, arm_hold=mode == "base")
        self.env.set_base_velocity(0.0, 0.0, 0.0)
        nbx, nby, nbyaw = self.base_in_odom()
        rec = {"target": target.tolist(), "ik_ok": bool(ik_ok), "pos_error_m": 0.0, "rot_error_rad": 0.0,
               "base_error_m": float(math.hypot(bx - nbx, by - nby)), "base_yaw_error_rad": float(abs(wrap(byaw - nbyaw)))}  # fmt: skip
        if mode == "arm":
            F = self.kin.flange_pose(self.env.get_state().arm_q)
            rec["pos_error_m"] = float(np.linalg.norm(target[:3] - F[:3, 3]))
            rec["rot_error_rad"] = float(np.linalg.norm(Rotation.from_matrix(arm_root[:3, :3] @ F[:3, :3].T).as_rotvec()))
        return rec

    def _tick(self, T_flange_world: np.ndarray | None, base_xy_world: np.ndarray, yaw_world: float, arm_root=None,
              arm_hold: bool = False) -> bool:  # fmt: skip
        env, sp = self.env, self.speed
        s = env.get_state()
        root = self.root()
        # Base: toward the root target, by the environment's own base controller if it has one.
        root_target = np.eye(4)
        root_target[:3, :3] = Rotation.from_euler("z", yaw_world).as_matrix()
        root_target[:3, 3] = [base_xy_world[0], base_xy_world[1], root[2, 3]]
        if not env.set_base_target(self.kin.base_for_root(root_target), sp.base_max_vel):
            err = base_xy_world - root[:2, 3]
            v = BASE_KP_LIN * err
            if np.linalg.norm(v) > sp.base_max_vel[0]:
                v *= sp.base_max_vel[0] / np.linalg.norm(v)
            dyaw = wrap(yaw_world - math.atan2(root[1, 0], root[0, 0]))
            w = float(np.clip(BASE_KP_ANG * dyaw, -sp.base_max_vel[2], sp.base_max_vel[2]))
            # The root's velocity, as the base center's: the arm mount is off center, so turning moves the root too.
            mx, my = root[:2, 3] - np.array(self._base_xy())
            env.set_base_velocity(float(v[0] + w * my), float(v[1] - w * mx), w)
        if arm_hold:  # "base" steps: the arm holds its joints, no IK
            q_hold = self._q_cmd if self._q_cmd is not None else np.asarray(s.arm_q, dtype=float)
            env.set_arm_joint_targets(q_hold)
            self._q_cmd = q_hold
            env.wait_for_next_tick(self.tick_s)
            if self.on_tick:
                self.on_tick()
            return True
        # Arm: one clipped step toward the flange target, relative to where the base is now ("arm" steps: in the
        # root frame already).
        F_target = arm_root if arm_root is not None else np.linalg.inv(root) @ T_flange_world
        # Step from the last command while the arm tracks it (stepping from the measured joints compounds the
        # servo's lag), from the measured joints when it doesn't (blocked, in contact).
        q_meas = np.asarray(s.arm_q, dtype=float)
        q = self._q_cmd if self._q_cmd is not None and np.abs(self._q_cmd - q_meas).max() < TRACKING_SLACK else q_meas
        F = self.kin.flange_pose(q)
        dp = F_target[:3, 3] - F[:3, 3]
        pos_clip, rot_clip, joint_clip = sp.arm_lin * self.tick_s, sp.arm_ang * self.tick_s, sp.joint * self.tick_s
        if np.linalg.norm(dp) > pos_clip:
            dp *= pos_clip / np.linalg.norm(dp)
        rot = Rotation.from_matrix(F_target[:3, :3] @ F[:3, :3].T).as_rotvec()
        if np.linalg.norm(rot) > rot_clip:
            rot *= rot_clip / np.linalg.norm(rot)
        Fd = np.eye(4)
        Fd[:3, :3] = Rotation.from_rotvec(rot).as_matrix() @ F[:3, :3]
        Fd[:3, 3] = F[:3, 3] + dp
        q_ik = self.kin.ik.solve_pose(Fd, seed=q)
        if q_ik is not None:
            self._q_cmd = q + np.clip(q_ik - q, -joint_clip, joint_clip)
            env.set_arm_joint_targets(self._q_cmd)
        env.wait_for_next_tick(self.tick_s)
        if self.on_tick:
            self.on_tick()
        return q_ik is not None

    def _base_xy(self) -> tuple[float, float]:
        p = self.env.get_state().base_pose
        return p.x, p.y
