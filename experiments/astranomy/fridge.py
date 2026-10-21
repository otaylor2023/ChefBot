"""Opening the fridge with physics, with a mobile base: the robot side.

No scripted actuation and no handle lock: the door (a 30 kg hinge, 0-90 deg,
damped) opens only if the gripper holds its handle and pulls it round.

Frames. odom: the robot root (arm base frame) where the episode started, fixed
for the episode: +x forward, +y left, +z up. Every target is absolute in odom:
the flange pose [x,y,z,qw,qx,qy,qz,gripper] as in the paper, plus the base
pose [base_x, base_y, base_yaw] = where the robot root should be (x, y, yaw in
odom). Each 15 Hz control step drives the base toward its target (velocity
control, clipped) and, every 20 Hz tick, re-solves one clipped IK step for the
flange target relative to where the base really is, so a held handle stays
held while the base moves.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from chefbot.envs.robocasa.build_map import ensure_map
from chefbot.envs.robocasa.config import RoboCasaConfig
from chefbot.envs.robocasa.serve import wire_stack
from chefbot.interface.types import Pose2D
from chefbot.robot.config import RobotConfig

from experiments.astranomy.host import CONTROL_HZ, IK_POS_CLIP, IK_ROT_CLIP, JOINT_CLIP, REPO, Host, R_of, quat_wxyz
from experiments.ortho_grasp.sim import ExperimentEnv, World

DOOR_JOINT = "fridge_main_group_fridge_door_joint"
HANDLE_GEOM = "fridge_main_group_fridge_door_handle_main"
SUCCESS_DEG = 60.0
BASE_KP_LIN, BASE_KP_ANG = 4.0, 4.0  # 1/s: base velocity per unit error
BASE_VMAX, BASE_WMAX = 0.5, 1.0  # the robot's own limits
TRACKING_SLACK = 0.15  # rad: further than this behind its command, the arm counts as blocked
# Start: about 0.8-1.0 m in front of the handle, a little to its left (clear of the
# swinging door), facing the fridge (world +y); randomized per seed.
START = {"x": 4.10, "y": -1.70, "yaw_deg": 90.0, "dx": 0.15, "dy": 0.15, "dyaw_deg": 15.0}


class FridgeWorld(World):
    """Layout 1, no loose objects, the robot placed per episode."""

    def __init__(self, results_dir: Path):
        cfg = RoboCasaConfig.load(layout=1, style=1)
        cfg = dataclasses.replace(cfg, objects=(), camera_width=960, camera_height=720)
        self.config = {"counter_z": 0.92}
        self.env = ExperimentEnv(cfg, render=True)
        self.objects = {}
        self.counter_z = 0.92
        map_dir = ensure_map(self.env, Path(results_dir) / "map")
        self.stack = wire_stack(self.env, map_dir, RobotConfig.load())
        self.arm = self.stack.robot._arm
        self.base = self.env.get_state().base_pose


class FridgeHost(Host):
    def __init__(self, out_dir: Path, record: bool = True):
        self.world = FridgeWorld(REPO / "experiments" / "astranomy" / "results")
        self.cfg = {"counter_z": 0.92}
        self.env, self.arm = self.world.env, self.world.arm
        self.out, self.record = Path(out_dir), record
        self.history, self.actions, self.frames, self.step_count = [], [], [], 0
        self.gap_open = 0.085
        m = self.env.model
        self.door_joint = m.joint(DOOR_JOINT).id
        self.door_qpos = int(m.jnt_qposadr[self.door_joint])
        self.handle_geom = m.geom(HANDLE_GEOM).id
        self.cup = None
        self._q_cmd = None

    # -- episode -------------------------------------------------------------------

    def reset_fridge(self, seed: int):
        rng = np.random.default_rng(seed)
        env = self.env
        env.reset_scene()
        pose = Pose2D(START["x"] + rng.uniform(-START["dx"], START["dx"]), START["y"] + rng.uniform(-START["dy"], START["dy"]),
                      math.radians(START["yaw_deg"] + rng.uniform(-START["dyaw_deg"], START["dyaw_deg"])))  # fmt: skip
        env.teleport_base(pose)
        env.set_arm_joints(np.array(self.world.stack.robot._cfg.arm_ready))
        env.set_gripper(False)
        with env.lock:
            env.data.qpos[self.door_qpos] = 0.0
            mujoco.mj_forward(env.model, env.data)
        for _ in range(10):
            env.set_base_velocity(0.0, 0.0, 0.0)
            env.wait_for_next_tick(0.05)
        self.gap_open = max(env.get_state().gripper_width, 1e-3)
        self.odom = self.root()  # fixed for the episode
        self.history, self.actions, self.frames, self.step_count = [], [], [], 0
        self._t_next = float(env.get_state().time)
        self._q_cmd = None
        self.out.mkdir(parents=True, exist_ok=True)
        self.seed, self.start_pose = seed, pose
        return self.observe()

    def door_deg(self) -> float:
        return float(np.degrees(self.env.data.qpos[self.door_qpos]))

    success_deg = SUCCESS_DEG

    def success(self) -> bool:
        return self.door_deg() >= self.success_deg

    def instruction(self) -> str:
        return "Open the fridge door all the way." if self.success_deg >= 80.0 else "Open the fridge door."

    def max_steps(self) -> int:
        return 900 if self.success_deg < 80.0 else 1350  # 60 s, or 90 s for opening it all the way

    # -- frames --------------------------------------------------------------------

    def root(self) -> np.ndarray:
        """World pose of the robot root (the arm base frame) where the base is now."""
        return self.arm.frames.arm_base_in_world(self.env.get_state().base_pose)

    def base_in_odom(self) -> tuple[float, float, float]:
        T = np.linalg.inv(self.odom) @ self.root()
        return float(T[0, 3]), float(T[1, 3]), float(math.atan2(T[1, 0], T[0, 0]))

    def flange_in_odom(self) -> np.ndarray:
        return np.linalg.inv(self.odom) @ self.root() @ self.flange_pose()

    def handle_in_odom(self) -> np.ndarray:
        p = self.env.data.geom_xpos[self.handle_geom]
        return (np.linalg.inv(self.odom) @ np.append(p, 1.0))[:3]

    # -- observation ---------------------------------------------------------------

    def state(self) -> dict:
        s = self.env.get_state()
        F = self.flange_in_odom()
        bx, by, byaw = self.base_in_odom()
        rpy = Rotation.from_matrix(F[:3, :3]).as_euler("xyz", degrees=True)
        return {
            "embodiment": "droid_mobile",
            "orientation_convention": "extrinsic xyz fixed-axis roll/pitch/yaw, degrees",
            "frame": "odom (the robot root where the episode started; fixed)",
            "effectors": {"arm": {"root_frame": "odom", "position_m": [round(float(v), 6) for v in F[:3, 3]],
                                  "roll_pitch_yaw_deg": [round(float(v), 4) for v in rpy],
                                  "gripper": "closed" if s.gripper_closed_cmd else "open"},
                          "base": {"root_frame": "odom", "base_pose": [round(bx, 4), round(by, 4), round(byaw, 4)],
                                   "in_contact": bool(s.base_in_contact)}},
            "geometry_state": {"ee_pos": F[:3, 3].tolist(), "ee_quat": quat_wxyz(F[:3, :3]), "gripper_pos": [self.closure()],
                               "base_pose": [bx, by, byaw]},
            "arm_joint_pos": [float(v) for v in s.arm_q],
        }  # fmt: skip

    # -- control -------------------------------------------------------------------

    def step(self, target: list[float]) -> dict:
        """One 15 Hz whole-body step: [x,y,z,qw,qx,qy,qz,gripper, base_x,base_y,base_yaw], all absolute in odom."""
        target = np.asarray(target, dtype=float)
        T_flange = np.eye(4)
        T_flange[:3, :3], T_flange[:3, 3] = R_of(target[3:7]), target[:3]
        T_flange_world = self.odom @ T_flange
        bx, by, byaw = (target[8:11] if len(target) >= 11 else self.base_in_odom())
        base_world = self.odom @ np.array([bx, by, 0.0, 1.0])
        yaw_world = math.atan2(self.odom[1, 0], self.odom[0, 0]) + byaw
        self.env.set_gripper(bool(target[7] >= 0.5))
        self._t_next += 1.0 / CONTROL_HZ
        ik_ok = True
        while self.env.get_state().time < self._t_next - 1e-6:
            ik_ok &= self._tick(T_flange_world, base_world[:2], yaw_world)
        self.env.set_base_velocity(0.0, 0.0, 0.0)
        self.step_count += 1
        F = self.flange_in_odom()
        nbx, nby, nbyaw = self.base_in_odom()
        rec = {"step": self.step_count, "target": target.tolist(), "ik_ok": bool(ik_ok),
               "pos_error_m": float(np.linalg.norm(target[:3] - F[:3, 3])),
               "rot_error_rad": float(np.linalg.norm(Rotation.from_matrix(R_of(target[3:7]) @ F[:3, :3].T).as_rotvec())),
               "base_error_m": float(math.hypot(bx - nbx, by - nby)), "base_yaw_error_rad": float(abs(_wrap(byaw - nbyaw))),
               "door_deg": self.door_deg()}  # fmt: skip
        self.actions.append(rec)
        return rec

    def _tick(self, T_flange_world: np.ndarray, base_xy_world: np.ndarray, yaw_world: float) -> bool:
        env = self.env
        s = env.get_state()
        # Base: P control toward the base target (the robot root's x, y, yaw), clipped to the robot's limits.
        root = self.root()
        err = base_xy_world - root[:2, 3]
        v = BASE_KP_LIN * err
        if np.linalg.norm(v) > BASE_VMAX:
            v *= BASE_VMAX / np.linalg.norm(v)
        dyaw = _wrap(yaw_world - math.atan2(root[1, 0], root[0, 0]))
        w = float(np.clip(BASE_KP_ANG * dyaw, -BASE_WMAX, BASE_WMAX))
        env.set_base_velocity(float(v[0]), float(v[1]), w)
        # Arm: one clipped IK step toward the flange target, relative to where the base is now.
        F_target = np.linalg.inv(root) @ T_flange_world  # in the current arm base frame
        # Step from the last commanded joints while the arm tracks them (the sim's joint
        # controller lags its target; stepping from the measured joints compounds the lag),
        # from the measured ones when it does not (blocked, in contact).
        q_meas = np.asarray(s.arm_q, dtype=float)
        q = self._q_cmd if self._q_cmd is not None and np.abs(self._q_cmd - q_meas).max() < TRACKING_SLACK else q_meas
        F = self.flange_pose(q)
        dp = F_target[:3, 3] - F[:3, 3]
        if np.linalg.norm(dp) > IK_POS_CLIP:
            dp *= IK_POS_CLIP / np.linalg.norm(dp)
        rot = Rotation.from_matrix(F_target[:3, :3] @ F[:3, :3].T).as_rotvec()
        if np.linalg.norm(rot) > IK_ROT_CLIP:
            rot *= IK_ROT_CLIP / np.linalg.norm(rot)
        Fd = np.eye(4)
        Fd[:3, :3] = Rotation.from_rotvec(rot).as_matrix() @ F[:3, :3]
        Fd[:3, 3] = F[:3, 3] + dp
        q_ik = self.arm.ik.solve_pose(self.tcp_from_flange(Fd), seed=q)
        if q_ik is not None:
            self._q_cmd = q + np.clip(q_ik - q, -JOINT_CLIP, JOINT_CLIP)
            env.set_arm_joint_targets(self._q_cmd)
        env.wait_for_next_tick(0.05)
        if self.record:
            self._frame()
        return q_ik is not None


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi
