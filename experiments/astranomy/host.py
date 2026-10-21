"""The robot side of Astranomy (NVIDIA SRL, "GPT-6 Astra as a robot policy") on
our RoboCasa cup scenes, in DROID's conventions so the paper's prompts apply.

Robot root: the arm base frame (+x forward, +y left, +z up). EEF pose: the
gripper's mounting flange (our 2F-85's base_mount), with DROID's local axes:
+X along the fingers (approach), Y across the two pads. Cameras: RoboCasa's
two base-mounted cameras behind the arm stand in for DROID's over-shoulder
pair, plus the wrist camera; rendered at 384 x 216 like the paper.

Control as in the paper: 15 Hz; each step holds an absolute flange target,
runs one local IK step from the measured state with the position error
clipped to 0.02 m, the rotation error to 0.1 rad and every joint's increment
to 0.05 rad, and applies the gripper command. The simulator only advances
inside a step: it is paused while the model thinks.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from chefbot.envs.robocasa.objects import object_points

REPO = Path(__file__).resolve().parents[2]
OG = REPO / "experiments" / "ortho_grasp"
CONTROL_HZ = 15.0
IK_POS_CLIP, IK_ROT_CLIP, JOINT_CLIP = 0.02, 0.1, 0.05
IMAGE_W, IMAGE_H = 384, 216
VIEWS = {"over_shoulder_left_camera": "robot0_agentview_left", "over_shoulder_right_camera": "robot0_agentview_right",
         "wrist_cam": "robot0_eye_in_hand"}  # fmt: skip
# Flange (DROID axes) from our mount (z along the fingers, x across the pads): columns are X_F, Y_F, Z_F in mount coords.
MOUNT_TO_FLANGE = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
GRIP_SITE = 0.1488  # m from the mount to the grip site (the IK's TCP), along the fingers
LIFT_SUCCESS = 0.05  # m: the cup's bottom above the counter, held


def quat_wxyz(R: np.ndarray) -> list[float]:
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    q = np.array([w, x, y, z])
    return (q if q[0] >= 0 else -q).tolist()


def R_of(q_wxyz) -> np.ndarray:
    w, x, y, z = np.asarray(q_wxyz, dtype=float) / np.linalg.norm(q_wxyz)
    return Rotation.from_quat([x, y, z, w]).as_matrix()


@dataclass
class Observation:
    step: int
    time: float
    state: dict
    images: dict[str, np.ndarray]  # view name -> RGB
    paths: dict[str, str] = field(default_factory=dict)


class Host:
    grip_site = GRIP_SITE

    def __init__(self, out_dir: Path, record: bool = True):
        from experiments.ortho_grasp.sim import World

        cfg = yaml.safe_load((OG / "configs" / "experiment.yaml").read_text())
        cfg["results_dir"] = str(REPO / cfg["results_dir"])
        self.cfg = cfg
        self.world = World(cfg)
        self.env, self.arm = self.world.env, self.world.arm
        self.out = Path(out_dir)
        self.record = record
        self.history: list[Observation] = []
        self.actions: list[dict] = []
        self.frames: list[np.ndarray] = []
        self.step_count = 0
        self.gap_open = 0.085

    # -- episode -------------------------------------------------------------------

    def reset(self, scene) -> Observation:
        from experiments.ortho_grasp import scenes as S  # noqa: F401

        self.scene = scene
        self.world.apply(scene)
        self.env.set_arm_joints(np.array(self.world.stack.robot._cfg.arm_ready))
        self.env.set_gripper(False)
        for _ in range(10):
            self.env.wait_for_next_tick(0.05)
        self.gap_open = max(self.env.get_state().gripper_width, 1e-3)
        self.cup = self.world.cup_root(scene)
        self.history, self.actions, self.frames, self.step_count = [], [], [], 0
        self._t_next = float(self.env.get_state().time)
        self.out.mkdir(parents=True, exist_ok=True)
        return self.observe()

    def cup_bottom(self) -> float:
        with self.env.lock:
            return float(object_points(self.env, self.cup)[:, 2].min())

    def success(self) -> bool:
        return self.cup_bottom() >= self.cfg["counter_z"] + LIFT_SUCCESS and self.cup in self.env._grasped_roots()

    # -- frames --------------------------------------------------------------------

    def root(self) -> np.ndarray:
        """World pose of the robot root (the arm base frame)."""
        return self.arm.frames.arm_base_in_world(self.world.base)

    def flange_pose(self, q: np.ndarray | None = None) -> np.ndarray:
        """Measured flange pose (4x4) in the robot root, DROID axes."""
        q = self.env.get_state().arm_q if q is None else q
        tcp = self.arm.ik.fk(np.asarray(q))  # arm base frame, our TCP axes (z along the fingers)
        mount = tcp.copy()
        mount[:3, 3] -= self.grip_site * tcp[:3, 2]
        out = np.eye(4)
        out[:3, :3] = mount[:3, :3] @ MOUNT_TO_FLANGE
        out[:3, 3] = mount[:3, 3]
        return out

    def tcp_from_flange(self, F: np.ndarray) -> np.ndarray:
        mount_R = F[:3, :3] @ MOUNT_TO_FLANGE.T
        tcp = np.eye(4)
        tcp[:3, :3] = mount_R
        tcp[:3, 3] = F[:3, 3] + self.grip_site * mount_R[:, 2]
        return tcp

    def closure(self) -> float:
        """gripper_pos[0]: normalized closure, 0 open .. 1 closed (from the pad gap)."""
        return float(np.clip(1.0 - self.env.get_state().gripper_width / self.gap_open, 0.0, 1.0))

    # -- observation ---------------------------------------------------------------

    def state(self) -> dict:
        s = self.env.get_state()
        F = self.flange_pose(s.arm_q)
        rpy = Rotation.from_matrix(F[:3, :3]).as_euler("xyz", degrees=True)
        return {
            "embodiment": "droid",
            "orientation_convention": "extrinsic xyz fixed-axis roll/pitch/yaw, degrees",
            "effectors": {"arm": {"root_frame": "DROID robot root", "position_m": [round(float(v), 6) for v in F[:3, 3]],
                                  "roll_pitch_yaw_deg": [round(float(v), 4) for v in rpy],
                                  "gripper": "closed" if s.gripper_closed_cmd else "open"}},
            "geometry_state": {"ee_pos": F[:3, 3].tolist(), "ee_quat": quat_wxyz(F[:3, :3]), "gripper_pos": [self.closure()]},
            "arm_joint_pos": [float(v) for v in s.arm_q],
        }  # fmt: skip

    def images(self) -> dict[str, np.ndarray]:
        return {name: self.world.render(cam, IMAGE_W, IMAGE_H) for name, cam in VIEWS.items()}

    def observe(self) -> Observation:
        obs = Observation(self.step_count, float(self.env.get_state().time), self.state(), self.images())
        d = self.out / "obs"
        d.mkdir(exist_ok=True)
        for name, img in obs.images.items():
            p = d / f"{obs.step:04d}_{name}.png"
            iio.imwrite(p, img)
            obs.paths[name] = str(p)
        self.history.append(obs)
        return obs

    def at(self, step: int) -> Observation:
        """The latest stored observation at or before control step `step`."""
        best = self.history[0]
        for o in self.history:
            if o.step <= step:
                best = o
        return best

    # -- control -------------------------------------------------------------------

    def step(self, target: list[float]) -> dict:
        """One 15 Hz control step toward an absolute flange target [x,y,z,qw,qx,qy,qz,gripper]."""
        target = np.asarray(target, dtype=float)
        s = self.env.get_state()
        F = self.flange_pose(s.arm_q)
        p_t, R_t = target[:3], R_of(target[3:7])
        dp = p_t - F[:3, 3]
        if np.linalg.norm(dp) > IK_POS_CLIP:
            dp *= IK_POS_CLIP / np.linalg.norm(dp)
        rot = Rotation.from_matrix(R_t @ F[:3, :3].T).as_rotvec()
        if np.linalg.norm(rot) > IK_ROT_CLIP:
            rot *= IK_ROT_CLIP / np.linalg.norm(rot)
        Fd = np.eye(4)
        Fd[:3, :3] = Rotation.from_rotvec(rot).as_matrix() @ F[:3, :3]
        Fd[:3, 3] = F[:3, 3] + dp
        q = np.asarray(s.arm_q, dtype=float)
        q_ik = self.arm.ik.solve_pose(self.tcp_from_flange(Fd), seed=q)
        ik_ok = q_ik is not None
        q_new = q + np.clip((q_ik - q) if ik_ok else 0.0, -JOINT_CLIP, JOINT_CLIP)
        self.env.set_arm_joint_targets(q_new)
        self.env.set_gripper(bool(target[7] >= 0.5))
        self._t_next += 1.0 / CONTROL_HZ
        while self.env.get_state().time < self._t_next - 1e-6:
            self.env.wait_for_next_tick(0.05)
            if self.record:
                self._frame()
        self.step_count += 1
        after = self.flange_pose()
        err_p = float(np.linalg.norm(p_t - after[:3, 3]))
        err_r = float(np.linalg.norm(Rotation.from_matrix(R_t @ after[:3, :3].T).as_rotvec()))
        rec = {"step": self.step_count, "target": target.tolist(), "ik_ok": ik_ok, "pos_error_m": err_p, "rot_error_rad": err_r}
        self.actions.append(rec)
        return rec

    # -- recording -----------------------------------------------------------------

    def _frame(self) -> None:
        views = [self.world.render(cam, 320, 180) for cam in VIEWS.values()]
        side = self.world.render("robot0_robotview", 320, 180)
        self.frames.append(np.concatenate([np.concatenate(views[:2], 1), np.concatenate([views[2], side], 1)], 0))

    def save_video(self, name: str = "episode.mp4") -> Path | None:
        if not self.frames:
            return None
        p = self.out / name
        iio.imwrite(p, np.stack(self.frames), fps=20, codec="libx264")
        return p

    def close(self) -> None:
        self.world.close()

    # -- calibration for the prompt (the paper's tables, measured on this robot) ---

    def calibration(self) -> dict:
        env, m, d = self.env, self.env.model, self.env.data
        out = {"pads": [], "cameras": {}}
        # Pad midpoint offset d along +X and the inner-face gap, through a slow close.
        env.set_gripper(False)
        for _ in range(10):
            env.wait_for_next_tick(0.05)
        samples = []
        env.set_gripper(True)
        for _ in range(12):
            env.gripper_hold = False
            env.wait_for_next_tick(0.05)
            env.gripper_hold = True
            for _ in range(3):
                env.wait_for_next_tick(0.05)
            samples.append(self._pad_sample())
        env.gripper_hold = False
        env.set_gripper(False)
        for _ in range(10):
            env.wait_for_next_tick(0.05)
        samples.insert(0, self._pad_sample())
        gaps = np.array([s[1] for s in samples])
        ds = np.array([s[0] for s in samples])
        g_open = gaps.max()
        closure = 1.0 - gaps / g_open
        order = np.argsort(closure)
        for g in (0.0, 0.25, 0.5, 0.75, 1.0):
            out["pads"].append({"g": g, "d": float(np.interp(g, closure[order], ds[order])),
                                "gap": float(np.interp(g, closure[order], gaps[order]))})  # fmt: skip
        # Cameras: OpenGL frames (MuJoCo's own), relative to the root (shoulders) or the flange (wrist).
        root = self.root()
        F_world = root @ self.flange_pose()
        for name, cam in VIEWS.items():
            i = m.camera(cam).id
            Rw, pw = d.cam_xmat[i].reshape(3, 3), d.cam_xpos[i]
            parent = F_world if name == "wrist_cam" else root
            Rp, pp = parent[:3, :3], parent[:3, 3]
            fy = 0.5 / math.tan(math.radians(float(m.cam_fovy[i])) / 2)
            out["cameras"][name] = {"parent": "flange" if name == "wrist_cam" else "robot_root",
                                    "t": (Rp.T @ (pw - pp)).tolist(), "q": quat_wxyz(Rp.T @ Rw),
                                    "fx_norm": fy * IMAGE_H / IMAGE_W, "fy_norm": fy}  # fmt: skip
        wr = out["cameras"]["wrist_cam"]
        view = R_of(wr["q"]) @ np.array([0.0, 0.0, -1.0])
        out["wrist_view_flange"] = view.tolist()
        out["wrist_view_angle_deg"] = float(np.degrees(np.arccos(np.clip(view[0], -1, 1))))
        Rdown = R_of([0.7071, 0, 0.7071, 0])
        out["wrist_view_down_root"] = (Rdown @ view).tolist()
        for key, pad in (("pad_uv_open", out["pads"][0]), ("pad_uv_closed", out["pads"][-1])):
            p = R_of(wr["q"]).T @ (np.array([pad["d"], 0.0, 0.0]) - np.array(wr["t"]))
            out[key] = [0.5 + wr["fx_norm"] * p[0] / -p[2], 0.5 - wr["fy_norm"] * p[1] / -p[2]] if p[2] < 0 else None
        return out

    def _pad_sample(self) -> tuple[float, float]:
        env, d = self.env, self.env.data
        mount = next(i for i in range(env.model.nbody) if env.model.body(i).name.endswith("base_mount"))
        R, p = d.xmat[mount].reshape(3, 3), d.xpos[mount]
        centers = [np.mean([d.geom_xpos[g] for g in gs], axis=0) for gs in env._pad_geoms.values()]
        mid = (centers[0] + centers[1]) / 2
        return float((R.T @ (mid - p))[2]), float(env.get_state().gripper_width)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, default=float))
