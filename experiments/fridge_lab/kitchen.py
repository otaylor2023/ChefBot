"""Any RoboCasa kitchen's fridge, opened with physics, on the whole-body controller of
experiments/astranomy/fridge.py (11-D absolute targets in odom, 15 Hz).

A scene is (layout, style) plus which fridge door the robot starts at; RoboCasa's
train layouts/styles (11-60) and test ones (1-10) use disjoint fridge models. The
fridge doors are the fixture's hinged "fridge ... door" joints (one, or two for a
French door); the freezer "doors" are drawers and are left alone. The episode
succeeds when any fridge door reaches the target angle.

The start pose is the one the Astranomy fridge used, relative to the door: the base
about 0.2 m past the handle (away from the hinge, clear of the swing) and 0.87 m out
from the door front, facing the fridge, randomized per seed (the YAM's: about 1 m).

The robot is RoboCasa's mobile Franka (PandaOmron), or with FRIDGE_ROBOT=yam TidyBot++
with an I2RT YAM arm (chefbot/envs/robocasa/yam_tidybot.py), which starts closer: its arm
reaches about 0.45 m. The YAM's cameras are MessyMem's (a ZED 2 on a pole, one low on the base,
the D405 on the wrist), each moved and its field of view changed a little per episode
(CAMERA_JITTER): the skill gets each camera's true pose and intrinsics, as a calibrated robot would.
"""

from __future__ import annotations

import dataclasses
import math
import os
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from robosuite.utils.camera_utils import get_camera_extrinsic_matrix

from chefbot.envs.robocasa.config import RoboCasaConfig
from chefbot.interface.types import Pose2D
from chefbot.robot.arm.arm_controller import ArmController
from chefbot.robot.arm.gripper import Gripper
from chefbot.robot.config import RobotConfig
from chefbot.robot.kinematics.reachability import ReachabilityChecker
from chefbot.robot.skills.common import ArmStack

from experiments.astranomy.fridge import FridgeHost
from experiments.fridge_lab import fisheye
from experiments.astranomy.host import REPO
from experiments.ortho_grasp.sim import ExperimentEnv, World



@dataclass(frozen=True)
class LabRobot:
    robocasa: str  # RoboCasa robot name
    config: str | None  # robot-layer config (None: chefbot/config/robot.yaml)
    start: dict  # nominal stance and per-seed jitter, relative to the handle
    start_outs: tuple  # stance distances out from the handle, tried in order
    grasp_outs: tuple  # the oracle's grasping stance distances, tried in order
    grip_site: float  # m from the flange (the skill's, DROID axes) to the grip site along the fingers
    pad_depth: tuple  # (open, closed): m from the flange to the pads' midpoint along the fingers
    cameras: dict  # the skill's camera names -> the model's
    camera_jitter: dict  # model camera -> (position m, rotation deg, fovy fraction): uniform +- per episode
    rgb_only: tuple = ()  # the skill's cameras with no depth (as on the real robot)
    fisheye: tuple = ()  # the skill's cameras with a fisheye lens (fisheye.py; their fovy jitter becomes focal-length jitter)


ROBOTS = {
    "panda": LabRobot("PandaOmron", None, {"along": 0.20, "out": 0.87, "d_along": 0.15, "d_out": 0.12, "d_yaw_deg": 15.0},
                      (0.87, 0.78, 0.70, 0.62, 0.55), (0.87, 0.78, 0.70, 0.62, 0.55), 0.1488, (0.1367, 0.1477),
                      {"left_shoulder": "robot0_agentview_left", "right_shoulder": "robot0_agentview_right", "wrist": "robot0_eye_in_hand"}, {}),
    # The YAM starts about 1 m out (base center), where its ZEDs see the fridge, and grasps from about
    # 0.57 m (its arm's base is 0.12 m ahead of the base center).
    "yam": LabRobot("YamTidybot", "chefbot/config/robot_yam.yaml", {"along": 0.20, "out": 1.0, "d_along": 0.12, "d_out": 0.10, "d_yaw_deg": 15.0},
                    (1.0, 0.9, 1.1, 0.8, 0.7, 0.6), (0.57, 0.62, 0.52, 0.67), 0.1447, (0.1335, 0.1335),
                    {"pole": "robot0_pole_zed", "base": "robot0_base_zed", "wrist": "robot0_eye_in_hand"},
                    {"robot0_pole_zed": (0.03, 4.0, 0.05), "robot0_base_zed": (0.03, 4.0, 0.05), "robot0_eye_in_hand": (0.005, 2.0, 0.03)},
                    rgb_only=("wrist",), fisheye=("wrist",)),  # MessyMem's wrist camera: an Arducam 160 degree fisheye USB camera, no depth
}
ROBOT = ROBOTS[os.environ.get("FRIDGE_ROBOT", "panda")]
START = ROBOT.start
# (along, out) from the handle, tried in order; along < 0 is in front of the door (toward the hinge)
START_ALONGS = (0.20, 0.35, 0.05, -0.10)
MIN_DOOR_RANGE = math.radians(80.0)
CAMERAS = ROBOT.cameras
CAMERA_JITTER = ROBOT.camera_jitter
CAM_W, CAM_H = 640, 360
FISHEYE_CAMS = {CAMERAS[n] for n in ROBOT.fisheye}
FISH_EXT = fisheye.pinhole_extent(fisheye.nominal_f() * (1 - CAMERA_JITTER.get(next(iter(FISHEYE_CAMS)), (0, 0, 0))[2])) if FISHEYE_CAMS else None


@dataclass
class Door:
    joint: str
    joint_id: int
    qpos: int
    handle_geoms: list[int]
    hinge: np.ndarray  # world, at episode start (door closed)
    axis: np.ndarray  # world; +angle opens
    handle: np.ndarray  # world center of the handle, door closed
    handle_lo: np.ndarray  # world AABB of the handle geoms
    handle_hi: np.ndarray
    out: np.ndarray  # unit, horizontal: out of the fridge front
    along: np.ndarray  # unit, horizontal: hinge -> handle
    width: float  # horizontal hinge -> handle distance


class LabWorld(World):
    """A kitchen with no loose objects; the arm stack without the navigation map."""

    def __init__(self, layout: int, style: int):
        cfg = dataclasses.replace(RoboCasaConfig.load(layout=layout, style=style), objects=(), camera_width=CAM_W, camera_height=CAM_H,
                                  robot=ROBOT.robocasa)  # fmt: skip
        self.config = {"counter_z": 0.92}
        self.env = ExperimentEnv(cfg, render=True)
        self.objects = {}
        self.counter_z = 0.92
        rc = RobotConfig.load(REPO / ROBOT.config if ROBOT.config else None)
        frames = ReachabilityChecker(rc)
        self.arm = ArmStack(self.env, rc, frames.ik, ArmController(self.env, frames.ik, rc), Gripper(self.env, rc), frames)
        self.robot_config = rc
        self.base = self.env.get_state().base_pose


def fridge_doors(env) -> tuple[str, str, list[Door]]:
    """(fixture name, model xml, the fridge doors) of the kitchen's fridge."""
    from robocasa.models.fixtures.fridge import Fridge

    m, d = env.model, env.data
    fridge = next(f for f in env.suite.fixtures.values() if isinstance(f, Fridge))
    R = Rotation.from_euler("z", fridge.rot).as_matrix()
    out = -R[:, 1]  # the fixture's front faces its local -y
    doors = []
    for name in fridge._fridge_door_joint_names:
        j = m.joint(name).id
        if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE or m.jnt_range[j, 1] < MIN_DOOR_RANGE:
            continue
        prefix = name[: -len("_joint")] + "_handle"
        geoms = [g for g in range(m.ngeom) if (m.geom(g).name or "").startswith(prefix)]
        if not geoms:
            continue
        lo, hi = _aabb(m, d, geoms)
        main = [g for g in geoms if m.geom(g).name.endswith("_handle_main")]
        handle = d.geom_xpos[main[0]].copy() if main else (lo + hi) / 2
        hinge = d.xanchor[j].copy()
        v = handle - hinge
        v[2] = 0.0
        v -= (v @ out) * out
        width = float(np.linalg.norm(v))
        doors.append(Door(name, j, int(m.jnt_qposadr[j]), geoms, hinge, d.xaxis[j].copy(), handle, lo, hi, out, v / width, width))
    return fridge.name, getattr(fridge, "xml", ""), doors


def _aabb(m, d, geoms: list[int]) -> tuple[np.ndarray, np.ndarray]:
    pts = []
    for g in geoms:
        c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
        R, p = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
        for s in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3):
            pts.append(p + R @ (c + s * h))
    pts = np.array(pts)
    return pts.min(0), pts.max(0)


class LabHost(FridgeHost):
    """FridgeHost on any kitchen: reset at a door, success on any fridge door, RGB-D observations."""

    def __init__(self, layout: int, style: int, out_dir: Path, record: bool = True):
        self.world = LabWorld(layout, style)
        self.layout, self.style = layout, style
        self.cfg = {"counter_z": 0.92}
        self.env, self.arm = self.world.env, self.world.arm
        self.out, self.record = Path(out_dir), record
        self.history, self.actions, self.frames, self.step_count = [], [], [], 0
        self.gap_open = 0.085
        self._fish_f = fisheye.nominal_f()
        self.grip_site = ROBOT.grip_site
        self.fixture, self.model_xml, self.doors = fridge_doors(self.env)
        self.cup = None
        self._q_cmd = None
        self.success_deg = 60.0
        self.frame_every = 2  # video at 10 fps

    # -- episode -------------------------------------------------------------------

    def start_pose(self, door: int, seed: int, along: float = START["along"], out: float = START["out"], jitter: bool = True) -> Pose2D:
        D = self.doors[door]
        rng = np.random.default_rng(seed)
        k = 1.0 if jitter else 0.0
        a = along + k * rng.uniform(-START["d_along"], START["d_along"])
        b = out + k * rng.uniform(-START["d_out"], START["d_out"])
        p = D.handle[:2] + a * D.along[:2] + b * D.out[:2]
        yaw = math.atan2(-D.out[1], -D.out[0]) + k * math.radians(rng.uniform(-START["d_yaw_deg"], START["d_yaw_deg"]))
        return Pose2D(float(p[0]), float(p[1]), float(yaw))

    def find_start(self, door: int, seed: int, jitter: bool = True, outs: tuple | None = None) -> Pose2D | None:
        """The seed's start pose, or (when the robot would touch something there, e.g. the cabinets
        beside a fridge in a housing) the first clear one of a few alternatives nearer the door's middle.
        jitter=False: the nominal stance (no per-seed randomization). outs: the distances out to try
        (default the robot's start distances)."""
        for along, out in [(a, o) for o in (outs or ROBOT.start_outs) for a in START_ALONGS]:
            pose = self.start_pose(door, seed, along, out, jitter)
            self.reset_at(door, seed, pose)
            if not self.debug()["robot_touching"]:
                return pose
        return None

    def reset_at(self, door: int, seed: int, pose: Pose2D | None = None) -> None:
        env = self.env
        env.reset_scene()
        pose = pose or self.start_pose(door, seed)
        env.teleport_base(pose)
        env.set_arm_joints(np.array(self.world.robot_config.arm_ready))
        env.set_gripper(False)
        with env.lock:
            for D in self.doors:
                env.data.qpos[D.qpos] = 0.0
            self._jitter_cameras(seed)
            mujoco.mj_forward(env.model, env.data)
        for _ in range(10):
            env.set_base_velocity(0.0, 0.0, 0.0)
            env.wait_for_next_tick(0.05)
        self.gap_open = max(env.get_state().gripper_width, 1e-3)
        self.odom = self.root()
        self.history, self.actions, self.frames, self.step_count = [], [], [], 0
        self._t_next = float(env.get_state().time)
        self._q_cmd = None
        self._ticks = 0
        self.out.mkdir(parents=True, exist_ok=True)
        self.door, self.seed, self.start = door, seed, pose

    def _jitter_cameras(self, seed: int) -> None:
        """Move and turn each camera a little and change its field of view, per (scene, seed)."""
        m = self.env.model
        if not hasattr(self, "_cam_nominal"):
            self._cam_nominal = {c: (m.cam_pos[m.camera(c).id].copy(), m.cam_quat[m.camera(c).id].copy(), float(m.cam_fovy[m.camera(c).id]))
                                 for c in CAMERA_JITTER}  # fmt: skip
        rng = np.random.default_rng([seed, self.layout, self.style, 7])
        for c, (dp, dr, df) in CAMERA_JITTER.items():
            pos, quat, fovy = self._cam_nominal[c]
            i = m.camera(c).id
            m.cam_pos[i] = pos + rng.uniform(-dp, dp, 3)
            turn = Rotation.from_rotvec(np.radians(rng.uniform(-dr, dr, 3)))
            x, y, z, w = (turn * Rotation.from_quat([*quat[1:], quat[0]])).as_quat()
            m.cam_quat[i] = [w, x, y, z]
            if c in FISHEYE_CAMS:  # the pinhole render covers the widest lens; the lens itself varies in focal length
                self._fish_f = fisheye.nominal_f() * (1.0 + rng.uniform(-df, df))
                m.cam_fovy[i] = fisheye.fovy_deg(FISH_EXT)
            else:
                m.cam_fovy[i] = fovy * (1.0 + rng.uniform(-df, df))

    def door_angles(self) -> list[float]:
        return [float(np.degrees(self.env.data.qpos[D.qpos])) for D in self.doors]

    def door_deg(self) -> float:
        return max(self.door_angles())

    def start_contact(self) -> bool:
        s = self.env.get_state()
        return bool(s.base_in_contact)

    # -- observation: what a real robot has ------------------------------------------

    def proprio(self) -> dict:
        s = self.env.get_state()
        F = self.flange_in_odom()
        bx, by, byaw = self.base_in_odom()
        return {"step": self.step_count, "sim_time_s": self.step_count / 15.0, "flange_pose_odom": F.tolist(),
                "base_pose_odom": [bx, by, byaw], "gripper_closed_cmd": bool(s.gripper_closed_cmd), "gripper_closure": self.closure(),
                "gripper_width_m": float(s.gripper_width), "arm_q": [float(v) for v in s.arm_q],
                "arm_q_cmd": None if self._q_cmd is None else [float(v) for v in self._q_cmd],
                "base_in_contact": bool(s.base_in_contact)}  # fmt: skip

    def rgbd(self) -> dict[str, dict]:
        """Each camera: rgb (H,W,3 uint8), depth (H,W float32 m; None for an RGB-only camera), K (3x3), cam_to_odom (4x4, OpenCV axes)."""
        inv = np.linalg.inv(self.odom)
        out = {}
        for name, cam in CAMERAS.items():
            if cam in FISHEYE_CAMS:
                out[name] = {"rgb": self._fisheye(cam, 1.0), "depth": None, "K": fisheye.K_of(self._fish_f), "D": list(fisheye.D),
                             "model": "fisheye", "cam_to_odom": inv @ get_camera_extrinsic_matrix(self.env.suite.sim, cam)}  # fmt: skip
                continue
            r = self.env.get_camera_rgbd(cam)
            out[name] = {"rgb": r.rgb, "depth": None if name in ROBOT.rgb_only else r.depth, "K": r.intrinsics, "D": None, "model": "pinhole",
                         "cam_to_odom": inv @ r.cam_to_world}  # fmt: skip
        return out

    def _fisheye(self, cam: str, scale: float) -> np.ndarray:
        """The fisheye image: a wide pinhole render, warped (fisheye.py)."""
        _, _, pw, ph = fisheye.maps(self._fish_f, scale, FISH_EXT)
        return fisheye.warp(self.world.render(cam, pw, ph), self._fish_f, scale, FISH_EXT)

    def debug(self) -> dict:
        """Privileged, for the researcher's logs only: never sent to the skill."""
        inv = np.linalg.inv(self.odom)
        m, d = self.env.model, self.env.data
        doors = []
        for D in self.doors:
            h = (inv @ np.append(d.geom_xpos[D.handle_geoms[0]], 1.0))[:3]
            doors.append({"angle_deg": float(np.degrees(d.qpos[D.qpos])), "handle_odom": h.tolist(),
                          "hinge_odom": (inv @ np.append(D.hinge, 1.0))[:3].tolist()})  # fmt: skip
        robot = self.env.robot_geoms
        touching = sorted({m.body(int(m.geom_bodyid[o])).name for a, b in self.env._touching() if (a in robot) != (b in robot)
                           for o in ((b,) if a in robot else (a,)) if o not in self.env._floor_geoms})  # fmt: skip
        return {"doors": doors, "robot_touching": touching[:8]}

    # -- recording -----------------------------------------------------------------

    def _frame(self) -> None:
        self._ticks = getattr(self, "_ticks", 0) + 1
        if self._ticks % self.frame_every:
            return
        views = [self._fisheye(cam, 0.5) if cam in FISHEYE_CAMS else self.world.render(cam, 320, 180) for cam in CAMERAS.values()]
        side = self.world.render("robot0_robotview", 320, 180)
        self.frames.append(np.concatenate([np.concatenate(views[:2], 1), np.concatenate([views[2], side], 1)], 0))

    def save_video(self, name: str = "episode.mp4") -> Path | None:
        import imageio.v3 as iio

        if not self.frames:
            return None
        p = self.out / name
        iio.imwrite(p, np.stack(self.frames), fps=10, codec="libx264")
        return p
