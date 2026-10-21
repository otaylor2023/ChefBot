"""Any RoboCasa kitchen's fridge, opened with physics, on the whole-body controller of
experiments/astranomy/fridge.py (11-D absolute targets in odom, 15 Hz).

A scene is (layout, style) plus which fridge door the robot starts at; RoboCasa's
train layouts/styles (11-60) and test ones (1-10) use disjoint fridge models. The
fridge doors are the fixture's hinged "fridge ... door" joints (one, or two for a
French door); the freezer "doors" are drawers and are left alone. The episode
succeeds when any fridge door reaches the target angle.

The start pose is the one the Astranomy fridge used, relative to the door: the base
about 0.2 m past the handle (away from the hinge, clear of the swing) and 0.87 m out
from the door front, facing the fridge, randomized per seed.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from chefbot.envs.robocasa.config import RoboCasaConfig
from chefbot.interface.types import Pose2D
from chefbot.robot.arm.arm_controller import ArmController
from chefbot.robot.arm.gripper import Gripper
from chefbot.robot.config import RobotConfig
from chefbot.robot.kinematics.reachability import ReachabilityChecker
from chefbot.robot.skills.common import ArmStack

from experiments.astranomy.fridge import FridgeHost
from experiments.ortho_grasp.sim import ExperimentEnv, World

START = {"along": 0.20, "out": 0.87, "d_along": 0.15, "d_out": 0.12, "d_yaw_deg": 15.0}
# (along, out) from the handle, tried in order; along < 0 is in front of the door (toward the hinge)
START_CANDIDATES = [(a, o) for o in (0.87, 0.78, 0.70, 0.62, 0.55) for a in (0.20, 0.35, 0.05, -0.10)]
MIN_DOOR_RANGE = math.radians(80.0)
CAMERAS = {"left_shoulder": "robot0_agentview_left", "right_shoulder": "robot0_agentview_right", "wrist": "robot0_eye_in_hand"}
CAM_W, CAM_H = 640, 360


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
        cfg = dataclasses.replace(RoboCasaConfig.load(layout=layout, style=style), objects=(), camera_width=CAM_W, camera_height=CAM_H)
        self.config = {"counter_z": 0.92}
        self.env = ExperimentEnv(cfg, render=True)
        self.objects = {}
        self.counter_z = 0.92
        rc = RobotConfig.load()
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
        self.fixture, self.model_xml, self.doors = fridge_doors(self.env)
        self.cup = None
        self._q_cmd = None
        self.success_deg = 60.0
        self.frame_every = 2  # video at 10 fps

    # -- episode -------------------------------------------------------------------

    def start_pose(self, door: int, seed: int, along: float = START["along"], out: float = START["out"]) -> Pose2D:
        D = self.doors[door]
        rng = np.random.default_rng(seed)
        a = along + rng.uniform(-START["d_along"], START["d_along"])
        b = out + rng.uniform(-START["d_out"], START["d_out"])
        p = D.handle[:2] + a * D.along[:2] + b * D.out[:2]
        yaw = math.atan2(-D.out[1], -D.out[0]) + math.radians(rng.uniform(-START["d_yaw_deg"], START["d_yaw_deg"]))
        return Pose2D(float(p[0]), float(p[1]), float(yaw))

    def find_start(self, door: int, seed: int) -> Pose2D | None:
        """The seed's start pose, or (when the robot would touch something there, e.g. the cabinets
        beside a fridge in a housing) the first clear one of a few alternatives nearer the door's middle."""
        for along, out in START_CANDIDATES:
            pose = self.start_pose(door, seed, along, out)
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
        """Each camera: rgb (H,W,3 uint8), depth (H,W float32 m), K (3x3), cam_to_odom (4x4, OpenCV axes)."""
        inv = np.linalg.inv(self.odom)
        out = {}
        for name, cam in CAMERAS.items():
            r = self.env.get_camera_rgbd(cam)
            out[name] = {"rgb": r.rgb, "depth": r.depth, "K": r.intrinsics, "cam_to_odom": inv @ r.cam_to_world}
        return out

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
        views = [self.world.render(cam, 320, 180) for cam in CAMERAS.values()]
        side = self.world.render("robot0_robotview", 320, 180)
        self.frames.append(np.concatenate([np.concatenate(views[:2], 1), np.concatenate([views[2], side], 1)], 0))

    def save_video(self, name: str = "episode.mp4") -> Path | None:
        import imageio.v3 as iio

        if not self.frames:
            return None
        p = self.out / name
        iio.imwrite(p, np.stack(self.frames), fps=10, codec="libx264")
        return p
