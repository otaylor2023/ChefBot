"""Lab robots: one folder per robot (robot.yaml, brief.md), loaded by name. A robot names a ChefBot robot config
(chefbot/config: arm, gripper, mount, postures, IK), the env's robot and gripper, its cameras and speed.

    robot = robots.load("panda_franka")
    robot.kin.flange_pose(q)        # the flange in the arm base frame
    robot.speed("slow")             # a speed profile, clipped to the robot's hardware limits

Frames. The robot root is the arm base frame (+x forward, +y left, +z up), placed on the base by the config's arm
mount. The flange is the arm's mounting flange as the IK has it (FR3 link 7's attachment site): +z out of the flange
toward the fingers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import yaml

from chefbot.interface.types import Pose2D
from chefbot.robot.arm.ik import ArmIK
from chefbot.robot.config import CONFIG_DIR, RobotConfig

ROOT = Path(__file__).resolve().parent
SPEED_PROFILES = ROOT / "speed_profiles.yaml"
# The controller's IK steps are a few mm: tighter than the robot layer's execution tolerance, or a step would count as
# reached at its seed and never move.
IK_POS_TOLERANCE, IK_ROT_TOLERANCE = 0.0005, math.radians(0.3)


def names() -> list[str]:
    return sorted(p.parent.name for p in ROOT.glob("*/robot.yaml"))


def load(name: str) -> "Robot":
    d = ROOT / name
    if not (d / "robot.yaml").exists():
        raise ValueError(f"no robot {name!r}; robots: {', '.join(names())}")
    return Robot(name, d, yaml.safe_load((d / "robot.yaml").read_text()))


@dataclass(frozen=True)
class Speed:
    name: str
    base_max_vel: tuple  # m/s, m/s, rad/s
    arm_lin: float  # m/s
    arm_ang: float  # rad/s
    joint: float  # rad/s


@dataclass
class Robot:
    name: str
    dir: Path
    raw: dict = field(repr=False)

    @property
    def cameras(self) -> dict:
        """The skill's camera names -> {sim, depth, noise}."""
        return self.raw["cameras"]

    @property
    def env(self) -> dict:
        """The env's robot and gripper: {robot, gripper}."""
        return self.raw["env"]

    @property
    def hand(self) -> dict:
        return self.raw["hand"]

    @property
    def brief(self) -> str:
        return (self.dir / "brief.md").read_text().split("-->", 1)[-1].strip()

    @cached_property
    def config(self) -> RobotConfig:
        return RobotConfig.load(CONFIG_DIR / self.raw["robot_config"])

    @cached_property
    def kin(self) -> "Kinematics":
        return Kinematics(self.config)

    def speed(self, name: str | None = None) -> Speed:
        """A speed profile (default: the robot's), clipped to its hardware limits."""
        name = name or self.raw["control"]["speed"]
        profiles = yaml.safe_load(SPEED_PROFILES.read_text())
        if name not in profiles:
            raise ValueError(f"no speed profile {name!r}; profiles: {', '.join(profiles)}")
        p = profiles[name]
        lim = self.raw["control"].get("limits", {})
        base = np.minimum(p["base_max_vel"], lim.get("base_max_vel", p["base_max_vel"]))
        return Speed(name, tuple(float(v) for v in base), float(p["arm_lin"]), float(p["arm_ang"]), float(p["joint"]))


class Kinematics:
    """The arm's IK on the flange and the frames everything shares: the root on the base, the flange."""

    def __init__(self, config: RobotConfig):
        self.config = config
        self.ik = ArmIK(tcp_xyz=(0.0, 0.0, 0.0), joint_limits=config.joint_limits, posture=np.asarray(config.arm_ready),
                        pos_tolerance=config.ik_pos_tolerance, pose_pos_tolerance=IK_POS_TOLERANCE,
                        pose_rot_tolerance=IK_ROT_TOLERANCE, max_iters=config.ik_max_iters, dt=config.ik_dt,
                        damping=config.ik_damping, arm=config.arm)  # fmt: skip
        self.mount_xyz = np.asarray(config.arm_mount_xyz, dtype=float)
        self.mount_yaw = config.arm_mount_yaw

    def root_in_world(self, base: Pose2D) -> np.ndarray:
        """The arm base frame (4x4) for a base pose (the base center on the floor)."""
        yaw = base.yaw + self.mount_yaw
        c, s = math.cos(yaw), math.sin(yaw)
        cb, sb = math.cos(base.yaw), math.sin(base.yaw)
        m = self.mount_xyz
        T = np.eye(4)
        T[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
        T[:3, 3] = [base.x + cb * m[0] - sb * m[1], base.y + sb * m[0] + cb * m[1], m[2]]
        return T

    def base_for_root(self, root_world: np.ndarray) -> Pose2D:
        """The base pose that puts the robot root at `root_world`."""
        m = self.mount_xyz
        yaw = math.atan2(root_world[1, 0], root_world[0, 0]) - self.mount_yaw
        c, s = math.cos(yaw), math.sin(yaw)
        return Pose2D(float(root_world[0, 3] - (c * m[0] - s * m[1])), float(root_world[1, 3] - (s * m[0] + c * m[1])), float(yaw))

    def flange_pose(self, q) -> np.ndarray:
        """The flange (4x4) in the arm base frame, for joints q."""
        return self.ik.fk(np.asarray(q, dtype=float))
