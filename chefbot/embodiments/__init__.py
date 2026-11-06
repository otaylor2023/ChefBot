"""Lab robots: one folder per robot (robot.yaml, brief.md), loaded by name. A robot names a ChefBot robot config
(chefbot/config: arm, gripper, mount, postures, IK), the env's robot and gripper, its cameras and speed, its body (the
parts the arm keeps clear of) and its controller's settings. Data only: the controller that reads it is
robot/control (wholebody.py, arm_solvers/, constraints.py, kinematics.py).

    robot = robots.load("panda_franka")
    robot.kin.flange_pose(q)        # the flange in the arm base frame
    robot.speed("slow")             # a speed profile, clipped to the robot's hardware limits

Frames. The robot root is the arm base frame (+x forward, +y left, +z up), placed on the base by the config's arm
mount. The flange is the arm's mounting flange as the IK has it (FR3 link 7's attachment site): +z out of the flange
toward the fingers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import yaml

from chefbot.robot.config import CONFIG_DIR, RobotConfig

ROOT = Path(__file__).resolve().parent
SPEED_PROFILES = ROOT / "speed_profiles.yaml"


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
    def body(self):
        """The robot's own fixed parts the arm keeps clear of (robot.yaml `body:`; interface/workspace.py Body), or
        None."""
        from chefbot.interface.workspace import Body

        return Body.from_config(self.raw["body"]) if "body" in self.raw else None

    @cached_property
    def kin(self) -> "Kinematics":
        from chefbot.robot.control import arm_solvers
        from chefbot.robot.control.kinematics import Kinematics

        return Kinematics(self.config, self.body, arm_solvers.chosen(self.raw.get("control", {}).get("arm_solver")))

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

    def on_real(self, speed: Speed) -> Speed:
        """`speed` clipped to the robot's real-only limits (robot.yaml control.real_limits), which the real host
        applies to every run on the robot; the sims are not slowed."""
        lim = self.raw["control"].get("real_limits") or {}
        base = np.minimum(speed.base_max_vel, lim.get("base_max_vel", speed.base_max_vel))
        return Speed(speed.name, tuple(float(v) for v in base), *(float(min(getattr(speed, k), lim.get(k, getattr(speed, k))))
                                                                 for k in ("arm_lin", "arm_ang", "joint")))  # fmt: skip
