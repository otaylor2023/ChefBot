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
LOW_CLEARANCE_M = 0.05  # an arm point passes over a low obstacle only with its underside this much above its top


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
        from chefbot.embodiments import arm_solvers

        return Kinematics(self.config, self.body, arm_solvers.chosen(self.raw.get("control", {}).get("arm_solver")))

    def speed(self, name: str | None = None, *, arm: str | None = None, base: str | None = None, use_arm: bool = True,
              use_base: bool = True) -> Speed:  # fmt: skip
        """The arm's rates from profile `arm` and the base's from `base` (each default `name`, then the robot's own).
        A base profile over the robot's hard cap (control.limits) is refused, not clipped. use_base=False (an arm-only
        run): the base's speed is 0, nothing can move it, and its profile isn't checked; use_arm=False likewise."""
        from chefbot.interface import limits

        default = self.raw["control"]["speed"]
        arm_name, base_name = arm or name or default, base or name or default
        arm_lin, arm_ang, joint = limits.arm_profile(arm_name) if use_arm else (0.0, 0.0, 0.0)
        vel = limits.base_profile(base_name, self.name)[0] if use_base else (0.0, 0.0, 0.0)
        label = arm_name if (arm_name == base_name or not use_base) else (base_name if not use_arm else f"arm {arm_name}, base {base_name}")
        return Speed(label, tuple(float(v) for v in vel), arm_lin, arm_ang, joint)

    # -- the base: what the robot layer drives it with ----------------------------------------------------------

    def footprint_below(self, arm_q=None):
        """footprint_below(top) for the robot layer (robot/robot.py, robot/nav/footprint.py): the robot seen from above,
        what of it is lower than an obstacle's top (inf: all of it, a wall). With a body (robot.yaml `body:`): the
        plate's outline, plus for joints `arm_q` the arm's points whose underside is under top + LOW_CLEARANCE_M
        (interface/workspace.py Body.footprint); without one, a circle of the config's footprint_radius."""
        from chefbot.interface.workspace import Footprint

        body = self.body
        if body is None:
            radius = self.config.footprint_radius
            return lambda top=math.inf: Footprint((0.0, 0.0, 0.0, 0.0), np.array([[0.0, 0.0, radius]]))
        q = None if arm_q is None else np.asarray(arm_q, dtype=float)
        return lambda top=math.inf: body.footprint(q, below=None if math.isinf(top) else top + LOW_CLEARANCE_M)

    def stop_model(self):
        """The base's measured stop (robot.yaml control.limits base_stop_*; robot/nav/speed.py StopModel), or None."""
        from chefbot.interface import limits
        from chefbot.robot.nav.speed import StopModel

        stop = limits.base_stop(self.name)
        return StopModel.from_limits(stop) if stop else None

    @property
    def base_accel(self) -> float:
        """The base's planar acceleration (m/s^2): its hard cap's (control.limits base_max_accel), else 0.1."""
        lim = (self.raw.get("control") or {}).get("limits") or {}
        return float(lim["base_max_accel"][0]) if "base_max_accel" in lim else 0.1

    def navigator(self, env, grid, speed: Speed, *, arm_q=None, config: dict | None = None, **robot_kwargs):
        """The robot layer's Robot (robot/robot.py) on `env` and map `grid`, built the one way every caller that drives
        this robot's base gets it: its footprint (footprint_below; with `arm_q`, the arm as it stands, which then stays
        as it is while the base drives: tuck_arm False), its measured stop and acceleration, `speed`'s base speeds
        (Robot.speed: refused over the hard cap). `config`: RobotConfig fields to set for this run (tolerances,
        margins); robot_kwargs go to Robot (reachability, proposers, centre_limit, drive_log), and may replace the
        footprint or the stop for a trial (real_trials' scaled footprint)."""
        from dataclasses import replace

        from chefbot.robot.robot import Robot as RobotLayer

        vel = speed.base_max_vel
        if not (vel[0] > 0 and vel[2] > 0):
            raise ValueError(f"speed {speed.name} gives the base no speed: navigation needs one")
        cfg = replace(self.config, max_lin_vel=float(vel[0]), max_ang_vel=float(vel[2]), **(config or {}))
        kw = {"footprint_below": self.footprint_below(arm_q), "stop_model": self.stop_model(), "base_accel": self.base_accel,
              "tuck_arm": arm_q is None} | robot_kwargs  # fmt: skip
        return RobotLayer(env, grid, cfg, **kw)

    def base_guard(self, map_dir=None):
        """A BaseGuard (robot/safety.py) for the whole-body controller driving the base (the real host with --base): it
        never drives without the env's localization (mocap); with a map folder (maps/README.md), also not into it
        (the plate at its heading: the arm moves with the skill and is the host's own guard's) and no faster than it
        can stop in the free space ahead (stop_model)."""
        from pathlib import Path

        from chefbot.robot.nav.footprint import FootprintSpace
        from chefbot.robot.nav.grid import OccupancyGrid
        from chefbot.robot.safety import BaseGuard

        space = None
        if map_dir is not None:
            if self.body is None:
                raise ValueError(f"{self.name} has no body: (its footprint) to guard the base on a map with")
            space = FootprintSpace.by_height(OccupancyGrid.load(Path(map_dir) / "grid.npy"), self.footprint_below())
        return BaseGuard(space, stop_model=self.stop_model(), accel=self.base_accel, arm_moved_rad=None)


class Kinematics:
    """The arm's IK on the flange and the frames everything shares: the root on the base, the flange."""

    def __init__(self, config: RobotConfig, body=None, solver: str = "qp"):
        from chefbot.embodiments import constraints

        self.config = config
        self.body = body  # interface/workspace.py Body or None
        self.constraints = constraints.for_robot(body)  # where the arm may not go: every solver avoids them
        self.solver = solver  # embodiments/arm_solvers: the controller's arm solver (set before the first arm_step)
        self._solver = None
        self.log = None  # where the router says it's going around (or can't)
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

    def arm_step(self, target: np.ndarray, q, dt: float, max_lin: float, max_ang: float, max_joint: float) -> np.ndarray:
        """The controller's arm step: the joints after dt from q toward the flange pose `target`, within the joint
        limits and clear of the constraints, by the chosen solver (embodiments/arm_solvers). Raises mink's
        NoSolutionFound when it can't (the controller holds)."""
        if self._solver is None:
            from chefbot.embodiments import arm_solvers

            hand = self.body.hand_m if self.body is not None else 0.0
            self._solver = arm_solvers.make(self.solver, self.ik, self.constraints, hand, self.log)
        return self._solver.step(target, q, dt, max_lin, max_ang, max_joint)
