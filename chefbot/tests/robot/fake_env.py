"""FakeEnv: a kinematic integrator standing in for a simulator.

The base integrates its velocity command exactly. The arm follows its joint
targets at a limited rate, the gripper closes at a constant speed, and a
`FakeObject` under the fingers stops them and then rides along with the hand.
"""

from __future__ import annotations

import math
import time

import numpy as np

from dataclasses import dataclass

from chefbot.interface.env import Env
from chefbot.interface.types import RGBD, EnvState, LocalizationReading, Pose2D, wrap_angle
from chefbot.robot.config import RobotConfig
from chefbot.robot.nav.grid import OccupancyGrid


GRIPPER_OPEN = 0.085
_GRIPPER_SPEED = 0.2  # m/s
_ARM_SPEED = 3.0  # rad/s: how fast the fake servo closes on its target
_GRASP_RADIUS = 0.02  # the grip point must be this close to an object to catch it


@dataclass
class FakeObject:
    position: np.ndarray  # (3,) world
    width: float = 0.05
    held: bool = False


class FakeEnv(Env):
    """Integrates the commanded base velocity exactly.

    `obstacles` (optional) are physical obstacles the base cannot enter; running
    into one holds the base in place and sets `base_in_contact`. `tick_sleep`
    slows ticks down in wall time so tests can interrupt a motion.

    `localizer=True` models the real robot: get_state reports odometry (the true pose seen from `odom_origin`, the
    world pose of odometry's zero; change it to make odometry drift) and the true pose as EnvState.localization
    (None while `localization_available` is False). Commands are then in odometry's frame.
    """

    def __init__(
        self,
        pose: Pose2D = Pose2D(0.0, 0.0, 0.0),
        obstacles: OccupancyGrid | None = None,
        tick_sleep: float = 0.0,
        arm_q: np.ndarray | None = None,
        tcp_world=None,
        localizer: bool = False,
        odom_origin: Pose2D = Pose2D(0.0, 0.0, 0.0),
    ):
        """`tcp_world(base_pose, arm_q) -> (3,)` locates the grip point; needed only for grasping."""
        # Starts stowed, so base motions do not have to tuck the arm first.
        self.arm_q = np.array(RobotConfig().arm_stow if arm_q is None else arm_q, dtype=float)
        self.arm_qd = np.zeros(7)
        self.arm_target = self.arm_q.copy()
        self.arm_targets: list[tuple[float, np.ndarray]] = []  # (time, target) as commanded
        self.arm_in_contact = False
        self.contact_when = None  # callable(env) -> bool, evaluated every tick
        self.gripper_width = GRIPPER_OPEN
        self.gripper_closed = False
        self.gripper_commands: list[bool] = []
        self.objects: list[FakeObject] = []
        self._tcp_world = tcp_world
        self.pose = pose
        self.obstacles = obstacles
        self.tick_sleep = tick_sleep
        self.velocity = np.zeros(3)  # world frame
        self.in_contact = False
        self.time = 0.0
        self.ticks = 0
        self.commands: list[tuple[float, np.ndarray]] = []  # (time, command in base_pose's frame)
        self.localizer = localizer
        self.odom_origin = odom_origin
        self.localization_available = True

    def _rot(self, vx: float, vy: float, yaw: float) -> tuple[float, float]:
        c, s = math.cos(yaw), math.sin(yaw)
        return c * vx - s * vy, s * vx + c * vy

    def odom(self) -> Pose2D:
        """The true pose in odometry's frame (the true pose itself without a localizer)."""
        if not self.localizer:
            return self.pose
        o = self.odom_origin
        x, y = self._rot(self.pose.x - o.x, self.pose.y - o.y, -o.yaw)
        return Pose2D(x, y, wrap_angle(self.pose.yaw - o.yaw))

    def get_state(self) -> EnvState:
        v = self.velocity.copy()
        if self.localizer:
            v[0], v[1] = self._rot(v[0], v[1], -self.odom_origin.yaw)
        return EnvState(
            self.odom(),
            v,
            self.in_contact,
            self.time,
            arm_q=self.arm_q.copy(),
            arm_qd=self.arm_qd.copy(),
            gripper_width=self.gripper_width,
            gripper_closed_cmd=self.gripper_closed,
            arm_in_contact=self.arm_in_contact,
            localization=(LocalizationReading(self.pose if self.localization_available else None)
                          if self.localizer else None),  # fmt: skip
        )

    def set_base_velocity(self, vx: float, vy: float, vyaw: float, frame: str = "world") -> None:
        self.commands.append((self.time, np.array([vx, vy, vyaw], dtype=float)))
        if frame == "base":
            vx, vy = self._rot(vx, vy, self.pose.yaw)
        elif frame == "world":
            if self.localizer:  # odometry's frame, into the true world's
                vx, vy = self._rot(vx, vy, self.odom_origin.yaw)
        else:
            raise ValueError(f"unknown frame {frame!r}")
        self.velocity = np.array([vx, vy, vyaw], dtype=float)

    def set_arm_joint_targets(self, q: np.ndarray) -> None:
        self.arm_target = np.array(q, dtype=float)
        self.arm_targets.append((self.time, self.arm_target.copy()))

    def set_gripper(self, closed: float) -> None:
        self.gripper_closed = float(closed) >= 0.5
        self.gripper_commands.append(self.gripper_closed)

    def tcp(self) -> np.ndarray | None:
        return None if self._tcp_world is None else np.asarray(self._tcp_world(self.pose, self.arm_q))

    def get_camera_rgbd(self, cam: str) -> RGBD:
        raise NotImplementedError("FakeEnv has no cameras")

    def wait_for_next_tick(self, dt: float) -> None:
        if self.tick_sleep:
            time.sleep(self.tick_sleep)
        vx, vy, vyaw = self.velocity
        x, y = self.pose.x + vx * dt, self.pose.y + vy * dt
        if self.obstacles is not None and not self.obstacles.is_free(x, y):
            self.in_contact = True
            x, y = self.pose.x, self.pose.y
        else:
            self.in_contact = False
        self.pose = Pose2D(x, y, wrap_angle(self.pose.yaw + vyaw * dt))
        # Arm: move toward the target, at most _ARM_SPEED per joint.
        step = np.clip(self.arm_target - self.arm_q, -_ARM_SPEED * dt, _ARM_SPEED * dt)
        self.arm_q, self.arm_qd = self.arm_q + step, step / dt
        # Gripper: close until the pads meet or an object is between them.
        tcp = self.tcp()
        between = None
        if tcp is not None:
            between = next(
                (o for o in self.objects if o.held or np.linalg.norm(o.position - tcp) <= _GRASP_RADIUS), None
            )
        if self.gripper_closed:
            floor = between.width if between is not None else 0.0
            self.gripper_width = max(floor, self.gripper_width - _GRIPPER_SPEED * dt)
            if between is not None and self.gripper_width <= between.width:
                between.held = True
        else:
            self.gripper_width = min(GRIPPER_OPEN, self.gripper_width + _GRIPPER_SPEED * dt)
            for obj in self.objects:
                obj.held = False
        for obj in self.objects:
            if obj.held and tcp is not None:
                obj.position = tcp.copy()
        self.time += dt
        self.ticks += 1
        if self.contact_when is not None:
            self.arm_in_contact = bool(self.contact_when(self))
