"""Plain serializable types shared by the agent, robot, and env layers.

Everything that crosses a layer boundary is one of these (or a builtin /
numpy array). No sim handles, no env-specific names.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def wrap_angle(a: float) -> float:
    """Wrap an angle in radians to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


@dataclass(frozen=True)
class Pose2D:
    """Base pose in the world frame: x, y in meters, yaw in radians."""

    x: float
    y: float
    yaw: float = 0.0

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)

    def distance_to(self, other: "Pose2D") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass
class Detection:
    label: str
    position_world: np.ndarray  # (3,) its middle
    confidence: float = 1.0
    # For something too big to work on at its middle (a long counter): the part
    # of it nearest the robot, (3,). None: work at `position_world`.
    near_point_world: np.ndarray | None = None


class MotionStatus:
    IDLE = "idle"
    MOVING = "moving"
    SUCCESS = "success"
    STOPPED = "stopped"
    BLOCKED = "blocked"
    NO_PATH = "no_path"
    UNREACHABLE = "unreachable"
    TIMEOUT = "timeout"
    ERROR = "error"
    # Manipulation (MANIP_SPEC §4.4)
    GRASP_FAILED = "grasp_failed"
    NO_GRASP = "no_grasp"
    NO_PLACEMENT = "no_placement"
    NOT_HOLDING = "not_holding"
    ALREADY_HOLDING = "already_holding"
    COLLISION = "collision"
    UNSUPPORTED = "unsupported"

    ALL = (
        IDLE, MOVING, SUCCESS, STOPPED, BLOCKED, NO_PATH, UNREACHABLE, TIMEOUT, ERROR,
        GRASP_FAILED, NO_GRASP, NO_PLACEMENT, NOT_HOLDING, ALREADY_HOLDING, COLLISION, UNSUPPORTED,
    )  # fmt: skip


@dataclass
class MotionResult:
    status: str
    final_pose: Pose2D | None = None
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)  # extra facts about the motion, e.g. {"scripted": True}


@dataclass
class EnvState:
    base_pose: Pose2D
    base_velocity: np.ndarray  # (3,) world frame: vx, vy (m/s), vyaw (rad/s)
    base_in_contact: bool
    time: float  # seconds on the env's clock
    arm_q: np.ndarray = field(default_factory=lambda: np.zeros(7))  # (7,) rad
    arm_qd: np.ndarray = field(default_factory=lambda: np.zeros(7))  # (7,) rad/s
    gripper_width: float = 0.0  # m between the fingertip pads
    gripper_closed_cmd: bool = False
    # Unexpected contact on the arm links or the gripper body. Finger contacts
    # are excluded: fingers touch the target and its support during a grasp.
    arm_in_contact: bool = False
    ee_wrench: np.ndarray | None = None  # (6,) force, torque at the wrist, if measured


@dataclass
class RGBD:
    rgb: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray  # (H, W) float32, meters
    intrinsics: np.ndarray  # (3, 3)
    cam_to_world: np.ndarray  # (4, 4), OpenCV convention (x right, y down, z forward)


@dataclass
class Grasp:
    """A grasp candidate. The TCP frame has z along the approach direction and
    x along the direction the fingers close."""

    tcp_pose_world: np.ndarray  # (4, 4) at the moment the fingers close
    pregrasp_offset: float  # m to back off along the approach axis before the final approach
    width_hint: float  # m: expected distance between the pads once closed on the object
    score: float = 1.0
    # World-frame displacement (3,) to carry the object clear after closing.
    # None: straight up by the robot's `lift_height`.
    retreat: np.ndarray | None = None


@dataclass
class Placement:
    tcp_pose_world: np.ndarray  # (4, 4) at release
    preplace_offset: float  # m above the release pose to approach from (world z)
    score: float = 1.0


@dataclass
class HeldObject:
    label: str
    grasp_width: float  # m between the pads when the grasp was verified
    picked_from: np.ndarray  # (3,) world position the object was detected at


@dataclass
class ActuationResult:
    success: bool
    scripted: bool
    message: str = ""
    fixture: str = ""  # stable name of the fixture that was actuated
    # World xy points (N, 2) the fixture newly occupies in this state, e.g. an
    # open door, so the base can plan around it. None: nothing to add.
    obstacle_xy: np.ndarray | None = None


@dataclass
class ToolResult:
    success: bool
    error: str | None = None
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
