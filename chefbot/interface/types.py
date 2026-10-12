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
    position_world: np.ndarray  # (3,)
    confidence: float = 1.0


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

    ALL = (IDLE, MOVING, SUCCESS, STOPPED, BLOCKED, NO_PATH, UNREACHABLE, TIMEOUT, ERROR)


@dataclass
class MotionResult:
    status: str
    final_pose: Pose2D | None = None
    message: str = ""


@dataclass
class EnvState:
    base_pose: Pose2D
    base_velocity: np.ndarray  # (3,) world frame: vx, vy (m/s), vyaw (rad/s)
    base_in_contact: bool
    time: float  # seconds on the env's clock


@dataclass
class RGBD:
    rgb: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray  # (H, W) float32, meters
    intrinsics: np.ndarray  # (3, 3)
    cam_to_world: np.ndarray  # (4, 4), OpenCV convention (x right, y down, z forward)


@dataclass
class ToolResult:
    success: bool
    error: str | None = None
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
