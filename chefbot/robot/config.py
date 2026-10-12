"""Robot-layer configuration (robot.yaml)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "robot.yaml"


@dataclass(frozen=True)
class RobotConfig:
    # Footprint / grid inflation
    footprint_radius: float = 0.35
    safety_margin: float = 0.05
    # Arm base in the base frame (base frame: origin on the floor under the base center)
    arm_mount_xyz: tuple[float, float, float] = (0.0, 0.0, 0.5)
    arm_mount_yaw: float = 0.0
    tcp_offset: float = 0.17
    # Base controller
    control_rate: float = 20.0
    kp_pos: float = 2.0
    kp_yaw: float = 2.5
    max_lin_vel: float = 0.5
    max_ang_vel: float = 1.0
    max_lin_acc: float = 1.0
    max_ang_acc: float = 2.0
    pos_tolerance: float = 0.015
    yaw_tolerance: float = math.radians(1.5)
    lookahead: float = 0.3
    deviation_threshold: float = 0.2
    motion_timeout_s: float = 60.0
    # Goal handling
    snap_radius: float = 0.5
    standoff_min: float = 0.45
    standoff_max: float = 0.8
    standoff_rings: int = 4
    standoff_angles: int = 24
    # IK
    ik_pos_tolerance: float = 0.01
    ik_max_iters: int = 150
    ik_dt: float = 0.1
    ik_damping: float = 1e-3

    @property
    def inflation_radius(self) -> float:
        return self.footprint_radius + self.safety_margin

    @property
    def control_dt(self) -> float:
        return 1.0 / self.control_rate

    @classmethod
    def load(cls, path: str | Path | None = None) -> "RobotConfig":
        raw = yaml.safe_load(Path(path or DEFAULT_PATH).read_text())
        mount, gains, limits = raw["arm_mount"], raw["gains"], raw["limits"]
        tol, standoff, ik = raw["tolerances"], raw["standoff"], raw["ik"]
        return cls(
            footprint_radius=raw["footprint_radius"],
            safety_margin=raw["safety_margin"],
            arm_mount_xyz=tuple(mount["xyz"]),
            arm_mount_yaw=math.radians(mount["yaw_deg"]),
            tcp_offset=raw["tcp_offset"],
            control_rate=raw["control_rate"],
            kp_pos=gains["kp_pos"],
            kp_yaw=gains["kp_yaw"],
            max_lin_vel=limits["max_lin_vel"],
            max_ang_vel=limits["max_ang_vel"],
            max_lin_acc=limits["max_lin_acc"],
            max_ang_acc=limits["max_ang_acc"],
            pos_tolerance=tol["pos"],
            yaw_tolerance=math.radians(tol["yaw_deg"]),
            lookahead=raw["lookahead"],
            deviation_threshold=raw["deviation_threshold"],
            motion_timeout_s=raw["motion_timeout_s"],
            snap_radius=raw["snap_radius"],
            standoff_min=standoff["min"],
            standoff_max=standoff["max"],
            standoff_rings=standoff["rings"],
            standoff_angles=standoff["angles"],
            ik_pos_tolerance=ik["pos_tolerance"],
            ik_max_iters=ik["max_iters"],
            ik_dt=ik["dt"],
            ik_damping=ik["damping"],
        )
