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
    # TCP relative to the flange
    tcp_xyz: tuple[float, float, float] = (0.0, 0.0, 0.16)
    tcp_yaw: float = 0.0
    joint_limits: tuple | None = None  # 7 x (low, high); None: the FR3's own
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
    reach_clearance: float = 0.20
    top_down_max_height: float = 1.2
    standoff_min: float = 0.45
    standoff_max: float = 0.8
    standoff_rings: int = 4
    standoff_angles: int = 24
    # Arm
    arm_stow: tuple = (0.0, -1.16, 0.0, -2.32, 0.0, 1.16, 0.785)
    arm_carry: tuple = (0.0, -1.16, 0.0, -2.32, 0.0, 1.16, 0.785)
    arm_ready: tuple = (0.0, -0.75, 0.0, -2.16, 0.0, 1.41, 0.785)
    max_joint_vel: float = 1.2
    max_joint_acc: float = 3.0
    cart_speed: float = 0.12
    cart_acc: float = 0.5
    cart_step: float = 0.01
    max_joint_jump: float = 0.25
    joint_tolerance: float = 0.02
    tracking_tolerance: float = 0.25
    arm_settle_s: float = 1.5
    # Gripper
    gripper_open_width: float = 0.085
    min_holding_width: float = 0.012
    gripper_settle_s: float = 1.2
    grasp_settle_s: float = 0.3
    # Skills
    lift_height: float = 0.12
    retreat_height: float = 0.10
    skill_timeout_s: float = 60.0
    # Arm motion checked and planned against a depth camera's view (robot/kinematics/collision.py)
    planning_cameras: tuple = ()  # none: no world model, arm moves go straight unchecked
    planning_voxel: float = 0.02
    self_filter_margin: float = 0.07
    base_box: tuple[float, float, float] = (0.80, 0.60, 1.0)
    planning_timeout_s: float = 5.0
    shorter_offsets: tuple = (0.06, 0.04)
    # IK
    ik_pose_pos_tolerance: float = 0.002
    ik_pose_rot_tolerance: float = math.radians(1.0)
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
        tcp, arm, gripper, skills = raw["tcp"], raw["arm"], raw["gripper"], raw["skills"]
        joint_limits = raw.get("joint_limits")
        planning = raw.get("planning") or {}
        return cls(
            footprint_radius=raw["footprint_radius"],
            safety_margin=raw["safety_margin"],
            arm_mount_xyz=tuple(mount["xyz"]),
            arm_mount_yaw=math.radians(mount["yaw_deg"]),
            tcp_xyz=tuple(tcp["xyz"]),
            tcp_yaw=math.radians(tcp["yaw_deg"]),
            joint_limits=tuple(tuple(pair) for pair in joint_limits) if joint_limits else None,
            arm_stow=tuple(arm["stow"]),
            arm_carry=tuple(arm["carry"]),
            arm_ready=tuple(arm["ready"]),
            max_joint_vel=arm["max_joint_vel"],
            max_joint_acc=arm["max_joint_acc"],
            cart_speed=arm["cart_speed"],
            cart_acc=arm["cart_acc"],
            cart_step=arm["cart_step"],
            max_joint_jump=arm["max_joint_jump"],
            joint_tolerance=arm["joint_tolerance"],
            tracking_tolerance=arm["tracking_tolerance"],
            arm_settle_s=arm["settle_s"],
            gripper_open_width=gripper["open_width"],
            min_holding_width=gripper["min_holding_width"],
            gripper_settle_s=gripper["settle_s"],
            grasp_settle_s=gripper["grasp_settle_s"],
            lift_height=skills["lift_height"],
            retreat_height=skills["retreat_height"],
            skill_timeout_s=skills["motion_timeout_s"],
            planning_cameras=tuple(planning.get("cameras") or ()),
            planning_voxel=planning.get("voxel", cls.planning_voxel),
            self_filter_margin=planning.get("self_filter_margin", cls.self_filter_margin),
            base_box=tuple(planning.get("base_box", cls.base_box)),
            planning_timeout_s=planning.get("timeout_s", cls.planning_timeout_s),
            shorter_offsets=tuple(planning.get("shorter_offsets", cls.shorter_offsets)),
            ik_pose_pos_tolerance=ik["pose_pos_tolerance"],
            ik_pose_rot_tolerance=math.radians(ik["pose_rot_tolerance_deg"]),
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
            reach_clearance=raw["reach_clearance"],
            top_down_max_height=raw["top_down_max_height"],
            standoff_min=standoff["min"],
            standoff_max=standoff["max"],
            standoff_rings=standoff["rings"],
            standoff_angles=standoff["angles"],
            ik_pos_tolerance=ik["pos_tolerance"],
            ik_max_iters=ik["max_iters"],
            ik_dt=ik["dt"],
            ik_damping=ik["damping"],
        )
