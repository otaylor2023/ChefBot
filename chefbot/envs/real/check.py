"""Read-only check of the real robot through the onboard server. Nothing moves.

    ssh -L 8765:localhost:8765 tidybot01        # with the onboard server running there
    python -m chefbot.envs.real.check [--frames]

Prints what the robot reports, and checks the arm's frames: our FR3 kinematics give the flange
from the measured joints, the arm's controller reports the pose of the point it moves; the offset
between them must be robot_real.yaml's arm `driver.control_point`, or every pose target misses by
the difference. --frames saves one frame per camera under runs/real_check/.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from chefbot.envs.real.env import CONFIG, RealEnv
from chefbot.robot.arm.ik import ArmIK, rotation_angle
from chefbot.robot.config import REAL_PATH, RobotConfig

REPO = Path(__file__).resolve().parents[3]
POS_TOLERANCE, ROT_TOLERANCE = 0.005, math.radians(2.0)


def control_offset(cfg: RobotConfig) -> np.ndarray:
    """The arm controller's point in the flange frame, from the config."""
    (x, y, z), yaw = cfg.driver_control_point
    T = np.eye(4)
    c, s = math.cos(yaw), math.sin(yaw)
    T[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    T[:3, 3] = [x, y, z]
    return T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--robot-config", default=str(REAL_PATH))
    parser.add_argument("--url")
    parser.add_argument("--frames", action="store_true", help="save one frame from each camera")
    args = parser.parse_args()
    cfg = RobotConfig.load(args.robot_config)
    flange = ArmIK(tcp_xyz=(0.0, 0.0, 0.0), joint_limits=cfg.joint_limits, arm=cfg.arm)  # its "tcp" is the flange
    expected = control_offset(cfg)
    env = RealEnv.from_config(expected, args.config, args.url, flange_in_base=None)
    try:
        s = env.raw_state()
        print(f"Robot: {cfg.arm_name} + {cfg.gripper_name}; server cameras: {env.server_cameras or 'none'}")
        print(f"  controller: {s['controller'] or '-'}    gripper: width {s['gripper_width']}, ready {s['gripper_ready']}")
        if s["base_pose"] is not None:
            x, y, yaw = s["base_pose"]
            where = "world (mocap)" if s["localized"] else "odometry: no mocap yet"
            print(f"  base: x {x:.3f}, y {y:.3f}, yaw {math.degrees(yaw):.1f} deg  [{where}]")
        else:
            print("  base: no pose (is the hb1 driver up?)")
        limit = env.server_limits.get("base_max_vel_xy")
        if limit is not None and cfg.max_lin_vel > limit + 1e-9:
            print(f"  note: robot config max_lin_vel {cfg.max_lin_vel} > the base driver's {limit}; the driver's wins")

        print("\nArm frames:")
        if s["arm_q"] is None:
            print("  no joints (is the Franka arm driver up?)")
        elif s["driver_position"] is None:
            print(f"  q = {np.round(s['arm_q'], 3)}; the arm controller reports no pose (is OpenSai up?)")
        else:
            T_flange = flange.fk(np.asarray(s["arm_q"], dtype=float))
            T_driver = np.eye(4)
            T_driver[:3, :3], T_driver[:3, 3] = s["driver_orientation"], s["driver_position"]
            measured = np.linalg.inv(T_flange) @ T_driver
            dp = float(np.linalg.norm(measured[:3, 3] - expected[:3, 3]))
            dr = rotation_angle(expected[:3, :3].T @ measured[:3, :3])
            yaw = math.degrees(math.atan2(measured[1, 0], measured[0, 0]))
            print(f"  controller point in the flange frame: xyz {np.round(measured[:3, 3], 4)}, yaw {yaw:.1f} deg, "
                  f"tilt {math.degrees(rotation_angle(measured[:3, :3])):.1f} deg total")  # fmt: skip
            print(f"  config driver.control_point:         xyz {np.round(expected[:3, 3], 4)}")
            if dp <= POS_TOLERANCE and dr <= ROT_TOLERANCE:
                print(f"  OK: they agree ({dp * 1000:.1f} mm, {math.degrees(dr):.1f} deg)")
            else:
                print(f"  MISMATCH: {dp * 1000:.1f} mm, {math.degrees(dr):.1f} deg. Set config/arms/{cfg.arm_name}.yaml "
                      "driver.control_point to the measured offset before sending pose targets.")  # fmt: skip

        if args.frames:
            out = REPO / "runs" / "real_check" / time.strftime("%Y-%m-%d_%H%M%S")
            out.mkdir(parents=True, exist_ok=True)
            for cam in env.camera_mounts:
                try:
                    frame = env.get_camera_rgbd(cam)
                    np.savez_compressed(out / f"{cam}.npz", rgb=frame.rgb, depth=frame.depth, intrinsics=frame.intrinsics,
                                        cam_to_world=frame.cam_to_world)  # fmt: skip
                    print(f"  {cam}: saved {out / (cam + '.npz')}")
                except Exception as exc:
                    print(f"  {cam}: {type(exc).__name__}: {exc}")
            if not env.camera_mounts:
                print("\nCameras: no mounts in envs/real config")
    finally:
        env.close()


if __name__ == "__main__":
    main()
