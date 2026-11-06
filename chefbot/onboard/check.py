"""Read-only check, on the robot computer: which drivers are up, and what they report. Nothing moves.

    python -m chefbot.onboard.check [--config chefbot/onboard/config.yaml] [--frames]

--frames also grabs one frame from each configured camera.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from chefbot.onboard import keys as K
from chefbot.onboard.stack import CONFIG, build_cameras, build_robot, load_config


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, np.ndarray):
        return np.array2string(v, precision=3, suppress_small=True, max_line_width=200)
    return str(v)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--frames", action="store_true", help="grab one frame from each camera")
    args = parser.parse_args()
    config = load_config(args.config)
    robot, cameras = build_robot(config), build_cameras(config)
    r = robot.r
    s = robot.read_state()

    def line(name: str, up: bool, detail: str) -> None:
        print(f"  {'ok ' if up else '-- '} {name:<22} {detail}")

    print("Drivers (Redis):")
    line("Franka arm driver", s["arm_q"] is not None, f"q = {_fmt(s['arm_q'])}")
    config_file = r.get(K.CONFIG_FILE_NAME)
    line("OpenSai controller", bool(s["controller"]), f"active = {s['controller'] or '-'}, config = {config_file or '-'}")
    line("  its control point", s["driver_position"] is not None, f"{_fmt(s['driver_position'])} (arm base frame)")
    line("Franka gripper driver", s["gripper_width"] is not None,
         f"width = {_fmt(s['gripper_width'])}, max = {_fmt(s['gripper_max_width'])}, ready = {s['gripper_ready'] or '-'}")  # fmt: skip
    odom = r.get(K.BASE_POSE)
    line("hb1 base driver", odom is not None, f"odometry = {odom or '-'}")
    line("Mocap", s["mocap_valid"], f"position = {r.get(K.MOCAP_POSITION) or '-'}, tracking_valid = {r.get(K.MOCAP_VALID) or '-'}")
    if s["mocap_pose"] is not None:
        x, y, yaw = s["mocap_pose"]
        print(f"\nBase pose by mocap: x = {x:.3f}, y = {y:.3f}, yaw = {math.degrees(yaw):.1f} deg")

    print("\nCameras:")
    if not cameras.cameras:
        print("  none configured (onboard config `cameras:`)")
    for name in cameras.cameras:
        if not args.frames:
            print(f"  {name}: configured (--frames to grab one)")
            continue
        try:
            frame = cameras.grab(name)
            depth = frame["depth"][frame["depth"] > 0]
            print(f"  {name}: {frame['rgb'].shape[1]}x{frame['rgb'].shape[0]}, depth {depth.min():.2f}-{depth.max():.2f} m"
                  if depth.size else f"  {name}: no valid depth")  # fmt: skip
        except Exception as exc:
            print(f"  {name}: {type(exc).__name__}: {exc}")
    cameras.close()


if __name__ == "__main__":
    main()
