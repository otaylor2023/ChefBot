"""Drive the base by gamepad, slowly: tidybot2's gamepad_teleop.py as it is, with lower speed limits.

    ~/miniforge3/envs/tidybot2/bin/python -m chefbot.onboard.bringup.gamepad [--vel 0.2] [--yaw-vel 0.5]

gamepad_teleop.py hard-codes max_vel (1.0 m/s, 1.0 m/s, 3.14 rad/s) at full stick; this runs the same script with
Vehicle's limits replaced (default 0.2 m/s, 0.5 rad/s, gentler acceleration). Buttons as in that script (Logitech
F710, XInput): Start begins control, Back stops it, hold a bumper to drive (left: base frame, right: global),
left stick translates, right stick turns. Releasing the bumper stops commanding. Nothing else here moves anything.

Base motor power and the e-stop as in the lab's instructions. The base driver (redis_driver.py) must not be running:
both would hold the CAN bus.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tidybot2", type=Path, default=Path.home() / "tidybot2")
    ap.add_argument("--vel", type=float, default=0.2, help="m/s at full stick, each of x and y")
    ap.add_argument("--yaw-vel", type=float, default=0.5, help="rad/s at full stick")
    ap.add_argument("--accel", type=float, default=0.15, help="m/s^2")
    ap.add_argument("--yaw-accel", type=float, default=0.4, help="rad/s^2")
    args = ap.parse_args()
    if not (0 < args.vel <= 0.5 and 0 < args.yaw_vel <= 1.0):
        sys.exit("refusing: --vel must be in (0, 0.5] m/s and --yaw-vel in (0, 1.0] rad/s")

    sys.path.insert(0, str(args.tidybot2))
    teleop = importlib.import_module("gamepad_teleop")
    original = teleop.Vehicle

    def slow_vehicle(**kwargs):
        kwargs.update(max_vel=(args.vel, args.vel, args.yaw_vel), max_accel=(args.accel, args.accel, args.yaw_accel))
        print(f"Vehicle limits: {kwargs['max_vel']} (m/s, m/s, rad/s), accel {kwargs['max_accel']}", flush=True)
        return original(**kwargs)

    teleop.Vehicle = slow_vehicle  # gamepad_teleop builds its Vehicle by this name, on Start
    print(f"gamepad_teleop from {teleop.__file__}, at most {args.vel} m/s and {args.yaw_vel} rad/s", flush=True)
    pad = teleop.GamepadTeleop()
    try:
        pad.run()
    finally:
        if pad.vehicle:
            pad.vehicle.stop_control()


if __name__ == "__main__":
    main()
