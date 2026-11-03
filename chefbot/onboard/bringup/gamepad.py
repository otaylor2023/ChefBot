"""Drive the base by gamepad, slowly: tidybot2's gamepad_teleop.py as it is, with lower speed limits.

    ~/miniforge3/envs/tidybot2/bin/python -m chefbot.onboard.bringup.gamepad [--vel 0.2] [--yaw-vel 0.5]

gamepad_teleop.py hard-codes max_vel (1.0 m/s, 1.0 m/s, 3.14 rad/s) at full stick; this runs the same script with
Vehicle's limits replaced (default 0.2 m/s, 0.5 rad/s, gentler acceleration). Buttons as in that script (Logitech
F710, XInput): Start begins control, Back stops it, hold a bumper to drive (left: base frame, right: global),
left stick translates, right stick turns. Releasing the bumper stops commanding. Nothing else here moves anything.

--one-axis (on by default; --free turns it off): one direction at a time, for calibration drives. The left stick
snaps to its larger axis (pushed mostly up = exactly forward, the sideways part dropped), and the right stick turns
only while the left stick is centred. With the left bumper (base frame) the command is then purely forward, back,
left, right or a turn in place, however the stick is held.

Base motor power and the e-stop as in the lab's instructions. The base driver (redis_driver.py) must not be running:
both would hold the CAN bus.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

# gamepad_teleop.py's axes: 1 and 0 the left stick (forward, sideways), 3 the right stick (turn).
FORWARD, SIDEWAYS, TURN = 1, 0, 3
CENTRED = 0.05  # the script's own deadzone


def one_axis(forward: float, sideways: float, turn: float) -> tuple[float, float, float]:
    """The stick, reduced to one motion: the larger left-stick axis alone, else the turn alone."""
    if max(abs(forward), abs(sideways)) > CENTRED:
        return (forward, 0.0, 0.0) if abs(forward) >= abs(sideways) else (0.0, sideways, 0.0)
    return 0.0, 0.0, turn


class OneAxisJoystick:
    """The gamepad, read through one_axis; buttons and everything else as they are."""

    def __init__(self, joy):
        self._joy = joy

    def get_axis(self, i: int) -> float:
        f, s, t = one_axis(self._joy.get_axis(FORWARD), self._joy.get_axis(SIDEWAYS), self._joy.get_axis(TURN))
        return {FORWARD: f, SIDEWAYS: s, TURN: t}.get(i, self._joy.get_axis(i))

    def __getattr__(self, name):
        return getattr(self._joy, name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tidybot2", type=Path, default=Path.home() / "tidybot2")
    ap.add_argument("--vel", type=float, default=0.2, help="m/s at full stick, each of x and y")
    ap.add_argument("--yaw-vel", type=float, default=0.5, help="rad/s at full stick")
    ap.add_argument("--accel", type=float, default=0.15, help="m/s^2")
    ap.add_argument("--yaw-accel", type=float, default=0.4, help="rad/s^2")
    ap.add_argument("--free", dest="one_axis", action="store_false", help="all stick axes at once (default: one at a time)")
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
    if args.one_axis:
        pad.joy = OneAxisJoystick(pad.joy)
        print("one axis at a time: forward/back, left/right, or turn (--free for all at once)", flush=True)
    try:
        pad.run()
    finally:
        if pad.vehicle:
            pad.vehicle.stop_control()


if __name__ == "__main__":
    main()
