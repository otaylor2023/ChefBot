"""Drive the base by gamepad, slowly: tidybot2's gamepad_teleop.py as it is, with lower speed limits.

    ~/miniforge3/envs/tidybot2/bin/python -m chefbot.onboard.bringup.gamepad [--vel 0.2] [--yaw-vel 0.5]
    ~/miniforge3/envs/tidybot2/bin/python -m chefbot.onboard.bringup.gamepad --echo   # no base: prints the stick only

gamepad_teleop.py hard-codes max_vel (1.0 m/s, 1.0 m/s, 3.14 rad/s) at full stick; this runs the same script with
Vehicle's limits replaced (default 0.2 m/s, 0.5 rad/s, gentler acceleration). Buttons as in that script (Logitech
F710, XInput): Start begins control, Back stops it, hold a bumper to drive (left: base frame, right: global),
left stick translates, right stick turns. Releasing the bumper stops commanding. Nothing else here moves anything.

--one-axis (on by default; --free turns it off): one direction at a time, for calibration drives. The left stick
snaps to its larger axis (pushed mostly up = exactly forward, the sideways part dropped), and the right stick turns
only while the left stick is centred. With the left bumper (base frame) the command is then purely forward, back,
left, right or a turn in place, however the stick is held.

--echo opens only the gamepad (no Vehicle, no CAN): it prints the raw stick and the command one-axis mode would send,
in m/s and rad/s, ten times a second. Ctrl-C stops it.

Base motor power and the e-stop as in the lab's instructions. The base driver (redis_driver.py) must not be running:
both would hold the CAN bus.
"""

from __future__ import annotations

import argparse
import importlib
import math
import sys
import time
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


def command(f: float, s: float, t: float, vel: float, yaw_vel: float, snap: bool) -> tuple[float, float, float]:
    """What gamepad_teleop would send for these raw axes (its signs, deadzone and scaling), in m/s, m/s, rad/s."""
    if snap:
        f, s, t = one_axis(f, s, t)

    def dz(v):
        return 0.0 if abs(v) <= CENTRED else math.copysign((abs(v) - CENTRED) / (1 - CENTRED), v)

    return -dz(f) * vel, -dz(s) * vel, -dz(t) * yaw_vel  # x forward, y left, yaw left (CCW) positive


def echo(vel: float, yaw_vel: float, snap: bool) -> None:
    import pygame

    pygame.init()
    joy = pygame.joystick.Joystick(0)
    print(f"gamepad: {joy.get_name()} ({joy.get_numaxes()} axes); no base. Ctrl-C stops.", flush=True)
    names = {True: "base frame (left bumper)", False: "global (right bumper)"}
    try:
        while True:
            pygame.event.pump()
            f, s, t = joy.get_axis(FORWARD), joy.get_axis(SIDEWAYS), joy.get_axis(TURN)
            vx, vy, w = command(f, s, t, vel, yaw_vel, snap)
            held = joy.get_button(4) or joy.get_button(5)
            frame = names[bool(joy.get_button(4))] if held else "no bumper: base would not move"
            moves = [n for n, v in (("forward" if vx > 0 else "back", vx), ("left" if vy > 0 else "right", vy),
                                    ("turn left" if w > 0 else "turn right", w)) if v]  # fmt: skip
            print(f"raw stick fwd {-f:+.2f} side {-s:+.2f} turn {-t:+.2f}  ->  vx {vx:+.3f} vy {vy:+.3f} m/s  "
                  f"yaw {w:+.3f} rad/s  [{', '.join(moves) or 'still'}]  {frame}", flush=True)  # fmt: skip
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tidybot2", type=Path, default=Path.home() / "tidybot2")
    ap.add_argument("--vel", type=float, default=0.2, help="m/s at full stick, each of x and y")
    ap.add_argument("--yaw-vel", type=float, default=0.5, help="rad/s at full stick")
    ap.add_argument("--accel", type=float, default=0.15, help="m/s^2")
    ap.add_argument("--yaw-accel", type=float, default=0.4, help="rad/s^2")
    ap.add_argument("--free", dest="one_axis", action="store_false", help="all stick axes at once (default: one at a time)")
    ap.add_argument("--echo", action="store_true", help="print the stick and the command only; no base")
    args = ap.parse_args()
    if not (0 < args.vel <= 0.5 and 0 < args.yaw_vel <= 1.0):
        sys.exit("refusing: --vel must be in (0, 0.5] m/s and --yaw-vel in (0, 1.0] rad/s")
    if args.echo:
        echo(args.vel, args.yaw_vel, args.one_axis)
        return

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
