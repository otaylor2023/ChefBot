"""Drive the base by gamepad at a speed profile: gamepad_teleop.py (copied unchanged) with its Vehicle held to the profile
(default `teleop`), which may not be over the robot's base cap (interface/limits.py: the embodiment's control.limits).

    cd chefbot/onboard/drivers/base && ~/miniforge3/envs/zitibot/bin/python teleop.py [--profile teleop] [--one-axis] [--echo]

Run from this folder (gamepad_teleop imports base_controller, and it constants and utils, by name), with the repo on
PYTHONPATH (for chefbot.interface.limits). The gamepad: Start begins, Back stops, hold a bumper to drive (left: the
base's frame, right: the world's), the left stick moves, the right stick turns; full stick is the profile's speed.

--one-axis: one motion at a time, for calibration drives (onboard/calibration/base.py): the left stick snaps to its
larger axis (pushed mostly up is exactly forward, the sideways part dropped), and the right stick turns only while the
left one is centred. With the left bumper the command is then purely forward, back, left, right or a turn in place.
--echo: the gamepad only (no Vehicle, no CAN: the base driver may run): prints the stick and the command it would
send, in m/s and rad/s, ten times a second, until Ctrl-C.
"""

import argparse
import math
import sys
import time

from chefbot.interface import limits

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="teleop", help="the speed profile (embodiments/speed_profiles.yaml)")
    ap.add_argument("--embodiment", default="tidybot_fr3")
    ap.add_argument("--one-axis", action="store_true", help="one motion at a time (forward/back, left/right, or a turn)")
    ap.add_argument("--echo", action="store_true", help="print the stick and the command only; no base")
    a = ap.parse_args()
    try:
        vel, accel = limits.base_profile(a.profile, a.embodiment)
    except ValueError as e:
        sys.exit(f"[gamepad] {e}")
    if a.echo:
        echo(vel[0], vel[2], a.one_axis)
        return
    import gamepad_teleop as teleop

    class Held(teleop.Vehicle):  # the script builds Vehicle(max_vel=(1.0, 1.0, 3.14), ...) on Start: the profile's instead
        def __init__(self, *args, **kwargs):
            kwargs.update(max_vel=vel)
            if accel is not None:
                kwargs.update(max_accel=accel)
            super().__init__(*args, **kwargs)

    teleop.Vehicle = Held
    print(f"[gamepad] profile {a.profile}: at most {vel[0]} m/s, {vel[2]} rad/s (accel {accel}); Start begins, hold a bumper to drive", flush=True)
    t = teleop.GamepadTeleop()
    if a.one_axis:
        t.joy = OneAxisJoystick(t.joy)
        print("[gamepad] one axis at a time: forward/back, left/right, or turn", flush=True)
    try:
        t.run()
    finally:
        if t.vehicle:
            t.vehicle.stop_control()


if __name__ == "__main__":
    main()
