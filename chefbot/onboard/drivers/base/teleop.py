"""Drive the base by gamepad at a speed profile: gamepad_teleop.py (copied unchanged) with its Vehicle held to the profile
(default `teleop`), which may not be over the robot's base cap (interface/limits.py: the embodiment's control.limits).

    cd chefbot/onboard/drivers/base && ~/miniforge3/envs/zitibot/bin/python teleop.py [--profile teleop]

Run from this folder (gamepad_teleop imports base_controller, and it constants and utils, by name), with the repo on
PYTHONPATH (for chefbot.interface.limits). The gamepad: Start begins, Back stops, hold a bumper to drive (left: the
base's frame, right: the world's), the left stick moves, the right stick turns; full stick is the profile's speed.
"""

import argparse
import sys

from chefbot.interface import limits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="teleop", help="the speed profile (embodiments/speed_profiles.yaml)")
    ap.add_argument("--embodiment", default="tidybot_fr3")
    a = ap.parse_args()
    try:
        vel, accel = limits.base_profile(a.profile, a.embodiment)
    except ValueError as e:
        sys.exit(f"[gamepad] {e}")
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
    try:
        t.run()
    finally:
        if t.vehicle:
            t.vehicle.stop_control()


if __name__ == "__main__":
    main()
