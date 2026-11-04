#!/usr/bin/env bash
# Drive the base by gamepad: ChefBot's copy of ZitiBot's gamepad_teleop.py (onboard/drivers/base, unchanged), in the
# robot's zitibot env, at the script's own speeds (full stick: 1 m/s, 3.14 rad/s). --slow holds it to the base driver's
# limits instead (onboard/config.yaml base.driver_max_*; MAX_VEL_XY / MAX_VEL_YAW / MAX_ACCEL_XY / MAX_ACCEL_YAW
# override them).
#
# The gamepad (Logitech F710, XInput): Start begins control, Back stops it; hold the left bumper to drive in the
# base's frame, the right one in the world's; the left stick moves, the right stick turns; let go of the bumper and
# it stops commanding. Ctrl-C stops it. Base motor power and the e-stop as in the lab's instructions.
#
# The base driver must not be running (both would hold the CAN bus): this refuses if it is (launch.sh or base.sh
# started it; or start launch.sh with --no-base).
#
# Usage: chefbot/onboard/bringup/gamepad.sh [--slow]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DRIVER_DIR="$(cd "$HERE/../drivers/base" && pwd)"
CONFIG="$(cd "$HERE/.." && pwd)/config.yaml"
BASE_PYTHON="${BASE_PYTHON:-$HOME/miniforge3/envs/zitibot/bin/python}"  # phoenix6, ruckig, pygame
SLOW=false
for arg in "$@"; do
	case "$arg" in
		--slow) SLOW=true ;;
		-h | --help) sed -n '2,15p' "$0"; exit 0 ;;
		*) echo "unknown option: $arg" >&2; exit 1 ;;
	esac
done

[[ -x "$BASE_PYTHON" ]] || { echo "no python at $BASE_PYTHON (set BASE_PYTHON)" >&2; exit 1; }
if pgrep -f "[d]rivers/base/driver.py" >/dev/null || pgrep -f "[t]idybot_base/redis_driver.py" >/dev/null; then
	echo "a base driver is running (it holds the CAN bus): stop it first (launch.sh --no-base, or Ctrl-C base.sh)" >&2
	exit 1
fi

cd "$DRIVER_DIR"  # gamepad_teleop imports base_controller, which imports constants and utils, by name
if ! $SLOW; then
	echo "[gamepad] the teleop's own speeds (full stick 1 m/s, 3.14 rad/s; --slow for the driver's limits); Start begins, hold a bumper to drive"
	exec "$BASE_PYTHON" "$DRIVER_DIR/gamepad_teleop.py"
fi
exec "$BASE_PYTHON" -c '
import sys, yaml
b = yaml.safe_load(open(sys.argv[1]))["base"]
import os
v_xy = float(os.environ.get("MAX_VEL_XY", b["driver_max_vel_xy"]))
v_yaw = float(os.environ.get("MAX_VEL_YAW", b["driver_max_vel_yaw"]))
a_xy = float(os.environ.get("MAX_ACCEL_XY", b["driver_max_accel_xy"]))
a_yaw = float(os.environ.get("MAX_ACCEL_YAW", b["driver_max_accel_yaw"]))
if not (0 < v_xy <= 0.5 and 0 < v_yaw <= 1.0):
    sys.exit(f"refusing: {v_xy} m/s, {v_yaw} rad/s (at most 0.5 m/s, 1.0 rad/s)")
import gamepad_teleop as teleop

class Limited(teleop.Vehicle):  # the script builds Vehicle(max_vel=(1.0, 1.0, 3.14), ...) on Start: ours instead
    def __init__(self, *args, **kwargs):
        kwargs.update(max_vel=(v_xy, v_xy, v_yaw), max_accel=(a_xy, a_xy, a_yaw))
        super().__init__(*args, **kwargs)

teleop.Vehicle = Limited
print(f"[gamepad] at most {v_xy} m/s, {v_yaw} rad/s (accel {a_xy}, {a_yaw}); Start begins, hold a bumper to drive", flush=True)
t = teleop.GamepadTeleop()
try:
    t.run()
finally:
    if t.vehicle:
        t.vehicle.stop_control()
' "$CONFIG"
