#!/usr/bin/env bash
# Drive the base by gamepad: ChefBot's copy of ZitiBot's gamepad_teleop.py (onboard/drivers/base, unchanged) through
# onboard/drivers/base/teleop.py, in the robot's zitibot env, at a speed profile (default `teleop`: full stick is its
# speed), which may not be over the robot's base cap (embodiments/tidybot_fr3/robot.yaml control.limits).
#
# The gamepad (Logitech F710, XInput): Start begins control, Back stops it; hold the left bumper to drive in the
# base's frame, the right one in the world's; the left stick moves, the right stick turns; let go of the bumper and
# it stops commanding. Ctrl-C stops it. Base motor power and the e-stop as in the lab's instructions.
#
# The base driver must not be running (both would hold the CAN bus): this refuses if it is (launch.sh or base.sh
# started it; or start launch.sh with --no-base).
#
# Usage: chefbot/onboard/bringup/gamepad.sh [--profile NAME]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DRIVER_DIR="$(cd "$HERE/../drivers/base" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
BASE_PYTHON="${BASE_PYTHON:-$HOME/miniforge3/envs/zitibot/bin/python}"  # phoenix6, ruckig, pygame
PROFILE=teleop
while [[ $# -gt 0 ]]; do
	case "$1" in
		--profile) PROFILE="$2"; shift 2 ;;
		-h | --help) sed -n '2,14p' "$0"; exit 0 ;;
		*) echo "unknown option: $1" >&2; exit 1 ;;
	esac
done

[[ -x "$BASE_PYTHON" ]] || { echo "no python at $BASE_PYTHON (set BASE_PYTHON)" >&2; exit 1; }
if pgrep -f "[d]rivers/base/driver.py" >/dev/null || pgrep -f "[t]idybot_base/redis_driver.py" >/dev/null; then
	echo "a base driver is running (it holds the CAN bus): stop it first (launch.sh --no-base, or Ctrl-C base.sh)" >&2
	exit 1
fi

cd "$DRIVER_DIR"  # gamepad_teleop imports base_controller, which imports constants and utils, by name
PYTHONPATH="$REPO_ROOT" exec "$BASE_PYTHON" "$DRIVER_DIR/teleop.py" --profile "$PROFILE"
