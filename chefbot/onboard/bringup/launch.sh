#!/usr/bin/env bash
# Start the robot's drivers (the existing ones, in place), then the onboard server.
#
#   1) OpenSai with chefbot_fr3.xml, next to this script (arm joint + Cartesian controllers, Redis, web UI)
#   2) the Franka gripper driver (OpenSai's drivers/FrankaPanda/redis_driver/launch_gripper.sh)
#   3) the hb1 base driver (ZitiBot's controllers/tidybot_base/redis_driver.py)
#   4) the mocap publisher (mocap.sh, next to this script; skipped with a warning off SRC wifi)
#   5) python -m chefbot.onboard.server, in the foreground
# Ctrl-C stops them all. Not started here, as with ZitiBot: the Franka arm driver (real-time torque
# loop); see README.md.
#
# Usage: chefbot/onboard/bringup/launch.sh [--no-gripper] [--no-base] [--no-mocap] [--no-server]
# Paths and limits come from the environment, with these defaults:
set -euo pipefail

OPENSAI_DIR="${OPENSAI_DIR:-$HOME/OpenSai}"
OPENSAI_CONFIG="${OPENSAI_CONFIG:-$(cd "$(dirname "$0")" && pwd)/chefbot_fr3.xml}"  # ChefBot's own (not ZitiBot's zitibot_panda.xml)
BASE_DRIVER_DIR="${BASE_DRIVER_DIR:-$OPENSAI_DIR/ZitiBot/controllers/tidybot_base}"
BASE_PYTHON="${BASE_PYTHON:-$HOME/miniforge3/envs/zitibot/bin/python}"  # has phoenix6, ruckig
CHEFBOT_PYTHON="${CHEFBOT_PYTHON:-$HOME/miniforge3/envs/chefbot/bin/python}"
MAX_VEL_XY="${MAX_VEL_XY:-0.125}"      # keep equal to onboard/config.yaml base.driver_max_vel_xy
MAX_VEL_YAW="${MAX_VEL_YAW:-0.4}"      # ... and base.driver_max_vel_yaw
MAX_ACCEL_XY="${MAX_ACCEL_XY:-0.1}"
MAX_ACCEL_YAW="${MAX_ACCEL_YAW:-0.5}"
READY_TIMEOUT_S=90

GRIPPER=true BASE=true MOCAP=true SERVER=true
for arg in "$@"; do
	case "$arg" in
		--no-gripper) GRIPPER=false ;;
		--no-base) BASE=false ;;
		--no-mocap) MOCAP=false ;;
		--no-server) SERVER=false ;;
		-h | --help) sed -n '2,17p' "$0"; exit 0 ;;
		*) echo "unknown option: $arg" >&2; exit 1 ;;
	esac
done

GRIPPER_DIR="$OPENSAI_DIR/drivers/FrankaPanda/redis_driver"
for f in "$OPENSAI_DIR/bin/OpenSai_main" "$OPENSAI_DIR/scripts/launch.sh"; do
	[[ -e "$f" ]] || { echo "missing $f (OPENSAI_DIR=$OPENSAI_DIR)" >&2; exit 1; }
done
$GRIPPER && { [[ -f "$GRIPPER_DIR/launch_gripper.sh" ]] || { echo "missing $GRIPPER_DIR/launch_gripper.sh" >&2; exit 1; }; }
$BASE && { [[ -f "$BASE_DRIVER_DIR/redis_driver.py" && -x "$BASE_PYTHON" ]] || { echo "missing the base driver or $BASE_PYTHON" >&2; exit 1; }; }

PIDS=()
LOGS="$(mktemp -d /tmp/chefbot-bringup.XXXXXX)"
cleanup() {
	trap - INT TERM EXIT
	echo "stopping drivers..." >&2
	for pid in "${PIDS[@]}"; do
		pkill -TERM -P "$pid" 2>/dev/null || true
		kill -TERM "$pid" 2>/dev/null || true
	done
	tmux kill-session -t interfaces_server 2>/dev/null || true
	pkill -TERM -f "OpenSai_main.*$(basename "$OPENSAI_CONFIG")" 2>/dev/null || true
	wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_for_key() {  # key, what must be in its value, timeout
	local deadline=$((SECONDS + $3))
	until [[ "$(redis-cli GET "$1" 2>/dev/null)" == *"$2"* ]]; do
		(( SECONDS < deadline )) || { echo "timed out waiting for $1; logs in $LOGS" >&2; exit 1; }
		sleep 0.5
	done
}

echo "1) OpenSai: $OPENSAI_CONFIG"
(cd "$OPENSAI_DIR" && exec bash scripts/launch.sh "$OPENSAI_CONFIG") >"$LOGS/opensai.log" 2>&1 &
PIDS+=($!)
wait_for_key "::sai-interfaces-webui::config_file_name" "$(basename "$OPENSAI_CONFIG")" "$READY_TIMEOUT_S"

if $GRIPPER; then
	echo "2) Franka gripper driver"
	(cd "$GRIPPER_DIR" && exec bash launch_gripper.sh) >"$LOGS/gripper.log" 2>&1 &
	PIDS+=($!)
else
	echo "2) gripper driver skipped"
fi

if $BASE; then
	echo "3) hb1 base driver (max vel $MAX_VEL_XY m/s, $MAX_VEL_YAW rad/s)"
	pkill -TERM -f "$BASE_DRIVER_DIR/redis_driver.py" 2>/dev/null || true  # a stale one holds the CAN bus
	(cd "$BASE_DRIVER_DIR" && exec "$BASE_PYTHON" redis_driver.py --max-vel-xy "$MAX_VEL_XY" --max-vel-yaw "$MAX_VEL_YAW" \
		--max-accel-xy "$MAX_ACCEL_XY" --max-accel-yaw "$MAX_ACCEL_YAW") >"$LOGS/base.log" 2>&1 &
	PIDS+=($!)
	wait_for_key "hb1::current_pose" "[" 15
else
	echo "3) base driver skipped"
fi

if $MOCAP; then
	echo "4) mocap publisher"
	redis-cli DEL chefbot-tidy::tracking_valid >/dev/null  # a stale value from an earlier run would look live
	bash "$(dirname "$0")/mocap.sh" --quiet 2>"$LOGS/mocap.log" &
	PIDS+=($!)
	deadline=$((SECONDS + 10))
	until [[ -n "$(redis-cli GET chefbot-tidy::tracking_valid 2>/dev/null)" ]]; do
		if ! kill -0 "${PIDS[-1]}" 2>/dev/null || (( SECONDS >= deadline )); then
			echo "   no mocap (not localized; see $LOGS/mocap.log)" >&2
			break
		fi
		sleep 0.5
	done
else
	echo "4) mocap skipped"
fi

echo "drivers up; logs in $LOGS"
if $SERVER; then
	echo "5) onboard server"
	"$CHEFBOT_PYTHON" -m chefbot.onboard.server
else
	echo "5) server skipped; Ctrl-C stops the drivers"
	wait
fi
