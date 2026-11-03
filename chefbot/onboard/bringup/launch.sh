#!/usr/bin/env bash
# Start the robot's drivers (the existing ones, in place), then the onboard server.
#
#   0) with --arm-driver: the Franka arm driver (real-time torque loop: OpenSai's safe_launch_driver.sh, which
#      clears stale commands and seeds a hold at the arm's joints; asks for sudo once, for the CPU governor)
#   1) OpenSai with chefbot_fr3.xml, next to this script (arm joint + Cartesian controllers, Redis, web UI)
#   2) the Franka gripper driver (OpenSai's drivers/FrankaPanda/redis_driver/launch_gripper.sh)
#   3) the hb1 base driver (ZitiBot's controllers/tidybot_base/redis_driver.py)
#   4) the mocap publisher (mocap.sh, next to this script; skipped with a warning off SRC wifi)
#   5) python -m chefbot.onboard.server, in the foreground
# Ctrl-C stops them all (the arm driver first: the arm's brakes hold it). Desk first: joints unlocked, FCI on.
# arm.sh, next to this script, is just 0) and 1).
#
# Usage: chefbot/onboard/bringup/launch.sh [--arm-driver] [--no-gripper] [--no-base] [--no-mocap] [--no-server]
# Paths and limits come from the environment, with these defaults:
set -euo pipefail

OPENSAI_DIR="${OPENSAI_DIR:-$HOME/OpenSai}"
OPENSAI_CONFIG="${OPENSAI_CONFIG:-$(cd "$(dirname "$0")" && pwd)/chefbot_fr3.xml}"  # ChefBot's own (not ZitiBot's zitibot_panda.xml)
BASE_DRIVER_DIR="${BASE_DRIVER_DIR:-$OPENSAI_DIR/ZitiBot/controllers/tidybot_base}"
BASE_PYTHON="${BASE_PYTHON:-$HOME/miniforge3/envs/zitibot/bin/python}"  # has phoenix6, ruckig
CHEFBOT_PYTHON="${CHEFBOT_PYTHON:-$HOME/chefbot_calib/env/bin/python}"  # numpy, redis, pyrealsense2, websockets, msgpack(-numpy)
MAX_VEL_XY="${MAX_VEL_XY:-0.125}"      # keep equal to onboard/config.yaml base.driver_max_vel_xy
MAX_VEL_YAW="${MAX_VEL_YAW:-0.4}"      # ... and base.driver_max_vel_yaw
MAX_ACCEL_XY="${MAX_ACCEL_XY:-0.1}"
MAX_ACCEL_YAW="${MAX_ACCEL_YAW:-0.5}"
READY_TIMEOUT_S=90

ARM_DRIVER=false GRIPPER=true BASE=true MOCAP=true SERVER=true
for arg in "$@"; do
	case "$arg" in
		--arm-driver) ARM_DRIVER=true ;;
		--no-gripper) GRIPPER=false ;;
		--no-base) BASE=false ;;
		--no-mocap) MOCAP=false ;;
		--no-server) SERVER=false ;;
		-h | --help) sed -n '2,19p' "$0"; exit 0 ;;
		*) echo "unknown option: $arg" >&2; exit 1 ;;
	esac
done

GRIPPER_DIR="$OPENSAI_DIR/drivers/FrankaPanda/redis_driver"  # the arm driver's too
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
	if $ARM_DRIVER; then  # first: once its torque loop stops, the arm's brakes hold it
		# SIGTERM, not SIGINT: started in the background by safe_launch_driver.sh, the driver ignores SIGINT
		# (2026-11-03: Ctrl-C left it running, the arm floating on zero torque)
		pkill -TERM -f "[s]ai_franka_robot_redis_driver" 2>/dev/null || true
		for _ in 1 2 3 4 5 6; do
			pgrep -f "[s]ai_franka_robot_redis_driver" >/dev/null || break
			sleep 0.5
		done
		if pgrep -f "[s]ai_franka_robot_redis_driver" >/dev/null; then
			pkill -KILL -f "[s]ai_franka_robot_redis_driver" 2>/dev/null || true
			echo "the arm driver didn't stop on SIGTERM: killed" >&2
		fi
		pgrep -f "[s]ai_franka_robot_redis_driver" >/dev/null && echo "THE ARM DRIVER IS STILL RUNNING: stop it (E-stop, or Desk)" >&2 || echo "arm driver stopped" >&2
	fi
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

if $ARM_DRIVER; then
	if pgrep -f "[s]ai_franka_robot_redis_driver" >/dev/null; then
		echo "an arm driver is already running: stop it first (or leave out --arm-driver)" >&2
		exit 1
	fi
	echo "0) Franka arm driver"
	sudo -v  # its launcher sets the CPU governor with sudo: ask here, not in the background
	pgrep -x redis-server >/dev/null || redis-server --daemonize yes >/dev/null
	redis-cli DEL opensai::sensors::FrankaRobot::joint_positions >/dev/null  # a stale value would look live
	(cd "$GRIPPER_DIR" && exec sh safe_launch_driver.sh) >"$LOGS/arm_driver.log" 2>&1 &
	PIDS+=($!)
	deadline=$((SECONDS + 30))
	until grep -q "safe hold is armed" "$LOGS/arm_driver.log" 2>/dev/null; do
		if ! kill -0 "${PIDS[-1]}" 2>/dev/null || (( SECONDS >= deadline )); then
			echo "the arm driver didn't come up (Desk: joints unlocked, FCI on?):" >&2
			tail -5 "$LOGS/arm_driver.log" >&2
			exit 1
		fi
		sleep 0.2
	done
	echo "   live; the arm floats until OpenSai holds it"
fi
if $ARM_DRIVER; then  # hold where the arm is now (it may have sagged since the driver seeded its hold)
	redis-cli SET opensai::controllers::FrankaRobot::joint_controller::joint_task::goal_position \
		"$(redis-cli --raw GET opensai::sensors::FrankaRobot::joint_positions)" >/dev/null
fi

echo "1) OpenSai: $OPENSAI_CONFIG"
# OpenSai takes only configs in its own config folder: link ours in (as the other projects' configs sit there).
OPENSAI_CONFIG_NAME="$(basename "$OPENSAI_CONFIG")"
[[ -e "$OPENSAI_DIR/config_folder/xml_config_files/$OPENSAI_CONFIG_NAME" && ! -L "$OPENSAI_DIR/config_folder/xml_config_files/$OPENSAI_CONFIG_NAME" ]] ||
	ln -sfn "$OPENSAI_CONFIG" "$OPENSAI_DIR/config_folder/xml_config_files/$OPENSAI_CONFIG_NAME"
(cd "$OPENSAI_DIR" && exec bash scripts/launch.sh "$OPENSAI_CONFIG_NAME") >"$LOGS/opensai.log" 2>&1 &
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
