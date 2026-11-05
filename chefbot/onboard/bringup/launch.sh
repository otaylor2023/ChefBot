#!/usr/bin/env bash
# Start the robot's drivers (the existing ones, in place), then the onboard server.
#
#   0) the Franka arm driver (real-time torque loop: OpenSai's safe_launch_driver.sh, which clears stale commands
#      and seeds a hold at the arm's joints; asks for sudo once, for the CPU governor)
#   1) OpenSai with chefbot_fr3.xml, next to this script (arm joint + Cartesian controllers, Redis, web UI)
#   2) the Franka gripper driver (OpenSai's drivers/FrankaPanda/redis_driver/launch_gripper.sh)
#   3) the hb1 base driver (ChefBot's copy of ZitiBot's: onboard/drivers/base/driver.py, in the zitibot env)
#   4) the mocap publisher (mocap.sh, next to this script; skipped with a warning off SRC wifi)
#   5) python -m chefbot.onboard.server, in the foreground
# Ctrl-C stops them all (the arm driver first: the arm's brakes hold it). Desk first: joints unlocked, FCI on.
# arm.sh, next to this script, is just 0) and 1); float.sh just 0) (the arm floating, gravity compensated: move it by hand).
#
# Usage: chefbot/onboard/bringup/launch.sh [--no-arm-driver] [--no-opensai] [--no-gripper] [--no-base] [--no-mocap] [--no-server]
#   (--no-arm-driver: one is already running, started by hand)
# Paths and limits come from the environment, with these defaults:
set -euo pipefail

OPENSAI_DIR="${OPENSAI_DIR:-$HOME/OpenSai}"
OPENSAI_CONFIG="${OPENSAI_CONFIG:-$(cd "$(dirname "$0")" && pwd)/chefbot_fr3.xml}"  # ChefBot's own (not ZitiBot's zitibot_panda.xml)
BASE_DRIVER_DIR="${BASE_DRIVER_DIR:-$(cd "$(dirname "$0")/../drivers/base" && pwd)}"  # ChefBot's copy (its README.md)
BASE_PYTHON="${BASE_PYTHON:-$HOME/miniforge3/envs/zitibot/bin/python}"  # has phoenix6, ruckig
CHEFBOT_PYTHON="${CHEFBOT_PYTHON:-$HOME/chefbot_calib/env/bin/python}"  # numpy, redis, pyrealsense2, websockets, msgpack(-numpy)
ONBOARD_CONFIG="$(cd "$(dirname "$0")/.." && pwd)/config.yaml"
# The base driver's limits: onboard/config.yaml's base.driver_max_* (their one copy; MAX_VEL_XY etc. override for a run).
read -r CFG_VXY CFG_VYAW CFG_AXY CFG_AYAW < <("$CHEFBOT_PYTHON" -c 'import sys, yaml
b = yaml.safe_load(open(sys.argv[1]))["base"]
print(b["driver_max_vel_xy"], b["driver_max_vel_yaw"], b["driver_max_accel_xy"], b["driver_max_accel_yaw"])' "$ONBOARD_CONFIG") || true
[[ -n "${CFG_AYAW:-}" ]] || { echo "couldn't read the base driver's limits from $ONBOARD_CONFIG" >&2; exit 1; }
MAX_VEL_XY="${MAX_VEL_XY:-$CFG_VXY}"
MAX_VEL_YAW="${MAX_VEL_YAW:-$CFG_VYAW}"
MAX_ACCEL_XY="${MAX_ACCEL_XY:-$CFG_AXY}"
MAX_ACCEL_YAW="${MAX_ACCEL_YAW:-$CFG_AYAW}"
READY_TIMEOUT_S=90
SERVER_PORT="${SERVER_PORT:-8765}"  # keep equal to onboard/config.yaml server.port

ARM_DRIVER=true OPENSAI=true GRIPPER=true BASE=true MOCAP=true SERVER=true
for arg in "$@"; do
	case "$arg" in
		--no-arm-driver) ARM_DRIVER=false ;;
		--no-opensai) OPENSAI=false ;;  # nothing holds the arm: it floats on the driver's zero torques
		--no-gripper) GRIPPER=false ;;
		--no-base) BASE=false ;;
		--no-mocap) MOCAP=false ;;
		--no-server) SERVER=false ;;
		-h | --help) sed -n '2,20p' "$0"; exit 0 ;;
		*) echo "unknown option: $arg" >&2; exit 1 ;;
	esac
done

GRIPPER_DIR="$OPENSAI_DIR/drivers/FrankaPanda/redis_driver"  # the arm driver's too
for f in "$OPENSAI_DIR/bin/OpenSai_main" "$OPENSAI_DIR/scripts/launch.sh"; do
	[[ -e "$f" ]] || { echo "missing $f (OPENSAI_DIR=$OPENSAI_DIR)" >&2; exit 1; }
done
$GRIPPER && { [[ -f "$GRIPPER_DIR/launch_gripper.sh" ]] || { echo "missing $GRIPPER_DIR/launch_gripper.sh" >&2; exit 1; }; }
$BASE && { [[ -f "$BASE_DRIVER_DIR/driver.py" && -x "$BASE_PYTHON" ]] || { echo "missing the base driver or $BASE_PYTHON" >&2; exit 1; }; }

PIDS=()
declare -A NAMES  # pid -> which part
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
	pkill -TERM -f "[g]amepad_teleop.py" 2>/dev/null || true  # its Vehicle stops (its SIGTERM handler)
	for pid in "${PIDS[@]}"; do
		pkill -TERM -P "$pid" 2>/dev/null || true
		kill -TERM "$pid" 2>/dev/null || true
	done
	tmux kill-session -t interfaces_server 2>/dev/null || true
	pkill -TERM -f "OpenSai_main.*$(basename "$OPENSAI_CONFIG")" 2>/dev/null || true
	wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT TERM  # Ctrl-C: stop everything, then out (not back to waiting on what was just stopped)

ACTIVE_CONTROLLER_KEY="opensai::controllers::FrankaRobot::active_controller_name"
JOINT_GOAL_KEY="opensai::controllers::FrankaRobot::joint_controller::joint_task::goal_position"
JOINT_GOAL_VEL_KEY="opensai::controllers::FrankaRobot::joint_controller::joint_task::goal_velocity"
JOINTS_KEY="opensai::sensors::FrankaRobot::joint_positions"

arm_float() {  # OpenSai's float_controller (chefbot_fr3.xml): no stiffness, light damping
	redis-cli SET "$ACTIVE_CONTROLLER_KEY" float_controller >/dev/null
	ARM_MODE="floating"
	echo "=== ARM FLOATING: move it by hand. s holds it where it is, h takes it home. (A laptop run's moves are ignored meanwhile.) ==="
}

ARM_MODE="held"
REPO_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"

arm_home() {  # the joint task's goal to the ready pose (interface/workspace.py HOME), if the straight way there is clear
	local q verdict
	q="$(redis-cli --raw GET "$JOINTS_KEY")"
	[[ "$q" == \[* ]] || { echo "=== no joint reading from the arm driver: not going home ===" >&2; return; }
	verdict="$(PYTHONPATH="$REPO_DIR" "$CHEFBOT_PYTHON" -c 'import json, sys
from chefbot.interface.workspace import HOME, path_clear
print(("clear " if path_clear(json.loads(sys.argv[1]), HOME) else "blocked ") + json.dumps(list(HOME)))' "$q" 2>&1)" || true
	if [[ "$verdict" != clear* ]]; then
		echo "=== NOT going home: the straight way there passes the pole's keep-out (${verdict%% *}). f, move the arm nearer home by hand, then h ===" >&2
		return
	fi
	if [[ "$ARM_MODE" == floating ]]; then  # hold first: a switch resets the controller's goal to where the arm is
		arm_hold >/dev/null
		sleep 0.3
	fi
	redis-cli SET "$JOINT_GOAL_VEL_KEY" "[0,0,0,0,0,0,0]" >/dev/null
	redis-cli SET "$JOINT_GOAL_KEY" "${verdict#clear }" >/dev/null
	ARM_MODE="held"
	echo "=== ARM GOING HOME (OpenSai's joint task, at most 0.5 rad/s); f floats it, s holds it where it is ==="
}

arm_hold() {  # back to the joint controller, its goal where the arm is now (OpenSai re-initializes it there too)
	local q
	q="$(redis-cli --raw GET "$JOINTS_KEY")"
	if [[ "$q" != \[* ]]; then
		echo "=== no joint reading from the arm driver: not switching (still $(redis-cli --raw GET "$ACTIVE_CONTROLLER_KEY")) ===" >&2
		return
	fi
	redis-cli SET "$JOINT_GOAL_KEY" "$q" >/dev/null
	redis-cli SET "$JOINT_GOAL_VEL_KEY" "[0,0,0,0,0,0,0]" >/dev/null
	redis-cli SET "$ACTIVE_CONTROLLER_KEY" joint_controller >/dev/null
	ARM_MODE="held"
	echo "=== ARM HOLDING where it is (f floats it, h takes it home) ==="
}

BASE_PID="" GAMEPAD_PID=""

start_base_driver() {  # ours (onboard/drivers/base), a part like the others; a stale one of either holds the CAN bus
	pkill -TERM -f "[d]rivers/base/driver.py" 2>/dev/null || true
	pkill -TERM -f "[t]idybot_base/redis_driver.py" 2>/dev/null || true
	redis-cli DEL hb1::current_pose >/dev/null  # a stale pose would look live (before the driver writes its first)
	(cd "$BASE_DRIVER_DIR" && exec "$BASE_PYTHON" "$BASE_DRIVER_DIR/driver.py" --max-vel-xy "$MAX_VEL_XY" --max-vel-yaw "$MAX_VEL_YAW" \
		--max-accel-xy "$MAX_ACCEL_XY" --max-accel-yaw "$MAX_ACCEL_YAW") >>"$LOGS/base.log" 2>&1 &
	BASE_PID=$!
	PIDS+=("$BASE_PID")
	NAMES[$BASE_PID]="the base driver"
	wait_for_key "hb1::current_pose" "[" 15
}

stop_base_driver() {  # out of the watched parts first (its exit is ours), then until it has let go of the CAN bus
	local keep=() pid
	for pid in "${PIDS[@]}"; do [[ "$pid" == "$BASE_PID" ]] || keep+=("$pid"); done
	PIDS=("${keep[@]}")
	pkill -TERM -P "$BASE_PID" 2>/dev/null || true
	kill -TERM "$BASE_PID" 2>/dev/null || true
	for _ in $(seq 1 30); do
		pgrep -f "[d]rivers/base/driver.py" >/dev/null || break
		sleep 0.1
	done
	pkill -KILL -f "[d]rivers/base/driver.py" 2>/dev/null || true
	BASE_PID=""
}

gamepad_start() {  # the base to the gamepad (onboard/drivers/base/gamepad_teleop.py, its own speeds): the driver stops
	if [[ -n "$GAMEPAD_PID" ]]; then
		echo "=== the gamepad already has the base (b gives it back) ==="
		return
	fi
	echo "=== GAMEPAD: stopping the base driver..."
	stop_base_driver
	(cd "$BASE_DRIVER_DIR" && exec "$BASE_PYTHON" "$BASE_DRIVER_DIR/gamepad_teleop.py") 2>&1 | grep --line-buffered -v -i "pkg_resources\|import os\|^pygame \|hello from the pygame" &
	GAMEPAD_PID=$!
	echo "=== GAMEPAD has the base: Start on the gamepad begins, hold a bumper to drive (left: base frame, right: world),"
	echo "    left stick moves, right stick turns; b gives the base back to its driver. Laptop base targets wait meanwhile. ==="
}

gamepad_stop() {  # the gamepad's Vehicle stops (its SIGTERM handler), then the driver takes the base again
	[[ -n "$GAMEPAD_PID" ]] || { echo "=== the gamepad isn't running (g starts it) ==="; return; }
	pkill -TERM -f "[g]amepad_teleop.py" 2>/dev/null || true
	for _ in $(seq 1 30); do
		pgrep -f "[g]amepad_teleop.py" >/dev/null || break
		sleep 0.1
	done
	pkill -KILL -f "[g]amepad_teleop.py" 2>/dev/null || true
	wait "$GAMEPAD_PID" 2>/dev/null || true
	GAMEPAD_PID=""
	echo "=== gamepad stopped; the base driver: starting..."
	start_base_driver
	echo "=== the base driver has the base again (its odometry starts over from where it is) ==="
}

watch_parts() {  # until a part exits (then say which, and stop everything); meanwhile f / h float and hold the arm
	local keys="" pid k
	if { $OPENSAI || $BASE; } && [[ -t 0 ]]; then
		keys=1
		$OPENSAI && echo "keys: f floats the arm (move it by hand), s holds it where it is, h takes it home"
		$BASE && echo "keys: g gives the base to the gamepad, b gives it back to its driver"
	fi
	while true; do
		for pid in "${PIDS[@]}"; do
			if ! kill -0 "$pid" 2>/dev/null; then
				echo "" >&2
				echo "=== ${NAMES[$pid]:-a part} STOPPED (its log: $LOGS): stopping everything ===" >&2
				exit 1
			fi
		done
		if [[ -n "$GAMEPAD_PID" ]] && ! kill -0 "$GAMEPAD_PID" 2>/dev/null; then
			echo "=== the gamepad exited by itself: giving the base back to its driver ===" >&2
			GAMEPAD_PID="x"
			gamepad_stop
		fi
		if [[ -n "$keys" ]]; then
			k=""
			read -rsn1 -t 0.5 k || true
			case "$k" in
				f | F) $OPENSAI && arm_float ;;
				s | S) $OPENSAI && arm_hold ;;
				h | H) $OPENSAI && arm_home ;;
				g | G) $BASE && gamepad_start ;;
				b | B) $BASE && gamepad_stop ;;
			esac
		else
			sleep 0.5
		fi
	done
}

wait_for_key() {  # key, what must be in its value, timeout
	local deadline=$((SECONDS + $3))
	until [[ "$(redis-cli GET "$1" 2>/dev/null)" == *"$2"* ]]; do
		(( SECONDS < deadline )) || { echo "timed out waiting for $1; logs in $LOGS" >&2; exit 1; }
		sleep 0.5
	done
}

if $ARM_DRIVER; then
	if pgrep -f "[s]ai_franka_robot_redis_driver" >/dev/null; then
		echo "an arm driver is already running: stop it first (or pass --no-arm-driver)" >&2
		exit 1
	fi
	echo "0) Franka arm driver: starting..."
	sudo -v  # its launcher sets the CPU governor with sudo: ask here, not in the background
	pgrep -x redis-server >/dev/null || redis-server --daemonize yes >/dev/null
	redis-cli DEL opensai::sensors::FrankaRobot::joint_positions >/dev/null  # a stale value would look live
	(cd "$GRIPPER_DIR" && exec sh safe_launch_driver.sh) >"$LOGS/arm_driver.log" 2>&1 &
	PIDS+=($!)
	NAMES[$!]="the arm driver"
	deadline=$((SECONDS + 30))
	until grep -q "safe hold is armed" "$LOGS/arm_driver.log" 2>/dev/null; do
		if ! kill -0 "${PIDS[-1]}" 2>/dev/null || (( SECONDS >= deadline )); then
			echo "the arm driver didn't come up (Desk: joints unlocked, FCI on?):" >&2
			tail -5 "$LOGS/arm_driver.log" >&2
			exit 1
		fi
		sleep 0.2
	done
	echo "0) Franka arm driver: up (the arm floats until OpenSai holds it)"
fi
if $ARM_DRIVER; then  # hold where the arm is now (it may have sagged since the driver seeded its hold)
	redis-cli SET opensai::controllers::FrankaRobot::joint_controller::joint_task::goal_position \
		"$(redis-cli --raw GET opensai::sensors::FrankaRobot::joint_positions)" >/dev/null
fi

if $OPENSAI; then
echo "1) OpenSai ($(basename "$OPENSAI_CONFIG")): starting..."
# OpenSai takes only configs in its own config folder: link ours in (as the other projects' configs sit there).
OPENSAI_CONFIG_NAME="$(basename "$OPENSAI_CONFIG")"
[[ -e "$OPENSAI_DIR/config_folder/xml_config_files/$OPENSAI_CONFIG_NAME" && ! -L "$OPENSAI_DIR/config_folder/xml_config_files/$OPENSAI_CONFIG_NAME" ]] ||
	ln -sfn "$OPENSAI_CONFIG" "$OPENSAI_DIR/config_folder/xml_config_files/$OPENSAI_CONFIG_NAME"
(cd "$OPENSAI_DIR" && exec bash scripts/launch.sh "$OPENSAI_CONFIG_NAME") >"$LOGS/opensai.log" 2>&1 &
PIDS+=($!)
NAMES[$!]="OpenSai"
wait_for_key "::sai-interfaces-webui::config_file_name" "$(basename "$OPENSAI_CONFIG")" "$READY_TIMEOUT_S"
echo "1) OpenSai: up (holding the arm)"
else
	# zero torques: floating (SET, never DEL: the driver reads this key every tick, and a missing one kills it,
	# "Could not deserialize custom string to eigen data", 1.1 s in: 2026-11-03)
	redis-cli SET opensai::commands::FrankaRobot::control_torques "[0,0,0,0,0,0,0]" >/dev/null
	echo "1) OpenSai skipped: THE ARM FLOATS (gravity compensated; it falls slowly if the load is off)"
fi

if $GRIPPER; then
	echo "2) Franka gripper driver: starting..."
	redis-cli DEL opensai::FrankaRobot::gripper::ready >/dev/null  # a stale "1" would look ready
	(cd "$GRIPPER_DIR" && exec bash launch_gripper.sh) >"$LOGS/gripper.log" 2>&1 &
	PIDS+=($!)
	NAMES[$!]="the gripper driver"
	wait_for_key "opensai::FrankaRobot::gripper::ready" "1" 30
	echo "2) Franka gripper driver: up (homed)"
else
	echo "2) gripper driver skipped"
fi

if $BASE; then
	echo "3) hb1 base driver (max vel $MAX_VEL_XY m/s, $MAX_VEL_YAW rad/s): starting..."
	start_base_driver
	echo "3) hb1 base driver: up"
else
	echo "3) base driver skipped"
fi

if $MOCAP; then
	echo "4) mocap publisher: starting..."
	redis-cli DEL chefbot-tidy::tracking_valid >/dev/null  # a stale value from an earlier run would look live
	bash "$(dirname "$0")/mocap.sh" --quiet 2>"$LOGS/mocap.log" &
	PIDS+=($!)
	NAMES[$!]="the mocap publisher"
	deadline=$((SECONDS + 10))
	until [[ -n "$(redis-cli GET chefbot-tidy::tracking_valid 2>/dev/null)" ]]; do
		if ! kill -0 "${PIDS[-1]}" 2>/dev/null || (( SECONDS >= deadline )); then
			echo "4) mocap publisher: NOT up (not localized; see $LOGS/mocap.log)" >&2
			break
		fi
		sleep 0.5
	done
	[[ -n "$(redis-cli GET chefbot-tidy::tracking_valid 2>/dev/null)" ]] && echo "4) mocap publisher: up"
else
	echo "4) mocap skipped"
fi

if $SERVER; then
	echo "5) onboard server: starting..."
	# the repo on its path (three folders up from this script): wherever launch.sh is run from
	PYTHONPATH="$(cd "$(dirname "$0")/../../.." && pwd)${PYTHONPATH:+:$PYTHONPATH}" "$CHEFBOT_PYTHON" -m chefbot.onboard.server &
	PIDS+=($!)
	NAMES[$!]="the onboard server"
	SERVER_PID=$!
	deadline=$((SECONDS + 30))
	until (exec 3<>/dev/tcp/127.0.0.1/"$SERVER_PORT") 2>/dev/null; do  # its port open: it takes clients
		if ! kill -0 "$SERVER_PID" 2>/dev/null || (( SECONDS >= deadline )); then
			echo "5) onboard server: didn't come up" >&2
			exit 1
		fi
		sleep 0.3
	done
	echo "5) onboard server: up (port $SERVER_PORT)"
	echo ""
	echo "=== ALL UP: the robot is ready (logs in $LOGS). Ctrl-C stops everything. ==="
	watch_parts
else
	echo "5) server skipped"
	echo ""
	echo "=== ALL UP: the drivers are ready (logs in $LOGS). Ctrl-C stops them. ==="
	watch_parts
fi
