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
# arm.sh, next to this script, is just 0) and 1); float.sh just 0) (the arm floating, gravity compensated: move it by hand);
# base.sh just 3) to 5) (--no-arm).
#
# Usage: chefbot/onboard/bringup/launch.sh [--no-arm] [--allow-arm] [--no-arm-driver] [--no-opensai] [--no-gripper]
#        [--no-base] [--no-mocap] [--no-server]
#   --no-arm-driver: one is already running, started by hand. --no-arm: no arm at all (no arm driver, OpenSai or
#   gripper): refused while an arm driver runs (the server's halt would hold an arm someone else started) unless
#   --allow-arm.
# Paths and limits come from the environment, with these defaults:
set -euo pipefail

OPENSAI_DIR="${OPENSAI_DIR:-$HOME/OpenSai}"
OPENSAI_CONFIG="${OPENSAI_CONFIG:-$(cd "$(dirname "$0")" && pwd)/chefbot_fr3.xml}"  # ChefBot's own (not ZitiBot's zitibot_panda.xml)
BASE_DRIVER_DIR="${BASE_DRIVER_DIR:-$(cd "$(dirname "$0")/../drivers/base" && pwd)}"  # ChefBot's copy (its README.md)
BASE_PYTHON="${BASE_PYTHON:-$HOME/miniforge3/envs/zitibot/bin/python}"  # has phoenix6, ruckig
CHEFBOT_PYTHON="${CHEFBOT_PYTHON:-$HOME/chefbot_calib/env/bin/python}"  # numpy, redis, pyrealsense2, websockets, msgpack(-numpy)
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
# The base driver runs at the robot's hard base cap: embodiments/tidybot_fr3/robot.yaml control.limits, read by
# interface/limits.py (its one copy: the laptop's speed profiles, the onboard server and the gamepad are held to it too).
read -r MAX_VEL_XY MAX_VEL_YAW MAX_ACCEL_XY MAX_ACCEL_YAW < <(PYTHONPATH="$REPO_ROOT" "$CHEFBOT_PYTHON" -c 'from chefbot.interface.limits import base_cap
c = base_cap("tidybot_fr3")
print(c["vel"][0], c["vel"][2], c["accel"][0], c["accel"][2])') || true
[[ -n "${MAX_ACCEL_YAW:-}" ]] || { echo "couldn't read the base cap (embodiments/tidybot_fr3/robot.yaml control.limits)" >&2; exit 1; }
READY_TIMEOUT_S=90
SERVER_PORT="${SERVER_PORT:-8765}"  # keep equal to onboard/config.yaml server.port

ARM_DRIVER=true OPENSAI=true GRIPPER=true BASE=true MOCAP=true SERVER=true NO_ARM=false ALLOW_ARM=false
for arg in "$@"; do
	case "$arg" in
		--no-arm) NO_ARM=true ARM_DRIVER=false OPENSAI=false GRIPPER=false ;;
		--allow-arm) ALLOW_ARM=true ;;
		--no-arm-driver) ARM_DRIVER=false ;;
		--no-opensai) OPENSAI=false ;;  # nothing holds the arm: it floats on the driver's zero torques
		--no-gripper) GRIPPER=false ;;
		--no-base) BASE=false ;;
		--no-mocap) MOCAP=false ;;
		--no-server) SERVER=false ;;
		-h | --help) sed -n '2,24p' "$0"; exit 0 ;;
		*) echo "unknown option: $arg" >&2; exit 1 ;;
	esac
done

GRIPPER_DIR="$OPENSAI_DIR/drivers/FrankaPanda/redis_driver"  # the arm driver's too
if $OPENSAI; then
	for f in "$OPENSAI_DIR/bin/OpenSai_main" "$OPENSAI_DIR/scripts/launch.sh"; do
		[[ -e "$f" ]] || { echo "missing $f (OPENSAI_DIR=$OPENSAI_DIR)" >&2; exit 1; }
	done
fi
if $NO_ARM && ! $ALLOW_ARM && pgrep -f "[s]ai_franka_robot_redis_driver" >/dev/null; then
	echo "the Franka arm driver is running: stop it (and lock the joints) or pass --allow-arm" >&2
	exit 1
fi
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
	pkill -TERM -f "[d]rivers/base/teleop.py" 2>/dev/null || true  # its Vehicle stops (its SIGTERM handler)
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

arm_float() {
	stop_ready_mover  # OpenSai's float_controller (chefbot_fr3.xml): no stiffness, light damping
	redis-cli SET "$ACTIVE_CONTROLLER_KEY" float_controller >/dev/null
	ARM_MODE="floating"
	echo "=== ARM FLOATING: move it by hand. h holds it where it is, r takes it to the ready position. (A laptop run's moves are ignored meanwhile.) ==="
}

ARM_MODE="held"
REPO_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"

READY_PID=""
stop_ready_mover() {  # a move to the ready position still under way: stopped (f and h take the arm from it)
	if [[ -n "$READY_PID" ]] && kill -0 "$READY_PID" 2>/dev/null; then
		pkill -TERM -P "$READY_PID" 2>/dev/null || true
		kill -TERM "$READY_PID" 2>/dev/null || true
	fi
	READY_PID=""
}

arm_ready() {  # to the ready position (interface/workspace.py READY), around the pole's keep-out if the straight way isn't
	# clear: the calibration's arm (onboard/calibration/arm.py: the joint task's goal a step at a time, each leg planned
	# by interface/workspace.py joint_path, the planner the real host uses), in the background so f and h still work
	[[ "$(redis-cli --raw GET "$JOINTS_KEY")" == \[* ]] || { echo "=== no joint reading from the arm driver: not going to the ready position ===" >&2; return; }
	stop_ready_mover
	if [[ "$ARM_MODE" == floating ]]; then  # hold first: a switch resets the controller's goal to where the arm is
		arm_hold >/dev/null
		sleep 0.3
	fi
	ARM_MODE="held"
	echo "=== ARM GOING TO THE READY POSITION (OpenSai's joint task; around the keep-out if it must); f floats it, h holds it where it is ==="
	(
		if PYTHONPATH="$REPO_DIR" "$CHEFBOT_PYTHON" -c 'from chefbot.onboard.calibration.arm import Arm
from chefbot.interface.workspace import READY
Arm().move_to(READY, "the ready position", confirm=False)' >>"$LOGS/ready.log" 2>&1; then
			echo "=== ARM AT THE READY POSITION ==="
		else
			echo "=== ARM NOT AT THE READY POSITION: $(tail -1 "$LOGS/ready.log") ===" >&2
		fi
	) &
	READY_PID=$!
}

arm_hold() {
	stop_ready_mover  # back to the joint controller, its goal where the arm is now (OpenSai re-initializes it there too)
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
	echo "=== ARM HOLDING where it is (f floats it, r takes it to the ready position) ==="
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

gamepad_start() {  # the base to the gamepad (onboard/drivers/base/teleop.py at the teleop speed profile): the driver stops
	if [[ -n "$GAMEPAD_PID" ]]; then
		echo "=== the gamepad already has the base (b gives it back) ==="
		return
	fi
	echo "=== GAMEPAD: stopping the base driver..."
	stop_base_driver
	(cd "$BASE_DRIVER_DIR" && PYTHONPATH="$REPO_ROOT" exec "$BASE_PYTHON" "$BASE_DRIVER_DIR/teleop.py") 2>&1 | grep --line-buffered -v -i "pkg_resources\|import os\|^pygame \|hello from the pygame" &
	GAMEPAD_PID=$!
	echo "=== GAMEPAD has the base: Start on the gamepad begins, hold a bumper to drive (left: base frame, right: world),"
	echo "    left stick moves, right stick turns; b gives the base back to its driver. Laptop base targets wait meanwhile. ==="
}

gamepad_stop() {  # the gamepad's Vehicle stops (its SIGTERM handler), then the driver takes the base again
	[[ -n "$GAMEPAD_PID" ]] || { echo "=== the gamepad isn't running (g starts it) ==="; return; }
	pkill -TERM -f "[d]rivers/base/teleop.py" 2>/dev/null || true
	for _ in $(seq 1 30); do
		pgrep -f "[d]rivers/base/teleop.py" >/dev/null || break
		sleep 0.1
	done
	pkill -KILL -f "[d]rivers/base/teleop.py" 2>/dev/null || true
	wait "$GAMEPAD_PID" 2>/dev/null || true
	GAMEPAD_PID=""
	echo "=== gamepad stopped; the base driver: starting..."
	start_base_driver
	echo "=== the base driver has the base again (its odometry starts over from where it is) ==="
}

changing() {  # key, seconds: its value changes within that long (a driver that publishes it is running)
	local first deadline
	first="$(redis-cli GET "$1" 2>/dev/null)"
	deadline=$((SECONDS + $2))
	while ((SECONDS < deadline)); do
		sleep 0.2
		[[ "$(redis-cli GET "$1" 2>/dev/null)" != "$first" ]] && return 0
	done
	return 1
}

live_check() {  # before ALL UP: each driver started is publishing (its values change), not just started
	local ok=0
	if $BASE && ! changing hb1::heartbeat 5; then
		echo "the base driver isn't publishing (hb1::heartbeat not changing): see $LOGS/base.log" >&2; ok=1
	fi
	if $ARM_DRIVER && ! changing opensai::sensors::FrankaRobot::joint_positions 5; then
		echo "the arm driver isn't publishing joints: see $LOGS/arm_driver.log" >&2; ok=1
	fi
	return $ok
}

GRIP_W="opensai::FrankaRobot::gripper::desired_width" GRIP_V="opensai::FrankaRobot::gripper::desired_speed"
GRIP_F="opensai::FrankaRobot::gripper::desired_force" GRIP_MODE="opensai::FrankaRobot::gripper::mode"
GRIP_NOW="opensai::FrankaRobot::gripper::current_width"
gripper_key() {  # o open, c close gently (a 10 N grasp), [ / ] 1 cm narrower / wider: the hand's fingers don't
	# backdrive, so floating the arm can't free them; these move them instead (onboard/robot_io.py's keys and limits)
	local now width
	now="$(redis-cli --raw GET "$GRIP_NOW" 2>/dev/null)"
	now="${now:-0.04}"
	case "$1" in
		o) width=0.08 ;;
		c) redis-cli SET "$GRIP_W" 0.0 >/dev/null; redis-cli SET "$GRIP_V" 0.05 >/dev/null; redis-cli SET "$GRIP_F" 10.0 >/dev/null
		   redis-cli SET opensai::FrankaRobot::gripper::grasp_success pending >/dev/null; redis-cli SET "$GRIP_MODE" g >/dev/null
		   echo "=== GRIPPER CLOSING (10 N grasp) ==="; return ;;
		"[") width="$(awk -v w="$now" 'BEGIN { w -= 0.01; if (w < 0) w = 0; printf "%.3f", w }')" ;;
		"]") width="$(awk -v w="$now" 'BEGIN { w += 0.01; if (w > 0.08) w = 0.08; printf "%.3f", w }')" ;;
	esac
	redis-cli SET "$GRIP_W" "$width" >/dev/null
	redis-cli SET "$GRIP_V" 0.05 >/dev/null
	redis-cli SET "$GRIP_F" 20.0 >/dev/null
	redis-cli SET "$GRIP_MODE" m >/dev/null
	echo "=== GRIPPER TO ${width} m (was ${now}) ==="
}

watch_parts() {  # until a part exits (then say which, and stop everything); meanwhile f / h float and hold the arm
	local keys="" pid k
	if { $OPENSAI || $BASE || $GRIPPER; } && [[ -t 0 ]]; then
		keys=1
		$OPENSAI && echo "keys: f floats the arm (move it by hand), h holds it where it is, r takes it to the ready position"
		$BASE && echo "keys: g gives the base to the gamepad, b gives it back to its driver"
		$GRIPPER && echo "keys: o opens the gripper, c closes it (gently), [ and ] move the fingers 1 cm closer / apart"
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
				h | H) $OPENSAI && arm_hold ;;
				r | R) $OPENSAI && arm_ready ;;
				g | G) $BASE && gamepad_start ;;
				b | B) $BASE && gamepad_stop ;;
				o | O) $GRIPPER && gripper_key o ;;
				c | C) $GRIPPER && gripper_key c ;;
				"[") $GRIPPER && gripper_key "[" ;;
				"]") $GRIPPER && gripper_key "]" ;;
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
elif $NO_ARM; then
	echo "0-2) no arm (--no-arm): no arm driver, OpenSai or gripper"
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
	$NO_ARM || echo "2) gripper driver skipped"
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
	live_check || exit 1
	echo ""
	echo "=== ALL UP: the robot is ready (logs in $LOGS). Ctrl-C stops everything. ==="
	watch_parts
else
	echo "5) server skipped"
	live_check || exit 1
	echo ""
	echo "=== ALL UP: the drivers are ready (logs in $LOGS). Ctrl-C stops them. ==="
	watch_parts
fi
