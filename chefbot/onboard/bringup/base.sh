#!/usr/bin/env bash
# Just the base: ChefBot's base driver (onboard/drivers/base/driver.py, with its command timeout), the mocap
# publisher (mocap.sh), then the onboard server in the foreground. No OpenSai, arm or gripper. Ctrl-C stops all
# three: the driver first (hb1::kill: it brakes and exits), then the server and mocap (TERM, then KILL after 3 s).
#
# Stop it with Ctrl-C in its terminal, or `kill -TERM <its pid>` (SIGINT does not reach a backgrounded script).
#
# Usage: chefbot/onboard/bringup/base.sh [--allow-arm]
#   Refuses while the Franka arm driver runs (the server's halt also holds the arm, which then belongs to whoever
#   started it), unless --allow-arm; and while anything else holds the base's CAN bus or publishes mocap.
# Paths and limits from the environment, with these defaults:
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
BASE_PYTHON="${BASE_PYTHON:-$HOME/miniforge3/envs/zitibot/bin/python}"            # phoenix6, ruckig, redis
CHEFBOT_PYTHON="${CHEFBOT_PYTHON:-$HOME/miniforge3/envs/chefbot-movement/bin/python}"
ONBOARD_CONFIG="$REPO/chefbot/onboard/config.yaml"
# The base driver's limits: onboard/config.yaml's base.driver_max_* (their one copy, as launch.sh reads them;
# MAX_VEL_XY etc. override for a run).
read -r CFG_VXY CFG_VYAW CFG_AXY CFG_AYAW < <("$CHEFBOT_PYTHON" -c 'import sys, yaml
b = yaml.safe_load(open(sys.argv[1]))["base"]
print(b["driver_max_vel_xy"], b["driver_max_vel_yaw"], b["driver_max_accel_xy"], b["driver_max_accel_yaw"])' "$ONBOARD_CONFIG") || true
[[ -n "${CFG_AYAW:-}" ]] || { echo "couldn't read the base driver's limits from $ONBOARD_CONFIG" >&2; exit 1; }
MAX_VEL_XY="${MAX_VEL_XY:-$CFG_VXY}"
MAX_VEL_YAW="${MAX_VEL_YAW:-$CFG_VYAW}"
MAX_ACCEL_XY="${MAX_ACCEL_XY:-$CFG_AXY}"
MAX_ACCEL_YAW="${MAX_ACCEL_YAW:-$CFG_AYAW}"
LOGS="${LOGS:-$REPO/runs/bringup_base/$(date +%Y%m%d_%H%M%S)}"

ALLOW_ARM=false
for arg in "$@"; do
	case "$arg" in
		--allow-arm) ALLOW_ARM=true ;;
		-h | --help) sed -n '2,11p' "$0"; exit 0 ;;
		*) echo "unknown option: $arg" >&2; exit 1 ;;
	esac
done

busy() { ps -eo args | grep -E "$1" | grep -v grep >/dev/null; }
busy 'redis_driver\.py|drivers/base/driver\.py|gamepad_teleop|bringup\.gamepad' &&
	{ echo "something already drives the base (redis_driver / driver.py / gamepad): stop it first" >&2; exit 1; }
busy 'build/SampleClient' && { echo "a mocap publisher is already running: stop it first" >&2; exit 1; }
busy 'chefbot\.onboard\.server' && { echo "an onboard server is already running: stop it first" >&2; exit 1; }
if busy 'sai_franka_robot_redis_driver' && ! $ALLOW_ARM; then
	echo "the Franka arm driver is running: stop it (and lock the joints) or pass --allow-arm" >&2; exit 1
fi

mkdir -p "$LOGS"
PIDS=()
stop_pid() {  # pid: TERM, up to 3 s, then KILL. (A script's background jobs ignore SIGINT, so INT would not do.)
	kill -TERM "$1" 2>/dev/null || return 0
	for _ in $(seq 30); do kill -0 "$1" 2>/dev/null || return 0; sleep 0.1; done
	echo "  pid $1 did not stop: KILL" >&2
	kill -KILL "$1" 2>/dev/null || true
}
cleanup() {
	trap - INT TERM EXIT
	echo "stopping..." >&2
	redis-cli SET hb1::kill kill >/dev/null || true   # first: the driver brakes and exits by itself
	[[ -n "${PIDS[0]:-}" ]] && for _ in $(seq 30); do kill -0 "${PIDS[0]}" 2>/dev/null || break; sleep 0.1; done
	[[ -n "${SERVER_PID:-}" ]] && stop_pid "$SERVER_PID"
	for pid in "${PIDS[@]}"; do stop_pid "$pid"; done
	pkill -TERM -f "[b]uild/SampleClient_unicast" 2>/dev/null || true   # mocap.sh execs the client
	redis-cli SET hb1::kill ok >/dev/null || true
	echo "stopped; logs in $LOGS" >&2
}
trap cleanup EXIT INT TERM

changes() {  # key, seconds: true if the key's value changes within that long
	local first; first="$(redis-cli GET "$1")"
	local deadline=$((SECONDS + $2))
	while (( SECONDS < deadline )); do
		sleep 0.2
		[[ "$(redis-cli GET "$1")" != "$first" ]] && return 0
	done
	return 1
}

echo "1) base driver (max $MAX_VEL_XY m/s, $MAX_VEL_YAW rad/s; accel $MAX_ACCEL_XY m/s^2, $MAX_ACCEL_YAW rad/s^2; brakes without fresh commands)"
"$BASE_PYTHON" -u "$REPO/chefbot/onboard/drivers/base/driver.py" --max-vel-xy "$MAX_VEL_XY" --max-vel-yaw "$MAX_VEL_YAW" \
	--max-accel-xy "$MAX_ACCEL_XY" --max-accel-yaw "$MAX_ACCEL_YAW" \
	>"$LOGS/base.log" 2>&1 &
PIDS+=($!)
changes hb1::heartbeat 15 || { echo "the base driver did not start; $LOGS/base.log:" >&2; tail -20 "$LOGS/base.log" >&2; exit 1; }

echo "2) mocap publisher"
(cd "$HOME/chefbot_mocap" 2>/dev/null || cd /tmp; exec bash "$HERE/mocap.sh" --quiet) >"$LOGS/mocap.log" 2>&1 &
PIDS+=($!)
changes chefbot-tidy::pos 10 || echo "   warning: no mocap frames yet (Motive streaming? chefbot-tidy tracked?)" >&2

echo "3) onboard server (Ctrl-C stops everything); logs in $LOGS"
cd "$REPO"
"$CHEFBOT_PYTHON" -u -m chefbot.onboard.server > >(tee "$LOGS/server.log") 2>&1 &
SERVER_PID=$!  # the server itself (not tee), so cleanup's Ctrl-C reaches it
wait "$SERVER_PID"
