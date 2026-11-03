#!/usr/bin/env bash
# Just the base: ChefBot's base driver (onboard/drivers/base/driver.py, with its command timeout), the mocap
# publisher (mocap.sh), then the onboard server in the foreground. No OpenSai, arm or gripper. Ctrl-C stops all
# three: the server first, then the driver (kill: it brakes and exits), then mocap.
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
MAX_VEL_XY="${MAX_VEL_XY:-0.125}"      # keep equal to onboard/config.yaml base.driver_max_vel_xy
MAX_VEL_YAW="${MAX_VEL_YAW:-0.4}"      # ... and base.driver_max_vel_yaw
LOGS="${LOGS:-$REPO/runs/bringup_base/$(date +%Y%m%d_%H%M%S)}"

ALLOW_ARM=false
for arg in "$@"; do
	case "$arg" in
		--allow-arm) ALLOW_ARM=true ;;
		-h | --help) sed -n '2,9p' "$0"; exit 0 ;;
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
cleanup() {
	trap - INT TERM EXIT
	echo "stopping..." >&2
	[[ -n "${SERVER_PID:-}" ]] && kill -INT "$SERVER_PID" 2>/dev/null && wait "$SERVER_PID" 2>/dev/null || true
	redis-cli SET hb1::kill kill >/dev/null || true   # the driver brakes and exits
	sleep 1
	for pid in "${PIDS[@]}"; do kill -INT "$pid" 2>/dev/null || true; done
	wait 2>/dev/null || true
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

echo "1) base driver (max $MAX_VEL_XY m/s, $MAX_VEL_YAW rad/s; brakes without fresh commands)"
"$BASE_PYTHON" -u "$REPO/chefbot/onboard/drivers/base/driver.py" --max-vel-xy "$MAX_VEL_XY" --max-vel-yaw "$MAX_VEL_YAW" \
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
