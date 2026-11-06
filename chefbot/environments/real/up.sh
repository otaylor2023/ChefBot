#!/usr/bin/env bash
# The laptop's side of the real robot (the robot computer's is onboard/bringup/launch.sh): what run_real and the lab
# skills need on this computer, each said "starting..." then "up", then watched until Ctrl-C.
#
#   1) the ssh tunnel to the onboard server: localhost:8765 -> the robot's 8765 (environments/real/config.yaml's url).
#      Restarted when it drops (the robot rebooting, wifi): run_real reconnects to the same port.
#   2) the vision server (runtime/vision_server.py: Grounding DINO, SAM 2) on .env's VISION_URL port, in its own env.
#      Takes a minute to load its models; up once its port answers.
#   3) Franka Desk: localhost:8443 -> the arm's control box (DESK_HOST) through the robot, opened in the browser at
#      https://localhost:8443/desk/. Start at /desk/: Desk's root redirects to https://localhost/desk/, dropping the
#      port. (Not local port 443, which would need no redirect: macOS lets a user bind 443 only on every interface,
#      which would put Desk on the wifi.) Accept the browser's warning (Desk's own certificate).
# A part already up (its port answering) is left alone and not stopped at the end; Ctrl-C stops what this started.
#
# Usage: chefbot/environments/real/up.sh [--no-tunnel] [--no-vision] [--no-desk] [--desk-only] [--check]
#   (--check: say what's up, start nothing; --desk-only: just Desk)
# From the environment, with these defaults:
#   ROBOT_HOST       tidybot01 (an ssh alias). Its password, if it takes one, comes from the macOS Keychain
#                    (service = account = ROBOT_HOST; `security add-generic-password -s tidybot01 -a tidybot01 -w`), via
#                    sshpass, never printed; with ssh keys neither is needed.
#   VISION_PYTHON    the vision env's python (torch, transformers): agentic-skills-exploration's env/vision/.venv
#   DESK_HOST        the arm's control box as the robot reaches it: 172.16.0.12
#   DESK_PORT        the local port Desk is forwarded to: 8443
# Runs on macOS's bash 3.2.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
ROBOT_HOST="${ROBOT_HOST:-tidybot01}"
VISION_PYTHON="${VISION_PYTHON:-$REPO/../../research/agentic-skills-exploration/env/vision/.venv/bin/python}"
TUNNEL_PORT="$(sed -nE 's#^url: *ws://[^:]+:([0-9]+).*#\1#p' "$REPO/chefbot/environments/real/config.yaml")"
VISION_PORT="$(sed -nE 's#^VISION_URL=.*:([0-9]+)/?.*#\1#p' "$REPO/.env" 2>/dev/null)"
VISION_READY_S=240
DESK_HOST="${DESK_HOST:-172.16.0.12}"
DESK_PORT="${DESK_PORT:-8443}"
DESK_URL="https://localhost:$DESK_PORT/desk/"
TUNNEL=true VISION=true DESK=true CHECK=false
for arg in "$@"; do
	case "$arg" in
		--no-tunnel) TUNNEL=false ;;
		--no-vision) VISION=false ;;
		--no-desk) DESK=false ;;
		--desk-only) TUNNEL=false VISION=false ;;
		--check) CHECK=true ;;
		-h | --help) sed -n '2,24p' "$0"; exit 0 ;;
		*) echo "unknown option: $arg" >&2; exit 1 ;;
	esac
done
[[ -n "$TUNNEL_PORT" ]] || { echo "no url port in environments/real/config.yaml" >&2; exit 1; }
$VISION && [[ -z "$VISION_PORT" ]] && { echo "no VISION_URL in $REPO/.env (skip it with --no-vision)" >&2; exit 1; }

LOGS="$(mktemp -d /tmp/chefbot-laptop.XXXXXX)"
TUNNEL_PID="" VISION_PID="" DESK_PID=""

port_open() { nc -z -G 2 127.0.0.1 "$1" >/dev/null 2>&1; }
server_on_robot() {  # the onboard server listening on the robot (through our own ssh: the tunnel says nothing of it)
	_ssh -o ConnectTimeout=6 "$ROBOT_HOST" "exec 3<>/dev/tcp/127.0.0.1/$TUNNEL_PORT" >/dev/null 2>&1
}
_ssh() {
	if command -v sshpass >/dev/null && SSHPASS="$(security find-generic-password -s "$ROBOT_HOST" -a "$ROBOT_HOST" -w 2>/dev/null)"; then
		SSHPASS="$SSHPASS" sshpass -e ssh "$@"
	else
		ssh -o BatchMode=yes "$@"
	fi
}

say_status() {
	if port_open "$TUNNEL_PORT"; then
		if server_on_robot; then echo "tunnel: up (localhost:$TUNNEL_PORT), the onboard server: up"
		else echo "tunnel: up (localhost:$TUNNEL_PORT), the onboard server: NOT up (start launch.sh on the robot)"; fi
	else echo "tunnel: down"; fi
	if [[ -n "$VISION_PORT" ]]; then port_open "$VISION_PORT" && echo "vision server: up (port $VISION_PORT)" || echo "vision server: down"; fi
	port_open "$DESK_PORT" && echo "Franka Desk: forwarded ($DESK_URL)" || echo "Franka Desk: not forwarded"
}
if $CHECK; then say_status; exit 0; fi

cleanup() {
	trap - INT TERM EXIT
	echo "" >&2
	echo "stopping..." >&2
	if [[ -n "$TUNNEL_PID" ]]; then  # sshpass and its ssh: the ssh holds the port
		pkill -TERM -P "$TUNNEL_PID" 2>/dev/null
		kill "$TUNNEL_PID" 2>/dev/null
		echo "tunnel stopped" >&2
	fi
	if [[ -n "$DESK_PID" ]]; then
		pkill -TERM -P "$DESK_PID" 2>/dev/null
		kill "$DESK_PID" 2>/dev/null
		echo "Desk forward stopped" >&2
	fi
	if [[ -n "$VISION_PID" ]]; then
		kill "$VISION_PID" 2>/dev/null
		for _ in 1 2 3 4 5 6; do kill -0 "$VISION_PID" 2>/dev/null || break; sleep 0.5; done
		kill -9 "$VISION_PID" 2>/dev/null
		echo "vision server stopped" >&2
	fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

FORWARD_PID=""
forward() {  # local port, remote host:port as the robot sees it, log name: its own ssh connection (no shared master:
	# it must outlive other sessions), kept alive; FORWARD_PID is its pid once the port answers
	local port="$1" to="$2" name="$3"
	_ssh -N -o ControlPath=none -o ExitOnForwardFailure=yes -o ServerAliveInterval=5 -o ServerAliveCountMax=3 \
		-o ConnectTimeout=8 -L "$port:$to" "$ROBOT_HOST" >>"$LOGS/$name.log" 2>&1 &
	FORWARD_PID=$!
	local deadline=$((SECONDS + 15))
	until port_open "$port"; do
		if ! kill -0 "$FORWARD_PID" 2>/dev/null || ((SECONDS >= deadline)); then
			pkill -TERM -P "$FORWARD_PID" 2>/dev/null
			kill "$FORWARD_PID" 2>/dev/null
			FORWARD_PID=""
			return 1
		fi
		sleep 0.3
	done
}
start_tunnel() { forward "$TUNNEL_PORT" "localhost:$TUNNEL_PORT" tunnel && TUNNEL_PID=$FORWARD_PID; }
start_desk() { forward "$DESK_PORT" "$DESK_HOST:443" desk && DESK_PID=$FORWARD_PID; }

if $TUNNEL; then
	if port_open "$TUNNEL_PORT"; then
		echo "1) tunnel: already up (localhost:$TUNNEL_PORT; not started here, left alone)"
	else
		echo "1) tunnel to $ROBOT_HOST: starting..."
		if start_tunnel; then echo "1) tunnel: up (localhost:$TUNNEL_PORT)"
		else echo "1) tunnel: FAILED (is $ROBOT_HOST on? $(tail -1 "$LOGS/tunnel.log" 2>/dev/null))" >&2; exit 1; fi
	fi
	server_on_robot && echo "   the onboard server on the robot: up" || echo "   the onboard server on the robot: not up yet (launch.sh)"
else
	echo "1) tunnel skipped"
fi

if $VISION; then
	if port_open "$VISION_PORT"; then
		echo "2) vision server: already up (port $VISION_PORT; not started here, left alone)"
	else
		[[ -x "$VISION_PYTHON" ]] || { echo "2) vision server: no python at $VISION_PYTHON (set VISION_PYTHON)" >&2; exit 1; }
		echo "2) vision server: starting... (loading its models, up to ${VISION_READY_S} s)"
		(cd "$REPO" && exec "$VISION_PYTHON" chefbot/runtime/vision_server.py --port "$VISION_PORT") >"$LOGS/vision.log" 2>&1 &
		VISION_PID=$!
		deadline=$((SECONDS + VISION_READY_S))
		until port_open "$VISION_PORT"; do
			if ! kill -0 "$VISION_PID" 2>/dev/null || ((SECONDS >= deadline)); then
				echo "2) vision server: didn't come up; its log:" >&2
				tail -5 "$LOGS/vision.log" >&2
				exit 1
			fi
			sleep 1
		done
		echo "2) vision server: up (port $VISION_PORT)"
	fi
else
	echo "2) vision server skipped"
fi

if $DESK; then
	if port_open "$DESK_PORT"; then
		echo "3) Franka Desk: port $DESK_PORT already in use here (a forward left running? not started, left alone)"
	else
		echo "3) Franka Desk ($DESK_HOST via $ROBOT_HOST): starting..."
		if start_desk; then
			echo "3) Franka Desk: $DESK_URL (opening it)"
			open "$DESK_URL" 2>/dev/null
		else echo "3) Franka Desk: FAILED ($(tail -1 "$LOGS/desk.log" 2>/dev/null)); the rest keeps running" >&2; fi
	fi
else
	echo "3) Franka Desk skipped"
fi

echo ""
echo "=== LAPTOP UP (logs in $LOGS). Ctrl-C stops what this started. ==="

while true; do  # watch: a dropped tunnel is restarted (the robot rebooted, wifi); a dead vision server is said
	sleep 3 &
	wait $! # (a signal ends the wait at once: Ctrl-C isn't held for the sleep)
	if [[ -n "$TUNNEL_PID" ]] && ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
		echo "[$(date +%H:%M:%S)] tunnel dropped: restarting..." >&2
		TUNNEL_PID=""
		until start_tunnel; do sleep 5; done
		echo "[$(date +%H:%M:%S)] tunnel: up again" >&2
	fi
	if [[ -n "$DESK_PID" ]] && ! kill -0 "$DESK_PID" 2>/dev/null; then
		echo "[$(date +%H:%M:%S)] Desk forward dropped: restarting..." >&2
		DESK_PID=""
		until start_desk; do sleep 5; done
		echo "[$(date +%H:%M:%S)] Desk forward: up again" >&2
	fi
	if [[ -n "$VISION_PID" ]] && ! kill -0 "$VISION_PID" 2>/dev/null; then
		echo "[$(date +%H:%M:%S)] the vision server EXITED (its log: $LOGS/vision.log); stopping" >&2
		VISION_PID=""
		exit 1
	fi
done
