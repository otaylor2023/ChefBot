#!/usr/bin/env bash
# Franka Desk on this computer: localhost:8443 -> the arm's control box (DESK_HOST) through the robot, as a background
# ssh forward that outlives this shell, then Desk opened at https://localhost:8443/desk/.
#
# Usage: chefbot/environments/real/desk.sh [start | stop | status]   (start by default)
#
# Start at /desk/: Desk's root redirects to https://localhost/desk/, dropping the port. (Not local port 443, which
# would need no redirect: macOS lets a user bind 443 only on every interface, which would put Desk on the wifi.)
# Accept the browser's warning (Desk's own certificate). The forward doesn't restart itself if the robot drops:
# run start again.
# From the environment, with these defaults:
#   ROBOT_HOST   tidybot01 (an ssh alias); its password, if it takes one, from the macOS Keychain (service = account =
#                ROBOT_HOST) via sshpass, never printed
#   DESK_HOST    the arm's control box as the robot reaches it: 172.16.0.12
#   DESK_PORT    the local port: 8443
# Runs on macOS's bash 3.2.
set -uo pipefail

ROBOT_HOST="${ROBOT_HOST:-tidybot01}"
DESK_HOST="${DESK_HOST:-172.16.0.12}"
DESK_PORT="${DESK_PORT:-8443}"
URL="https://localhost:$DESK_PORT/desk/"
SPEC="$DESK_PORT:$DESK_HOST:443"

port_open() { nc -z -G 2 127.0.0.1 "$DESK_PORT" >/dev/null 2>&1; }
pids() { pgrep -f -- "-L $SPEC" | grep -v "^$$\$"; }

case "${1:-start}" in
	start)
		if port_open; then
			echo "Desk: already forwarded ($URL)"
		else
			if command -v sshpass >/dev/null && SSHPASS="$(security find-generic-password -s "$ROBOT_HOST" -a "$ROBOT_HOST" -w 2>/dev/null)"; then
				SSHPASS="$SSHPASS" sshpass -e ssh -f -N -o ControlPath=none -o ExitOnForwardFailure=yes -o ServerAliveInterval=5 \
					-o ServerAliveCountMax=3 -o ConnectTimeout=8 -L "$SPEC" "$ROBOT_HOST" || { echo "Desk: FAILED (is $ROBOT_HOST on?)" >&2; exit 1; }
			else
				ssh -f -N -o BatchMode=yes -o ControlPath=none -o ExitOnForwardFailure=yes -o ServerAliveInterval=5 \
					-o ServerAliveCountMax=3 -o ConnectTimeout=8 -L "$SPEC" "$ROBOT_HOST" || { echo "Desk: FAILED (is $ROBOT_HOST on?)" >&2; exit 1; }
			fi
			for _ in 1 2 3 4 5 6 7 8 9 10; do port_open && break; sleep 0.5; done
			port_open || { echo "Desk: the forward started but the port doesn't answer" >&2; exit 1; }
			echo "Desk: forwarded in the background ($URL); stop it with: $0 stop"
		fi
		open "$URL" 2>/dev/null
		;;
	stop)
		p="$(pids)"
		if [[ -n "$p" ]]; then kill $p && echo "Desk: forward stopped"; else echo "Desk: no forward running"; fi
		;;
	status)
		if port_open; then echo "Desk: forwarded ($URL)"; else echo "Desk: not forwarded"; fi
		;;
	-h | --help) sed -n '2,16p' "$0" ;;
	*) echo "usage: $0 [start | stop | status]" >&2; exit 1 ;;
esac
