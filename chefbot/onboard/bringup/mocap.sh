#!/usr/bin/env bash
# The mocap publisher: the SRC kitchen's OptiTrack (Motive) into the NUC's Redis. NatNet's SampleClient (with the
# lab's Redis output: <rigid body>::pos, ::ori as xyzw, ::tracking_valid, ::lin_vel, ::ang_vel, per rigid body
# Motive streams; ChefBot reads chefbot-tidy::*). It only writes Redis.
#
# The NUC must be on SRC wifi (the stream reaches only SRC). Motive's Streaming > Transmission Type must match
# MOCAP_MODE: the kitchen's is unicast. The client is built once into $MOCAP_DIR from the NatNet SDK's sample
# (the SDK's own build is multicast; it is left as it is).
#
# Usage: chefbot/onboard/bringup/mocap.sh [--quiet]     (--quiet: no per-frame printing; Ctrl-C stops it)
set -euo pipefail

NATNET_SDK="${NATNET_SDK:-$HOME/external/NatNet_SDK_4.1_ubuntu}"
MOCAP_DIR="${MOCAP_DIR:-$HOME/chefbot_mocap}"
MOTIVE_IP="${MOTIVE_IP:-172.24.69.102}"  # the SRC kitchen's Motive computer
MOCAP_MODE="${MOCAP_MODE:-unicast}"      # unicast | multicast, as Motive streams
WIFI="${WIFI:-wlp88s0}"

QUIET=false
for arg in "$@"; do
	case "$arg" in
		--quiet) QUIET=true ;;
		-h | --help) sed -n '2,10p' "$0"; exit 0 ;;
		*) echo "unknown option: $arg" >&2; exit 1 ;;
	esac
done
case "$MOCAP_MODE" in unicast) TYPE=Unicast ;; multicast) TYPE=Multicast ;; *) echo "MOCAP_MODE: unicast or multicast" >&2; exit 1 ;; esac

MY_IP="$(ip -4 -br addr show "$WIFI" 2>/dev/null | awk '{print $3}' | cut -d/ -f1)"
[[ "$MY_IP" == 172.24.* ]] || { echo "the NUC is not on SRC wifi ($WIFI: ${MY_IP:-no address}): sudo nmcli con up SRC" >&2; exit 1; }

CLIENT="$MOCAP_DIR/build/SampleClient_$MOCAP_MODE"
if [[ ! -x "$CLIENT" ]]; then
	echo "building $CLIENT"
	mkdir -p "$MOCAP_DIR/build"
	# The sample has the multicast line live and the unicast one commented out: keep one, set to $TYPE.
	sed -e '\|^//static const ConnectionType kDefaultConnectionType = ConnectionType_Unicast;|d' \
		-e "s|^static const ConnectionType kDefaultConnectionType = ConnectionType_Multicast;|static const ConnectionType kDefaultConnectionType = ConnectionType_$TYPE;|" \
		"$NATNET_SDK/samples/SampleClient/SampleClient.cpp" >"$MOCAP_DIR/SampleClient_$MOCAP_MODE.cpp"
	grep -q "^static const ConnectionType kDefaultConnectionType = ConnectionType_$TYPE;" "$MOCAP_DIR/SampleClient_$MOCAP_MODE.cpp" ||
		{ echo "could not set the connection type in the SDK's SampleClient.cpp" >&2; exit 1; }
	g++ "$MOCAP_DIR/SampleClient_$MOCAP_MODE.cpp" -I"$NATNET_SDK/include" -L"$NATNET_SDK/lib" -lNatNet -std=c++17 \
		-I/usr/include/eigen3 -I/usr/include/jsoncpp -I/usr/include/hiredis -ljsoncpp -lpthread -lhiredis -o "$CLIENT"
fi

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$NATNET_SDK/lib"
echo "mocap: $MOCAP_MODE from Motive $MOTIVE_IP to $MY_IP"
if $QUIET; then
	exec "$CLIENT" "$MOTIVE_IP" "$MY_IP" </dev/null >/dev/null 2>&1
fi
exec "$CLIENT" "$MOTIVE_IP" "$MY_IP"
