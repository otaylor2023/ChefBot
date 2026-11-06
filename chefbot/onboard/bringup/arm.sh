#!/usr/bin/env bash
# Just the arm: the Franka arm driver, then OpenSai with chefbot_fr3.xml holding it (no gripper, base, mocap or
# server). Ctrl-C stops both, the driver first. Desk first: joints unlocked, FCI on.
#
# Usage: chefbot/onboard/bringup/arm.sh   (the whole robot: launch.sh)
exec bash "$(dirname "$0")/launch.sh" --no-gripper --no-base --no-mocap --no-server "$@"
