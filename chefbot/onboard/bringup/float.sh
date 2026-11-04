#!/usr/bin/env bash
# Just the Franka arm driver: no OpenSai, so nothing holds the arm and it floats (zero torques, gravity
# compensated by the driver): move it by hand. Ctrl-C stops the driver; the brakes then hold the arm.
# Desk first: joints unlocked, FCI on.
#
# Usage: chefbot/onboard/bringup/float.sh   (held by OpenSai: arm.sh; the whole robot: launch.sh)
exec bash "$(dirname "$0")/launch.sh" --no-opensai --no-gripper --no-base --no-mocap --no-server "$@"
