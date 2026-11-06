#!/usr/bin/env bash
# Just the base: launch.sh --no-arm. ChefBot's base driver (onboard/drivers/base/driver.py, with its command timeout,
# at the robot's hard base cap), the mocap publisher (mocap.sh), then the onboard server; no arm driver, OpenSai or
# gripper. Ctrl-C stops them all; g / b hand the base to the gamepad and back, as in launch.sh.
#
# Stop it with Ctrl-C in its terminal, or `kill -TERM <its pid>` (SIGINT does not reach a backgrounded script).
#
# Usage: chefbot/onboard/bringup/base.sh [--allow-arm] [launch.sh's other options]
#   Refuses while the Franka arm driver runs (the server's halt also holds the arm, which then belongs to whoever
#   started it), unless --allow-arm. Paths and limits as launch.sh's (CHEFBOT_PYTHON, BASE_PYTHON).
exec bash "$(dirname "$0")/launch.sh" --no-arm "$@"
