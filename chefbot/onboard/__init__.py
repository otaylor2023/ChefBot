"""Everything that runs on the robot computer (tidybot01), and nothing else does.

The robot's own drivers do the control: OpenSai (the FR3's joint and Cartesian controllers), the
Franka gripper driver, the hb1 base driver and the mocap publisher, all on the local Redis. This
package is a thin server in front of them (server.py): it streams their state, passes commands
through (robot_io.py), grabs camera frames on request (cameras.py), ties mocap to the base's
odometry (frames.py), and holds the robot when its client goes quiet. The robot layer, planning
and the agent run elsewhere, against envs/real's RealEnv.

Imports: chefbot.interface only (types and the wire protocol); see tests/test_boundaries.py.
Install on the robot: pip install -e ".[onboard]". Start-up: bringup/README.md.
"""
