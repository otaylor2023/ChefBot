"""Wire protocol between envs/real (RealEnv, on the computer running the robot layer) and the
onboard server (chefbot/onboard, on the robot): the Env's operations, as msgpack dicts
(protocol.pack / unpack).

Robot -> client:  "hello" once (cameras, limits), then "state" at a fixed rate; "rgbd" replies;
                  "halted" when the watchdog or a halt stopped the robot.
Client -> robot:  commands, fire-and-forget, each holding until the next; "heartbeat" while idle;
                  "rgbd" requests, answered by id.
"""

from __future__ import annotations

VERSION = 1

# client -> robot
HELLO = "hello"
HEARTBEAT = "heartbeat"
ARM_JOINTS = "arm_joints"  # q: joint targets (rad)
ARM_POSE = "arm_pose"  # position (3,), orientation (3, 3): the arm controller's point, arm base frame
GRIPPER = "gripper"  # closed: bool
BASE_TARGET = "base_target"  # pose: (x, y, yaw) world, max_vel: (vx, vy, vyaw)
BASE_HOLD = "base_hold"
HALT = "halt"
RGBD = "rgbd"  # id, camera

COMMANDS = frozenset({ARM_JOINTS, ARM_POSE, GRIPPER, BASE_TARGET, BASE_HOLD, HALT})

# robot -> client
STATE = "state"
HALTED = "halted"
ERROR = "error"
