"""Wire protocol between envs/real (RealEnv, on the computer running the robot layer) and the
onboard server (chefbot/onboard, on the robot): the Env's operations, as msgpack dicts
(protocol.pack / unpack).

Robot -> client:  "hello" once (cameras, limits), then "state" at a fixed rate; "rgbd" replies;
                  "halted" when the watchdog or a halt stopped the robot.
Client -> robot:  commands, fire-and-forget, each holding until the next; "heartbeat" while idle;
                  "rgbd" requests, answered by id; "cameras_keep", the cameras to hold open for it (a skill's).
"""

from __future__ import annotations

VERSION = 1

# client -> robot
HELLO = "hello"
HEARTBEAT = "heartbeat"
ARM_JOINTS = "arm_joints"  # q: joint targets (rad)
ARM_POSE = "arm_pose"  # position (3,), orientation (3, 3): the arm controller's point, arm base frame
GRIPPER = "gripper"  # closed: bool (a grasp, or a move to full width, at the onboard config's speed and force)
GRIPPER_MOVE = "gripper_move"  # width (m), speed (m/s): libfranka's move
GRIPPER_GRASP = "gripper_grasp"  # speed (m/s), force (N): libfranka's grasp
BASE_TARGET = "base_target"  # pose: (x, y, yaw) world, max_vel: (vx, vy, vyaw); refused without live mocap
BASE_TARGET_ODOM = "base_target_odom"  # pose: (x, y, yaw) in the base driver's odometry, max_vel; needs the driver only
BASE_HOLD = "base_hold"
HALT = "halt"
RGBD = "rgbd"  # id, camera
CAMERAS_KEEP = "cameras_keep"  # cameras: names opened now and kept open (not closed when idle) until another list or the client goes

COMMANDS = frozenset({ARM_JOINTS, ARM_POSE, GRIPPER, GRIPPER_MOVE, GRIPPER_GRASP, BASE_TARGET, BASE_TARGET_ODOM, BASE_HOLD, HALT})

# robot -> client
STATE = "state"
HALTED = "halted"
ERROR = "error"
