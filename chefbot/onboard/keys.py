"""Redis keys of the drivers on the robot computer, as ZitiBot uses them
(ZitiBot/controllers/zitibot_core/redis_keys.py, tidybot_base/redis_io.py, tidybot_base/mocap.py).

Formats: OpenSai and the Franka driver use JSON lists ("[0.1, 0.2]", a 3x3 as a list of rows); the
gripper driver plain numbers ("0.08"); the hb1 base driver "[x, y, yaw]"; the mocap publisher
"[x, y, z]" and an x, y, z, w quaternion.
"""

_ARM = "opensai::controllers::FrankaRobot"

# OpenSai arm controllers (onboard/bringup/chefbot_fr3.xml)
ACTIVE_CONTROLLER = f"{_ARM}::active_controller_name"
JOINT_CONTROLLER = "joint_controller"
CARTESIAN_CONTROLLER = "cartesian_controller"
JOINT_GOAL_POSITION = f"{_ARM}::joint_controller::joint_task::goal_position"
JOINT_GOAL_VELOCITY = f"{_ARM}::joint_controller::joint_task::goal_velocity"
JOINT_GOAL_ACCELERATION = f"{_ARM}::joint_controller::joint_task::goal_acceleration"
CARTESIAN_GOAL_POSITION = f"{_ARM}::cartesian_controller::cartesian_task::goal_position"
CARTESIAN_GOAL_ORIENTATION = f"{_ARM}::cartesian_controller::cartesian_task::goal_orientation"
CARTESIAN_CURRENT_POSITION = f"{_ARM}::cartesian_controller::cartesian_task::current_position"
CARTESIAN_CURRENT_ORIENTATION = f"{_ARM}::cartesian_controller::cartesian_task::current_orientation"
CONFIG_FILE_NAME = "::sai-interfaces-webui::config_file_name"

# Franka arm driver (1 kHz)
JOINT_POSITIONS = "opensai::sensors::FrankaRobot::joint_positions"
JOINT_VELOCITIES = "opensai::sensors::FrankaRobot::joint_velocities"

# Franka gripper driver
GRIPPER_MODE = "opensai::FrankaRobot::gripper::mode"
GRIPPER_DESIRED_WIDTH = "opensai::FrankaRobot::gripper::desired_width"
GRIPPER_DESIRED_SPEED = "opensai::FrankaRobot::gripper::desired_speed"
GRIPPER_DESIRED_FORCE = "opensai::FrankaRobot::gripper::desired_force"
GRIPPER_MAX_WIDTH = "opensai::FrankaRobot::gripper::max_width"
GRIPPER_CURRENT_WIDTH = "opensai::FrankaRobot::gripper::current_width"
GRIPPER_GRASP_SUCCESS = "opensai::FrankaRobot::gripper::grasp_success"  # "1" / "0" / "pending"
GRIPPER_READY = "opensai::FrankaRobot::gripper::ready"  # "1" once homed
GRIPPER_MOVE, GRIPPER_GRASP = "m", "g"
GRIPPER_PENDING = "pending"

# hb1 base driver (ZitiBot/controllers/tidybot_base/redis_driver.py): wheel odometry
BASE_POSE = "hb1::current_pose"
BASE_VELOCITY = "hb1::current_vel"
BASE_DESIRED_POSE = "hb1::desired_pose"
BASE_STOP = "hb1::stop"  # "stop": decelerate and hold
BASE_MAX_VEL_SCALE = "hb1::max_vel_scale"  # times the driver's --max-vel-xy / --max-vel-yaw
# ChefBot's base driver (onboard/drivers/base/driver.py) only: a target counts while this keeps changing, and the
# driver's loop counter, so a live driver can be told from odometry an earlier run left in Redis.
BASE_CMD_SEQ = "hb1::cmd_seq"
BASE_HEARTBEAT = "hb1::heartbeat"

# OptiTrack (Motive / NatNet publisher): the base's rigid body, `chefbot-tidy` in the SRC kitchen's Motive
MOCAP_POSITION = "chefbot-tidy::pos"
MOCAP_ORIENTATION = "chefbot-tidy::ori"
MOCAP_VALID = "chefbot-tidy::tracking_valid"
