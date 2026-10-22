"""The ARX5 L5 arm for the fridge lab's real driver, as an RPC server on the TidyBot++ NUC.

A joint-space variant of tidybot2-arx's arm_server.py (same SDK, CAN bus and limits), for the driver
in experiments/fridge_lab/real: it reports measured joints and gripper width and takes joint targets,
and all kinematics happen on the driver's side (models/arx5_L5.xml). It controls the same arm as
arm_server.py, so stop that one first. Copy this file into the tidybot2-arx checkout on the NUC
(next to constants.py) and run it there:

    python arx_arm_server.py            # listens on ARM_RPC_HOST, port 50002

The joint controller interpolates each target over one control period (default_preview_time), so
targets sent at 15 Hz give smooth motion; the driver limits how far each target may be from the last.
"""

import math
import sys
import time
from multiprocessing.managers import BaseManager as MPBaseManager

import numpy as np

from constants import ARM_RPC_HOST, RPC_AUTHKEY

sys.path.append('/home/iprl/ria/tidybot2-arx/arx5-sdk/python')  # as in arm_server.py; adjust if the SDK lives elsewhere
import arx5_interface as arx5  # noqa: E402

PORT = 50002
CONTROL_PERIOD = 1.0 / 15.0
OPEN_WIDTH = 0.088  # m, arx5-sdk/include/app/config.h
MAX_JOINT_STEP = 0.15  # rad per command: the driver sends far smaller steps; anything bigger is refused


class Arm:
    def __init__(self):
        robot_config = arx5.RobotConfigFactory.get_instance().get_config('L5')
        controller_config = arx5.ControllerConfigFactory.get_instance().get_config('joint_controller', robot_config.joint_dof)
        controller_config.default_preview_time = CONTROL_PERIOD

        # Limits as in arm_server.py
        joint_pos_max = robot_config.joint_pos_max.copy()
        joint_pos_max[1] = math.radians(210)
        joint_pos_max[2] = math.radians(166.24)
        robot_config.joint_pos_max = joint_pos_max
        robot_config.gripper_vel_max = 0.5
        robot_config.gripper_torque_max = 5.5

        self.arm = arx5.Arx5JointController(robot_config, controller_config, 'can1')
        self.cmd = arx5.JointState(robot_config.joint_dof)
        self.q_min = np.array(robot_config.joint_pos_min)
        self.q_max = np.array(joint_pos_max)

    def state(self):
        """Measured joints (rad) and gripper opening (m)."""
        js = self.arm.get_joint_state()
        return {'q': js.pos().tolist(), 'gripper_width': float(js.gripper_pos)}

    def limits(self):
        return {'q_min': self.q_min.tolist(), 'q_max': self.q_max.tolist(), 'open_width': OPEN_WIDTH}

    def command(self, q, gripper_width):
        """One joint target (rad) and gripper opening (m). Refused if it is far from where the arm is."""
        q = np.clip(np.asarray(q, dtype=float), self.q_min, self.q_max)
        q_now = np.array(self.arm.get_joint_state().pos())
        if np.abs(q - q_now).max() > MAX_JOINT_STEP:
            return False
        self.cmd.pos()[:] = q
        self.cmd.gripper_pos = float(np.clip(gripper_width, 0.0, OPEN_WIDTH))
        self.arm.set_joint_cmd(self.cmd)
        return True

    def move_to(self, q, seconds=3.0, gripper_width=OPEN_WIDTH):
        """Interpolate to a joint configuration (reset poses; not for the episode)."""
        q0 = np.array(self.arm.get_joint_state().pos())
        q1 = np.clip(np.asarray(q, dtype=float), self.q_min, self.q_max)
        n = max(1, int(seconds / CONTROL_PERIOD))
        for i in range(n):
            self.cmd.pos()[:] = q0 + (i + 1) / n * (q1 - q0)
            self.cmd.gripper_pos = gripper_width
            self.arm.set_joint_cmd(self.cmd)
            time.sleep(CONTROL_PERIOD)
        return self.state()

    def home(self):
        """The SDK's home pose, gripper open."""
        self.cmd.gripper_pos = OPEN_WIDTH
        self.arm.set_joint_cmd(self.cmd)
        time.sleep(0.5)
        self.arm.reset_to_home()
        return self.state()

    def hold(self):
        """Hold where the arm is now."""
        js = self.arm.get_joint_state()
        self.cmd.pos()[:] = js.pos()
        self.cmd.gripper_pos = js.gripper_pos
        self.arm.set_joint_cmd(self.cmd)

    def close(self):
        self.hold()


class ArmManager(MPBaseManager):
    pass


ArmManager.register('Arm', Arm)

if __name__ == '__main__':
    manager = ArmManager(address=(ARM_RPC_HOST, PORT), authkey=RPC_AUTHKEY)
    server = manager.get_server()
    print(f'ARX joint-space arm server at {ARM_RPC_HOST}:{PORT}')
    server.serve_forever()
