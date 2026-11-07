"""PORT: the lab's whole-body calls on the real robot, where the skill runs arm only (the episode refuses act and
act_base): act's 11-D rows go to act_arm, each flange target (odom) put in the root frame of where the base is (it
holds, so that is where it stays), the base columns dropped; act_base's rows become as many steps of the arm holding
its commanded joints, the gripper as its last command left it (closed after a grasp, open after a move). Everything
else passes through. In sim these calls also moved the base (up to 7 cm forward and sideways, or Astra's choice):
here the arm reaches from where the base stands."""

import numpy as np
from scipy.spatial.transform import Rotation


class ArmOnly:
    def __init__(self, robot):
        self._robot = robot

    def __getattr__(self, name):
        return getattr(self._robot, name)

    def _root_inv(self) -> np.ndarray:
        x, y, yaw = self._robot.proprio()["base_pose_odom"]
        c, s = np.cos(yaw), np.sin(yaw)
        T = np.eye(4)
        T[:2, :2] = [[c, -s], [s, c]]
        T[:2, 3] = [x, y]
        return np.linalg.inv(T)

    def act(self, rows) -> dict:
        rows = np.asarray(rows, dtype=float).reshape(-1, 11)
        inv = self._root_inv()
        out = []
        for r in rows:
            F = np.eye(4)
            F[:3, :3] = Rotation.from_quat([r[4], r[5], r[6], r[3]]).as_matrix()  # row: x y z qw qx qy qz
            F[:3, 3] = r[:3]
            G = inv @ F
            qx, qy, qz, qw = Rotation.from_matrix(G[:3, :3]).as_quat()
            out.append([*G[:3, 3], qw, qx, qy, qz, r[7]])
        return self._robot.act_arm(out)

    def act_base(self, rows) -> dict:
        n = len(np.asarray(rows, dtype=float).reshape(-1, 3))
        p = self._robot.proprio()
        q = p.get("arm_q_cmd") or p["arm_q"]
        g = 1.0 if p.get("gripper_closed_cmd") else 0.0
        return self._robot.act_joints([list(map(float, q)) + [g]] * n)
