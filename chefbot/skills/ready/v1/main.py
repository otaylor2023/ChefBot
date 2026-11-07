"""Ready: the arm to the FR3's ready pose (joints 0, -43, 0, -124, 0, 81, 45 deg: the hand pointing down, 30 cm in
front of the arm's base, 66 cm up), the start of the pick skills. It gives the controller the pose, not a joint path,
so the controller takes the arm around the robot's own body (the pole, the camera on its clamp) itself; a straight
joint-space line to the same pose can pass through them. The arm only, the gripper as it is. No models, no cameras."""

import numpy as np

# The ready pose's flange in the root frame (the FR3's kinematics at the joints above).
READY = np.array([[0.707388, -0.706825, 0.0, 0.298804],
                  [-0.706825, -0.707388, 0.0, 0.0],
                  [0.0, 0.0, -1.0, 0.656365],
                  [0.0, 0.0, 0.0, 1.0]])  # fmt: skip
POS_TOL, ROT_TOL_DEG = 0.01, 3.0


def _row(F, gripper):
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(F[:3, :3]).as_quat()
    return [*F[:3, 3].tolist(), w, x, y, z, gripper]


def _off(F):
    pos = float(np.linalg.norm(F[:3, 3] - READY[:3, 3]))
    c = (np.trace(F[:3, :3].T @ READY[:3, :3]) - 1.0) / 2.0
    return pos, float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def run(robot, max_s: float = 30.0):
    p = robot.proprio()
    g = 1.0 if p["gripper_closed_cmd"] else 0.0
    pos, rot = _off(np.array(p["flange_pose_root"]))
    robot.log(f"from {pos * 100:.1f} cm, {rot:.0f} deg off the ready pose")
    for _ in range(int(max_s)):  # one second of control at a time, until it is there
        if pos <= POS_TOL and rot <= ROT_TOL_DEG:
            break
        robot.act_arm([_row(READY, g)] * 15)
        pos, rot = _off(np.array(robot.proprio()["flange_pose_root"]))
    ok = pos <= POS_TOL and rot <= ROT_TOL_DEG
    robot.log(f"{'at' if ok else 'not at'} the ready pose: {pos * 100:.1f} cm, {rot:.1f} deg off")
    return {"ok": ok, "reason": "at the ready pose" if ok else f"stopped {pos * 100:.1f} cm, {rot:.0f} deg short of the ready pose",
            "pos_error_m": round(pos, 4), "rot_error_deg": round(rot, 1)}
