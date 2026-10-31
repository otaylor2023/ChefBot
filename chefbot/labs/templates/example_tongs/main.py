"""An example tongs-lab skill: how to find the tongs and the food with a detector, turn a box into a 3-D point in the
robot's frame, and move the arm there. It does not pick anything up: it holds the hand 15 cm above the tongs,
fingers down, and returns."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

PADS_M = 0.103  # the fingertip pads' centers, out from the flange along its +z


def planar(x, y, yaw):
    T = np.eye(4)
    c, s = np.cos(yaw), np.sin(yaw)
    T[:2, :2], T[:2, 3] = [[c, -s], [s, c]], [x, y]
    return T


def pixel_to_root(cam: dict, u: float, v: float, T_root: np.ndarray) -> np.ndarray | None:
    """The 3-D point (robot root frame) seen at pixel (u, v), from depth and the camera's calibration."""
    z = float(cam["depth"][int(v), int(u)])
    if not z > 0:
        return None
    K = cam["K"]
    p_cam = np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z, 1.0])
    return (np.linalg.inv(T_root) @ cam["cam_to_odom"] @ p_cam)[:3]


def arm_rows(F0: np.ndarray, F1: np.ndarray, n: int, gripper: float) -> list[list[float]]:
    """n act_arm rows from flange pose F0 to F1 (root frame): positions in a line, orientations slerped."""
    turn = Slerp([0.0, 1.0], Rotation.from_matrix([F0[:3, :3], F1[:3, :3]]))
    rows = []
    for t in np.linspace(0.0, 1.0, n + 1)[1:]:
        x, y, z, w = turn([t]).as_quat()[0]
        rows.append([*(F0[:3, 3] + t * (F1[:3, 3] - F0[:3, 3])), w, x, y, z, gripper])
    return rows


def find(robot, cam, T_root, what):
    dets = robot.detect(cam["rgb"], [what], model="grounding-dino", threshold=0.25)
    robot.log(what, "detections:", dets[:3])
    if not dets:
        return None
    x0, y0, x1, y1 = dets[0]["box"]
    return pixel_to_root(cam, (x0 + x1) / 2, (y0 + y1) / 2, T_root)


def run(robot, object="food", where="on the counter", tool="tongs"):  # noqa: A002 (the task's arguments)
    obs = robot.observe(["pole", "wrist"])
    p = obs["proprio"]
    T_root = planar(*p["base_pose_odom"])
    for name, cam in obs["cameras"].items():
        robot.save_image(name, cam["rgb"])
    tongs = find(robot, obs["cameras"]["pole"], T_root, tool) or find(robot, obs["cameras"]["wrist"], T_root, tool)
    food = find(robot, obs["cameras"]["pole"], T_root, object) or find(robot, obs["cameras"]["wrist"], T_root, object)
    robot.log("tongs at", tongs, "food at", food, where)
    if tongs is None:
        return {"ok": False, "reason": "tongs not found"}
    F0 = np.array(p["flange_pose_root"])
    F1 = np.eye(4)
    F1[:3, :3] = np.diag([1.0, -1.0, -1.0])  # the flange's +z (toward the fingers) straight down
    F1[:3, 3] = tongs + [0.0, 0.0, 0.15 + PADS_M]  # the pads 15 cm above the tongs
    robot.act_arm(arm_rows(F0, F1, 60, 0.0))  # 4 s
    return {"ok": False, "reason": "the example only hovers"}
