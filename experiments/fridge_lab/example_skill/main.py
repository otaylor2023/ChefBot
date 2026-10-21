"""An example skill: how to see, find 3-D points, and move. It does not open the door."""

import numpy as np
from scipy.spatial.transform import Rotation


def pixel_to_odom(cam: dict, u: float, v: float) -> np.ndarray | None:
    """The 3-D point (odom) seen at pixel (u, v), from depth and the camera's calibration."""
    z = float(cam["depth"][int(v), int(u)])
    if not z > 0:
        return None
    K = cam["K"]
    p_cam = np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z, 1.0])
    return (cam["cam_to_odom"] @ p_cam)[:3]


def row(F: np.ndarray, gripper: float, base) -> list[float]:
    """An 11-D action row from a 4x4 flange pose (odom), a gripper command and a base pose [x, y, yaw]."""
    x, y, z, w = Rotation.from_matrix(F[:3, :3]).as_quat()
    return [*F[:3, 3], w, x, y, z, gripper, *base]


def run(robot):
    obs = robot.observe()
    cams, p = obs["cameras"], obs["proprio"]
    view = next(n for n in cams if n != "wrist")  # the first base camera
    robot.save_image(view, cams[view]["rgb"])
    try:
        dets = robot.detect(cams[view]["rgb"], ["fridge door handle"], model="grounding-dino", threshold=0.25)
    except RuntimeError as e:  # no vision server on this machine
        robot.log("detect unavailable:", e)
        dets = []
    robot.log("detections:", dets[:3])
    if dets:
        x0, y0, x1, y1 = dets[0]["box"]
        point = pixel_to_odom(cams[view], (x0 + x1) / 2, (y0 + y1) / 2)
        robot.log("handle box center in odom:", None if point is None else np.round(point, 3).tolist())

    # Move the flange 5 cm forward over 15 steps (1 s), base held where it is.
    F = np.array(p["flange_pose_odom"])
    base = p["base_pose_odom"]
    rows = []
    for i in range(1, 16):
        G = F.copy()
        G[0, 3] += 0.05 * i / 15
        rows.append(row(G, 0.0, base))
    r = robot.act(rows)
    robot.log("tracking error after the move:", r["steps"][-1], "budget:", robot.budget())
