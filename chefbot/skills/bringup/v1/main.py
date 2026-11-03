"""Bring-up: the episode host works on this robot. Reads proprio, takes one frame from each camera, moves the flange
up 3 cm and back (arm only), opens and closes the gripper. No models."""

import numpy as np


def _row(F, gripper):
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(F[:3, :3]).as_quat()
    return [*F[:3, 3].tolist(), w, x, y, z, gripper]


def run(robot, lift_m: float = 0.03):
    p = robot.proprio()
    F = np.array(p["flange_pose_root"])
    robot.log(f"flange in the root frame: {np.round(F[:3, 3], 3).tolist()}; gripper {p['gripper_width_m']:.3f} m")
    seen = robot.observe()
    for name, cam in seen["cameras"].items():
        robot.save_image(name, cam["rgb"])
        z = cam["depth"]
        robot.log(f"{name}: {cam['rgb'].shape}, depth median {float(np.median(z[z > 0])) if z is not None and (z > 0).any() else None}, "
                  f"camera at {np.round(np.array(cam['cam_to_odom'])[:3, 3], 3).tolist()} (odom)")
    g = 1.0 if p["gripper_closed_cmd"] else 0.0
    n = 30  # 2 s each way
    up = F.copy()
    up[2, 3] += lift_m
    path = [F + (up - F) * (k / n) for k in range(1, n + 1)]
    r1 = robot.act_arm([_row(T, g) for T in path])
    r2 = robot.act_arm([_row(T, g) for T in path[::-1]] + [_row(F, g)] * 10)
    robot.gripper_move(0.08, 0.05)
    robot.gripper_move(0.04, 0.05)
    robot.gripper_move(0.08, 0.05)
    end = np.array(robot.proprio()["flange_pose_root"])
    err = float(np.linalg.norm(end[:3, 3] - F[:3, 3]))
    robot.log(f"back within {err * 1000:.1f} mm of the start")
    return {"ok": err < 0.01, "back_within_mm": round(err * 1000, 1), "cameras": sorted(seen["cameras"])}
