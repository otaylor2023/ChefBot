"""Bring-up for the real driver: small moves of each part, back to the start, logging how well each is
tracked. The flange 5 cm forward, 5 cm up and back; the gripper closed and opened; the base 5 cm
forward and back, turned 5 degrees and back with the flange held still in odom."""

import numpy as np
from scipy.spatial.transform import Rotation


def row(F, gripper, base):
    x, y, z, w = Rotation.from_matrix(F[:3, :3]).as_quat()
    return [*F[:3, 3], w, x, y, z, gripper, *base]


def move(robot, name, F0, F1, g, b0, b1, steps=30, settle=10):
    rows = [row(F0 + (F1 - F0) * (i + 1) / steps, g, list(np.array(b0) + (np.array(b1) - b0) * (i + 1) / steps)) for i in range(steps)]
    r = robot.act(rows + [rows[-1]] * settle)
    s = r["steps"][-1]
    p = r["proprio"]
    robot.log(f"{name}: pos err {s['pos_error_m'] * 100:.1f} cm, rot err {np.degrees(s['rot_error_rad']):.1f} deg, "
              f"base err {s['base_error_m'] * 100:.1f} cm, ik ok {all(x['ik_ok'] for x in r['steps'])}, "
              f"gripper width {p['gripper_width_m'] * 1000:.0f} mm")  # fmt: skip


def run(robot):
    p = robot.proprio()
    F, b = np.array(p["flange_pose_odom"]), list(p["base_pose_odom"])
    robot.log("start: flange", np.round(F[:3, 3], 3).tolist(), "base", np.round(b, 3).tolist())
    obs = robot.observe()
    for name, c in obs["cameras"].items():
        robot.save_image(name, c["rgb"])
    fwd, up = F.copy(), F.copy()
    fwd[:3, 3] += F[:3, 0] * 0.05  # along the fingers
    up[2, 3] += 0.05
    move(robot, "flange forward 5 cm", F, fwd, 0.0, b, b)
    move(robot, "flange back", fwd, F, 0.0, b, b)
    move(robot, "flange up 5 cm", F, up, 0.0, b, b)
    move(robot, "flange down", up, F, 0.0, b, b)
    move(robot, "gripper closed", F, F, 1.0, b, b, steps=5, settle=15)
    move(robot, "gripper open", F, F, 0.0, b, b, steps=5, settle=15)
    b_fwd = [b[0] + 0.05 * np.cos(b[2]), b[1] + 0.05 * np.sin(b[2]), b[2]]
    F_fwd = F.copy()
    F_fwd[:3, 3] += [0.05 * np.cos(b[2]), 0.05 * np.sin(b[2]), 0.0]  # the flange rides along
    move(robot, "base forward 5 cm", F, F_fwd, 0.0, b, b_fwd, steps=45)
    move(robot, "base back", F_fwd, F, 0.0, b_fwd, b, steps=45)
    b_turn = [b[0], b[1], b[2] + np.radians(5)]
    move(robot, "base turned 5 deg, flange held", F, F, 0.0, b, b_turn, steps=30, settle=15)
    move(robot, "base turned back", F, F, 0.0, b_turn, b, steps=30, settle=15)
    obs = robot.observe()
    for name, c in obs["cameras"].items():
        robot.save_image(name + "_end", c["rgb"])
