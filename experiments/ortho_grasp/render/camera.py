"""B0 inputs (spec §7.3): the unmodified shoulder and wrist camera images, each
camera's intrinsics and camera-to-robot transform as text, and the TCP pose.

The shoulder image is taken with the arm folded back (out of its view), the
wrist image with the arm at its ready pose above the counter, where the TCP
pose reported in the prompt is.
"""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np

from experiments.ortho_grasp.frames import pose_to_robot

WRIST_CAMERA = "robot0_eye_in_hand"


def _calibration(name: str, rgbd, base) -> str:
    K = rgbd.intrinsics
    T = pose_to_robot(rgbd.cam_to_world, base)
    h, w = rgbd.depth.shape
    rows = "; ".join("[" + ", ".join(f"{v:.4f}" for v in row) + "]" for row in T[:3, :3])
    return (f"{name}: {w} x {h} px; intrinsics fx = {K[0, 0]:.1f}, fy = {K[1, 1]:.1f}, cx = {K[0, 2]:.1f}, cy = {K[1, 2]:.1f} px. "
            f"Camera-to-robot transform (camera axes in the OpenCV convention: x right, y down, z forward out of the lens): "
            f"rotation rows {rows}; camera position [{T[0, 3]:.4f}, {T[1, 3]:.4f}, {T[2, 3]:.4f}] m.")  # fmt: skip


def capture(world, out_dir: Path) -> dict:
    """Writes shoulder.png and wrist.png; returns {"images", "camera_text", "tcp_position", "tcp_quaternion"}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    env = world.env
    shoulder = env.get_camera_rgbd(env.config.shoulder_camera)
    iio.imwrite(out_dir / "shoulder.png", shoulder.rgb)
    q_ready = np.array(world.stack.robot._cfg.arm_ready)
    env.set_arm_joints(q_ready)
    env.wait_for_next_tick(0.3)
    wrist = env.get_camera_rgbd(WRIST_CAMERA)
    iio.imwrite(out_dir / "wrist.png", wrist.rgb)
    tcp_world = world.arm.frames.pose_to_world(world.base, world.arm.ik.fk(q_ready))
    tcp = pose_to_robot(tcp_world, world.base)
    env.set_arm_joints(np.array(world.config["observe_arm"]))
    env.wait_for_next_tick(0.3)
    text = "\n".join([_calibration("Shoulder camera (first image)", shoulder, world.base),
                      _calibration("Wrist camera (second image)", wrist, world.base)])  # fmt: skip
    return {"images": [str(out_dir / "shoulder.png"), str(out_dir / "wrist.png")], "camera_text": text,
            "tcp_position": tcp[:3, 3].tolist(), "tcp_quaternion": _quat(tcp[:3, :3]).tolist()}  # fmt: skip


def _quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z)."""
    t = np.trace(R)
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q)
