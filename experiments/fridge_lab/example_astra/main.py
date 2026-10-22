"""An example Astra-lab policy: how to ask Astra about an image, turn its answer into a 3-D point, and move.
It does not open the door."""

import numpy as np
from scipy.spatial.transform import Rotation

ANSWER = {"type": "object", "additionalProperties": False, "required": ["handle_px", "note"],
          "properties": {"handle_px": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                         "note": {"type": "string"}}}  # fmt: skip


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
    rgb = cams[view]["rgb"]
    robot.save_image(view, rgb)
    h, w = rgb.shape[:2]
    # Ask while the robot moves: start the request, act, then collect the answer.
    t = robot.astra_start(f"This {w}x{h} image is from a robot's camera, facing a fridge. Give the pixel (x, y) of the "
                          "fridge door handle, and a short note on what you see.", images=[rgb], schema=ANSWER)
    F = np.array(p["flange_pose_odom"])
    base = p["base_pose_odom"]
    rows = []
    for i in range(1, 16):  # move the flange 5 cm forward over 1 s
        G = F.copy()
        G[0, 3] += 0.05 * i / 15
        rows.append(row(G, 0.0, base))
    robot.act(rows)
    a = robot.astra_wait(t)
    robot.log("astra:", a.get("json") or a.get("failure"), "latency", a.get("latency_s"), "s")
    if a.get("ok") and a.get("json"):
        u, v = a["json"]["handle_px"]
        point = pixel_to_odom(cams[view], min(max(u, 0), w - 1), min(max(v, 0), h - 1))
        robot.log("handle point in odom:", None if point is None else np.round(point, 3).tolist())
