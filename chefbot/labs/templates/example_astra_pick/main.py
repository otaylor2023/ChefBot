"""An example Astra pick-lab policy: how to ask Astra about an image, turn its answer into a 3-D point in the robot's
frame, and move the arm there (arm only: robot.act_arm). It does not pick anything up: it holds the hand 15 cm above
the point Astra chose, fingers down, and returns."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

PADS_M = 0.093  # the fingertip pads' centers, out from the flange along its +z
ANSWER = {"type": "object", "additionalProperties": False, "required": ["px", "note"],
          "properties": {"px": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                         "note": {"type": "string"}}}  # fmt: skip


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


def run(robot, object="object"):  # noqa: A002 (the task's argument: what to pick up)
    obs = robot.observe(["wrist"])
    cam, p = obs["cameras"]["wrist"], obs["proprio"]
    T_root = planar(*p["base_pose_odom"])
    h, w = cam["rgb"].shape[:2]
    r = robot.astra(f"The robot's wrist camera ({w} x {h} px). Where is the {object}? Give a pixel on it.", images=[cam["rgb"]],
                    schema=ANSWER, effort="low")  # fmt: skip
    a = r.get("json") if r.get("ok") else None
    robot.log(f"astra {r.get('latency_s')} s: {a or r.get('failure')}")
    if not a:
        return {"ok": False, "reason": "no answer"}
    P = pixel_to_root(cam, *a["px"], T_root)
    if P is None:
        return {"ok": False, "reason": "no depth there"}
    F0 = np.array(p["flange_pose_root"])
    F1 = np.eye(4)
    F1[:3, :3] = np.diag([1.0, -1.0, -1.0])  # the flange's +z (toward the fingers) straight down
    F1[:3, 3] = P + [0.0, 0.0, 0.15 + PADS_M]  # the pads 15 cm above the point
    robot.act_arm(arm_rows(F0, F1, 60, 0.0))  # 4 s
    return {"ok": False, "reason": "the example only hovers"}
