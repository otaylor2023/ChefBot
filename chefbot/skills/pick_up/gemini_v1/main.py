"""A basic top-down pick: one wrist RGB-D frame, Gemini Robotics-ER points at the object, the depth around the point
gives where it is and how tall, then above it, down, grasp, up.

From where the arm is (the ready pose: the hand pointing down, the object in the wrist camera's view):
  1. open the gripper; one wrist frame
  2. Gemini points at the object (x, y normalized 0..1000)
  3. the depth there, through the camera's pose: the point in odom; around it, the surface it stands on (a low
     percentile of the heights) and its top; its middle from the points above that surface near the point
  4. the fingertip pads' center to its top less half its height (at most 2.5 cm in, at least 1.5 cm above the
     surface), the hand as it is (pointing down, the fingers closing as they do now)
  5. 10 cm above it, then down, a pause for the arm to settle, the grasp, 5 cm up, a pause
  6. held if the fingers stopped on something (not closed on nothing, not still open)
"""

import json
import math

import numpy as np

MODEL = "gemini-robotics-er-2-preview"
PAD = 0.13065  # m along the flange's z: the fingertip pads' center (the Franka Hand's 0.1034 on the 27.25 mm holder)
SPEED = 0.02  # m/s along a move
HZ = 15.0


def _quat_wxyz(R):
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = math.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = math.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x, y, z = math.copysign(x, R[2, 1] - R[1, 2]), math.copysign(y, R[0, 2] - R[2, 0]), math.copysign(z, R[1, 0] - R[0, 1])
    return [w, x, y, z]


def _root_from_odom(p):
    """odom -> root (the arm base now): the base's planar pose in odom."""
    x, y, yaw = p["base_pose_odom"]
    T = np.eye(4)
    T[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    T[:2, 3] = [x, y]
    return np.linalg.inv(T)


def _move(robot, F_from, F_to, gripper, hold=0):
    """The flange from one pose to the other in a straight line (root frame), at SPEED, then `hold` steps there."""
    d = float(np.linalg.norm(F_to[:3, 3] - F_from[:3, 3]))
    n = max(10, int(math.ceil(d / SPEED * HZ)))
    q = _quat_wxyz(F_to[:3, :3])
    rows = [[*(F_from[:3, 3] + (F_to[:3, 3] - F_from[:3, 3]) * (k / n)).tolist(), *q, gripper] for k in range(1, n + 1)]
    rows += [[*F_to[:3, 3].tolist(), *q, gripper]] * hold
    return robot.act_arm(rows)


def _point(robot, rgb, obj):
    prompt = (f"Point to the {obj}. Return JSON only: {{\"x\": x, \"y\": y}}, a point on the middle of its top as seen "
              "here. Coordinates are integers normalized 0 to 1000, x left to right, y top to bottom. "
              "If it is not in the image, return {\"x\": null, \"y\": null}.")
    r = robot.llm(MODEL, prompt, [rgb], max_output_tokens=200, thinking_budget=0, temperature=0.0)
    text = r["text"].strip().strip("`")
    text = text[text.find("{"): text.rfind("}") + 1]
    a = json.loads(text)
    robot.log(f"gemini: {r['text'].strip()[:200]} (${r['cost_usd']:.4f}, {r['latency_s']:.1f} s)")
    if a.get("x") is None or a.get("y") is None:
        return None
    h, w = rgb.shape[:2]
    return float(a["x"]) * w / 1000.0, float(a["y"]) * h / 1000.0


def _cloud(cam, u, v, r):
    """Odom points of the depth pixels within r of (u, v)."""
    z = np.asarray(cam["depth"], dtype=float)
    h, w = z.shape
    v0, v1, u0, u1 = max(0, int(v) - r), min(h, int(v) + r + 1), max(0, int(u) - r), min(w, int(u) + r + 1)
    vv, uu = np.mgrid[v0:v1, u0:u1]
    d = z[v0:v1, u0:u1]
    ok = (d > 0.05) & (d < 2.0)
    K = np.asarray(cam["K"])
    x = (uu[ok] - K[0, 2]) * d[ok] / K[0, 0]
    y = (vv[ok] - K[1, 2]) * d[ok] / K[1, 1]
    P = np.stack([x, y, d[ok], np.ones_like(x)], 1)
    return (np.asarray(cam["cam_to_odom"]) @ P.T).T[:, :3]


def _marked(rgb, u, v):
    img = np.array(rgb, copy=True)
    h, w = img.shape[:2]
    for du in range(-12, 13):
        for t in (-1, 0, 1):
            for (a, b) in ((int(v) + t, int(u) + du), (int(v) + du, int(u) + t)):
                if 0 <= a < h and 0 <= b < w:
                    img[a, b] = (255, 0, 255)
    return img


def run(robot, object: str, force_n: float = 20.0):
    robot.gripper_move(0.08, 0.05)
    obs = robot.observe(("wrist",))
    cam = obs["cameras"]["wrist"]
    rgb = cam["rgb"]
    robot.save_image("wrist", rgb)
    pt = _point(robot, rgb, object)
    if pt is None:
        return {"ok": False, "reason": f"gemini doesn't see a {object}"}
    u, v = pt
    robot.save_image("point", _marked(rgb, u, v))
    near = _cloud(cam, u, v, 4)
    if len(near) < 5:
        return {"ok": False, "reason": "no depth at the point", "point_px": [round(u), round(v)]}
    P = np.median(near, 0)  # the point on the object, in odom
    around = _cloud(cam, u, v, 90)
    flat = np.linalg.norm(around[:, :2] - P[:2], axis=1)
    surface = float(np.percentile(around[flat < 0.15, 2], 10))
    body = around[(flat < 0.06) & (around[:, 2] > surface + 0.01)]
    if len(body) < 10:
        return {"ok": False, "reason": f"nothing stands above the surface at the point ({P[2] - surface:.3f} m)", "point_px": [round(u), round(v)]}
    top = float(np.percentile(body[:, 2], 95))
    height = top - surface
    center = np.median(body[:, :2], 0)
    pad_z = max(surface + 0.015, top - min(0.025, height / 2))
    robot.log(f"point {np.round(P, 3).tolist()}; surface z {surface:.3f}, top {top:.3f} ({height * 100:.1f} cm tall), "
              f"middle {np.round(center, 3).tolist()}; pads to z {pad_z:.3f} (odom)")
    p = robot.proprio()
    to_root = _root_from_odom(p)
    F0 = np.array(p["flange_pose_root"])
    R = F0[:3, :3]
    if R[2, 2] > -0.9:
        return {"ok": False, "reason": "the hand isn't pointing down (start from the ready pose)"}
    grasp_pad = (to_root @ np.array([center[0], center[1], pad_z, 1.0]))[:3]
    F_grasp = np.eye(4)
    F_grasp[:3, :3] = R
    F_grasp[:3, 3] = grasp_pad - R[:, 2] * PAD
    F_above = F_grasp.copy()
    F_above[2, 3] += 0.10
    F_lift = F_grasp.copy()
    F_lift[2, 3] += 0.05
    _move(robot, F0, F_above, 0.0)
    _move(robot, F_above, F_grasp, 0.0, hold=15)  # 1 s: the arm settles on the grasp pose
    g = robot.gripper_grasp(force_n, 0.05)
    _move(robot, F_grasp, F_lift, 1.0, hold=15)
    w = float(robot.proprio()["gripper_width_m"])
    held = 0.004 < w < 0.075
    robot.log(f"after the lift: fingers {w * 1000:.1f} mm apart: {'held' if held else 'not held'}")
    return {"ok": held, "reason": "held" if held else ("closed on nothing" if w <= 0.004 else "didn't close"), "width_m": round(w, 4),
            "point_px": [round(u), round(v)], "grasp_pad_odom": [round(float(center[0]), 3), round(float(center[1]), 3), round(pad_z, 3)],
            "object_height_m": round(height, 3)}
