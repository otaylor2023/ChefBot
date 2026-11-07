"""Hand-eye check: find a small object with the wrist camera and put the open hand exactly over it, the fingertips' ends
level with its top and the object centred between the pads, so a look shows whether the wrist camera's calibration
(environments/real/config.yaml cameras.wrist) and the hand agree. Nothing touches the object: the fingers stop
`clearance_m` above its top. The pole camera's estimate of the same object is logged too (the two calibrations
agreeing), when it sees it.

1. From where the arm is (run_real --ready): the wrist camera finds `object` (Grounding DINO, then SAM's mask), its
   points from depth give its centre and top (the root frame).
2. The hand straight down over it, 10 cm up, the gripper opened to `open_m` (1 inch).
3. Down until the fingertips' ends are `clearance_m` over the top; a wrist and a pole image there; held `hold_s`.
4. Back up the same way.
"""

import numpy as np
from scipy.spatial.transform import Rotation

TIPS = 0.134  # m: the printed fingers' ends out from the flange (measured 2026-11-06; robot.yaml body hand_m 0.135)
HOVER = 0.10  # m above the final pose, first
MOVE_CAP = 300  # control steps a move may take while still progressing


def _planar(x, y, yaw):
    T = np.eye(4)
    c, s = np.cos(yaw), np.sin(yaw)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:2, 3] = [x, y]
    return T


def _row(F, g):
    x, y, z, w = Rotation.from_matrix(F[:3, :3]).as_quat()
    return [*F[:3, 3].tolist(), w, x, y, z, g]


def _points(cam, base_pose):
    """The camera's depth as points (H, W, 3) in the root frame."""
    d = np.nan_to_num(np.asarray(cam["depth"], dtype=float))
    v, u = np.indices(d.shape)
    K = np.asarray(cam["K"])
    q = np.stack([(u - K[0, 2]) * d / K[0, 0], (v - K[1, 2]) * d / K[1, 1], d], -1)
    T = np.linalg.inv(_planar(*base_pose)) @ np.asarray(cam["cam_to_odom"])
    return q @ T[:3, :3].T + T[:3, 3], d > 0


def _find(robot, cam, base_pose, object, name):
    """(centre xy, top z, n points, box) of `object` in one camera, or None."""
    if cam.get("depth") is None:
        return None
    ds = sorted(robot.detect(cam["rgb"], [object], model="grounding-dino", threshold=0.2), key=lambda d: -d["score"])
    robot.log(f"{name} detections", ds[:3])
    if not ds:
        return None
    mask = robot.segment(cam["rgb"], boxes=[ds[0]["box"]])[0]
    pts, valid = _points(cam, base_pose)
    sel = pts[mask & valid]
    if len(sel) < 20:
        robot.log(f"{name}: only {len(sel)} depth points on the mask")
        return None
    sel = sel[np.abs(sel[:, 2] - np.median(sel[:, 2])) < 0.05]  # depth outliers off the edges
    overlay = cam["rgb"].copy()
    overlay[mask] = (0.5 * overlay[mask] + [0, 127, 0]).astype(np.uint8)
    robot.save_image(f"{name}_found", overlay)
    return np.median(sel[:, :2], axis=0), float(np.percentile(sel[:, 2], 95)), len(sel), ds[0]["box"]


def _move(robot, F, g, n=40):
    """F (root frame) until within 3 mm and 3 deg, or no longer progressing (up to MOVE_CAP steps)."""
    r = robot.act_arm([_row(F, g)] * 15)
    best, stalled, used = np.inf, 0, 15
    while used < max(n, MOVE_CAP):
        e = r["steps"][-1]
        if e["pos_error_m"] < 0.003 and e["rot_error_rad"] < 0.05:
            break
        score = e["pos_error_m"] + 0.1 * e["rot_error_rad"]
        stalled = stalled + 1 if score > best - 0.0007 else 0
        best = min(best, score)
        if stalled >= 4:
            break
        r = robot.act_arm([_row(F, g)] * 5)
        used += 5
    e = r["steps"][-1]
    robot.log("move", np.round(F[:3, 3], 4).tolist(), "error", round(e["pos_error_m"] * 1000, 1), "mm",
              round(np.degrees(e["rot_error_rad"]), 1), "deg")  # fmt: skip
    return r["proprio"], e


def run(robot, object: str = "purple chapstick cap", open_m: float = 0.0254, clearance_m: float = 0.005, hold_s: float = 5.0):
    seen = robot.observe(["wrist", "pole"])
    base = seen["proprio"]["base_pose_odom"]
    for name, cam in seen["cameras"].items():
        robot.save_image(f"{name}_start", cam["rgb"])
    wrist = _find(robot, seen["cameras"]["wrist"], base, object, "wrist")
    if wrist is None:
        return {"ok": False, "reason": f"the wrist camera doesn't see a {object} (with depth)"}
    xy, top, n, box = wrist
    robot.log(f"wrist: {object} at x {xy[0]:.4f} y {xy[1]:.4f}, top z {top:.4f} (root frame, m), {n} points")
    out = {"wrist_xy": np.round(xy, 4).tolist(), "wrist_top_z": round(top, 4)}
    pole = _find(robot, seen["cameras"]["pole"], base, object, "pole") if "pole" in seen["cameras"] else None
    if pole is not None:
        d = np.r_[pole[0] - xy, pole[1] - top] * 1000
        robot.log(f"pole: at x {pole[0][0]:.4f} y {pole[0][1]:.4f}, top z {pole[1]:.4f}: pole - wrist {np.round(d, 1).tolist()} mm")
        out |= {"pole_xy": np.round(pole[0], 4).tolist(), "pole_top_z": round(pole[1], 4), "pole_minus_wrist_mm": np.round(d, 1).tolist()}

    # The hand straight down, keeping the heading it has now (its x on the floor), so the wrist turns little.
    F0 = np.array(robot.proprio()["flange_pose_root"])
    z = np.array([0.0, 0.0, -1.0])
    x = F0[:3, 0] - (F0[:3, 0] @ z) * z
    x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([1.0, 0.0, 0.0])
    R = np.column_stack([x, np.cross(z, x), z])
    F = np.eye(4)
    F[:3, :3] = R
    F[:3, 3] = [xy[0], xy[1], top + clearance_m + TIPS]  # the fingertips' ends clearance_m over the top
    H = F.copy()
    H[2, 3] += HOVER

    _move(robot, H, 0.0, 60)
    robot.gripper_move(open_m, 0.05)
    robot.act_arm([_row(H, 0.0)] * 20)  # the fingers get there
    p, e = _move(robot, F, 0.0, 60)
    width = p["gripper_width_m"]
    there = robot.observe(["wrist", "pole"])
    for name, cam in there["cameras"].items():
        robot.save_image(f"{name}_aligned", cam["rgb"])
    A = np.array(there["proprio"]["flange_pose_root"])
    tips = A[:3, 3] + TIPS * A[:3, 2]
    robot.log(f"aligned: the fingertips' ends at {np.round(tips, 4).tolist()} (where the wrist put the {object}'s top "
              f"centre + {clearance_m * 1000:.0f} mm), gripper {width * 1000:.1f} mm open; holding {hold_s:.0f} s")  # fmt: skip
    robot.act_arm([_row(F, 0.0)] * int(hold_s * 15))
    _move(robot, H, 0.0, 60)
    out |= {"ok": e["pos_error_m"] < 0.005, "reached_within_mm": round(e["pos_error_m"] * 1000, 1),
            "fingertips": np.round(tips, 4).tolist(), "gripper_open_mm": round(width * 1000, 1)}  # fmt: skip
    return out
