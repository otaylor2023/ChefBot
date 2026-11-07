"""Place what the hand holds into a container (a bowl) in view: find the container's rim with the wrist and pole
cameras (Grounding DINO's box, SAM's mask, the depth under it), look again from close above if it is far, hold the
hand over the rim's centre, lower until the held thing is just above the rim, open, and back off up.

The arm only: the base stays where it is (the agent parks it first). Positions are in the odom frame the cameras
and the flange share (cam_to_odom, flange_pose_odom); arm_only.py puts them in the arm's root frame.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from arm_only import ArmOnly

PADS = 0.130  # the printed pads' middle out from the flange (fingertips 0.148 by touch, calibration/tcp.py)
CLOSE_VIEW_M = 0.45  # beyond this from the wrist camera its depth flattens things: look again from closer
VIEW_M = 0.32  # the close view: the wrist camera this far above the rim's centre (its floor within CLOSE_VIEW_M)
HOVER_M = 0.12  # the pads' middle this far above the rim before lowering
HANG_M = 0.04  # how far the held thing reaches below the pads' middle (a tomato's half, ~4 cm)
MOVE_CAP = 300
GRASPED_MIN_M = 0.004  # fingers closed further than this: nothing between them


def cloud(cam):
    d = cam["depth"]
    v, u = np.indices(d.shape)
    K = cam["K"]
    q = np.stack([(u - K[0, 2]) * d / K[0, 0], (v - K[1, 2]) * d / K[1, 1], d], -1)
    T = cam["cam_to_odom"]
    return q @ T[:3, :3].T + T[:3, 3]


def row(F, g):
    x, y, z, w = Rotation.from_matrix(F[:3, :3]).as_quat()
    return [*F[:3, 3], w, x, y, z, g, 0.0, 0.0, 0.0]  # the base columns: arm_only.py drops them


def move(robot, F, g, n=30):
    """The flange to F (odom), until it is there, stops getting closer, or MOVE_CAP steps. Returns the proprio."""
    r = robot.act([row(F, g)] * min(n, 15))
    left, prev, stalled = MOVE_CAP - min(n, 15), None, 0
    while left > 0:
        e = r["steps"][-1]
        if e["pos_error_m"] < 0.003 and e["rot_error_rad"] < 0.06:
            break
        score = e["pos_error_m"] + 0.1 * e["rot_error_rad"]
        stalled = stalled + 1 if prev is not None and prev - score < 0.0007 else 0
        if stalled >= 3:
            break
        prev = score
        r = robot.act([row(F, g)] * min(5, left))
        left -= 5
    p = r["proprio"]
    robot.log("move", np.round(F[:3, 3], 4).tolist(), "err", r["steps"][-1],
              "at", np.round(np.array(p["flange_pose_odom"])[:3, 3], 4).tolist())  # fmt: skip
    return p


def hand_down(F):
    """F's position with the hand pointing straight down, turned about the vertical as F's is."""
    x = np.array(F[:3, 0], dtype=float)
    x[2] = 0.0
    x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([1.0, 0.0, 0.0])
    z = np.array([0.0, 0.0, -1.0])
    G = np.array(F, dtype=float).copy()
    G[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    return G


def find_rim(robot, target, cameras=("wrist", "pole")):
    """The container's rim in each camera that sees it: {camera, center (odom xyz at the rim's height), top, radius,
    distance, score}, best first (the wrist's when it is close enough to measure, else the higher score)."""
    obs = robot.observe(list(cameras))
    hand = np.array(obs["proprio"]["flange_pose_odom"])[:3, 3]
    found = []
    for name in cameras:
        cam = obs["cameras"].get(name)
        if cam is None or cam["depth"] is None:
            continue
        robot.save_image(f"look_{name}", cam["rgb"])
        ds = robot.detect(cam["rgb"], [target], model="grounding-dino", threshold=0.25)
        ds.sort(key=lambda d: d["score"], reverse=True)
        robot.log(name, "detections", ds[:3])
        if not ds:
            continue
        mask = robot.segment(cam["rgb"], boxes=[ds[0]["box"]])[0]
        robot.save_image(f"mask_{name}", mask.astype(np.uint8) * 255)
        xyz = cloud(cam)
        valid = (cam["depth"] > 0.05) & (cam["depth"] < 2.5)
        pts = xyz[mask & valid]
        pts = pts[np.linalg.norm(pts - hand, axis=1) > 0.12]  # not what the hand holds, nor the fingers
        if len(pts) < 50:
            robot.log(name, "too few depth points on the", target, len(pts))
            continue
        top = float(np.percentile(pts[:, 2], 97))
        rim = pts[pts[:, 2] > top - 0.015]
        center, radius = np.median(pts[:, :2], axis=0), None
        if len(rim) >= 30:  # a circle through the rim (Kasa): x^2 + y^2 = 2ax + 2by + c
            A = np.column_stack([2 * rim[:, :2], np.ones(len(rim))])
            a, b, c = np.linalg.lstsq(A, (rim[:, :2] ** 2).sum(1), rcond=None)[0]
            r = float(np.sqrt(max(c + a * a + b * b, 0.0)))
            resid = float(np.sqrt(np.mean((np.linalg.norm(rim[:, :2] - [a, b], axis=1) - r) ** 2)))
            robot.log(name, "rim circle", [float(a), float(b)], "radius", r, "residual", resid)
            if 0.04 < r < 0.20 and resid < 0.25 * r and np.linalg.norm(np.array([a, b]) - center) < 0.08:
                center, radius = np.array([a, b]), r
        dist = float(np.median(cam["depth"][mask & valid]))
        found.append({"camera": name, "center": [float(center[0]), float(center[1]), top], "top": top,
                      "radius": radius, "distance": dist, "score": float(ds[0]["score"])})  # fmt: skip
    found.sort(key=lambda f: (not (f["camera"] == "wrist" and f["distance"] <= CLOSE_VIEW_M), -f["score"]))
    robot.log("rims", found)
    return found, obs


def run(robot, target="bowl", object="the held object", where="on the counter"):
    robot = ArmOnly(robot)
    p = robot.proprio()
    if p["gripper_width_m"] < GRASPED_MIN_M:
        return {"ok": False, "reason": f"the hand holds nothing (fingers {p['gripper_width_m'] * 1000:.0f} mm apart)"}
    found, obs = find_rim(robot, target)
    if not found:
        return {"ok": False, "reason": f"no {target} seen by the wrist or pole camera"}
    best = found[0]
    if not (best["camera"] == "wrist" and best["distance"] <= CLOSE_VIEW_M):
        c = np.array(best["center"])  # a close view from above: the wrist camera VIEW_M over the rim's centre
        F = np.array(obs["proprio"]["flange_pose_odom"])
        cam_in_flange = np.linalg.inv(F) @ obs["cameras"]["wrist"]["cam_to_odom"]
        V = hand_down(F)
        V[:3, 3] = 0.0
        C = V @ cam_in_flange  # the camera's pose with the hand down, the flange at the origin
        V[:3, 3] = c + np.array([0.0, 0.0, VIEW_M]) - C[:3, 3]
        robot.log(f"the {target} from the {best['camera']} camera at {best['distance']:.2f} m: a close view first")
        lifted = np.array(robot.proprio()["flange_pose_odom"])
        lifted[2, 3] = max(lifted[2, 3], V[2, 3])
        move(robot, lifted, 1, 30)
        q = move(robot, V, 1, 45)
        if np.linalg.norm(np.array(q["flange_pose_odom"])[:3, 3] - V[:3, 3]) < 0.05:
            close, _ = find_rim(robot, target, ("wrist",))
            if close:
                best = close[0]
        else:
            robot.log("the close view not reached: placing by the first view")
    c = np.array(best["center"])
    F = hand_down(np.array(robot.proprio()["flange_pose_odom"]))
    H = F.copy()
    H[:3, 3] = [c[0], c[1], best["top"] + PADS + HOVER_M]
    S = np.array(robot.proprio()["flange_pose_odom"])
    S[2, 3] = max(S[2, 3], H[2, 3])
    move(robot, S, 1, 30)
    p = move(robot, H, 1, 45)
    off = float(np.linalg.norm(np.array(p["flange_pose_odom"])[:2, 3] - H[:2, 3]))
    if off > 0.03:
        return {"ok": False, "reason": f"could not get over the {target} ({off * 100:.1f} cm off): out of reach from here",
                "rim": best}  # fmt: skip
    for k, cam in robot.observe(["wrist", "pole"])["cameras"].items():
        robot.save_image(f"over_{k}", cam["rgb"])
    L = H.copy()
    L[2, 3] = best["top"] + PADS + HANG_M + 0.02  # the held thing's bottom ~2 cm above the rim
    p = move(robot, L, 1, 30)
    robot.gripper_move(0.08, 0.05)
    p = robot.act_base([[0.0, 0.0, 0.0]] * 20)["proprio"]  # the hand opens while the arm holds
    for k, cam in robot.observe(["wrist", "pole"])["cameras"].items():
        robot.save_image(f"released_{k}", cam["rgb"])
    U = np.array(p["flange_pose_odom"])
    U[2, 3] += 0.15
    move(robot, U, 0, 30)
    return {"ok": True, "reason": f"released {object} over the {target}'s centre, {L[2, 3] - PADS - best['top']:.2f} m above its rim",
            "rim": best}  # fmt: skip
