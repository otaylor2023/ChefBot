"""Marks -> 6-DOF grasp (+ waypoints) (spec §13).

Marks come in per view as plot coordinates: top {name: (x, y)}, side
{name: (x, z)} (C-ind) or {name: z} (C-seq). Lifting takes x and y from the
top view and z from the side view.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from experiments.ortho_grasp.frames import MAX_OPEN, box_corners, box_distance, gripper_boxes, tcp_pose

PREGRASP_BACK = 0.10  # m along the approach, backed out
LIFT = 0.12  # m straight up
PARALLEL_COS = np.cos(np.radians(45.0))
FROM_BELOW_AZ = -0.2
GS_LARGE_DEG = 10.0
X_DISAGREE = 0.01
APPROACH_STEP_DEG = 10.0
CLEARANCE = 0.002  # m: the gripper's body and knuckles stay this far from the cloud
FINGER_TOLERANCE = 0.002  # m: a finger may press this far into the cloud (its marks sit on the surface)


@dataclass
class Resolved:
    ok: bool
    reason: str | None = None  # why not, when not ok
    points: dict[str, list[float]] = field(default_factory=dict)  # lifted marks (robot frame)
    pose: np.ndarray | None = None  # TCP (4x4, robot frame) at the grasp
    width: float | None = None  # m, finger opening at the grasp
    waypoints: list[dict] = field(default_factory=list)  # {"pose", "gripper": open|closed, "name"}
    flags: dict[str, bool] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def fingertips(self) -> tuple[np.ndarray, np.ndarray] | None:
        """A, B: the pads, at +-width/2 along the closing axis."""
        if self.pose is None:
            return None
        m, x = self.pose[:3, 3], self.pose[:3, 0]
        return m - x * self.width / 2, m + x * self.width / 2

    def to_json(self) -> dict:
        return {
            "ok": self.ok, "reason": self.reason, "points": self.points,
            "pose": None if self.pose is None else self.pose.tolist(), "width": self.width,
            "waypoints": [{**w, "pose": np.asarray(w["pose"]).tolist()} for w in self.waypoints],
            "flags": self.flags, "metrics": self.metrics,
        }  # fmt: skip


def lift(top: dict, side: dict, names: list[str]) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """3D points from per-view marks; per-point |x_top - x_side| (C-ind only)."""
    points, disagree = {}, {}
    for n in names:
        x, y = float(top[n][0]), float(top[n][1])
        s = side[n]
        if np.ndim(s) == 0:  # C-seq: z only, x is the top view's
            z = float(s)
        else:
            z = float(s[1])
            disagree[n] = abs(float(s[0]) - x)
        points[n] = np.array([x, y, z])
    return points, disagree


def _finish(pose: np.ndarray, width: float, flags: dict, metrics: dict, points: dict) -> Resolved:
    if width > MAX_OPEN:
        flags["width_over"] = True
        width = MAX_OPEN
    r = Resolved(True, None, {k: list(map(float, v)) for k, v in points.items()}, pose, float(width), [], flags, metrics)
    r.waypoints = t1_waypoints(pose)
    return r


def t1_waypoints(pose: np.ndarray) -> list[dict]:
    pre = pose.copy()
    pre[:3, 3] -= PREGRASP_BACK * pose[:3, 2]
    lift_ = pose.copy()
    lift_[2, 3] += LIFT
    return [{"name": "pre-grasp", "pose": pre, "gripper": "open"}, {"name": "grasp", "pose": pose, "gripper": "closed"},
            {"name": "lift", "pose": lift_, "gripper": "closed"}]  # fmt: skip


def resolve_p1(points: dict[str, np.ndarray], disagree: dict[str, float]) -> Resolved:
    A, B, F = points["A"], points["B"], points["F"]
    flags = {"x_disagree": any(d > X_DISAGREE for d in disagree.values())}
    metrics = {f"x_disagree_{k}": float(v) for k, v in disagree.items()}
    M = (A + B) / 2
    if np.linalg.norm(F - M) < 1e-6 or np.linalg.norm(B - A) < 1e-6:
        return Resolved(False, "degenerate", {k: list(map(float, v)) for k, v in points.items()}, flags=flags, metrics=metrics)
    a = (F - M) / np.linalg.norm(F - M)  # toward the flange
    c0 = (B - A) / np.linalg.norm(B - A)
    gs = float(np.degrees(np.arcsin(min(1.0, abs(c0 @ a)))))
    metrics["gs_angle_deg"] = gs
    flags["gs_large"] = gs > GS_LARGE_DEG
    flags["parallel"] = abs(c0 @ a) > PARALLEL_COS
    flags["from_below"] = bool(a[2] < FROM_BELOW_AZ)
    lifted = {k: list(map(float, v)) for k, v in points.items()}
    if flags["parallel"]:
        return Resolved(False, "parallel", lifted, flags=flags, metrics=metrics)
    if flags["from_below"]:
        return Resolved(False, "from_below", lifted, flags=flags, metrics=metrics)
    metrics["flange_distance"] = float(np.linalg.norm(F - M))
    pose = tcp_pose(M, -a, c0)
    return _finish(pose, float(np.linalg.norm(B - A)), flags, metrics, points)


def approach_candidates(closing: np.ndarray) -> list[np.ndarray]:
    """Unit directions perpendicular to `closing`, every APPROACH_STEP_DEG (toward the flange)."""
    c = closing / np.linalg.norm(closing)
    e1 = np.cross(c, [0.0, 0.0, 1.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(c, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(c, e1)
    return [np.cos(t) * e1 + np.sin(t) * e2 for t in np.radians(np.arange(0.0, 360.0, APPROACH_STEP_DEG))]


def gripper_clear(pose: np.ndarray, width: float, cloud: np.ndarray, counter_z: float, back: float = PREGRASP_BACK,
                  step: float = 0.01, touch_last: float = 0.0) -> bool:  # fmt: skip
    """The open gripper clear of the cloud and above the counter at `pose` and
    along the straight approach from `back` m behind it. The fingers may touch
    the cloud within the last `touch_last` m."""
    for d in np.arange(0.0, back + 1e-9, step):
        p = pose.copy()
        p[:3, 3] -= d * pose[:3, 2]
        boxes = gripper_boxes(p, width)
        for i, box in enumerate(boxes):
            fingers = i >= 2
            if fingers and d < touch_last:
                continue
            if fingers:  # a finger may touch the cloud (marks on the surface); only cutting into it counts
                box = (box[0], box[1], np.maximum(box[2] - FINGER_TOLERANCE, 1e-4))
                if len(cloud) and np.any(box_distance(cloud, box) <= 0.0):
                    return False
            elif len(cloud) and np.any(box_distance(cloud, box) <= CLEARANCE):
                return False
            if box_corners(box)[:, 2].min() < counter_z - 1e-4:
                return False
    return True


def resolve_p2(points: dict[str, np.ndarray], disagree: dict[str, float], cloud: np.ndarray, counter_z: float) -> Resolved:
    A, B = points["A"], points["B"]
    flags = {"x_disagree": any(d > X_DISAGREE for d in disagree.values())}
    metrics = {f"x_disagree_{k}": float(v) for k, v in disagree.items()}
    lifted = {k: list(map(float, v)) for k, v in points.items()}
    if np.linalg.norm(B - A) < 1e-6:
        return Resolved(False, "degenerate", lifted, flags=flags, metrics=metrics)
    M, c = (A + B) / 2, (B - A) / np.linalg.norm(B - A)
    width = min(float(np.linalg.norm(B - A)), MAX_OPEN)
    valid = []
    for a in approach_candidates(c):
        pose = tcp_pose(M, -a, c)
        if gripper_clear(pose, width, cloud, counter_z):
            valid.append((float(a[2]), -float(a[0]), a))  # straight down first (flange up), then from the robot's side
    if not valid:
        return Resolved(False, "no_approach", lifted, flags=flags, metrics=metrics)
    valid.sort(key=lambda v: (-round(v[0], 6), -round(v[1], 6)))
    a = valid[0][2]
    metrics["approach_tilt_deg"] = float(np.degrees(np.arccos(np.clip(a[2], -1, 1))))
    return _finish(tcp_pose(M, -a, c), float(np.linalg.norm(B - A)), flags, metrics, points)


def resolve_pose(position, rotation: np.ndarray, width: float) -> Resolved:
    """A pose given directly (B0, references)."""
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, position
    return _finish(pose, float(width), {}, {}, {"M": np.asarray(position, dtype=float)})


def quat_to_matrix(wxyz) -> np.ndarray:
    w, x, y, z = np.asarray(wxyz, dtype=float) / np.linalg.norm(wxyz)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])  # fmt: skip


def resolve_waypoints(per_waypoint: list[Resolved], grippers: list[str]) -> Resolved:
    """T3: one Resolved per waypoint (pre-grasp, grasp, lift); the grasp one is the grasp."""
    names = ["pre-grasp", "grasp", "lift"]
    for i, r in enumerate(per_waypoint):
        if not r.ok:
            return Resolved(False, f"waypoint {i + 1}: {r.reason}", r.points, flags=r.flags, metrics=r.metrics)
    grasp = per_waypoint[1]
    out = Resolved(True, None, {f"{k}{i + 1}": v for i, r in enumerate(per_waypoint) for k, v in r.points.items()},
                   grasp.pose, grasp.width, [], {}, {})  # fmt: skip
    for r in per_waypoint:
        for k, v in r.flags.items():
            out.flags[k] = out.flags.get(k, False) or v
    out.metrics = {f"{k}_wp{i + 1}": v for i, r in enumerate(per_waypoint) for k, v in r.metrics.items()}
    out.waypoints = [{"name": names[i], "pose": r.pose, "gripper": grippers[i]} for i, r in enumerate(per_waypoint)]
    return out


def heuristic(box: np.ndarray, rim_z: float) -> Resolved:
    """R-heur: top-down at the cup's middle at rim height, fingers along x, open 85 mm."""
    middle = (box[0] + box[1]) / 2
    return resolve_pose([middle[0], middle[1], rim_z], tcp_pose(np.zeros(3), [0, 0, -1], [1, 0, 0])[:3, :3], MAX_OPEN)



RIM_DEPTH = 0.015  # m: a rim grasp's grip point below the rim (as the repo's proposer)
RIM_ANGLES_DEG = (0, 25, -25, 50, -50, 80, -80, 115, -115, 150, -150, 180)  # around the rim, from the robot's side


def rim_grasps(full: np.ndarray, rim_z: float) -> list[Resolved]:
    """Top-down grasps astride one wall at the rim, from the ground-truth cloud,
    nearest the robot (at the origin) first. Like the repo proposer's rim grasp,
    but measuring the wall locally, so it works on cups narrower than the gripper."""
    lo, hi = full.min(axis=0), full.max(axis=0)
    middle = (lo[:2] + hi[:2]) / 2
    toward_robot = np.arctan2(-middle[1], -middle[0])
    near_top = full[full[:, 2] >= rim_z - RIM_DEPTH - 0.019]
    out = []
    for offset in RIM_ANGLES_DEG:
        t = toward_robot + np.radians(offset)
        radial, tangent = np.array([np.cos(t), np.sin(t)]), np.array([-np.sin(t), np.cos(t)])
        rel = near_top[:, :2] - middle
        band = near_top[(np.abs(rel @ tangent) <= 0.006) & (rel @ radial > 0)]
        if len(band) == 0:
            continue
        r = (band[:, :2] - middle) @ radial
        outer = float(r.max())
        wall = r[r > outer - 0.015]  # this wall only (the far one is a diameter away)
        inner = float(wall.min())
        center = middle + (outer + inner) / 2 * radial
        pose = tcp_pose(np.array([center[0], center[1], rim_z - RIM_DEPTH]), [0, 0, -1], [radial[0], radial[1], 0.0])
        out.append(_finish(pose, min(MAX_OPEN, outer - inner + 0.02), {}, {"wall_thickness": outer - inner}, {"M": pose[:3, 3]}))
    return out


def correct_flange(A, B, F, cloud: np.ndarray, counter_z: float, reachable) -> Resolved:
    """Keep the fingers (A, B: midpoint, closing line, width) and move only the
    flange point F until the grasp is valid: first F's direction made
    perpendicular to the closing line (the least change), then other approaches
    around the closing line, least turned from the model's first. Valid: not from
    below, the gripper clear of `cloud` (what the model saw) and the counter along
    the approach, and `reachable(pose)` (IK at pre-grasp and grasp)."""
    A, B, F = (np.asarray(p, dtype=float) for p in (A, B, F))
    lifted = {"A": list(map(float, A)), "B": list(map(float, B)), "F_model": list(map(float, F))}
    if np.linalg.norm(B - A) < 1e-6:
        return Resolved(False, "degenerate", lifted)
    M, c = (A + B) / 2, (B - A) / np.linalg.norm(B - A)
    width = float(np.linalg.norm(B - A))
    a_model = F - M
    a_model = a_model / np.linalg.norm(a_model) if np.linalg.norm(a_model) > 1e-9 else np.array([0.0, 0.0, 1.0])
    a0 = a_model - (a_model @ c) * c
    a0 = a0 / np.linalg.norm(a0) if np.linalg.norm(a0) > 1e-6 else np.array([0.0, 0.0, 1.0]) - c[2] * c
    a0 /= np.linalg.norm(a0)
    candidates = sorted(approach_candidates(c), key=lambda a: -float(a @ a0))
    for a in [a0] + candidates:
        if a[2] < FROM_BELOW_AZ:
            continue
        pose = tcp_pose(M, -a, c)
        if not gripper_clear(pose, min(MAX_OPEN, min(width, MAX_OPEN) + 0.01), cloud, counter_z, touch_last=0.01):
            continue
        r = _finish(pose, width, {}, {}, {**{k: np.array(v) for k, v in lifted.items()}, "F": M + FLANGE_DIST * a})
        if not reachable(r):
            continue
        r.metrics["flange_correction_deg"] = float(np.degrees(np.arccos(np.clip(a @ a_model, -1.0, 1.0))))
        r.metrics["gs_angle_model_deg"] = float(np.degrees(np.arcsin(min(1.0, abs(a_model @ c)))))
        return r
    return Resolved(False, "no_valid_flange", lifted)


FLANGE_DIST = 0.156
