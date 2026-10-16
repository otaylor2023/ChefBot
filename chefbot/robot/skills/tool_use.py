"""Utensil skills: lower_into, stir, lift_out (a held spoon or spatula in a pan or bowl).

lower_into dips a long utensil (a spoon held across its middle) in tip first:
one end goes to the lowest gap in the container, just above what the depth
cameras see there (the food), the gripper out over the rim. Driving the tip
into the food would lever the spoon around in the grip.
A short one goes in like place (the placement proposer says where it would rest
in the container) but stops just short and keeps hold. stir traces small
circles at that depth; lift_out goes straight up.
"""

from __future__ import annotations

import math
import threading

import numpy as np

from chefbot.interface.proposers import PlacementProposer
from chefbot.interface.types import Detection, HeldObject, MotionStatus
from chefbot.robot.arm.arm_controller import OK, ArmResult
from chefbot.robot.safety import require_base_at_rest
from chefbot.robot.skills.common import ArmStack, distances, follow_path, plan_approach, shifted, to_ready
from chefbot.robot.skills.place import _held_box

HOVER = 0.02  # stop this far above where the utensil would rest
STIR_RADIUS = 0.03  # m
STIR_POINTS = 8  # waypoints per circle
STIR_LIFT = 0.015  # m: circle this far above where the utensil came to rest (through the food, not grinding on it)
LIFT = 0.15  # lift_out: straight up this far
LONG = 0.15  # m: a held object longer than this goes in tip first
TIP_TILTS = (20.0, 35.0, 50.0)  # deg below horizontal: the far end rides high, clear of the stove
TIP_YAWS = 12  # headings tried around the container
TIP_ABOVE = 0.04  # m: the tip goes this far above the highest point seen under it (over the food, not into it)
TIP_SPOTS = (0.0, 0.04, 0.07)  # m: rings of spots around the container's middle to put the tip in
TIP_SPOTS_TRIED = 4  # the lowest spots, planned in order
SEEN_MARGIN = 0.05  # m around the grasp-time box to look for the held object's points
MIN_SEEN = 30  # points: fewer and the grasp-time box is used
FAR_CLEAR = 0.05  # m: the far end stays this far above what is under it (the grip lets it sag ~15 deg)
TIP_APPROACH = 0.12  # m: straight down the last part
CONTENTS = (0.20, 0.08, 0.12)  # m: around the container's middle (xy, below, above): it and what is in it, contact expected


def run_lower_into(
    stack: ArmStack, proposer: PlacementProposer | None, target: Detection, held: HeldObject | None,
    preempt: threading.Event,
) -> ArmResult:  # fmt: skip
    cfg, arm = stack.config, stack.arm
    if held is None:
        return ArmResult(MotionStatus.NOT_HOLDING, "the gripper is not holding anything")
    moving = require_base_at_rest(stack.env)
    if not moving.ok:
        return moving
    middle = np.asarray(target.position_world, dtype=float)
    xy, below, up = CONTENTS
    contents = np.array([middle - [xy, xy, below], middle + [xy, xy, up]])
    seen = stack.workspace()
    world = stack.workspace([_held_box(stack, held), contents], with_held=True)
    plan = None
    points = _world_points(stack, seen.points)
    ends = _held_ends(stack, held, points)
    poses = [pose for spot in _tip_spots(points, middle) for pose in _tip_poses(stack, ends, spot, points)]
    for down in poses:
        above = [shifted(down, [0.0, 0.0, h]) for h in distances(TIP_APPROACH, cfg.shorter_offsets)]
        plan = plan_approach(stack, world, above, down, [shifted(down, [0.0, 0.0, LIFT])])
        if plan is not None:
            break
    if plan is None:
        placements = proposer.propose(target, held) if proposer is not None else []
        for placement in sorted(placements, key=lambda p: -p.score):
            down = shifted(stack.to_arm(placement.tcp_pose_world), [0.0, 0.0, HOVER])
            above = [shifted(down, [0.0, 0.0, h]) for h in distances(placement.preplace_offset, cfg.shorter_offsets)]
            plan = plan_approach(stack, world, above, down, [shifted(down, [0.0, 0.0, LIFT])])
            if plan is not None:
                break
    if plan is None:
        return ArmResult(MotionStatus.UNREACHABLE, f"the arm cannot reach into the {target.label} from here; align to it first")
    for step in (lambda: to_ready(stack, world, preempt), lambda: follow_path(stack, plan.into, preempt)):
        result = step()
        if not result.ok:
            return result
    result = arm.follow(plan.line_in, preempt, expect_contact=True)
    if result.status == MotionStatus.TIMEOUT:
        # Stopped short on the way down: the utensil is resting on what is in it.
        arm.hold()
        return ArmResult(MotionStatus.SUCCESS, f"the {held.label} is in the {target.label}, on what is in it")
    if not result.ok:
        return result
    return ArmResult(MotionStatus.SUCCESS, f"the {held.label} is in the {target.label}")


def _world_points(stack: ArmStack, points_arm: np.ndarray) -> np.ndarray:
    if not len(points_arm):
        return np.zeros((0, 3))
    arm_base = stack.frames.arm_base_in_world(stack.base_pose())
    return points_arm @ arm_base[:3, :3].T + arm_base[:3, 3]


def _top_under(points: np.ndarray, xy: np.ndarray, radius: float = 0.03) -> float | None:
    under = points[np.linalg.norm(points[:, :2] - xy[:2], axis=1) <= radius]
    return float(under[:, 2].max()) if len(under) else None


def _tip_spots(points: np.ndarray, middle: np.ndarray) -> list[np.ndarray]:
    """World points to put the utensil's tip at: spots around the container's
    middle, each just above the highest depth point under it, lowest first."""
    xy, below, up = CONTENTS
    near = points[(np.abs(points[:, :2] - middle[:2]).max(axis=1) <= xy)
                  & (points[:, 2] >= middle[2] - below) & (points[:, 2] <= middle[2] + up)]  # fmt: skip
    spots = []
    for radius in TIP_SPOTS:
        for k in range(1 if radius == 0.0 else 8):
            angle = 2.0 * math.pi * k / 8
            spot = middle + [radius * math.cos(angle), radius * math.sin(angle), 0.0]
            top = _top_under(near, spot)
            top = float(middle[2]) if top is None else top
            spots.append((top, radius, spot[:2]))
    spots.sort(key=lambda s: (round(s[0], 2), s[1]))
    return [np.array([*spot, top + TIP_ABOVE]) for top, _, spot in spots[:TIP_SPOTS_TRIED]]


def _held_ends(stack: ArmStack, held: HeldObject, points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """The two ends (TCP frame) of a long held object, as the depth cameras see
    it now: it hangs in the grip at its own angle, not as it lay when grasped.
    From the grasp-time box if too little of it is seen. None if short or unknown."""
    if held.box_in_tcp is None:
        return None
    corners = np.asarray(held.box_in_tcp, dtype=float)
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    axis = int(np.argmax(hi - lo))
    if hi[axis] - lo[axis] < LONG:
        return None
    a, b = (lo + hi) / 2.0, (lo + hi) / 2.0
    a[axis], b[axis] = hi[axis], lo[axis]
    mine = held_points(stack, held, points)
    if mine is None:
        return a, b
    middle = mine.mean(axis=0)
    direction = np.linalg.svd(mine - middle)[2][0]
    if direction @ (a - b) < 0:
        direction = -direction
    along = (mine - middle) @ direction
    return middle + along.max() * direction, middle + along.min() * direction


def held_points(stack: ArmStack, held: HeldObject, points: np.ndarray) -> np.ndarray | None:
    """The depth points (TCP frame) on the held object now, in and around its
    grasp-time box. None without a box or if too few are seen."""
    if held.box_in_tcp is None or not len(points):
        return None
    corners = np.asarray(held.box_in_tcp, dtype=float)
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    tcp = stack.tcp_world()
    local = (points - tcp[:3, 3]) @ tcp[:3, :3]
    mine = local[np.all((local >= lo - SEEN_MARGIN) & (local <= hi + SEEN_MARGIN), axis=1)]
    return mine if len(mine) >= MIN_SEEN else None


def _tip_poses(
    stack: ArmStack, held_ends: tuple[np.ndarray, np.ndarray] | None, middle: np.ndarray, points: np.ndarray
) -> list[np.ndarray]:
    """Arm-frame TCP poses that put an end of a long held object at `middle`,
    tilted down into it, with the other end clear of what the depth cameras see
    under it (the held object is not in the arm's collision model): by tilt,
    then the least turn of the wrist. None without `held_ends`."""
    if held_ends is None:
        return []
    tcp = stack.tcp_world()
    ends = [held_ends, held_ends[::-1]]
    ranked = []
    for t, tilt in enumerate(TIP_TILTS):
        for k in range(TIP_YAWS):
            yaw = 2.0 * math.pi * k / TIP_YAWS
            turn = min(yaw, 2.0 * math.pi - yaw)
            for tip, far in ends:
                rotation = _rot_z(yaw) @ tcp[:3, :3]
                out = rotation @ tip
                level = np.array([out[0], out[1], 0.0])
                if np.linalg.norm(level) < 1e-6:
                    continue
                # Tip down by `tilt` (from its pitch now): turn about the level axis across it.
                pitch = math.atan2(-out[2], float(np.linalg.norm(level)))
                across = np.cross(level / np.linalg.norm(level), [0.0, 0.0, 1.0])
                rotation = _rot_axis(across, pitch - math.radians(tilt)) @ rotation
                pose = np.eye(4)
                pose[:3, :3] = rotation
                pose[:3, 3] = middle - rotation @ tip
                far_end = pose[:3, 3] + rotation @ far
                top = _top_under(points, far_end)
                if top is not None and far_end[2] < top + FAR_CLEAR:
                    continue
                ranked.append((t, turn, len(ranked), stack.to_arm(pose)))
    return [pose for *_, pose in sorted(ranked, key=lambda r: r[:3])]


def _rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    k = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    skew = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * skew @ skew


def run_stir(stack: ArmStack, seconds: float, held: HeldObject | None, preempt: threading.Event) -> ArmResult:
    """Small circles around where the utensil is now, for `seconds` of run time."""
    if held is None:
        return ArmResult(MotionStatus.NOT_HOLDING, "the gripper is not holding anything")
    arm, env = stack.arm, stack.env
    q = rest = arm.current_target()
    up = arm.plan_line(q, shifted(stack.ik.fk(q), [0.0, 0.0, STIR_LIFT]))
    if up is not None:
        result = arm.follow(up, preempt, expect_contact=True)
        if not result.ok:
            return result
        q = up[-1]
    center = stack.ik.fk(q)
    start = env.get_state().time
    laps = 0
    while env.get_state().time - start < seconds:
        if preempt.is_set():
            return ArmResult(MotionStatus.STOPPED, "motion was stopped")
        waypoints = []
        for k in range(STIR_POINTS + 1):
            angle = 2.0 * math.pi * k / STIR_POINTS
            pose = shifted(center, [STIR_RADIUS * math.cos(angle), STIR_RADIUS * math.sin(angle), 0.0])
            line = arm.plan_line(q, pose)
            if line is None:
                return ArmResult(MotionStatus.UNREACHABLE, "cannot stir here: the arm is at the edge of its reach")
            waypoints += line[1:]
            q = line[-1]
        result = arm.follow(waypoints, preempt, expect_contact=True)
        if not result.ok:
            return result
        laps += 1
    back = arm.plan_line(q, stack.ik.fk(rest))  # down where it was: stirring again starts from the same place
    if back is not None:
        arm.follow(back, preempt, expect_contact=True)
    return ArmResult(MotionStatus.SUCCESS, f"stirred for {seconds:.0f} s ({laps} circles)")


def run_lift_out(stack: ArmStack, preempt: threading.Event) -> ArmResult:
    """Straight up, then back to `ready` around what the cameras see (the held
    utensil included), so the arm never swings it blind out of the pan."""
    arm = stack.arm
    q = arm.current_target()
    up = None
    for lift in (LIFT, LIFT * 2 / 3, LIFT / 3):
        up = arm.plan_line(q, shifted(stack.ik.fk(q), [0.0, 0.0, lift]))
        if up is not None:
            break
    if up is None:
        return ArmResult(MotionStatus.UNREACHABLE, "cannot lift straight up from here")
    result = arm.follow(up, preempt, expect_contact=True)
    if not result.ok:
        return result
    back = to_ready(stack, stack.workspace(with_held=True), preempt)
    return OK if back.ok else back
