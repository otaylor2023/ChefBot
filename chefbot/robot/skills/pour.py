"""pour: tip what the held container (a pan held by its handle, a bowl) holds
into or onto a target (a plate, another bowl).

The container's bowl goes beside the target's middle, level, then the wrist
turns it about its long axis (the handle's) so it tips over sideways about its
lip, which stays put just above the target's middle (as one pours from a pan);
held there while it all falls out, then back level and to `ready`. Where the bowl is in the hand comes from the depth cameras (the
container can hang in the grip at its own angle), else from its grasp-time box.
"""

from __future__ import annotations

import math
import threading

import numpy as np

from chefbot.interface.types import Detection, HeldObject, MotionStatus
from chefbot.robot.arm.arm_controller import ArmResult
from chefbot.robot.kinematics.collision import depth_points
from chefbot.robot.safety import require_base_at_rest
from chefbot.robot.skills.common import ArmStack, follow_path, to_ready
from chefbot.robot.skills.tool_use import _rot_axis, _rot_z, _top_under, _world_points, held_points

TARGET_RADIUS = 0.12  # m around the target's middle the bowl's bottom has to clear
TARGET_TALL = 0.25  # m: what is higher above the target's middle than this is not part of it
ABOVE = 0.04  # m: the bowl's bottom this far above the top of the target (food rolls if dropped from higher)
TILTS = (100.0, 85.0)  # deg about the long axis, most first (much more and the bowl swings into the forearm)
YAWS = 12  # headings of the container tried over the target
TILT_STEP = 10.0  # deg between IK samples while tipping
HOLD_S = 1.5  # s held tipped while it all falls out
WRIST_FOLD = (5, 0.5)  # (joint, rad): below this the wrist folds the hand into the forearm (Panda joint 6)
SPEED = 0.3  # of the arm's speed limits while tipping: the food slides out instead of being flung
LIP_SHORT = 0.04  # m: the lip stops this far short of the target's middle (what slides out carries on past it)


def run_pour(stack: ArmStack, target: Detection, held: HeldObject | None, preempt: threading.Event) -> ArmResult:
    if held is None:
        return ArmResult(MotionStatus.NOT_HOLDING, "the gripper is not holding anything")
    moving = require_base_at_rest(stack.env)
    if not moving.ok:
        return moving
    bowl = _bowl(stack, held)
    if bowl is None:
        return ArmResult(MotionStatus.UNSUPPORTED, f"cannot tell where the {held.label}'s bowl is in the gripper")
    middle, axis, radius, below = bowl
    goal = np.asarray(target.position_world, dtype=float)
    seen = stack.workspace()
    points = _world_points(stack, seen.points)
    points = points[points[:, 2] <= goal[2] + TARGET_TALL]  # not the wall cabinet over it
    top = _top_under(points, goal, radius=TARGET_RADIUS)  # a bowl's rim, not its floor
    top = float(goal[2]) if top is None else top
    world = stack.workspace(with_held=True)
    for level, point, line, angle in _pour_poses(stack, middle, axis, radius, below, goal, top):
        q_level = stack.ik.solve_pose(level, seed=stack.ready)
        if q_level is None or _folded(q_level) or not world.free(q_level):
            continue
        tip = _turn(stack, q_level, point, line, angle)
        if tip is None:
            continue
        path = world.path(stack.arm.current_target(), q_level)
        if path is None:
            continue
        result = follow_path(stack, path, preempt)
        if not result.ok:
            return result
        return _tip(stack, tip, preempt, held, target)
    return ArmResult(MotionStatus.UNREACHABLE, f"the arm cannot hold the {held.label} over the {target.label} from here; align to it first")


def _bowl(stack: ArmStack, held: HeldObject) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """In the TCP frame: the bowl's middle, the container's long axis (from the
    grip out to the bowl), the bowl's radius, and how far its bottom is below its middle."""
    points = held_points(stack, held, _raw_points(stack))
    if points is None:
        if held.box_in_tcp is None:
            return None
        points = np.asarray(held.box_in_tcp, dtype=float)
    center = points.mean(axis=0)
    axis = np.linalg.svd(points - center)[2][0]
    if axis @ center < 0:
        axis = -axis  # from the grip out
    along = points @ axis
    across = points - np.outer(along, axis)
    radius = float(np.percentile(np.linalg.norm(across - across.mean(axis=0), axis=1), 95))
    radius = min(radius, float(np.ptp(along)) / 2.0)
    middle = across.mean(axis=0) + (float(along.max()) - radius) * axis
    tcp = stack.tcp_world()
    height = points @ (tcp[:3, :3].T @ [0.0, 0.0, 1.0])  # world-up in the TCP frame
    below = float((middle @ (tcp[:3, :3].T @ [0.0, 0.0, 1.0])) - height.min())
    return middle, axis, radius, below


def _raw_points(stack: ArmStack) -> np.ndarray:
    """World points from the planning cameras, the arm and what it holds included."""
    try:
        images = [stack.env.get_camera_rgbd(camera) for camera in stack.config.planning_cameras]
    except (NotImplementedError, RuntimeError):
        return np.zeros((0, 3))
    return np.concatenate([depth_points(image) for image in images]) if images else np.zeros((0, 3))


def _pour_poses(stack, middle, axis, radius, below, goal, top):
    """Arm-frame (level TCP pose, point, axis, angle): the bowl level beside the
    target's middle, to be turned by `angle` about a line along its long axis
    through its lip (`point`, `axis`), over the target; by tilt, then the least
    turn of the container."""
    tcp = stack.tcp_world()
    ranked = []
    for t, tilt in enumerate(TILTS):
        for k in range(YAWS):
            yaw = 2.0 * math.pi * k / YAWS
            turn = min(yaw, 2.0 * math.pi - yaw)
            rotation = _rot_z(yaw) @ tcp[:3, :3]
            long = rotation @ axis
            long = np.array([long[0], long[1], 0.0])
            if np.linalg.norm(long) < 1e-6:
                continue
            long /= np.linalg.norm(long)
            for sign in (1.0, -1.0):
                side = sign * np.cross([0.0, 0.0, 1.0], long)  # the side the lip goes down on
                # The lip on that side stays put by the target's middle while the bowl turns over it.
                lip = np.array([*(goal[:2] - LIP_SHORT * side[:2]), top + ABOVE + below])
                level = np.eye(4)
                level[:3, :3] = rotation
                level[:3, 3] = lip - radius * side - rotation @ middle
                ranked.append((t, turn, len(ranked), level, lip, long, -sign * math.radians(tilt)))  # side down
    to_arm = np.linalg.inv(stack.frames.arm_base_in_world(stack.base_pose()))
    for *_, level, at, long, angle in sorted(ranked, key=lambda r: r[:3]):
        yield to_arm @ level, to_arm[:3, :3] @ at + to_arm[:3, 3], to_arm[:3, :3] @ long, angle


def _about(pose: np.ndarray, point: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """`pose` turned by `angle` about the line through `point` along `axis`."""
    turn = _rot_axis(axis, angle)
    out = np.eye(4)
    out[:3, :3] = turn @ pose[:3, :3]
    out[:3, 3] = point + turn @ (pose[:3, 3] - point)
    return out


def _turn(stack: ArmStack, q_level: np.ndarray, point: np.ndarray, axis: np.ndarray, angle: float) -> list[np.ndarray] | None:
    """Joint waypoints from level to tipped, the TCP turning about the line
    through `point` along `axis` (sampled every TILT_STEP degrees). None if any
    sample has no IK solution or the solution jumps."""
    level = stack.ik.fk(q_level)
    steps = max(2, math.ceil(abs(math.degrees(angle)) / TILT_STEP))
    waypoints = [np.asarray(q_level, dtype=float)]
    for i in range(1, steps + 1):
        q = stack.ik.solve_pose(_about(level, point, axis, angle * i / steps), seed=waypoints[-1])
        if q is None or np.abs(q - waypoints[-1]).max() > stack.config.max_joint_jump or _folded(q):
            return None
        waypoints.append(q)
    return waypoints


def _folded(q: np.ndarray) -> bool:
    joint, least = WRIST_FOLD
    return len(q) > joint and q[joint] < least


def _tip(stack: ArmStack, tip: list[np.ndarray], preempt, held: HeldObject, target: Detection) -> ArmResult:
    arm = stack.arm
    result = arm.follow(tip, preempt, expect_contact=True, speed=SPEED)
    if result.status == MotionStatus.TIMEOUT:
        # Stopped short (the bowl against something): tipped as far as it goes.
        arm.hold()
        tip = _upto(stack, tip)
    elif not result.ok:
        return result
    stack.env.wait_for_next_tick(HOLD_S)
    result = arm.follow([stack.arm.current_target(), *tip[-2::-1]], preempt, expect_contact=True, speed=SPEED)
    if not result.ok:
        return result
    home = to_ready(stack, stack.workspace(with_held=True), preempt)
    if not home.ok:
        return home
    return ArmResult(MotionStatus.SUCCESS, f"poured the {held.label} out over the {target.label}")


def _upto(stack: ArmStack, tip: list[np.ndarray]) -> list[np.ndarray]:
    """The waypoints of `tip` up to the one nearest where the arm is now."""
    q = np.asarray(stack.env.get_state().arm_q, dtype=float)
    nearest = int(np.argmin([np.abs(np.asarray(w) - q).max() for w in tip]))
    return tip[: nearest + 1]
