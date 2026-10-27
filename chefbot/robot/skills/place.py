"""place: set the held object down on a detected surface (MANIP_SPEC §5.5)."""

from __future__ import annotations

import threading

import numpy as np

from chefbot.interface.proposers import PlacementProposer
from chefbot.interface.types import Detection, HeldObject, MotionStatus
from chefbot.robot.arm.arm_controller import ArmResult
from chefbot.robot.safety import require_base_at_rest
from chefbot.robot.skills.common import ArmStack, distances, follow_path, plan_approach, shifted, to_ready


def run_place(
    stack: ArmStack,
    proposer: PlacementProposer | None,
    surface: Detection,
    held: HeldObject | None,
    preempt: threading.Event,
) -> tuple[ArmResult, HeldObject | None]:
    """Returns the result and what the gripper holds afterwards."""
    cfg, arm, gripper = stack.config, stack.arm, stack.gripper
    # Believed held, but the fingers are shut on nothing: it is gone.
    if held is None or not gripper.is_holding():
        return ArmResult(MotionStatus.NOT_HOLDING, "the gripper is not holding anything"), None
    moving = require_base_at_rest(stack.env)
    if not moving.ok:
        return moving, held
    placements = proposer.propose(surface, held) if proposer is not None else []
    if not placements:
        return ArmResult(MotionStatus.NO_PLACEMENT, f"no free spot for the {held.label} on {surface.label}"), held

    # What the camera sees now, less the object in the hand.
    world = stack.workspace([_held_box(stack, held)])
    plan = None
    for placement in sorted(placements, key=lambda p: -p.score):
        release = stack.to_arm(placement.tcp_pose_world)
        heights = distances(placement.preplace_offset, cfg.shorter_offsets)
        above = [shifted(release, [0.0, 0.0, h]) for h in heights]
        # Leave no higher than the approach came from: that much room is known to be free.
        ups = distances(min(cfg.retreat_height, placement.preplace_offset), cfg.shorter_offsets)
        plan = plan_approach(stack, world, above, release, [shifted(release, [0.0, 0.0, up]) for up in ups])
        if plan is not None:
            break
    if plan is None:
        where = " clear of what is around it" if world.checked else ""
        return ArmResult(
            MotionStatus.UNREACHABLE,
            f"the arm cannot reach a spot on {surface.label}{where} from here; align to it first",
        ), held

    for step in (
        lambda: to_ready(stack, world, preempt),
        lambda: follow_path(stack, plan.into, preempt),
        # Setting the object down: contact is expected.
        lambda: arm.follow(plan.line_in, preempt, expect_contact=True),
    ):
        result = step()
        if not result.ok:
            return result, held if gripper.is_holding() else None
    if not gripper.open(preempt):
        return ArmResult(MotionStatus.STOPPED, "motion was stopped"), None
    for step in (
        lambda: arm.follow(plan.line_out, preempt, expect_contact=True),
        lambda: follow_path(stack, plan.back, preempt),
    ):
        result = step()
        if not result.ok:
            return result, None
    if not gripper.is_open():
        return ArmResult(MotionStatus.ERROR, "the gripper did not open fully after the release"), None
    return ArmResult(MotionStatus.SUCCESS, f"placed the {held.label} on {surface.label}"), None


def _held_box(stack: ArmStack, held: HeldObject) -> np.ndarray | None:
    """World box (2, 3) around the held object where it hangs now, or None if unknown."""
    if held.box_in_tcp is None:
        return None
    tcp = stack.tcp_world()
    corners = np.asarray(held.box_in_tcp, dtype=float) @ tcp[:3, :3].T + tcp[:3, 3]
    return np.array([corners.min(axis=0), corners.max(axis=0)])
