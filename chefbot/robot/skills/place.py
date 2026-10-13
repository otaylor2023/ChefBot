"""place: set the held object down on a detected surface (MANIP_SPEC §5.5)."""

from __future__ import annotations

import threading

from chefbot.interface.proposers import PlacementProposer
from chefbot.interface.types import Detection, HeldObject, MotionStatus
from chefbot.robot.arm.arm_controller import ArmResult
from chefbot.robot.safety import require_base_at_rest
from chefbot.robot.skills.common import ArmStack, plan_approach, shifted


def run_place(
    stack: ArmStack,
    proposer: PlacementProposer | None,
    surface: Detection,
    held: HeldObject | None,
    preempt: threading.Event,
) -> tuple[ArmResult, HeldObject | None]:
    """Returns the result and what the gripper holds afterwards."""
    cfg, arm, gripper = stack.config, stack.arm, stack.gripper
    if held is None:
        return ArmResult(MotionStatus.NOT_HOLDING, "the gripper is not holding anything"), None
    moving = require_base_at_rest(stack.env)
    if not moving.ok:
        return moving, held
    placements = proposer.propose(surface, held) if proposer is not None else []
    if not placements:
        return ArmResult(MotionStatus.NO_PLACEMENT, f"no free spot for the {held.label} on {surface.label}"), held

    plan = None
    for placement in sorted(placements, key=lambda p: -p.score):
        release = stack.to_arm(placement.tcp_pose_world)
        above = shifted(release, [0.0, 0.0, placement.preplace_offset])
        # Leave no higher than the approach came from: that much room is known to be free.
        up = min(cfg.retreat_height, placement.preplace_offset)
        plan = plan_approach(stack, above, release, shifted(release, [0.0, 0.0, up]))
        if plan is not None:
            break
    if plan is None:
        return ArmResult(
            MotionStatus.UNREACHABLE, f"the arm cannot reach a spot on {surface.label} from here; align to it first"
        ), held

    for step in (
        lambda: arm.move_joints(stack.ready, preempt),
        lambda: arm.move_joints(plan.q_approach, preempt),
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
        lambda: arm.move_joints(stack.ready, preempt),
    ):
        result = step()
        if not result.ok:
            return result, None
    if not gripper.is_open():
        return ArmResult(MotionStatus.ERROR, "the gripper did not open fully after the release"), None
    return ArmResult(MotionStatus.SUCCESS, f"placed the {held.label} on {surface.label}"), None
