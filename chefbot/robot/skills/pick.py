"""pick: grasp a detected object and lift it (MANIP_SPEC §5.4)."""

from __future__ import annotations

import threading

import numpy as np

from chefbot.interface.proposers import GraspProposer
from chefbot.interface.types import Detection, HeldObject, MotionStatus
from chefbot.robot.arm.arm_controller import ArmResult
from chefbot.robot.safety import require_base_at_rest
from chefbot.robot.skills.common import ArmStack, plan_approach, shifted


def run_pick(
    stack: ArmStack,
    proposer: GraspProposer | None,
    target: Detection,
    held: HeldObject | None,
    preempt: threading.Event,
) -> tuple[ArmResult, HeldObject | None]:
    """Returns the result and what the gripper holds afterwards."""
    cfg, arm, gripper = stack.config, stack.arm, stack.gripper
    if held is not None:
        return ArmResult(MotionStatus.ALREADY_HOLDING, f"already holding a {held.label}"), held
    moving = require_base_at_rest(stack.env)
    if not moving.ok:
        return moving, None
    grasps = proposer.propose(target) if proposer is not None else []
    if not grasps:
        return ArmResult(MotionStatus.NO_GRASP, f"no graspable object found for {target.label}"), None

    # The best-scored grasp whose approach, grasp and lift all have joint solutions.
    plan = None
    to_arm = np.linalg.inv(stack.frames.arm_base_in_world(stack.base_pose()))[:3, :3]
    for grasp in sorted(grasps, key=lambda g: -g.score):
        at_grasp = stack.to_arm(grasp.tcp_pose_world)
        pregrasp = shifted(at_grasp, -grasp.pregrasp_offset * at_grasp[:3, 2])
        lift = [0.0, 0.0, cfg.lift_height] if grasp.retreat is None else to_arm @ grasp.retreat
        plan = plan_approach(stack, pregrasp, at_grasp, shifted(at_grasp, lift))
        if plan is not None:
            break
    if plan is None:
        return ArmResult(
            MotionStatus.UNREACHABLE, f"the arm cannot reach any grasp on the {target.label} from here; align to it first"
        ), None

    for step in (
        lambda: arm.move_joints(stack.ready, preempt),
        lambda: arm.move_joints(plan.q_approach, preempt),
    ):
        result = step()
        if not result.ok:
            return result, None
    if not gripper.open(preempt):
        return _stopped(), None
    # Final approach: the fingers are expected to touch the object and what it stands on.
    result = arm.follow(plan.line_in, preempt, expect_contact=True)
    if not result.ok:
        return result, None
    if not gripper.close(preempt):
        return _stopped(), None

    if not gripper.is_holding():
        return _give_up(stack, plan, preempt, "the gripper closed on nothing"), None
    width = gripper.width()
    result = arm.follow(plan.line_out, preempt, expect_contact=True)
    if result.ok:
        result = arm.move_joints(stack.ready, preempt)
    if not result.ok:
        # Still gripping something, but the lift was interrupted: report what is in the hand.
        holding = HeldObject(target.label, width, np.asarray(target.position_world, dtype=float))
        return result, holding if gripper.is_holding() else None
    if not gripper.is_holding():
        gripper.open(preempt)
        return ArmResult(MotionStatus.GRASP_FAILED, f"the {target.label} slipped out of the gripper while lifting"), None
    return (
        ArmResult(MotionStatus.SUCCESS, f"holding the {target.label}"),
        HeldObject(target.label, gripper.width(), np.asarray(target.position_world, dtype=float)),
    )


def _stopped() -> ArmResult:
    return ArmResult(MotionStatus.STOPPED, "motion was stopped")


def _give_up(stack: ArmStack, plan, preempt: threading.Event, why: str) -> ArmResult:
    """Open, back out the way the arm came in, and report the failed grasp."""
    stack.gripper.open(preempt)
    stack.arm.follow(plan.line_in[::-1], preempt, expect_contact=True)
    stack.arm.move_joints(stack.ready, preempt)
    return ArmResult(MotionStatus.GRASP_FAILED, why)
