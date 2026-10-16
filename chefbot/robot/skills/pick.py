"""pick: grasp a detected object and lift it (MANIP_SPEC §5.4)."""

from __future__ import annotations

import threading

import numpy as np

from chefbot.interface.proposers import GraspProposer
from chefbot.interface.types import Detection, HeldObject, MotionStatus
from chefbot.robot.arm.arm_controller import ArmResult
from chefbot.robot.safety import require_base_at_rest
from chefbot.robot.skills.common import (
    ArmStack,
    box_corners,
    distances,
    follow_path,
    plan_approach,
    shifted,
    to_ready,
)


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

    # What the camera sees now, less the object itself (it is meant to be touched).
    world = stack.workspace([g.object_box for g in grasps])
    # The best-scored grasp whose approach, grasp and lift all have joint
    # solutions clear of the world, at full distances or, in a tight spot, shorter.
    plan, chosen = None, None
    to_arm = np.linalg.inv(stack.frames.arm_base_in_world(stack.base_pose()))[:3, :3]
    for grasp in sorted(grasps, key=lambda g: -g.score):
        at_grasp = stack.to_arm(grasp.tcp_pose_world)
        approaches = [
            shifted(at_grasp, -d * at_grasp[:3, 2]) for d in distances(grasp.pregrasp_offset, cfg.shorter_offsets)
        ]
        lift = np.array([0.0, 0.0, cfg.lift_height]) if grasp.retreat is None else to_arm @ grasp.retreat
        length = float(np.linalg.norm(lift))
        lifts = [shifted(at_grasp, lift * d / length) for d in distances(length, cfg.shorter_offsets)]
        plan = plan_approach(stack, world, approaches, at_grasp, lifts)
        if plan is not None:
            chosen = grasp
            break
    if plan is None:
        where = " clear of what is around it" if world.checked else ""
        return ArmResult(
            MotionStatus.UNREACHABLE,
            f"the arm cannot reach any grasp on the {target.label}{where} from here; align to it first",
        ), None

    for step in (
        lambda: to_ready(stack, world, preempt),
        lambda: follow_path(stack, plan.into, preempt),
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
    # Where the object hangs from the gripper, for keeping it out of later world models.
    box_in_tcp = None
    if chosen.object_box is not None:
        tcp = stack.tcp_world()
        box_in_tcp = (box_corners(chosen.object_box) - tcp[:3, 3]) @ tcp[:3, :3]
    result = arm.follow(plan.line_out, preempt, expect_contact=True)
    if result.ok:
        result = follow_path(stack, plan.back, preempt)
    if not result.ok:
        # Still gripping something, but the lift was interrupted: report what is in the hand.
        holding = HeldObject(target.label, width, np.asarray(target.position_world, dtype=float), box_in_tcp)
        return result, holding if gripper.is_holding() else None
    if not gripper.is_holding():
        gripper.open(preempt)
        return ArmResult(MotionStatus.GRASP_FAILED, f"the {target.label} slipped out of the gripper while lifting"), None
    return (
        ArmResult(MotionStatus.SUCCESS, f"holding the {target.label}"),
        HeldObject(target.label, gripper.width(), np.asarray(target.position_world, dtype=float), box_in_tcp),
    )


def _stopped() -> ArmResult:
    return ArmResult(MotionStatus.STOPPED, "motion was stopped")


def _give_up(stack: ArmStack, plan, preempt: threading.Event, why: str) -> ArmResult:
    """Open, back out the way the arm came in, and report the failed grasp."""
    stack.gripper.open(preempt)
    stack.arm.follow(plan.line_in[::-1], preempt, expect_contact=True)
    follow_path(stack, plan.into[::-1], preempt)
    return ArmResult(MotionStatus.GRASP_FAILED, why)
