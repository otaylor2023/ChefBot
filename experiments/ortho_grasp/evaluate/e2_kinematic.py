"""E2: kinematic checks, no physics (spec §14).

ik_ok: the repo's IK solves pre-grasp, then the grasp (seeded from it), from
the fixed base pose, within the joint limits. collision_ok: the open gripper's
boxes (the repo's collision model of the 2F-85) clear of the ground-truth cup
cloud and the counter at pre-grasp and along the straight approach; the
fingers may touch the cup only within the last 1 cm.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from experiments.ortho_grasp.resolve import PREGRASP_BACK, Resolved, gripper_clear

TOUCH_LAST = 0.01


def evaluate(r: Resolved, geometry, ik: Callable[[np.ndarray, np.ndarray | None], np.ndarray | None], e1_valid: bool) -> dict:
    """`ik(pose_robot, seed)` -> joint vector or None."""
    pre = r.waypoints[0]["pose"] if r.waypoints else None
    if pre is None:
        pre = r.pose.copy()
        pre[:3, 3] -= PREGRASP_BACK * r.pose[:3, 2]
    q_pre = ik(pre, None)
    q_grasp = ik(r.pose, q_pre) if q_pre is not None else None
    ik_ok = q_pre is not None and q_grasp is not None
    back = float(np.linalg.norm(pre[:3, 3] - r.pose[:3, 3]))
    width = min(0.085, r.width + 0.01)  # the pre-grasp opening (spec §13.5)
    collision_ok = gripper_clear(r.pose, width, geometry.full, geometry.counter_z, back=max(back, 0.0),
                                 step=0.005, touch_last=TOUCH_LAST)  # fmt: skip
    return {"ik_ok": bool(ik_ok), "collision_ok": bool(collision_ok), "e2_valid": bool(e1_valid and ik_ok and collision_ok),
            "q_pre": None if q_pre is None else list(map(float, q_pre)),
            "q_grasp": None if q_grasp is None else list(map(float, q_grasp))}  # fmt: skip
