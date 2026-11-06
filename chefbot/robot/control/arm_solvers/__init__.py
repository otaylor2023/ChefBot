"""The controller's arm solvers (robot/control/wholebody.py), one per file, interchangeable: each takes the arm one
control tick from its joints toward a flange pose, within its joint limits and clear of the robot's own parts (its
body: the pole, the camera on its clamp, the base's plate), at the speed profile's rates.

    qp      one QP per tick: the inertia-weighted motion that best closes the pose error, the joint limits and the
            body as hard constraints (qp.py; the default)
    osc     OpenSai's operational-space motion task (motionForceTask, onboard/bringup/chefbot_fr3.xml's gains), ported:
            a virtual arm driven by it at 1 kHz on the model, repelled from the body (osc.py)
    legacy  the controller's first: a short clipped Cartesian step solved exactly by the arm's IK (legacy.py); no body

The robot's robot.yaml `control.arm_solver` names its default; CHEFBOT_ARM_SOLVER overrides it everywhere (the sims'
labs too), run_real --solver for a real run. Every solver: `step(target, q, dt, max_lin, max_ang, max_joint)` -> the
joints after the tick, or mink's NoSolutionFound (the controller holds). The constraints (robot/control/constraints.py)
are defined once, outside the solvers, and every solver avoids them its own way; each runs inside the router
(route.py), which plans a way around them when the solver stalls against one.
"""

from __future__ import annotations

import os

NAMES = ("qp", "osc", "legacy")
DEFAULT = "qp"


def make(name: str, ik, constraints: list = (), hand_m: float = 0.0, log=None):
    """The solver `name` on an ArmIK, avoiding `constraints`, inside the router."""
    from chefbot.robot.control.arm_solvers.route import Router

    if name == "qp":
        from chefbot.robot.control.arm_solvers.qp import QPSolver as S
    elif name == "osc":
        from chefbot.robot.control.arm_solvers.osc import OSCSolver as S
    elif name == "legacy":
        from chefbot.robot.control.arm_solvers.legacy import LegacySolver as S
    else:
        raise ValueError(f"no arm solver {name!r}; solvers: {', '.join(NAMES)}")
    return Router(S(ik, constraints, hand_m), ik, constraints, hand_m, log)


def chosen(robot_default: str | None = None) -> str:
    """CHEFBOT_ARM_SOLVER if set, else the robot's default, else qp."""
    return os.environ.get("CHEFBOT_ARM_SOLVER") or robot_default or DEFAULT
