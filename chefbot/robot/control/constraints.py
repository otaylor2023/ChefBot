"""Where the arm may not go: constraints on the points along it (interface/workspace.py arm_points' 17: the joint
origins, the flange, the fingertips, the midpoints between, the base column's first two), defined once and read by
every arm solver (robot/control/arm_solvers: each avoids them its own way) and by the real host's guard (it stops on
them). A new constraint here is avoided by all the solvers, guarded on the real robot, and refused in the agent's planned
arm moves (robot/arm/arm_controller.py, by path_violation).

A constraint gives, for points (N x 3, arm base frame), each one's clearance (m, past a margin; negative: in
violation) and the unit direction that increases it:

    d, n = constraint.clearance(points, margin)

    for_robot(robot.body)   the robot's: its body's keep-out (the pole, the camera on its clamp) and its plate
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chefbot.interface.workspace import Body


@dataclass(frozen=True)
class KeepOut:
    """Out of the body's parts (each grown by the links' radius and some to spare)."""

    body: Body
    name: str = "keep-out"

    def clearance(self, P, margin: float = 0.0):
        d, n, which = self.body.clearance(P, margin)
        return d, n

    def nearest_part(self, P) -> str:
        d, _, which = self.body.clearance(P)
        return str(which[int(np.argmin(d))])


@dataclass(frozen=True)
class PlateClearance:
    """Every point (bar the base column, which stands on it) at least the body's plate clearance above the base's
    plate, over it (within its outline)."""

    body: Body
    name: str = "the base's plate"

    def clearance(self, P, margin: float = 0.0):
        P = np.atleast_2d(np.asarray(P, dtype=float))
        over = self.body.over_plate(P)
        over[:2] = False
        d = np.where(over, P[:, 2] - (self.body.plate_clear + margin), np.inf)
        return d, np.tile([0.0, 0.0, 1.0], (len(P), 1))


def for_robot(body: Body | None) -> list:
    """The robot's constraints (none without a body)."""
    return [] if body is None else [KeepOut(body), PlateClearance(body)]


def path_violation(constraints, hand_m: float, qs, step: float = 0.02) -> str | None:
    """Where joints moving along `qs` (waypoints, joined by straight joint lines) first go into a constraint, said,
    or None. The real host's rule (environments/real/host.py path_clear): never below the start's clearance (from a
    start inside one) or 0, and once it is out, never back in.
    Sampled so no joint moves more than `step` (rad) between samples. The points are the FR3's (workspace.arm_points),
    the only arm with a body today."""
    from chefbot.interface.workspace import arm_points

    if not constraints:
        return None
    qs = [np.asarray(q, dtype=float) for q in qs]
    samples = [qs[0]]
    for a, b in zip(qs, qs[1:]):
        n = max(1, int(np.ceil(np.abs(b - a).max() / step)))
        samples += [a + (b - a) * (i / n) for i in range(1, n + 1)]
    floor = None
    for q in samples:
        pts = arm_points(q, hand_m)
        d, _, which = nearest(constraints, pts)
        here = float(d.min())
        if floor is not None and here < floor - 1e-5:  # into one, or deeper than it has been
            c = constraints[int(which[int(np.argmin(d))])]
            what = f"the keep-out of the {c.nearest_part(pts)}" if isinstance(c, KeepOut) else c.name
            return f"{what} ({-here * 100:.1f} cm in)"
        floor = min(0.0, here) if floor is None else max(floor, min(0.0, here))
    return None


def nearest(constraints, P, margin: float = 0.0):
    """Each point's clearance from the nearest constraint, the direction out of it, and which constraint (index)."""
    P = np.atleast_2d(np.asarray(P, dtype=float))
    d = np.full(len(P), np.inf)
    n = np.zeros((len(P), 3))
    which = np.full(len(P), -1)
    for i, c in enumerate(constraints):
        di, ni = c.clearance(P, margin)
        closer = di < d
        d[closer], n[closer], which[closer] = di[closer], ni[closer], i
    return d, n, which
