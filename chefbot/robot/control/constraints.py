"""Where the arm may not go: constraints on the points along it (interface/workspace.py arm_points' 17: the joint
origins, the flange, the fingertips, the midpoints between, the base column's first two), defined once and read by
every arm solver (robot/control/arm_solvers: each avoids them its own way) and by the real host's guard (it stops on
them). A new constraint here is avoided by all the solvers and guarded on the real robot.

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
