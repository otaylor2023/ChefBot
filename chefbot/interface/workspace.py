"""Where the FR3 may go on ChefBot's base: its joint limits, the keep-out around the pole (and the pole camera on its
clamp) behind the arm on its left, and the checks on a pose or a joint-space path. Pure geometry (interface/fr3.py),
shared by the robot computer's calibration (onboard/calibration) and the laptop's real host (environments/real)."""

from __future__ import annotations

import numpy as np

from chefbot.interface import fr3

# The FR3's joint limits (Franka's datasheet), less a margin.
LIMITS = np.array([[-2.7437, 2.7437], [-1.7837, 1.7837], [-2.9007, 2.9007], [-3.0421, -0.1518], [-2.8065, 2.8065],
                   [0.5445, 4.5169], [-3.0159, 3.0159]])  # fmt: skip
MARGIN = 0.1
# Keep out of the pole behind the arm on its left (the arm's own frame: x forward, y left, z up from its mounting
# plate) and the pole camera on its clamp: no joint, link midpoint or fingertip within these. The pole's center
# measured 5 in back and 8 in left of the arm base's (2026-11-03); the camera where pole.py calibrated it
# (environments/real/config.yaml: 1.04 m up, so 0.70 m above the plate). Radii are for the arm's centerline: the
# pole's ~3 cm half-width, the links' ~8 cm, and ~4 cm to spare. (It replaced a quarter-space, everything behind
# and left of the arm base, that also walled off the plate beside the arm.)
POLE_XY = (-0.127, 0.203)
POLE_RADIUS = 0.15
POLE_TOP = 0.85  # m above the plate: the pole and its clamp
POLE_CAMERA = (-0.25, 0.26, 0.70)
POLE_CAMERA_RADIUS = 0.15
TCP_Z = 0.13065  # m along the flange's z: the fingertips


def arm_points(q) -> np.ndarray:
    """Points along the arm (joint origins, link midpoints, the fingertips) in the arm base frame."""
    fs = fr3.frames(q)
    p = [f[:3, 3] for f in fs] + [fs[-1][:3, 3] + fs[-1][:3, 2] * TCP_Z]
    return np.array(p + [(a + b) / 2 for a, b in zip(p, p[1:])])


def _segment_distance(p, a, b) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = np.clip((p - a) @ (b - a) / ((b - a) @ (b - a)), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * (b - a)), axis=1)


def _inside(p, margin: float) -> np.ndarray:
    """Points (N x 3, arm base frame) within the keep-out grown by margin: the pole's cylinder, and the camera on its
    clamp (a capsule from the pole's top to the camera)."""
    p = np.atleast_2d(p)
    radial = np.linalg.norm(p[:, :2] - np.asarray(POLE_XY), axis=1)
    pole = (radial < POLE_RADIUS + margin) & (p[:, 2] < POLE_TOP + margin)
    clamp = _segment_distance(p, (*POLE_XY, POLE_TOP - 0.2), POLE_CAMERA) < POLE_CAMERA_RADIUS + margin
    return pole | clamp


def in_keep_out(q, margin: float = 0.0) -> bool:
    """Any point of the arm in the keep-out; the hand's (flange to fingertips) with the keep-out grown by margin (m:
    for something held that sticks out of the hand)."""
    T = fr3.flange(q)
    hand = np.array([T[:3, 3] + T[:3, 2] * t for t in np.linspace(0.0, TCP_Z, 4)])
    return bool(np.any(_inside(arm_points(q), 0.0)) or np.any(_inside(hand, margin)))


def path_clear(q0, q1, samples: int = 30, margin: float = 0.0) -> bool:
    """The joint-space straight line from q0 to q1 (how the arm moves) stays out of the keep-out."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    return not any(in_keep_out(q0 + (q1 - q0) * t, margin) for t in np.linspace(0, 1, samples))
