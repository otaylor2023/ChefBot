"""Where the FR3 may go on ChefBot's base: its joint limits, and the robot's own fixed parts it must keep clear of
(the pole behind it on its left, the pole camera on its clamp, the base's plate). Pure geometry (interface/fr3.py),
shared by the controller's IK (embodiments/motion.py: it steers around them), the laptop's real host (its guard
stops on them) and the robot computer's calibration (it plans around them).

The parts themselves are the embodiment's (embodiments/tidybot_fr3/robot.yaml `body:`, measured on the real robot):
read here as data, so every user, the robot computer's too, has the one definition.

    body = load_body()                 # tidybot_fr3's
    body.inside(arm_points(q))         # which of the arm's points are in the keep-out
    d, n = body.clearance(points)      # how far each is outside it (m; negative: in), and which way is out
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from chefbot.interface import fr3

EMBODIMENTS = Path(__file__).resolve().parent.parent / "embodiments"

# The FR3's joint limits (Franka's datasheet), less a margin.
LIMITS = np.array([[-2.7437, 2.7437], [-1.7837, 1.7837], [-2.9007, 2.9007], [-3.0421, -0.1518], [-2.8065, 2.8065],
                   [0.5445, 4.5169], [-3.0159, 3.0159]])  # fmt: skip
MARGIN = 0.1
MIN_SPARE_M = 0.0254  # every keep-out clears its part by the links' radius and at least an inch more


@dataclass(frozen=True)
class Body:
    """The robot's fixed parts around the arm, in the arm base frame. Cylinders stand on the plate (xy, radius, top);
    capsules are segments (a, b, radius). Their radii are the parts' own: the keep-out grows each by `grow` (the
    links' radius and some to spare), as the checks are on points along the arm's centerline."""

    hand_m: float
    grow: float
    plate_x: tuple[float, float]
    plate_y: tuple[float, float]
    plate_clear: float
    cylinders: tuple = ()  # (name, (x, y), radius, top)
    capsules: tuple = ()  # (name, a, b, radius)

    @classmethod
    def from_config(cls, c: dict) -> "Body":
        cyl, cap = [], []
        for o in c.get("obstacles", []):
            if "cylinder" in o:
                k = o["cylinder"]
                cyl.append((o["name"], tuple(map(float, k["xy"])), float(k["radius"]), float(k["top"])))
            elif "capsule" in o:
                k = o["capsule"]
                cap.append((o["name"], tuple(map(float, k["a"])), tuple(map(float, k["b"])), float(k["radius"])))
            else:
                raise ValueError(f"obstacle {o.get('name')!r}: a cylinder or a capsule")
        if float(c["spare_m"]) < MIN_SPARE_M:
            raise ValueError(f"body spare_m {c['spare_m']} m: at least {MIN_SPARE_M} m (an inch) past the parts and the links")
        return cls(float(c["hand_m"]), float(c["link_radius_m"]) + float(c["spare_m"]), tuple(c["plate"]["x"]),
                   tuple(c["plate"]["y"]), float(c["plate_clear_m"]), tuple(cyl), tuple(cap))  # fmt: skip

    def clearance(self, p, margin: float = 0.0):
        """For points (N x 3): how far each is outside the keep-out grown by margin (m; negative: inside), the unit
        direction that takes it further out, and the nearest part's name."""
        p = np.atleast_2d(np.asarray(p, dtype=float))
        best = np.full(len(p), np.inf)
        normal = np.zeros((len(p), 3))
        which = np.full(len(p), "", dtype=object)
        for name, xy, r, top in self.cylinders:
            radial = p[:, :2] - np.asarray(xy)
            dist = np.linalg.norm(radial, axis=1)
            side = dist - (r + self.grow + margin)
            n_side = np.column_stack([radial / np.maximum(dist, 1e-9)[:, None], np.zeros(len(p))])
            over = p[:, 2] - (top + self.grow + margin)  # above the top: clear by the height
            d = np.where(over > 0, np.hypot(np.maximum(side, 0.0), over), side)
            n = np.where((over > 0)[:, None] & (side <= 0)[:, None], [0.0, 0.0, 1.0], n_side)
            self._keep(best, normal, which, d, n, name)
        for name, a, b, r in self.capsules:
            a, b = np.asarray(a), np.asarray(b)
            t = np.clip((p - a) @ (b - a) / ((b - a) @ (b - a)), 0.0, 1.0)
            off = p - (a + t[:, None] * (b - a))
            dist = np.linalg.norm(off, axis=1)
            self._keep(best, normal, which, dist - (r + self.grow + margin), off / np.maximum(dist, 1e-9)[:, None], name)
        return best, normal, which

    @staticmethod
    def _keep(best, normal, which, d, n, name):
        closer = d < best
        best[closer], normal[closer], which[closer] = d[closer], n[closer], name

    def inside(self, p, margin: float = 0.0) -> np.ndarray:
        """Which points (N x 3) are in the keep-out grown by margin."""
        return self.clearance(p, margin)[0] < 0.0

    def over_plate(self, p) -> np.ndarray:
        """Which points (N x 3) are over the base's plate (in its outline, at any height)."""
        p = np.atleast_2d(np.asarray(p, dtype=float))
        return ((p[:, 0] > self.plate_x[0]) & (p[:, 0] < self.plate_x[1]) & (p[:, 1] > self.plate_y[0]) &
                (p[:, 1] < self.plate_y[1]))  # fmt: skip


def load_body(embodiment: str = "tidybot_fr3") -> Body:
    """An embodiment's body (its robot.yaml `body:`)."""
    raw = yaml.safe_load((EMBODIMENTS / embodiment / "robot.yaml").read_text())
    if "body" not in raw:
        raise ValueError(f"embodiment {embodiment!r} has no body: in its robot.yaml")
    return Body.from_config(raw["body"])


BODY = load_body()  # the real robot's: what the robot computer's calibration and the real host check against
TCP_Z = BODY.hand_m  # m along the flange's z: the fingertips


def arm_points(q, hand_m: float = TCP_Z) -> np.ndarray:
    """Points along the arm (joint origins, the flange, the fingertips, the midpoints between) in the arm base frame:
    9 + 8, the base column's first (0, 1)."""
    fs = fr3.frames(q)
    p = [f[:3, 3] for f in fs] + [fs[-1][:3, 3] + fs[-1][:3, 2] * hand_m]
    return np.array(p + [(a + b) / 2 for a, b in zip(p, p[1:])])


def _inside(p, margin: float, body: Body = BODY) -> np.ndarray:
    """Points (N x 3, arm base frame) within the keep-out grown by margin."""
    return body.inside(p, margin)


def in_keep_out(q, margin: float = 0.0, body: Body = BODY) -> bool:
    """Any point of the arm in the keep-out; the hand's (flange to fingertips) with the keep-out grown by margin (m:
    for something held that sticks out of the hand)."""
    T = fr3.flange(q)
    hand = np.array([T[:3, 3] + T[:3, 2] * t for t in np.linspace(0.0, body.hand_m, 4)])
    return bool(np.any(body.inside(arm_points(q, body.hand_m))) or np.any(body.inside(hand, margin)))


def path_clear(q0, q1, samples: int = 30, margin: float = 0.0, body: Body = BODY) -> bool:
    """The joint-space straight line from q0 to q1 (how the arm moves) stays out of the keep-out."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    return not any(in_keep_out(q0 + (q1 - q0) * t, margin, body) for t in np.linspace(0, 1, samples))
