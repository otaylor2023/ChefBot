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
# The ready position (the real host's --ready; launch.sh's r): the flange 0.85 m over the plate above the arm's base, the
# hand and the wrist camera looking forward and 35 deg down at a counter ahead (2026-11-04: the arm where it was left,
# rounded; 2.5 cm clear of the keep-out). Before it: HAND_DOWN, the hand straight down, which sees only the floor in
# front of a counter; the calibrations keep their own poses.
READY = (0.0, -1.34, 0.0, -2.01, 0.0, 1.62, 0.785)
HAND_DOWN = (0.0, -0.75, 0.0, -2.16, 0.0, 1.41, 0.785)
MIN_SPARE_M = 0.0254  # every keep-out clears its part by the links' radius and at least an inch more
# An arm already in a keep-out may move out, never deeper: by more than this (a path out of it can wobble a mm deeper
# first; 2026-11-04 the way home from 2.8 cm in dipped 1 mm, and a 0.01 mm rule refused it).
DEEPER_SLACK_M = 0.005


@dataclass(frozen=True)
class Body:
    """The robot's fixed parts around the arm, in the arm base frame. Cylinders stand on the plate (xy, radius, top);
    capsules are segments (a, b, radius). Their radii are the parts' own: the keep-out grows each by `grow` (the
    links' radius and some to spare), as the checks are on points along the arm's centerline."""

    hand_m: float
    grow: float
    plate_height: float  # m: the arm's plate (the arm base frame's z = 0) above the floor
    plate_x: tuple[float, float]
    plate_y: tuple[float, float]
    plate_clear: float
    cylinders: tuple = ()  # (name, (x, y), radius, top)
    capsules: tuple = ()  # (name, a, b, radius)
    link_radius: float = 0.08  # m: the arm's links about their centerline (grow less the spare)

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
        return cls(float(c["hand_m"]), float(c["link_radius_m"]) + float(c["spare_m"]), float(c["plate_height_m"]), tuple(c["plate"]["x"]),
                   tuple(c["plate"]["y"]), float(c["plate_clear_m"]), tuple(cyl), tuple(cap), float(c["link_radius_m"]))  # fmt: skip

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

    @property
    def base_xy(self) -> tuple[float, float]:
        """The base's centre (the frame odometry, mocap and navigation use: x forward, y left) in the arm base frame:
        the plate's centre, ~18 cm behind the arm. ASSUMED: the base turns about its plate's centre (TidyBot++'s four
        casters sit symmetric under it); not measured against the arm."""
        return ((self.plate_x[0] + self.plate_x[1]) / 2, (self.plate_y[0] + self.plate_y[1]) / 2)

    def to_base(self, p) -> np.ndarray:
        """Points (N x 3) from the arm base frame to the base frame on the floor (x forward, y left, z up from the floor)."""
        p = np.atleast_2d(np.asarray(p, dtype=float))
        cx, cy = self.base_xy
        return p + np.array([-cx, -cy, self.plate_height])

    def footprint(self, q=None, below: float | None = None) -> "Footprint":
        """The robot seen from above, in the base frame: the plate's outline (the base under it, floor to
        plate_height), plus discs of link_radius about the arm's points for joints `q` (none: the plate alone). With
        `below` (m above the floor), only the arm's points whose underside is lower count: the rest pass over
        obstacles that low."""
        cx, cy = self.base_xy
        rect = (self.plate_x[0] - cx, self.plate_x[1] - cx, self.plate_y[0] - cy, self.plate_y[1] - cy)
        discs = np.zeros((0, 3))
        if q is not None:
            pts = self.to_base(arm_points(np.asarray(q, dtype=float), self.hand_m))
            if below is not None:
                pts = pts[pts[:, 2] - self.link_radius < below]
            discs = np.column_stack([pts[:, :2], np.full(len(pts), self.link_radius)])
        return Footprint(rect, discs)

    def over_plate(self, p) -> np.ndarray:
        """Which points (N x 3) are over the base's plate (in its outline, at any height)."""
        p = np.atleast_2d(np.asarray(p, dtype=float))
        return ((p[:, 0] > self.plate_x[0]) & (p[:, 0] < self.plate_x[1]) & (p[:, 1] > self.plate_y[0]) &
                (p[:, 1] < self.plate_y[1]))  # fmt: skip


@dataclass(frozen=True)
class Footprint:
    """The robot seen from above, in the base frame (x forward, y left, origin the base's centre): a rectangle (the
    plate: x_min, x_max, y_min, y_max) and discs (N x 3: x, y, radius; the arm). Body.footprint makes it."""

    rect: tuple[float, float, float, float]
    discs: np.ndarray

    def distance(self, xy) -> np.ndarray:
        """How far points (N x 2, base frame) are from the footprint (m; 0 inside it)."""
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        x0, x1, y0, y1 = self.rect
        dx = np.maximum.reduce([x0 - xy[:, 0], np.zeros(len(xy)), xy[:, 0] - x1])
        dy = np.maximum.reduce([y0 - xy[:, 1], np.zeros(len(xy)), xy[:, 1] - y1])
        d = np.hypot(dx, dy)
        if len(self.discs):
            to_disc = np.linalg.norm(xy[:, None, :] - self.discs[None, :, :2], axis=2) - self.discs[None, :, 2]
            d = np.minimum(d, np.maximum(to_disc.min(axis=1), 0.0))
        return d

    @property
    def radius(self) -> float:
        """The farthest the footprint reaches from the base's centre (m)."""
        x0, x1, y0, y1 = self.rect
        r = max(np.hypot(x, y) for x in (x0, x1) for y in (y0, y1))
        if len(self.discs):
            r = max(r, float((np.linalg.norm(self.discs[:, :2], axis=1) + self.discs[:, 2]).max()))
        return float(r)

    def reach(self, direction_deg: float = 0.0) -> float:
        """How far it reaches along a direction from the base's centre (m; 0 deg: forward)."""
        u = np.array([np.cos(np.radians(direction_deg)), np.sin(np.radians(direction_deg))])
        x0, x1, y0, y1 = self.rect
        r = max(np.array([x, y]) @ u for x in (x0, x1) for y in (y0, y1))
        if len(self.discs):
            r = max(r, float((self.discs[:, :2] @ u + self.discs[:, 2]).max()))
        return float(r)

    def summary(self) -> dict:
        x0, x1, y0, y1 = self.rect
        return {"plate": {"x": [round(x0, 4), round(x1, 4)], "y": [round(y0, 4), round(y1, 4)]}, "arm_discs": len(self.discs),
                "radius_m": round(self.radius, 4), "reach_ahead_m": round(self.reach(0.0), 4)}  # fmt: skip


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
    """The joint-space straight line from q0 to q1 (how the arm moves) stays out of the keep-out; from a start already
    in it, it never goes more than DEEPER_SLACK_M deeper and ends clear."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    if not in_keep_out(q0, margin, body):
        return not any(in_keep_out(q0 + (q1 - q0) * t, margin, body) for t in np.linspace(0, 1, samples))
    floor = clearance_at(q0, body)
    for t in np.linspace(0, 1, samples)[1:]:
        c = clearance_at(q0 + (q1 - q0) * t, body)
        if c < floor - DEEPER_SLACK_M:
            return False
        floor = max(floor, min(0.0, c))
    return not in_keep_out(q1, margin, body)


def clearance_at(q, body: Body = BODY) -> float:
    """The arm's least clearance (m; negative: in) from the keep-out at joints q."""
    return float(body.clearance(arm_points(q, body.hand_m))[0].min())


def joint_path(q0, q1, point_clear, segment_clear, limits, vias=(), tries: int = 60, rng=None) -> list | None:
    """Joint waypoints from q0 to q1 (the last is q1), each leg a straight joint-space line clear (segment_clear(a, b)):
    straight if it is, else through one waypoint clear itself (point_clear(v)): the given vias (a posture) first, then
    the shoulder's joints moved first or last, then random joints between; else None. The one planner around the
    keep-out: the arm solvers' router (embodiments/arm_solvers/route.py: the real host's way to the ready position) and
    the calibration's moves (onboard/calibration/arm.py) each pass their own checks."""
    q0, q1 = np.asarray(q0, dtype=float), np.asarray(q1, dtype=float)
    if segment_clear(q0, q1):
        return [q1]
    rng = np.random.default_rng(0) if rng is None else rng
    lim, n = np.asarray(limits, dtype=float), len(q0)
    shoulder_first, wrist_first = q0.copy(), q0.copy()
    shoulder_first[: min(4, n)] = q1[: min(4, n)]
    wrist_first[min(4, n):] = q1[min(4, n):]
    cands = [np.asarray(v, dtype=float) for v in vias] + [shoulder_first, wrist_first]
    cands += [np.clip(q0 + (q1 - q0) * rng.uniform(0.2, 0.8) + rng.normal(0, 0.5, n), lim[:, 0], lim[:, 1]) for _ in range(tries)]
    for v in cands:
        if point_clear(v) and segment_clear(q0, v) and segment_clear(v, q1):
            return [v, q1]
    return None
