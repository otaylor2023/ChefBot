"""From a memory-map entry to a park target: "go to tomato" (real_nav goto). Pure: no robot.

The park rule (robot/nav/goal.py counter_park_candidates) only gives stances for targets <= 0.25 m in from a counter
side and >= 0.35 m from its ends (the pick skill's trained start). An object deeper in (the island is 0.96 m deep) or
near an end gets the nearest side's closest legal spot instead: the base parks squared up to that side, in front of
the object, and the report says how far past the rule the object really is (`reach_extra_m`; `trained: false`).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import yaml

from chefbot.robot.nav.goal import PARK_END_MARGIN_M, counter_faces, counter_park_candidates

MAX_INSET_M = 0.25
# The base's plate in the base (turning-centre) frame: embodiments/tidybot_fr3/robot.yaml plate x [-0.462, 0.097] in the
# arm frame, the turning centre 0.1825 m behind the arm -> x +-0.28; y +-0.241.
PLATE_X, PLATE_Y = (-0.2795, 0.2795), (-0.241, 0.241)
CLEAR_M = 0.05  # and this to spare


class Floor:
    """Where the base may stand: the spec's bounds, minus every obstacle box (counters, fridge, no-go zones)."""

    def __init__(self, bounds, boxes):
        self.bounds = np.asarray(bounds, float)
        self.boxes = boxes  # (center, size, yaw rad, name)

    @classmethod
    def from_spec(cls, spec_path) -> "Floor":
        spec = yaml.safe_load(Path(spec_path).read_text())
        boxes = [(np.asarray(o["box"]["center"], float), np.asarray(o["box"]["size"], float),
                  math.radians(o["box"].get("yaw_deg", 0.0)), o["name"]) for o in spec.get("obstacles", []) if "box" in o]  # fmt: skip
        return cls(spec["bounds"], boxes)

    def footprint(self, x, y, yaw, margin=CLEAR_M, n=6) -> np.ndarray:
        xs = np.linspace(PLATE_X[0] - margin, PLATE_X[1] + margin, n)
        ys = np.linspace(PLATE_Y[0] - margin, PLATE_Y[1] + margin, n)
        g = np.array([(a, b) for a in xs for b in ys])
        c, s = math.cos(yaw), math.sin(yaw)
        return np.stack([x + c * g[:, 0] - s * g[:, 1], y + s * g[:, 0] + c * g[:, 1]], axis=1)

    def why_blocked(self, x, y, yaw) -> str | None:
        p = self.footprint(x, y, yaw)
        (x0, y0), (x1, y1) = self.bounds
        if (p[:, 0] < x0).any() or (p[:, 0] > x1).any() or (p[:, 1] < y0).any() or (p[:, 1] > y1).any():
            return "off the floor (map bounds)"
        for center, size, yaw_b, name in self.boxes:
            c, s = math.cos(yaw_b), math.sin(yaw_b)
            d = p - center
            q = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=1)
            if ((np.abs(q[:, 0]) <= size[0] / 2) & (np.abs(q[:, 1]) <= size[1] / 2)).any():
                return f"in {name}"
        return None


def load_objects(path) -> list[dict]:
    return json.loads(Path(path).read_text())["objects"]


def resolve(objects: list[dict], name: str) -> dict:
    """The entry called `name` (exact), else the only entry of that label; refuses unknown positions."""
    hits = [o for o in objects if o["name"] == name] or [o for o in objects if o["label"] == name]
    if not hits:
        raise LookupError(f"no object {name!r} in the map (have: {', '.join(sorted(o['name'] for o in objects)) or 'none'})")
    if len(hits) > 1:
        raise LookupError(f"{name!r} is {len(hits)} objects: say which ({', '.join(o['name'] for o in hits)})")
    o = hits[0]
    if o.get("xyz") is None:
        last = o.get("last_xyz")
        raise LookupError(f"{o['name']}: position unknown (not where it was last seen"
                          f"{f' at ({last[0]:.2f}, {last[1]:.2f})' if last else ''}); look for it first")  # fmt: skip
    return o


def park_target(center, size, yaw: float, xy, end_margin: float = PARK_END_MARGIN_M, max_inset: float = MAX_INSET_M,
                floor: Floor | None = None, out_m=(0.45, 0.48)) -> dict:  # fmt: skip
    """The target to park for: the object's own x/y if the rule has a stance for it, else the nearest side's closest
    point within the rule. With `floor`, only sides where the base can stand (a stance clear of walls, counters and
    no-go zones). Returns {target, side, inset_m, reach_extra_m, within_rule, stance, blocked}."""
    centre, r = np.asarray(center, float), np.asarray(xy, float) - np.asarray(center, float)
    best = None
    blocked = {}
    for f in counter_faces(center, size, yaw):
        inset = f["half_n"] - float(r @ f["n"])
        along = float(r @ f["t"])
        lim = f["half_along"] - end_margin
        if lim < 0:
            continue
        eps = 1e-3  # a clipped target 1 mm inside the rule's limits, not on them
        a2, i2 = float(np.clip(along, -lim + eps, lim - eps)), float(np.clip(inset, eps, max_inset - eps))
        extra = float(np.hypot(along - a2, inset - i2))
        tgt = centre + f["t"] * a2 + f["n"] * (f["half_n"] - i2)
        cand = {"target": [round(float(tgt[0]), 4), round(float(tgt[1]), 4)], "side": f["name"], "inset_m": round(inset, 3),
                "reach_extra_m": round(extra, 3), "within_rule": extra < 2e-3}  # fmt: skip
        stances = [p for p, info in counter_park_candidates(center, size, yaw, cand["target"], out_m=out_m) if info["side"] == f["name"]]
        if floor is not None:
            why = [floor.why_blocked(p.x, p.y, p.yaw) for p in stances]
            stances = [p for p, w in zip(stances, why) if w is None]
            if not stances:
                blocked[f["name"]] = next((w for w in why if w), "no stance")
                continue
        if stances:
            cand["stance"] = [round(stances[0].x, 3), round(stances[0].y, 3), round(math.degrees(stances[0].yaw), 1)]
        if best is None or extra < best["reach_extra_m"]:
            best = cand
    if best is None:
        raise ValueError(f"no side of the counter where the base can stand ({blocked or 'end margins'})")
    best["blocked"] = blocked
    return best
