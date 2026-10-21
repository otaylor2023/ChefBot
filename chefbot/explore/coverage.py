"""Exploration coverage, from the agent's own records only.

Which fixtures the agent has looked at (a detect with the fixture in view),
which of the fixtures with doors it has looked into (a detect while it
believed the door open), and which doors it left open. This is what
complete() checks in an exploration run: nothing here reads ground truth, so
the agent can be told exactly what is left.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from chefbot.agent.check.checker import CheckResult
from chefbot.agent.world.map import MapEntry, WorldMap
from chefbot.interface.observer import Observation, Observer
from chefbot.interface.types import Pose2D

_SAMPLE_SPACING = 0.4  # m between points sampled along a fixture's front


@dataclass
class Coverage:
    world_map: WorldMap
    open_labels: frozenset[str]
    look_range: float = 1.5
    look_half_angle: float = math.radians(60)
    # A cabinet whose floor is above the robot's camera cannot be looked into: not explorable.
    storage_max_bottom: float = math.inf
    seen: dict[str, set[int]] = field(default_factory=dict)  # fixture -> sample points seen
    looked_inside: set[str] = field(default_factory=set)

    def fixtures(self) -> list[MapEntry]:
        """The fixtures to explore: every map fixture but the stove's burners and cabinets out of view."""
        return [e for e in self.world_map.list() if e.kind == "fixture" and e.label != "burner"
                and (e.z is None or e.z[0] <= self.storage_max_bottom)]  # fmt: skip

    def openable(self, entry: MapEntry) -> bool:
        return entry.label in self.open_labels

    def samples(self, entry: MapEntry) -> list[np.ndarray]:
        """Points across a fixture's width (its middle without a size)."""
        middle = np.array(entry.location, dtype=float)
        if entry.size is None or entry.facing is None:
            return [middle]
        across = np.array([-entry.facing[1], entry.facing[0]])
        half = max(entry.size[0] / 2 - 0.15, 0.0)
        n = max(1, math.ceil(2 * half / _SAMPLE_SPACING) + 1)
        return [middle + t * across for t in np.linspace(-half, half, n)] if n > 1 else [middle]

    def _in_view(self, pose: Pose2D, point: np.ndarray) -> bool:
        dx, dy = point[0] - pose.x, point[1] - pose.y
        if math.hypot(dx, dy) > self.look_range:
            return False
        off = math.atan2(dy, dx) - pose.yaw
        return abs(math.atan2(math.sin(off), math.cos(off))) <= self.look_half_angle

    def record_detect(self, pose: Pose2D, states: dict[str, str]) -> None:
        """A detect from `pose`, with the door states the agent believed at the time."""
        for entry in self.fixtures():
            hits = {i for i, p in enumerate(self.samples(entry)) if self._in_view(pose, p)}
            if not hits:
                continue
            self.seen.setdefault(entry.name, set()).update(hits)
            if self.openable(entry) and states.get(entry.name) == "open":
                self.looked_inside.add(entry.name)

    def remaining(self, states: dict[str, str]) -> list[str]:
        """What is left, one line each; empty when the kitchen is explored."""
        out = []
        for entry in self.fixtures():
            samples = self.samples(entry)
            seen = self.seen.get(entry.name, set())
            if not seen:
                out.append(f"{entry.name}: not looked at yet")
            elif len(seen) < len(samples):
                missing = [samples[i] for i in range(len(samples)) if i not in seen]
                across = np.array([-entry.facing[1], entry.facing[0]])  # facing the fixture: to your right
                middle = np.array(entry.location, dtype=float)
                reach = max(entry.size[0] / 2 - _END_INSET, 1e-6)
                for part in ("left", "right", "middle"):
                    side = [p for p in missing if _side(p - middle, across) == part]
                    if side:
                        xy = np.mean(side, axis=0)
                        t = max(-1.0, min(1.0, float((xy - middle) @ across) / reach))
                        hint = (f'go_to("{entry.name}", part="{part}")' if abs(t) > 0.9 else
                                f'go_to("{entry.name}", along={t:.2f})')  # fmt: skip
                        out.append(f"{entry.name}: only partly seen; not the {part} part, around "
                                   f"({xy[0]:.2f}, {xy[1]:.2f}): {hint}, then detect")  # fmt: skip
            if self.openable(entry) and entry.name not in self.looked_inside:
                out.append(f"{entry.name}: not looked inside yet (open it, then detect)")
        for name, state in states.items():
            entry = self.world_map.get(name)
            if state in ("open", "ajar") and entry is not None and self.openable(entry):
                out.append(f"{name}: left {state} (close it)")
        return out


class CoverageObserver(Observer):
    """What complete() sees in an exploration run: the coverage report, not the world."""

    def __init__(self, coverage: Coverage, states):
        self._coverage = coverage
        self._states = states  # () -> {fixture: "open" | "closed" | ...}, the agent's belief

    def observe(self) -> Observation:
        left = self._coverage.remaining(self._states())
        return Observation(text="\n".join(left) or "every fixture explored", extra={"remaining": left})


class CoverageChecker:
    def check(self, condition: str, observation: Observation) -> CheckResult:
        left = observation.extra.get("remaining", [])
        if not left:
            return CheckResult(True, "every fixture has been looked at, every door looked behind and closed")
        return CheckResult(False, "still to do: " + "; ".join(left))


_END_INSET = 0.4  # m: as go_to's (agent/tools/nav.py)


def _side(offset: np.ndarray, across: np.ndarray) -> str:
    """Which part of a fixture a point is in, as seen facing it."""
    t = float(offset @ across)
    return "right" if t > 0.3 else "left" if t < -0.3 else "middle"
