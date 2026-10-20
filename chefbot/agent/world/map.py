"""Semantic map: named fixtures and objects with rough locations.

Env-agnostic: no sim names, no parts, no geometry beyond a location and an
optional facing direction (toward open floor).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from chefbot.interface.types import Pose2D

_SUFFIX = re.compile(r"_\d+$")


def label_of(name: str) -> str:
    """A map entry's label is its name minus any `_n` suffix, with underscores read
    as spaces: counter_2 -> counter, egg_cracker -> egg cracker."""
    return _SUFFIX.sub("", name).replace("_", " ")


@dataclass(frozen=True)
class MapEntry:
    name: str
    location: tuple[float, float]
    facing: tuple[float, float] | None = None
    kind: str = "fixture"  # "fixture" | "object"
    inside: str | None = None  # objects: name of the fixture holding it
    # Fixtures: footprint [width along the front, depth front to back], centered on
    # `location` and turned by `facing`. Lets an object be placed in the fixture it
    # stands on even when another fixture's middle is closer (a long counter).
    size: tuple[float, float] | None = None
    # Fixtures stacked over one footprint (cabinets, drawers): [bottom, top] height, so what
    # sits inside one is told from what sits on the counter above it. None: any height.
    z: tuple[float, float] | None = None

    @property
    def label(self) -> str:
        return label_of(self.name)

    def contains(self, x: float, y: float) -> bool:
        """Whether (x, y) is over this fixture's footprint (False without a size)."""
        if self.size is None or self.facing is None:
            return False
        dx, dy = x - self.location[0], y - self.location[1]
        fx, fy = self.facing
        depth = dx * fx + dy * fy
        along = -dx * fy + dy * fx
        return abs(along) <= self.size[0] / 2 and abs(depth) <= self.size[1] / 2

    def distance_to(self, pose: Pose2D) -> float:
        return math.hypot(self.location[0] - pose.x, self.location[1] - pose.y)


class WorldMap:
    def __init__(self, entries: list[MapEntry], frame: str = "world"):
        self.frame = frame
        self._entries = {e.name: e for e in entries}
        if len(self._entries) != len(entries):
            raise ValueError("duplicate names in map")

    @classmethod
    def from_dict(cls, raw: dict) -> "WorldMap":
        entries = []
        for kind, section in (("fixture", "fixtures"), ("object", "objects")):
            for name, spec in (raw.get(section) or {}).items():
                facing = spec.get("facing")
                size = spec.get("size")
                z = spec.get("z")
                entries.append(
                    MapEntry(
                        name=name,
                        location=(float(spec["location"][0]), float(spec["location"][1])),
                        facing=(float(facing[0]), float(facing[1])) if facing else None,
                        kind=kind,
                        inside=spec.get("in"),
                        size=(float(size[0]), float(size[1])) if size else None,
                        z=(float(z[0]), float(z[1])) if z else None,
                    )
                )
        return cls(entries, raw.get("frame", "world"))

    @classmethod
    def load(cls, path: str | Path) -> "WorldMap":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    def to_dict(self) -> dict:
        out: dict = {"frame": self.frame, "fixtures": {}, "objects": {}}
        for e in self._entries.values():
            spec: dict = {"location": [round(v, 3) for v in e.location]}
            if e.facing is not None:
                spec["facing"] = [round(v, 3) for v in e.facing]
            if e.size is not None:
                spec["size"] = [round(v, 3) for v in e.size]
            if e.z is not None:
                spec["z"] = [round(v, 3) for v in e.z]
            if e.inside is not None:
                spec["in"] = e.inside
            out["fixtures" if e.kind == "fixture" else "objects"][e.name] = spec
        return out

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    def get(self, name: str) -> MapEntry | None:
        return self._entries.get(name)

    def put(self, entry: MapEntry) -> None:
        """Add an entry, or replace the one with its name (an object put down or seen somewhere new)."""
        self._entries[entry.name] = entry

    def remove(self, name: str) -> None:
        """Forget an entry (an object that has been picked up is no longer where the map says)."""
        self._entries.pop(name, None)

    def list(self) -> list[MapEntry]:
        return list(self._entries.values())

    def labels(self) -> set[str]:
        return {e.label for e in self._entries.values()}

    def in_range(self, pose: Pose2D, radius: float) -> list[MapEntry]:
        """Entries within `radius` of `pose`, nearest first."""
        near = [e for e in self._entries.values() if e.distance_to(pose) <= radius]
        return sorted(near, key=lambda e: e.distance_to(pose))
