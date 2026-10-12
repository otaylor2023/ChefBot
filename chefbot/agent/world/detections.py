"""Detections: the results of the latest `detect` call, in the world frame."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from chefbot.interface.types import Pose2D


@dataclass
class DetectionEntry:
    id: str
    label: str
    position_world: np.ndarray  # (3,)
    confidence: float
    seen_from: Pose2D
    timestamp: float
    map_name: str | None = None  # set when matched to a map entry

    def distance_to(self, pose: Pose2D) -> float:
        return math.hypot(self.position_world[0] - pose.x, self.position_world[1] - pose.y)


class Detections:
    """Scratch buffer, overwritten on every `detect` (v0.1 simplification)."""

    def __init__(self) -> None:
        self._entries: dict[str, DetectionEntry] = {}

    def replace(self, entries: list[DetectionEntry]) -> None:
        self._entries = {e.id: e for e in entries}

    def get(self, id: str) -> DetectionEntry | None:
        return self._entries.get(id)

    def all(self) -> list[DetectionEntry]:
        return list(self._entries.values())

    def map_names(self) -> set[str]:
        return {e.map_name for e in self._entries.values() if e.map_name is not None}

    def refine(
        self, id: str, position_world: np.ndarray, confidence: float, seen_from: Pose2D, timestamp: float
    ) -> None:
        """Update one entry in place after a closer look; all others are untouched."""
        entry = self._entries[id]
        entry.position_world = np.asarray(position_world, dtype=float)
        entry.confidence = confidence
        entry.seen_from = seen_from
        entry.timestamp = timestamp
