"""Nearby view: what is around the robot right now. Built on demand, never stored."""

from __future__ import annotations

from dataclasses import dataclass

from chefbot.agent.world.detections import DetectionEntry, Detections
from chefbot.agent.world.map import MapEntry, WorldMap
from chefbot.interface.types import Pose2D


@dataclass
class NearbyView:
    visible: list[DetectionEntry]  # alignable; nearest first
    not_visible: list[MapEntry]  # on the map and in range, but not found by the last detect

    def find_visible(self, id: str) -> DetectionEntry | None:
        return next((e for e in self.visible if e.id == id), None)

    def find_not_visible(self, name: str) -> MapEntry | None:
        return next((e for e in self.not_visible if e.name == name), None)

    def is_empty(self) -> bool:
        return not self.visible and not self.not_visible

    def describe(self) -> str:
        if self.is_empty():
            return "Nothing nearby."
        parts = []
        if self.visible:
            parts.append("Visible: " + ", ".join(e.id for e in self.visible) + ".")
        if self.not_visible:
            parts.append(
                "Nearby but not visible: " + ", ".join(e.name for e in self.not_visible) + "."
            )
        return " ".join(parts)


def build_nearby(world_map: WorldMap, detections: Detections, pose: Pose2D, radius: float) -> NearbyView:
    """Filter the detections and the map against the current pose.

    Visible / not-visible status comes from the last `detect`; only the range
    filter tracks the robot as it moves.
    """
    visible = sorted(
        (e for e in detections.all() if e.distance_to(pose) <= radius),
        key=lambda e: e.distance_to(pose),
    )
    detected = detections.map_names()
    not_visible = [e for e in world_map.in_range(pose, radius) if e.name not in detected]
    return NearbyView(visible, not_visible)
