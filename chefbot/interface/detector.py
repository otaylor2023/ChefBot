"""Detector: what the robot currently sees, in ChefBot labels."""

from __future__ import annotations

from abc import ABC, abstractmethod

from chefbot.interface.types import Detection


class Detector(ABC):
    @abstractmethod
    def detect(self, classes: list[str]) -> list[Detection]:
        """Detect instances of `classes` (ChefBot labels) through the shoulder
        camera from the robot's current pose. Positions are in the world frame."""
