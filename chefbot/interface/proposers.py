"""Perception proposers: where to grasp an object and where to put it down.

Like `Detector`, these are provided by the env in sim (ground truth) and by
real perception later. The robot layer receives them by injection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from chefbot.interface.types import Detection, Grasp, HeldObject, Placement


class GraspProposer(ABC):
    @abstractmethod
    def propose(self, target: Detection) -> list[Grasp]:
        """Grasp candidates for the object at `target`, best first. Empty if
        nothing graspable is there."""


class PlacementProposer(ABC):
    @abstractmethod
    def propose(self, surface: Detection, held: HeldObject) -> list[Placement]:
        """Release poses that set the held object down on `surface`, best first.
        Empty if it cannot be placed there."""
