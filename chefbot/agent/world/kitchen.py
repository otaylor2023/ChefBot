"""What the agent believes about the kitchen beyond the map's geometry: the
state of doors, burners and appliances, and where objects are and why it
thinks so. Built only from the agent's own records (the map, its tool results,
detections), never from ground truth. Rendered as the KITCHEN block.
"""

from __future__ import annotations

from dataclasses import dataclass

YOU, SEEN, MAP = "you", "seen", "map"


@dataclass
class Fact:
    value: str  # a state ("open", "on") or, for an object, where it is (a map name or detection id)
    source: str  # YOU (a tool you called), SEEN (a detection), MAP (as mapped)
    t: float  # run time it was learned


class KitchenBelief:
    def __init__(self) -> None:
        self.states: dict[str, Fact] = {}  # fixture / burner name -> state
        self.where: dict[str, Fact] = {}  # object name -> what it is on or in

    def set_state(self, name: str, state: str, source: str, t: float) -> None:
        old = self.states.get(name)
        # A detection does not overwrite what you just did at the same moment.
        if old is not None and old.source == YOU and source == SEEN and old.value == state:
            return
        self.states[name] = Fact(state, source, t)

    def put(self, obj: str, holder: str, source: str, t: float) -> None:
        old = self.where.get(obj)
        if old is not None and old.value == holder and source == SEEN and old.source == YOU:
            return  # still where you put it
        self.where[obj] = Fact(holder, source, t)

    def forget(self, obj: str) -> None:
        self.where.pop(obj, None)
