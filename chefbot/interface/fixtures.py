"""FixtureActuator: open and close fixtures (doors, drawers).

Scripted in sim for v0.1; a real implementation (arm on the handle) keeps the
same interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from chefbot.interface.types import ActuationResult, Detection

OPEN = "open"
CLOSED = "closed"
UNKNOWN = "unknown"


class FixtureActuator(ABC):
    @abstractmethod
    def set_state(self, fixture: Detection, open: bool) -> ActuationResult:
        """Open or close the fixture `fixture` refers to (the fixture itself or
        one of its handles). `success=False` if it is not something that opens."""

    @abstractmethod
    def get_state(self, fixture: Detection) -> str:
        """OPEN | CLOSED | UNKNOWN (also for things that do not open)."""

    def swept_xy(self, fixture: Detection, open: bool) -> np.ndarray | None:
        """World xy points (N, 2) the fixture passes through on its way to the
        requested state (a door's arc). The robot keeps its base out of them
        while the fixture moves. None: nothing sweeps the floor."""
        return None
