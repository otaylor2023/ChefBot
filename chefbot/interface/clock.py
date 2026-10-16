"""Clock: run time for timers, cooking and logs (AGENT_SPEC §3)."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Clock(ABC):
    @abstractmethod
    def now(self) -> float:
        """Seconds since the run started. Sim: sim time. Real: wall time."""

    @abstractmethod
    def sleep(self, seconds: float) -> None:
        """Let `seconds` of run time pass while the robot is idle. Sim: steps the
        env (burners, cooking and timers progress); real: waits."""


class ManualClock(Clock):
    """A clock that only moves when told to. For tests and scripted runs."""

    def __init__(self, start: float = 0.0):
        self.t = float(start)

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(0.0, float(seconds))
