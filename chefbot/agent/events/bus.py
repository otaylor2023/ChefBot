"""Events, timers, monitors and safety monitors (AGENT_SPEC §6).

One queue, owned by the agent layer. Nothing here runs on its own: `poll()`
fires whatever is due at `clock.now()`, and is called while waiting, while a
blocking motion runs, and at the start of every turn.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Callable

from chefbot.agent.check.checker import CheckResult
from chefbot.interface.clock import Clock

TIMER, MONITOR, SAFETY, MOTION_DONE = "timer", "monitor", "safety", "motion_done"


@dataclass
class Event:
    id: int
    time: float
    kind: str
    message: str
    step_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"id": self.id, "time": round(self.time, 2), "kind": self.kind, "message": self.message,
                "step_id": self.step_id, "data": self.data}  # fmt: skip


@dataclass
class Timer:
    name: str
    due: float
    step_id: str | None = None


@dataclass
class Monitor:
    step_id: str
    condition: str
    every_s: float
    deadline: float
    next_check: float
    last_reason: str = ""


class EventBus:
    """`check(step_id, condition) -> CheckResult` runs a monitor's check (observe,
    judge, log). Safety monitors are callables `(now) -> list[(message, data)]`."""

    def __init__(self, clock: Clock, check: Callable[[str, str], CheckResult] | None = None):
        self.clock = clock
        self._check = check
        self._ids = itertools.count(1)
        self._queue: list[Event] = []
        self.timers: dict[str, Timer] = {}
        self.monitors: dict[str, Monitor] = {}
        self._safety: list[Callable[[float], list[tuple[str, dict]]]] = []
        self.delivered: list[Event] = []  # everything ever drained, for the run log

    # -- producers ----------------------------------------------------------

    def post(self, kind: str, message: str, step_id: str | None = None, data: dict | None = None) -> Event:
        event = Event(next(self._ids), self.clock.now(), kind, message, step_id, dict(data or {}))
        self._queue.append(event)
        return event

    def set_timer(self, name: str, seconds: float, step_id: str | None = None) -> Timer:
        timer = Timer(name, self.clock.now() + max(0.0, float(seconds)), step_id)
        self.timers[name] = timer  # setting a running timer again restarts it
        return timer

    def cancel_timer(self, name: str) -> bool:
        return self.timers.pop(name, None) is not None

    def watch(self, step_id: str, condition: str, every_s: float, timeout_s: float) -> Monitor:
        if self._check is None:
            raise RuntimeError("this run has no state checker; monitors cannot run")
        now = self.clock.now()
        monitor = Monitor(step_id, condition, max(1.0, float(every_s)), now + float(timeout_s), now + float(every_s))
        self.monitors[step_id] = monitor
        return monitor

    def unwatch(self, step_id: str) -> bool:
        return self.monitors.pop(step_id, None) is not None

    def add_safety(self, monitor: Callable[[float], list[tuple[str, dict]]]) -> None:
        self._safety.append(monitor)

    # -- consumers ----------------------------------------------------------

    def poll(self) -> list[Event]:
        """Fire everything due now; returns the events this produced (also queued)."""
        now = self.clock.now()
        fired: list[Event] = []
        for name, timer in sorted(self.timers.items(), key=lambda kv: kv[1].due):
            if timer.due <= now:
                del self.timers[name]
                fired.append(self.post(TIMER, f"timer '{name}' is up", timer.step_id, {"timer": name}))
        for step_id, monitor in list(self.monitors.items()):
            if monitor.next_check > now:
                continue
            result = self._check(step_id, monitor.condition)
            monitor.last_reason = result.reason
            if result.passed:
                del self.monitors[step_id]
                fired.append(self.post(MONITOR, f"monitor passed for {step_id}: {result.reason}", step_id,
                                       {"passed": True, "reason": result.reason}))  # fmt: skip
            elif now >= monitor.deadline:
                del self.monitors[step_id]
                fired.append(self.post(MONITOR, f"monitor timed out for {step_id}: {result.reason}", step_id,
                                       {"passed": False, "timeout": True, "reason": result.reason}))  # fmt: skip
            else:
                monitor.next_check = min(now + monitor.every_s, max(monitor.deadline, now))
        for safety in self._safety:
            for message, data in safety(now):
                fired.append(self.post(SAFETY, message, None, data))
        return fired

    def has_pending(self) -> bool:
        """Undelivered events in the queue."""
        return bool(self._queue)

    def drain(self) -> list[Event]:
        """Take every queued event (oldest first)."""
        events, self._queue = self._queue, []
        self.delivered.extend(events)
        return events

    def active(self) -> bool:
        """Anything that can still produce an event by itself (timers, monitors)."""
        return bool(self.timers or self.monitors)

    def next_due(self) -> float:
        """The run time of the next timer or monitor check (inf if none)."""
        times = [t.due for t in self.timers.values()]
        times += [min(m.next_check, m.deadline) for m in self.monitors.values()]
        return min(times, default=math.inf)
