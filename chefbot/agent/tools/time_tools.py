"""Time tools: set_timer, cancel_timer, watch, and the event-aware wait (AGENT_SPEC §6, §7.3, §7.4)."""

from __future__ import annotations

import math

from chefbot.agent.events.bus import MOTION_DONE, Event, EventBus
from chefbot.agent.plan.store import PlanError, PlanStore
from chefbot.interface.clock import Clock
from chefbot.interface.robot_api import RobotAPI
from chefbot.interface.types import MotionResult, MotionStatus, ToolResult

_MOTION_SLICE_S = 0.05  # wall seconds between event polls while a motion runs
_MAX_IDLE_STEP_S = 1.0  # run seconds per idle step while nothing is moving


class TimeTools:
    def __init__(self, robot: RobotAPI, bus: EventBus, clock: Clock, store: PlanStore | None = None):
        self._robot = robot
        self._bus = bus
        self._clock = clock
        self._store = store
        self._was_moving = False
        self.motion_name: str | None = None
        # Called with the final MotionResult of a motion left running (NavTools' bookkeeping).
        self.on_motion_end = None

    # -- tools ------------------------------------------------------------

    def set_timer(self, name: str, seconds: float, step_id: str | None = None) -> ToolResult:
        if seconds <= 0:
            return ToolResult(False, "invalid_arguments", "seconds must be positive")
        if step_id is not None and (err := self._known_step(step_id)) is not None:
            return err
        replaced = name in self._bus.timers
        self._bus.set_timer(name, seconds, step_id)
        what = "restarted" if replaced else "set"
        return ToolResult(True, None, f"Timer '{name}' {what}: {_duration(seconds)}.", {"timer": name, "seconds": seconds})

    def cancel_timer(self, name: str) -> ToolResult:
        if not self._bus.cancel_timer(name):
            running = ", ".join(self._bus.timers) or "none"
            return ToolResult(False, "unknown_timer", f"No timer '{name}'. Running: {running}.")
        return ToolResult(True, None, f"Timer '{name}' cancelled.")

    def watch(self, step_id: str, every_s: float, timeout_s: float) -> ToolResult:
        if self._store is None:
            return ToolResult(False, "unsupported", "There is no plan to watch steps of.")
        try:
            step = self._store.step(step_id)
        except PlanError as exc:
            return ToolResult(False, exc.code, exc.message)
        if every_s <= 0 or timeout_s <= 0:
            return ToolResult(False, "invalid_arguments", "every_s and timeout_s must be positive")
        try:
            self._bus.watch(step_id, step.done_when, every_s, timeout_s)
        except RuntimeError as exc:
            return ToolResult(False, "unsupported", str(exc))
        return ToolResult(
            True, None,
            f"Watching {step_id} every {_duration(every_s)} for up to {_duration(timeout_s)}: "
            f'"{step.done_when}". You get an event when it holds (then call complete) or when the time is up.',
            {"step_id": step_id},
        )  # fmt: skip

    def wait(self, timeout_s: float | None = None, finish=None) -> ToolResult:
        """Returns on the first of: the current motion ends, any event, the timeout.
        `finish(MotionResult) -> ToolResult` reports a motion that ended (NavTools)."""
        moving = self._robot.get_status().status == MotionStatus.MOVING
        if not moving and not self._bus.active() and not self._bus.has_pending() and timeout_s is None:
            return ToolResult(
                False, "nothing_to_wait_for",
                "Nothing is moving and no timer or monitor is running. Set a timer, watch a step, or pass timeout_s.",
            )  # fmt: skip
        result, events, why = self.await_events(timeout_s)
        data = {"status": result.status if result else MotionStatus.IDLE, "woke_for": why}
        if events:
            data["events"] = [e.to_dict() for e in events]
        if why == "motion_done" and result is not None:
            if finish is not None:
                out = finish(result)
                out.data["woke_for"] = why
                return out
            ok = result.status == MotionStatus.SUCCESS
            message = f"wait: the motion ended: {result.status}" + (f" ({result.message})" if result.message else "")
            return ToolResult(ok, None if ok else result.status, message, {**data, **result.data})
        if why == "event":
            return ToolResult(True, None, "wait: " + "; ".join(e.message for e in events), data)
        return ToolResult(True, None, f"wait: {_duration(timeout_s or 0)} passed, no events.", data)

    # -- waiting ----------------------------------------------------------

    def await_events(self, timeout_s: float | None = None) -> tuple[MotionResult | None, list[Event], str]:
        """Let time pass until the current motion ends ("motion_done"), an event
        arrives ("event") or `timeout_s` of run time passes ("timeout").
        Events are drained and returned; a motion keeps going after an event."""
        clock, bus = self._clock, self._bus
        deadline = clock.now() + timeout_s if timeout_s is not None else math.inf
        started_moving = self._robot.get_status().status == MotionStatus.MOVING
        while True:
            bus.poll()
            if bus.has_pending():
                return self._robot.get_status(), bus.drain(), "event"
            if started_moving:
                status = self._robot.wait(_MOTION_SLICE_S)
                if status.status != MotionStatus.MOVING:
                    self._was_moving = False
                    return status, [], "motion_done"
                if clock.now() >= deadline:
                    return status, [], "timeout"
                continue
            now = clock.now()
            if now >= deadline:
                return self._robot.get_status(), [], "timeout"
            if not bus.active() and math.isinf(deadline):
                return self._robot.get_status(), [], "nothing"
            clock.sleep(max(1e-3, min(bus.next_due(), deadline, now + _MAX_IDLE_STEP_S) - now))

    def note_motion(self, name: str, moving: bool) -> None:
        """A motion tool returned; remember a motion left running (wait=false or woken early)."""
        self._was_moving = moving
        self.motion_name = name if moving else None

    def poll_motion(self) -> None:
        """Turn start: a motion left running that has since ended becomes an event."""
        if not self._was_moving:
            return
        status = self._robot.get_status()
        if status.status != MotionStatus.MOVING:
            self._was_moving = False
            if self.on_motion_end is not None:
                self.on_motion_end(status)
            message = f"{self.motion_name or 'the motion'} ended: {status.status}"
            if status.message:
                message += f" ({status.message})"
            self._bus.post(MOTION_DONE, message, data={"status": status.status})

    def _known_step(self, step_id: str) -> ToolResult | None:
        if self._store is None:
            return None
        try:
            self._store.step(step_id)
        except PlanError as exc:
            return ToolResult(False, exc.code, exc.message)
        return None


def _duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}" if seconds >= 60 else f"{seconds} s"
