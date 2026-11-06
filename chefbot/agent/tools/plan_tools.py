"""Plan tools: complete, set_status, add_step, skip_step, request_replan, finish (AGENT_SPEC §7.2)."""

from __future__ import annotations

from typing import Callable

from chefbot.agent.check.checker import CheckResult
from chefbot.agent.events.bus import EventBus
from chefbot.agent.plan.model import FINISHED
from chefbot.agent.plan.store import PlanError, PlanStore
from chefbot.interface.types import ToolResult

# complete refused by the check this many times in a row is blocked until a call that changes the kitchen succeeds.
REFUSALS_BEFORE_BLOCK = 2
CHANGES_KITCHEN = {"pick_up", "place", "open", "close", "turn_on", "turn_off", "lower_into", "lift_out", "squeeze", "release",
                   "stir", "pour", "wait"}  # wait: time passes (food cooks)  # fmt: skip

# The operator's answer to request_replan: "continue", "abort" or "reload" (plan file edited).
Operator = Callable[[str], str]


def terminal_operator(reason: str) -> str:
    print(f"\nThe agent asks for a new plan: {reason}")
    while True:
        answer = input("continue / abort / reload (after editing the plan file)? ").strip().lower()
        if answer in ("continue", "abort", "reload"):
            return answer


class PlanTools:
    """`check(step_id, condition) -> CheckResult` observes the world and judges the
    condition (and logs the check); the agent never sees what it observed."""

    def __init__(
        self,
        store: PlanStore,
        check: Callable[[str, str], CheckResult],
        bus: EventBus | None = None,
        operator: Operator = terminal_operator,
        reload: Callable[[], None] | None = None,
        open_doors: Callable[[], list[str]] | None = None,
    ):
        self._store = store
        self._check = check
        self._bus = bus
        self._operator = operator
        self._reload = reload
        self._open_doors = open_doors  # doors the robot opened and hasn't closed (the kitchen belief)
        self._warned_open = False
        self.outcome: tuple[str, str] | None = None  # (outcome, summary) once the run should end
        # complete refused by the check, in a row, per step, and whether the kitchen changed since the last refusal
        self._refused: dict[str, int] = {}
        self._changed_since: dict[str, bool] = {}

    def note_action(self, tool: str, success: bool) -> None:
        """After every tool call (the runner): a successful call that changes the kitchen lets complete be tried again."""
        if success and tool in CHANGES_KITCHEN:
            self._changed_since = {k: True for k in self._changed_since}

    def complete(self, step_id: str) -> ToolResult:
        try:
            self._store.can_complete(step_id)
            step = self._store.step(step_id)
        except PlanError as exc:
            return _error(exc)
        refused = self._refused.get(step_id, 0)
        if refused >= REFUSALS_BEFORE_BLOCK and not self._changed_since.get(step_id, True):
            last = next((n for n in reversed(step.notes) if n.startswith("check failed:")), "check failed")
            return ToolResult(False, "complete_refused_again",
                              f"The check refused {step_id} {refused}x and nothing in the kitchen changed since ({last}). Calling "
                              "complete again gives the same answer: do what the check says is missing first, or skip_step / "
                              "add_step / request_replan if the plan is wrong.", {"step_id": step_id})  # fmt: skip
        result = self._check(step_id, step.done_when)
        if not result.passed:
            self._refused[step_id] = refused + 1
            self._changed_since[step_id] = False
            self._store.add_note(step_id, f"check failed: {result.reason}")
            return ToolResult(
                False, "check_failed", f"{step_id} is not done: {result.reason}",
                {"step_id": step_id, "reason": result.reason},
            )  # fmt: skip
        self._store.complete(step_id)
        self._refused.pop(step_id, None)
        self._changed_since.pop(step_id, None)
        if self._bus is not None:
            self._bus.unwatch(step_id)
        return ToolResult(True, None, f"{step_id} done: {result.reason}", {"step_id": step_id, "reason": result.reason})

    def set_status(self, step_id: str, status: str, note: str | None = None) -> ToolResult:
        try:
            step = self._store.set_status(step_id, status, note)
        except PlanError as exc:
            return _error(exc)
        message = f"{step_id} is {step.status}."
        if step.status == "waiting" and self._bus is not None and not self._watched(step_id):
            message += (" Nothing is timing or watching it, so no event will tell you when it is ready:"
                        " set_timer or watch it.")  # fmt: skip
        return ToolResult(True, None, message, {"step_id": step_id, "status": step.status})

    def _watched(self, step_id: str) -> bool:
        return step_id in self._bus.monitors or any(t.step_id == step_id for t in self._bus.timers.values())

    def add_step(
        self, goal: str, done_when: str, reason: str, after: str | None = None, depends_on: list[str] | None = None
    ) -> ToolResult:
        try:
            step = self._store.add_step(after, goal, done_when, reason, depends_on)
        except PlanError as exc:
            return _error(exc)
        return ToolResult(True, None, f"Added step {step.id}: {goal}", {"step_id": step.id})

    def skip_step(self, step_id: str, reason: str) -> ToolResult:
        try:
            self._store.skip(step_id, reason)
        except PlanError as exc:
            return _error(exc)
        return ToolResult(True, None, f"{step_id} skipped.", {"step_id": step_id})

    def request_replan(self, reason: str) -> ToolResult:
        answer = self._operator(reason)
        if answer == "abort":
            self.outcome = ("aborted", f"replan requested: {reason}")
            return ToolResult(True, None, "The operator ended the run.", {"answer": answer})
        if answer == "reload" and self._reload is not None:
            self._reload()
            return ToolResult(True, None, "The operator edited the plan; see the new plan below.", {"answer": answer})
        return ToolResult(True, None, "The operator says to continue with the current plan.", {"answer": "continue"})

    def finish(self, summary: str, abandon: bool = False) -> ToolResult:
        doors = self._open_doors() if self._open_doors is not None else []
        if doors and not self._warned_open:  # once: a door that won't close can't trap the run
            self._warned_open = True
            return ToolResult(False, "doors_open",
                              f"You opened {', '.join(doors)} and left {'it' if len(doors) == 1 else 'them'} open. Close "
                              f"{'it' if len(doors) == 1 else 'them'} before you finish (an open door blocks the floor, and "
                              "a kitchen is left as it was found).", {"open": doors})  # fmt: skip
        remaining = [s for s in self._store.live.steps if s.status not in FINISHED]
        if remaining and not abandon:
            listed = ", ".join(f"{s.id} ({s.status})" for s in remaining)
            return ToolResult(
                False, "steps_remaining",
                f"Not every step is done, skipped or failed: {listed}. Finish them, or finish with abandon=true.",
                {"remaining": [s.id for s in remaining]},
            )  # fmt: skip
        self.outcome = ("abandoned" if remaining else "finished", summary)
        return ToolResult(True, None, "Run finished.", {"outcome": self.outcome[0]})


def _error(exc: PlanError) -> ToolResult:
    return ToolResult(False, exc.code, exc.message)
