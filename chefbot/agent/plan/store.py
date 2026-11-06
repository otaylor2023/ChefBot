"""Reference plan, live plan and the edit log (AGENT_SPEC §4.2, §4.3).

The store enforces the status rules; a broken rule raises PlanError with the
error code the agent sees. Every change to the live plan is appended to the
edit log (and passed to `on_edit`, for the run log).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from chefbot.agent.plan.model import (
    ACTIVE,
    DONE,
    FAILED,
    FINISHED,
    PAUSED,
    PAUSING,
    PENDING,
    SKIPPED,
    STATUSES,
    WAITING,
    Plan,
    Step,
)

# status -> statuses set_status may move it to. DONE only through complete().
_ALLOWED = {
    PENDING: {ACTIVE, WAITING, FAILED, SKIPPED},
    ACTIVE: {PAUSING, PAUSED, WAITING, FAILED, SKIPPED},
    PAUSING: {PAUSED, ACTIVE, FAILED},
    PAUSED: {ACTIVE, FAILED, SKIPPED},
    WAITING: {ACTIVE, PAUSED, FAILED, SKIPPED},
    FAILED: {ACTIVE},  # retry
    DONE: set(),
    SKIPPED: set(),
}


class PlanError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class Edit:
    t: float
    step: str | None
    change: str  # "status: active -> paused", "added", "note", ...
    reason: str | None = None

    def to_dict(self) -> dict:
        return {"t": self.t, "step": self.step, "change": self.change, "reason": self.reason}


class PlanStore:
    def __init__(self, reference: Plan, now: Callable[[], float] = lambda: 0.0, on_edit=None):
        self.reference = reference.copy()
        self.live = reference.copy()
        self.edits: list[Edit] = []
        self._now = now
        self._on_edit = on_edit

    # -- queries ----------------------------------------------------------

    def step(self, step_id: str) -> Step:
        step = self.live.step(step_id)
        if step is None:
            known = ", ".join(s.id for s in self.live.steps)
            raise PlanError("unknown_step", f'No step "{step_id}". Steps: {known}.')
        return step

    def active(self) -> Step | None:
        """The step that is active or pausing, if any."""
        return next((s for s in self.live.steps if s.status in (ACTIVE, PAUSING)), None)

    def remaining(self) -> list[Step]:
        return [s for s in self.live.steps if s.status not in FINISHED]

    # -- changes ----------------------------------------------------------

    def set_status(self, step_id: str, status: str, note: str | None = None) -> Step:
        step = self.step(step_id)
        if status == DONE:
            raise PlanError("use_complete", f"Steps are marked done with complete({step_id!r}), which checks them.")
        if status not in STATUSES:
            raise PlanError("invalid_status", f'"{status}" is not a status. Use one of: {", ".join(STATUSES)}.')
        if status == step.status:
            if note:
                self.add_note(step_id, note)
            return step
        if status not in _ALLOWED[step.status]:
            raise PlanError(
                "invalid_transition", f"Step {step_id} cannot go from {step.status} to {status}."
            )
        if status in (ACTIVE, PAUSING):
            other = self.active()
            if other is not None and other.id != step_id:
                raise PlanError(
                    "another_step_active",
                    f"Step {other.id} is {other.status}. Pause it or finish it first (one active step at a time).",
                )
        if status == ACTIVE:
            self._require_dependencies(step)
        if status == PAUSED and not note:
            raise PlanError("note_required", "Pausing a step needs a resume note: what is left to do and where things are.")
        if status in (FAILED, SKIPPED) and not note:
            raise PlanError("note_required", f"Marking a step {status} needs a reason.")
        before = step.status
        step.status = status
        if status in (FAILED, SKIPPED):
            step.reason = note
        if status == PAUSED:
            step.log = f"paused: {note}"
        if note:
            step.notes.append(f"resume: {note}" if status == PAUSED else note)
        self._edit(step_id, f"status: {before} -> {status}", note)
        return step

    def complete(self, step_id: str, log: str | None = None) -> Step:
        """Mark a step done once its check has passed (the caller runs the check)."""
        step = self.step(step_id)
        if step.status in (DONE, SKIPPED):
            raise PlanError("invalid_transition", f"Step {step_id} is already {step.status}.")
        if step.status == PENDING:
            self._require_dependencies(step)
        before = step.status
        step.status = DONE
        notes = [n for n in step.notes if not n.startswith(("resume:", "check failed:", "did:"))]
        step.log = log or (notes[-1] if notes else "completed")
        self._edit(step_id, f"status: {before} -> done", step.log)
        return step

    def can_complete(self, step_id: str) -> None:
        """Raise PlanError if complete() would be refused (before spending a check on it)."""
        step = self.step(step_id)
        if step.status in (DONE, SKIPPED):
            raise PlanError("invalid_transition", f"Step {step_id} is already {step.status}.")
        if step.status == PENDING:
            self._require_dependencies(step)

    def add_note(self, step_id: str, note: str) -> Step:
        step = self.step(step_id)
        step.notes.append(note)
        self._edit(step_id, "note", note)
        return step

    def add_step(
        self, after: str | None, goal: str, done_when: str, reason: str, depends_on: list[str] | None = None
    ) -> Step:
        if not reason:
            raise PlanError("note_required", "Adding a step needs a reason.")
        ids = [s.id for s in self.live.steps]
        for dep in depends_on or []:
            self.step(dep)
        if after is not None:
            self.step(after)
            base = f"{after}a"
        else:
            base = f"s{len(ids) + 1}"
        new_id, suffix = base, 1
        while new_id in ids:
            suffix += 1
            new_id = f"{base}{suffix}" if after is not None else f"s{len(ids) + suffix}"
        step = Step(new_id, goal, done_when, list(depends_on or []), added_because=reason)
        index = ids.index(after) + 1 if after is not None else len(ids)
        self.live.steps.insert(index, step)
        self._edit(new_id, f"added after {after}" if after else "added", reason)
        return step

    def skip(self, step_id: str, reason: str) -> Step:
        return self.set_status(step_id, SKIPPED, reason)

    def _require_dependencies(self, step: Step) -> None:
        unmet = [d for d in step.depends_on if self.step(d).status not in (DONE, SKIPPED)]
        if unmet:
            raise PlanError(
                "dependencies_not_met", f"Step {step.id} depends on {', '.join(unmet)}, not done yet."
            )

    def replace(self, plan: Plan, reason: str, reference: bool = False) -> None:
        """A new plan from the planner: it becomes the live plan (a replan keeps the done and skipped steps in it).
        reference: also the reference plan (the first plan of a run planned from a request)."""
        self.live = plan.copy()
        if reference:
            self.reference = plan.copy()
        self._edit(None, f"new plan: {len(plan.steps)} steps", reason)

    def _edit(self, step_id: str | None, change: str, reason: str | None) -> None:
        edit = Edit(self._now(), step_id, change, reason)
        self.edits.append(edit)
        if self._on_edit is not None:
            self._on_edit(edit)
