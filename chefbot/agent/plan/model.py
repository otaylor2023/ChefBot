"""Plan and Step (AGENT_SPEC §4.1). `done_when` is plain text, judged by the StateChecker."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

PENDING, ACTIVE, PAUSING, PAUSED = "pending", "active", "pausing", "paused"
WAITING, DONE, FAILED, SKIPPED = "waiting", "done", "failed", "skipped"
STATUSES = (PENDING, ACTIVE, PAUSING, PAUSED, WAITING, DONE, FAILED, SKIPPED)
FINISHED = (DONE, FAILED, SKIPPED)  # a run can finish once every step is in one of these


@dataclass
class Step:
    id: str
    goal: str
    done_when: str
    depends_on: list[str] = field(default_factory=list)
    hint: str | None = None
    status: str = PENDING
    notes: list[str] = field(default_factory=list)  # agent notes, resume notes, check failures
    log: str | None = None  # one line, written when the step ends or pauses
    added_because: str | None = None  # set for steps added mid-run
    reason: str | None = None  # why it failed or was skipped


@dataclass
class Plan:
    goal: str
    steps: list[Step]

    def step(self, step_id: str) -> Step | None:
        return next((s for s in self.steps if s.id == step_id), None)

    def copy(self) -> "Plan":
        return copy.deepcopy(self)

    def to_dict(self) -> dict:
        return {"goal": self.goal, "steps": [asdict(s) for s in self.steps]}

    @classmethod
    def from_dict(cls, raw: dict) -> "Plan":
        steps = []
        for s in raw["steps"]:
            steps.append(
                Step(
                    id=str(s["id"]),
                    goal=s["goal"],
                    done_when=s["done_when"],
                    depends_on=[str(d) for d in s.get("depends_on") or []],
                    hint=s.get("hint"),
                    status=s.get("status", PENDING),
                    notes=list(s.get("notes") or []),
                    log=s.get("log"),
                    added_because=s.get("added_because"),
                    reason=s.get("reason"),
                )
            )
        ids = [s.id for s in steps]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate step ids in plan: {ids}")
        for s in steps:
            missing = [d for d in s.depends_on if d not in ids]
            if missing:
                raise ValueError(f"step {s.id} depends on unknown steps {missing}")
        return cls(raw["goal"], steps)

    @classmethod
    def load(cls, path: str | Path) -> "Plan":
        """A plan from a task file (its `goal` and `steps`)."""
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))
