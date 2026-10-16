"""The run directory (AGENT_SPEC §11).

    run.jsonl    the timeline: one small record per line
    start.json   initial ground truth, reference plan, config, model ids, git commit
    end.json     final ground truth, live plan, edit log, outcome, totals
    requests/    full LLM requests + responses, one file per turn
    checks/      every judge call: condition, observation, verdict

Ground truth (the Observer's text) goes into the log for people to read, never
back into the agent's context.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from chefbot.interface.clock import Clock
from chefbot.interface.observer import Observer


def _json(obj: Any) -> str:
    return json.dumps(obj, default=_default, ensure_ascii=False)


def _default(obj: Any):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


_DURATION = re.compile(r"\s*\((?:for )?\d+:\d\d(?: of heat)?\)")


def world_diff(before: list[str], after: list[str]) -> list[str]:
    """Relation lines removed (-) and added (+). A line whose only change is
    elapsed time ("on (for 0:03)" -> "on (for 0:06)") is not a change."""
    key = lambda line: _DURATION.sub("", line)  # noqa: E731
    old, new = {key(x) for x in before}, {key(x) for x in after}
    return [f"- {x}" for x in before if key(x) not in new] + [f"+ {x}" for x in after if key(x) not in old]


def git_commit(repo: Path) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True).stdout
        return out.stdout.strip() + ("-dirty" if dirty.strip() else "") if out.returncode == 0 else None
    except OSError:
        return None


class RunLog:
    def __init__(self, run_dir: str | Path, clock: Clock, observer: Observer | None = None, full_requests: bool = True):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._observer = observer
        self._full_requests = full_requests
        self._timeline = (self.dir / "run.jsonl").open("a", encoding="utf-8")
        self._checks = 0

    # -- ground truth -----------------------------------------------------

    def world(self) -> list[str]:
        """The Observer's relation lines now ([] without an observer)."""
        if self._observer is None:
            return []
        text = self._observer.observe().text or ""
        return [line.strip() for line in text.splitlines() if line.strip()]

    # -- files ------------------------------------------------------------

    def start(self, **info) -> None:
        (self.dir / "start.json").write_text(_json({"t": self._clock.now(), "world": self.world(), **info}))

    def end(self, **info) -> None:
        (self.dir / "end.json").write_text(_json({"t": self._clock.now(), "world": self.world(), **info}))

    def request(self, turn: int, messages: list[dict], tools: list[dict] | None, response: Any) -> str | None:
        if not self._full_requests:
            return None
        folder = self.dir / "requests"
        folder.mkdir(exist_ok=True)
        name = f"requests/{turn:04d}.json"
        raw = getattr(response, "raw", response)
        (self.dir / name).write_text(_json({"turn": turn, "messages": messages, "tools": tools, "response": raw}))
        return name

    # -- timeline records ---------------------------------------------------

    def record(self, kind: str, turn: int | None = None, **fields) -> dict:
        entry = {"t": round(self._clock.now(), 3), "turn": turn, "kind": kind, **fields}
        self._timeline.write(_json(entry) + "\n")
        self._timeline.flush()
        return entry

    def action(self, turn: int, **fields) -> dict:
        return self.record("action", turn, **fields)

    def event(self, turn: int, event) -> dict:
        fields = event.to_dict()
        fields["event"] = fields.pop("kind")
        fields["event_id"] = fields.pop("id")
        fields.pop("time", None)
        return self.record("event", turn, **fields)

    def check(self, turn: int, step_id: str, condition: str, observation: str | None, result, source: str) -> dict:
        self._checks += 1
        name = f"checks/{self._checks:04d}.json"
        (self.dir / "checks").mkdir(exist_ok=True)
        (self.dir / name).write_text(_json({
            "step": step_id, "condition": condition, "observation": observation,
            "passed": result.passed, "reason": result.reason, "judge_reply": result.raw, "source": source,
        }))  # fmt: skip
        return self.record("check", turn, step=step_id, condition=condition, passed=result.passed,
                           reason=result.reason, source=source, file=name)  # fmt: skip

    def plan_edit(self, turn: int, edit) -> dict:
        return self.record("plan_edit", turn, **edit.to_dict())

    def close(self) -> None:
        self._timeline.close()


def read_timeline(run_dir: str | Path) -> list[dict]:
    path = Path(run_dir) / "run.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
