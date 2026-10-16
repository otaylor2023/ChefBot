"""A scripted 'LLM' for the symbolic env: a fixed policy that reads ids from the
agent's own perception state, so whole tasks run through the loop without an LLM."""

from __future__ import annotations

import json
import re

import numpy as np

from chefbot.agent.check.checker import CheckResult
from chefbot.agent.loop.llm import LLMResponse, ToolCall


def reply(tool: str, **args) -> LLMResponse:
    return LLMResponse(None, [ToolCall(f"call_{tool}", tool, json.dumps(args))])


class Policy:
    """Steps are (tool, args) tuples or callables(session) -> (tool, args)."""

    def __init__(self, session_ref: dict, steps: list):
        self._ref = session_ref
        self._steps = list(steps)

    def chat(self, messages, tools=None, **kwargs) -> LLMResponse:
        if not self._steps:
            return reply("finish", summary="script ended", abandon=True)
        step = self._steps.pop(0)
        tool, args = step(self._ref["session"]) if callable(step) else step
        return reply(tool, **args)


def visible(session, label: str) -> list[str]:
    return [d["id"] for d in session.perception.visible if d.get("label") == label]


def first(label: str, tool: str, **extra):
    """`tool(id=<nearest visible id with this label>)`."""
    return lambda s: (tool, {"id": visible(s, label)[0], **extra})


def nearest(label: str, to_label: str, tool: str):
    """`tool(id=<visible `label` nearest the visible `to_label`>)`: e.g. the burner under the pan."""

    def step(s):
        entries = {e.id: e for e in s.registry.tools.detections.all()}
        anchor = entries[visible(s, to_label)[0]].position_world
        ids = visible(s, label)
        best = min(ids, key=lambda i: float(np.linalg.norm(entries[i].position_world[:2] - anchor[:2])))
        return tool, {"id": best}

    return step


class TruthChecker:
    """Judges a done_when by ground-truth regexes (tests only): {done_when: [regex, ...]}."""

    def __init__(self, rules: dict[str, list[str]]):
        self._rules = rules

    def check(self, condition: str, obs) -> CheckResult:
        lines = (obs.text or "").splitlines()
        missing = [p for p in self._rules.get(condition, []) if not any(re.search(p, line) for line in lines)]
        return CheckResult(not missing, "ok" if not missing else f"missing {missing}")
