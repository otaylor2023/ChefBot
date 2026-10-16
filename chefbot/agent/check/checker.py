"""StateChecker: does a step's `done_when` hold? (AGENT_SPEC §5.1)

v0.1 asks a small LLM to judge the condition against the Observer's text.
Later: a VLM judge over images, plus proprioception / force. Same interface.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from chefbot.interface.observer import Observation

JUDGE_PROMPT = """You check whether a condition holds in a kitchen, given a description of its current state.

The state lists relations, one per line ("egg: inside pan", "burner_1: on (for 2:10)").
Judge only from the state. If the state does not say something the condition needs, it does not hold.
Check every part of the condition; it holds only if all parts do.

Food goes through cook states in order, e.g. egg: raw -> cooking -> set -> overcooked;
bread: raw -> warm -> toasted -> burnt. A condition that names a cook state holds only if the
state shows exactly that one ("cooking" is not "set"; "overcooked" is not "set").

Reply with one JSON object and nothing else:
{"passed": true or false, "reason": "<one short sentence naming the relations you used>"}"""


@dataclass
class CheckResult:
    passed: bool
    reason: str
    confidence: float | None = None
    raw: str | None = None  # the judge's reply, for the run log


class StateChecker:
    def __init__(self, llm, model: str, temperature: float = 0.0):
        self._llm = llm
        self._model = model
        self._temperature = temperature

    def check(self, condition: str, obs: Observation) -> CheckResult:
        if not obs.text:
            return CheckResult(False, "no observation to judge from")
        messages = [
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": f"CONDITION\n{condition}\n\nSTATE\n{obs.text}"},
        ]
        try:
            reply = self._llm.chat(messages, model=self._model, temperature=self._temperature)
        except Exception as exc:  # a failed check, never a crash
            return CheckResult(False, f"judge error: {exc}")
        return parse_verdict(reply.text or "")


def parse_verdict(text: str) -> CheckResult:
    """The judge's JSON verdict, tolerating code fences and surrounding prose."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            verdict = json.loads(match.group(0))
            return CheckResult(bool(verdict.get("passed")), str(verdict.get("reason", "")).strip(), raw=text)
        except (json.JSONDecodeError, AttributeError):
            pass
    return CheckResult(False, f"could not read the judge's verdict: {text[:200]!r}", raw=text)
