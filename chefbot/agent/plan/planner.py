"""The planner: a user's request, in plain words, to a plan the executor (plan_loop) carries out (AGENT_SPEC §15:
"the planning agent"). An LLM with its own prompt (agents/<agent>/prompts/planner.md) and three tools: ask_user
(a clarifying question), cannot_do (the request can't be done here), submit_plan (checked before it is accepted;
a plan that fails a check goes back with the problems). It sees what the robot knows about the kitchen and what the
robot can do; it never sees the simulator's state.

    planner = Planner(llm, prompt, user=ask)              # user(question) -> answer, or None: nobody to ask
    result = planner.plan(request, kitchen_text, skills_text)
    result.plan        # Plan, or None (result.refused says why)

Replanning (request_replan) calls planner.replan with the executor's reason and the plan so far.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from chefbot.agent.plan.model import DONE, PENDING, SKIPPED, Plan, Step

MAX_TURNS = 8
MAX_QUESTIONS = 2
_ROBOT_ACTION = re.compile(r"\b(i|robot|we)\s+(have\s+)?(placed|picked|put|moved|turned|opened|closed|took|poured)\b", re.I)

_STEP = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "description": 'A short id: "s1", "s2", ...'},
        "goal": {"type": "string", "description": "What the step achieves, in a few words."},
        "done_when": {"type": "string", "description": "A checkable fact about the kitchen when the step is done."},
        "depends_on": {"type": "array", "items": {"type": "string"}, "description": "Ids of steps that must be done first."},
        "hint": {"type": "string", "description": "Optional: timing, what to watch, how to do it."},
    },
    "required": ["id", "goal", "done_when"],
}


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required, "additionalProperties": False}}}  # fmt: skip


TOOLS = [
    _tool("ask_user", "Ask the user one short question, only when the request is ambiguous in a way that changes the dish.",
          {"question": {"type": "string"}}, ["question"]),  # fmt: skip
    _tool("cannot_do", "The request cannot be done with this kitchen and this robot: say why.",
          {"reason": {"type": "string"}}, ["reason"]),  # fmt: skip
    _tool("submit_plan", "Submit the plan. It is checked; if it has problems you get them back to fix.",
          {"goal": {"type": "string", "description": "The dish or end state, in one line."},
           "steps": {"type": "array", "items": _STEP}}, ["goal", "steps"]),  # fmt: skip
]


@dataclass
class PlanResult:
    plan: Plan | None
    refused: str | None = None  # cannot_do's reason
    questions: list[tuple[str, str]] = field(default_factory=list)  # (question, answer)
    rejected: list[list[str]] = field(default_factory=list)  # each rejected submit_plan's problems
    turns: int = 0
    cost_usd: float = 0.0
    transcript: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"plan": self.plan.to_dict() if self.plan else None, "refused": self.refused, "questions": self.questions,
                "rejected": self.rejected, "turns": self.turns, "cost_usd": round(self.cost_usd, 5)}  # fmt: skip


def check_plan(goal: str, steps: list[dict], kept: list[Step] = ()) -> list[str]:
    """The problems with a submitted plan (empty: accepted). `kept`: steps already done (a replan keeps them)."""
    problems = []
    if not isinstance(goal, str) or not goal.strip():
        problems.append("the plan has no goal")
    if not steps:
        problems.append("the plan has no steps")
    ids = [str(s.get("id", "")).strip() for s in steps]
    known = {s.id for s in kept}
    for i, s in enumerate(steps):
        sid = ids[i]
        if not sid:
            problems.append(f"step {i + 1} has no id")
        elif sid in known or ids.count(sid) > 1:
            problems.append(f"step id {sid!r} is used twice")
        if not str(s.get("goal", "")).strip():
            problems.append(f"{sid or i + 1}: no goal")
        dw = str(s.get("done_when", "")).strip()
        if not dw:
            problems.append(f"{sid or i + 1}: no done_when")
        elif _ROBOT_ACTION.search(dw):
            problems.append(f"{sid}: done_when must be a fact about the kitchen, not the robot's action: {dw!r}")
        for d in s.get("depends_on") or []:
            if d not in ids and d not in known:
                problems.append(f"{sid}: depends on {d!r}, which is not a step")
            elif d == sid:
                problems.append(f"{sid}: depends on itself")
    graph = {ids[i]: [d for d in (s.get("depends_on") or []) if d in ids] for i, s in enumerate(steps)}
    if _cycle(graph):
        problems.append("the steps' dependencies form a cycle")
    return problems


def _cycle(graph: dict[str, list[str]]) -> bool:
    state: dict[str, int] = {}

    def visit(n: str) -> bool:
        if state.get(n) == 1:
            return True
        if state.get(n) == 2:
            return False
        state[n] = 1
        if any(visit(m) for m in graph.get(n, [])):
            return True
        state[n] = 2
        return False

    return any(visit(n) for n in graph)


def skills_text(schemas: list[dict], exclude: set[str]) -> str:
    """The robot's skills and tools, one line each (the first sentence of each description)."""
    lines = []
    for s in schemas:
        f = s["function"]
        if f["name"] in exclude:
            continue
        first = re.split(r"(?<=[.:])\s", f["description"].strip(), maxsplit=1)[0]
        args = ", ".join(f["parameters"]["properties"])
        lines.append(f"- {f['name']}({args}): {first}")
    return "\n".join(lines)


def plan_text(plan: Plan) -> str:
    lines = [f"GOAL: {plan.goal}"]
    for s in plan.steps:
        deps = f"  depends: {', '.join(s.depends_on)}" if s.depends_on else ""
        lines.append(f"  {s.id}  {s.status:<8}  {s.goal}{deps}\n            done_when: {s.done_when}"
                     + "".join(f"\n            note: {n}" for n in s.notes[-3:]))  # fmt: skip
    return "\n".join(lines)


class Planner:
    def __init__(self, llm, prompt: str | Path, user: Callable[[str], str] | None = None, max_turns: int = MAX_TURNS):
        self.llm = llm
        self.prompt = Path(prompt).read_text() if isinstance(prompt, Path) or str(prompt).endswith(".md") else str(prompt)
        self.user = user
        self.max_turns = max_turns

    def plan(self, request: str, kitchen: str, skills: str) -> PlanResult:
        text = f"REQUEST: {request}\n\nKITCHEN\n{kitchen}\n\nROBOT\n{skills}"
        return self._run(text, kept=[])

    def replan(self, request: str, reason: str, current: Plan, kitchen: str, skills: str) -> PlanResult:
        """A new plan for what is left. Done and skipped steps are kept as they are; new steps may depend on them."""
        kept = [s for s in current.steps if s.status in (DONE, SKIPPED)]
        text = (f"REPLAN. The executor asked for a new plan: {reason}\n\nREQUEST: {request}\n\nTHE PLAN SO FAR\n"
                f"{plan_text(current)}\n\nSteps that are done or skipped stay; submit only the steps still to do (new ids,"
                f" not {', '.join(s.id for s in kept) or 'any of the above'}). If the request cannot be done any more,"
                f" call cannot_do.\n\nKITCHEN (as the robot knows it now)\n{kitchen}\n\nROBOT\n{skills}")  # fmt: skip
        return self._run(text, kept=kept)

    def _run(self, text: str, kept: list[Step]) -> PlanResult:
        messages = [{"role": "system", "content": self.prompt}, {"role": "user", "content": text}]
        result = PlanResult(None)
        for turn in range(1, self.max_turns + 1):
            response = self.llm.chat(messages, tools=TOOLS)
            result.turns, result.cost_usd = turn, result.cost_usd + (response.cost_usd or 0.0)
            if not response.tool_calls:
                messages += [{"role": "assistant", "content": response.text or ""},
                             {"role": "user", "content": "Answer with exactly one tool call: ask_user, cannot_do or submit_plan."}]  # fmt: skip
                continue
            call = response.tool_calls[0]
            try:
                args = json.loads(call.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            messages.append({"role": "assistant", "content": response.text or None,
                             "tool_calls": [{"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}]})  # fmt: skip
            result.transcript.append({"turn": turn, "tool": call.name, "args": args})
            reply = self._answer(call.name, args, result, kept)
            if result.plan is not None or result.refused is not None:
                return result
            messages.append({"role": "tool", "tool_call_id": call.id, "content": reply})
        result.refused = result.refused or f"no plan after {self.max_turns} turns"
        return result

    def _answer(self, name: str, args: dict, result: PlanResult, kept: list[Step]) -> str:
        if name == "ask_user":
            q = str(args.get("question", "")).strip()
            if len(result.questions) >= MAX_QUESTIONS:
                return "You have asked enough questions: decide, and submit the plan."
            answer = self.user(q) if self.user is not None else None
            answer = answer or "Nobody can answer now. Choose what most people would mean, and say what you assumed in the goal."
            result.questions.append((q, answer))
            return f"The user answers: {answer}"
        if name == "cannot_do":
            result.refused = str(args.get("reason", "")).strip() or "no reason given"
            return ""
        if name == "submit_plan":
            steps = args.get("steps") or []
            steps = [s for s in steps if isinstance(s, dict)]
            problems = check_plan(args.get("goal", ""), steps, kept)
            if problems:
                result.rejected.append(problems)
                return "The plan was not accepted:\n- " + "\n- ".join(problems) + "\nFix these and submit again."
            new = [Step(id=str(s["id"]).strip(), goal=str(s["goal"]).strip(), done_when=str(s["done_when"]).strip(),
                        depends_on=[str(d) for d in s.get("depends_on") or []], hint=s.get("hint") or None, status=PENDING)
                   for s in steps]  # fmt: skip
            result.plan = Plan(str(args["goal"]).strip(), list(kept) + new)
            return "Accepted."
        return f'No tool named "{name}". Tools: ask_user, cannot_do, submit_plan.'
