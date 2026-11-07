"""chefbot-v1 skills as an agent's tools: each call runs the skill as one sandboxed episode (episode.py) on a host kept
open for the agent's whole run, continuing the world as the last skill left it (restart=False: nothing is reset).

    skills_tools = EpisodeSkills(host, run_dir / "skills", ledger, vision)
    schema, handler = skills_tools.tool(skills.load("pick_up/gemini_v1"))
    handler(object="apple")   # -> ToolResult: success if the episode ran to its end and the skill's `ok` isn't false

The skill's result goes to the LLM as is (what run returned: {ok, reason, ...}); the episode's files go to
<out>/<NN>_<skill>/ (result.json, trace.jsonl, skill_log.txt, its images).
"""

from __future__ import annotations

import json
from pathlib import Path

from chefbot.interface.types import ToolResult
from chefbot.runtime import models as M
from chefbot.runtime.episode import LIMITS, run_episode


class EpisodeSkills:
    def __init__(self, host, out: Path, ledger: M.Ledger, vision: M.Vision | None, limits: dict | None = None,
                 confirm: bool = False, arm_only: bool = True):  # fmt: skip
        self.host, self.out, self.ledger, self.vision = host, Path(out), ledger, vision
        self.limits = {**LIMITS, **(limits or {})}
        self.confirm, self.arm_only = confirm, arm_only
        self.calls = 0

    def move_to(self, out: Path) -> None:
        """Put the episodes' files (and the model ledger) under `out`: the agent's run folder."""
        self.out = Path(out)
        self.ledger = M.Ledger(self.out / "model_ledger.jsonl", self.ledger.cap)

    def tool(self, skill) -> tuple[dict, callable]:
        """The skill's schema and the handler that runs it."""
        if not skill.sandboxed:
            raise ValueError(f"skill {skill.path} is not sandboxed: it runs in the agent's process (ToolRegistry)")

        def handler(**args) -> ToolResult:
            return self.run(skill, args)

        handler.__name__ = skill.name
        handler.__signature__ = _signature(skill)  # the registry binds the call's arguments to it
        return skill.schema, handler

    def run(self, skill, args: dict) -> ToolResult:
        self.calls += 1
        out = self.out / f"{self.calls:02d}_{skill.name}"
        access = {**skill.access(), "confirm_moves": self.confirm}
        if self.arm_only:
            access["arm_only"] = True  # the base holds: no skill here drives it
        result = run_episode(self.host, None, 0, skill.dir, out, self.ledger, self.vision, access, interface=skill.interface,
                             limits=self.limits, record=False, tag="agent", restart=False, args=args or None)  # fmt: skip
        returned, end = result.get("returned"), result.get("end")
        said = f"{skill.name}({', '.join(f'{k}={json.dumps(v)}' for k, v in args.items())})"
        data = {"skill": skill.path, "end": end, "returned": returned, "robot_s": result.get("robot_time_s"),
                "model_usd": result.get("model_usd"), "files": str(out)}  # fmt: skip
        if end != "skill returned":
            return ToolResult(False, "skill_error", f"{said}: the skill did not finish ({end}); see {out / 'skill_log.txt'}", data)
        if isinstance(returned, dict) and returned.get("ok") is False:
            return ToolResult(False, "skill_failed", f"{said}: {returned.get('reason', 'failed')}", data)
        return ToolResult(True, None, f"{said}: {json.dumps(returned, default=str)}", data)


def _signature(skill):
    import inspect

    params = [inspect.Parameter(a, inspect.Parameter.KEYWORD_ONLY, **({"default": spec["default"]} if "default" in spec else {}))
              for a, spec in skill.args.items()]  # fmt: skip
    return inspect.Signature(params)
