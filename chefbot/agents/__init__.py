"""Agents: one folder per agent (agent.yaml, prompts/), loaded by name. An agent names the harness it runs in
(agent/harnesses), the robot APIs it runs with, its prompts, the tools it gets (chefbot/tools), its tools'
settings and its loop's settings.

    agent = agents.load("kitchen_planner")
    agent.harness.AgentSession.build(..., agent_config=agent.settings, prompts_dir=agent.prompts_dir,
                                     agent_tools=agent.tools)
    agent.loop_config(model="...")      # LoopConfig, with run-time overrides

Today's cooking agent is kitchen_planner (DEFAULT), the only one so far.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import yaml

from chefbot import tools as toolbox
from chefbot.agent import harnesses
from chefbot.agent.config import AgentConfig, LoopConfig
from chefbot.interface import INTERFACES

ROOT = Path(__file__).resolve().parent
DEFAULT = "kitchen_planner"
KEYS = {"name", "harness", "interfaces", "prompts", "tools", "settings", "loop"}


def names() -> list[str]:
    return sorted(p.parent.name for p in ROOT.glob("*/agent.yaml"))


def load(name: str) -> "Agent":
    """An agent by name (a folder here), or by path to its folder."""
    d = Path(name) if Path(name).is_absolute() else ROOT / name
    if not (d / "agent.yaml").exists():
        raise ValueError(f"no agent {name!r}; agents: {', '.join(names())}")
    agent = Agent(d.name, d, yaml.safe_load((d / "agent.yaml").read_text()) or {})
    agent.validate()
    return agent


@dataclass
class Agent:
    name: str
    dir: Path
    manifest: dict = field(repr=False)

    @property
    def path(self) -> Path:
        return self.dir / "agent.yaml"

    @property
    def harness_name(self) -> str:
        return self.manifest["harness"]

    @property
    def harness(self) -> ModuleType:
        return harnesses.get(self.harness_name)

    @property
    def interfaces(self) -> tuple[str, ...]:
        return tuple(self.manifest["interfaces"])

    @property
    def prompts_dir(self) -> Path:
        return self.dir / self.manifest.get("prompts", "prompts")

    @property
    def tools(self) -> tuple[str, ...]:
        return tuple(self.manifest.get("tools") or ())

    @property
    def settings(self) -> AgentConfig:
        return AgentConfig.load(self.path)

    def loop_config(self, **overrides) -> LoopConfig:
        return LoopConfig.load(self.path, **overrides)

    def validate(self) -> None:
        m, problems = self.manifest, []
        unknown = set(m) - KEYS
        if unknown:
            problems.append(f"unknown keys {sorted(unknown)}")
        if m.get("name") != self.name:
            problems.append(f"name {m.get('name')!r} is not its folder's, {self.name!r}")
        if m.get("harness") not in harnesses.HARNESSES:
            problems.append(f"no harness {m.get('harness')!r} (harnesses: {', '.join(harnesses.names())})")
        interfaces = m.get("interfaces") or []
        if not interfaces or set(interfaces) - set(INTERFACES):
            problems.append(f"interfaces must be some of {list(INTERFACES)}, not {interfaces!r}")
        listed = m.get("tools") or []
        unknown_tools = [t for t in listed if t not in toolbox.names()]
        if unknown_tools:
            problems.append(f"no tools {unknown_tools} in chefbot/tools (tools: {', '.join(toolbox.names())})")
        no_interface = [t for t in listed if t in toolbox.names() and not set(toolbox.get(t).interfaces) & set(interfaces)]
        if no_interface:
            problems.append(f"tools {no_interface} work with none of its interfaces {interfaces!r}")
        if len(set(listed)) != len(listed):
            problems.append("a tool is listed twice")
        if not (self.prompts_dir / "system.md").exists():
            problems.append(f"no system.md in {self.prompts_dir}")
        if problems:
            raise ValueError(f"agent {self.name}: " + "; ".join(problems))
        self.settings, self.loop_config()  # their own checks: unknown keys, time_mode
