"""Agents: one folder per agent (agent.yaml, prompts/), loaded by name. An agent names the harness it runs in
(agent/harnesses), the robot APIs it runs with, its prompts, the skills it gets (its registry, a view of chefbot/skills: `skills:`, the name it calls each by -> one
registered version, pinned, with its own description or arg docs if it wants), the tools
it gets (chefbot/tools), its tools' settings and its loop's settings. A run is a session (agents/session.py): an
agent, an env and one robot API.

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

from chefbot import skills as skillbook
from chefbot import tools as toolbox
from chefbot.agent import harnesses
from chefbot.agent.config import AgentConfig, LoopConfig
from chefbot.interface import INTERFACES

ROOT = Path(__file__).resolve().parent
DEFAULT = "kitchen_planner"
KEYS = {"name", "harness", "interfaces", "prompts", "skills", "tools", "settings", "loop"}


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
    def skill_map(self) -> dict[str, dict]:
        """Its registry: the name it calls each skill by -> {path, description?, args?} (agent.yaml `skills:`, where
        an entry is a registered skill's path, or a mapping with `path` and the overrides)."""
        out = {}
        for name, entry in (self.manifest.get("skills") or {}).items():
            out[name] = {"path": entry} if isinstance(entry, str) else dict(entry)
        return out

    @property
    def skills(self) -> list[skillbook.Skill]:
        """Its skills, as it sees them (each under its name, with its descriptions), in its registry's order."""
        return [skillbook.load(e["path"]).viewed(name, e.get("description"), e.get("args"))
                for name, e in self.skill_map.items()]  # fmt: skip

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
        problems += _registry_problems(m.get("skills"))
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


def _registry_problems(registry) -> list[str]:
    """What is wrong with an agent's `skills:` (none: [])."""
    if registry is None:
        return []
    if not isinstance(registry, dict):
        return [f"skills must map the name it calls each skill by to a registered skill's path, not {registry!r}"]
    problems, registered = [], skillbook.names()
    for name, entry in registry.items():
        if not (isinstance(name, str) and name.isidentifier()):
            problems.append(f"skill name {name!r} is not a name the LLM can call")
            continue
        e = {"path": entry} if isinstance(entry, str) else entry
        if not isinstance(e, dict) or set(e) - {"path", "description", "args"} or not isinstance(e.get("path"), str):
            problems.append(f"skills {name}: a registered skill's path, or {{path, description?, args?}}, not {entry!r}")
        elif e["path"] not in registered:
            near = [p for p in registered if p.split("/")[0] == e["path"].split("/")[0]]
            problems.append(f"skills {name}: no registered skill {e['path']!r} (a version, exactly"
                            + (f"; registered: {', '.join(near)})" if near else ")"))  # fmt: skip
        else:
            try:
                skillbook.load(e["path"]).viewed(name, e.get("description"), e.get("args"))
            except ValueError as err:
                problems.append(f"skills {name}: {err}")
    return problems
