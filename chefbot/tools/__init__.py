"""Tools: the agent's tools that are not skills, one file each (tools/<group>/<name>.yaml), loaded by name. A tool
is what the agent asks of the run rather than of the robot's body: looking (detect, nearby, holding, get_pose),
following a motion (wait, get_status, stop), editing the plan (complete, add_step, finish, ...) and keeping time
(set_timer, cancel_timer, watch). Moving the robot is a skill's job (skills/).

    tools.get("detect").schema      # the function-calling schema the LLM sees
    tools.get("detect").entry       # ("nav", "detect"): the handler, a method of one of the harness's tool owners

A tool file:
    name: detect                    # its file's name
    description: ...                # what the LLM reads
    parameters: {classes: {...}}    # JSON-schema properties (optional)
    required: [...]                 # (optional)
    entry: nav.detect               # owner.method; the owners are a harness's (nav, manip, plan, time)
    needs: [grasp]                  # robot capabilities it needs (interface/robot_api.py)
    interfaces: [robot-api-v1]      # the robot APIs it works with (interface.INTERFACES)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import yaml

from chefbot.interface import INTERFACES
from chefbot.interface.robot_api import ALL_CAPABILITIES

ROOT = Path(__file__).resolve().parent
KEYS = {"name", "description", "parameters", "required", "entry", "needs", "interfaces"}
OWNERS = ("nav", "manip", "plan", "time")
JSON_TYPES = {"string", "boolean", "number", "array"}  # what ToolRegistry's argument check knows


@dataclass(frozen=True)
class Tool:
    name: str
    group: str
    path: Path
    manifest: dict = field(repr=False, hash=False, compare=False)

    @property
    def description(self) -> str:
        return self.manifest["description"]

    @property
    def entry(self) -> tuple[str, str]:
        owner, method = self.manifest["entry"].split(".")
        return owner, method

    @property
    def needs(self) -> frozenset[str]:
        return frozenset(self.manifest.get("needs") or ())

    @property
    def interfaces(self) -> tuple[str, ...]:
        return tuple(self.manifest["interfaces"])

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.manifest.get("parameters") or {},
                    "required": self.manifest.get("required") or [],
                    "additionalProperties": False,
                },
            },
        }

    def validate(self) -> None:
        m, problems = self.manifest, []
        unknown = set(m) - KEYS
        if unknown:
            problems.append(f"unknown keys {sorted(unknown)}")
        if m.get("name") != self.name:
            problems.append(f"name {m.get('name')!r} is not its file's, {self.name!r}")
        if not isinstance(m.get("description"), str) or not m["description"].strip():
            problems.append("no description")
        params = m.get("parameters") or {}
        for key, spec in params.items():
            if not isinstance(spec, dict) or spec.get("type") not in JSON_TYPES:
                problems.append(f"parameter {key!r} must have a type, one of {sorted(JSON_TYPES)}")
        missing = set(m.get("required") or ()) - set(params)
        if missing:
            problems.append(f"required {sorted(missing)} are not parameters")
        owner, _, method = str(m.get("entry", "")).partition(".")
        if owner not in OWNERS or not method:
            problems.append(f"entry must be owner.method with owner one of {list(OWNERS)}, not {m.get('entry')!r}")
        bad_needs = set(m.get("needs") or ()) - set(ALL_CAPABILITIES)
        if bad_needs:
            problems.append(f"needs {sorted(bad_needs)} are not robot capabilities")
        interfaces = m.get("interfaces") or []
        if not interfaces or set(interfaces) - set(INTERFACES):
            problems.append(f"interfaces must be some of {list(INTERFACES)}, not {interfaces!r}")
        if problems:
            raise ValueError(f"tool {self.group}/{self.name}: " + "; ".join(problems))


def load_file(path: str | Path) -> Tool:
    path = Path(path)
    tool = Tool(path.stem, path.parent.name, path, yaml.safe_load(path.read_text()) or {})
    tool.validate()
    return tool


@cache
def _all(root: Path = ROOT) -> dict[str, Tool]:
    found: dict[str, Tool] = {}
    for path in sorted(root.glob("*/*.yaml")):
        tool = load_file(path)
        if tool.name in found:
            raise ValueError(f"tool {tool.name!r} is defined twice: {found[tool.name].path} and {path}")
        found[tool.name] = tool
    return found


def names() -> list[str]:
    return sorted(_all())


def get(name: str) -> Tool:
    if name not in _all():
        raise ValueError(f"no tool {name!r}; tools: {', '.join(names())}")
    return _all()[name]


def schema(name: str) -> dict:
    return get(name).schema
