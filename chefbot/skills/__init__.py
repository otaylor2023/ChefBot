"""Deployed skills: programs the robot runs, one folder each (skills/<name>/<version>/: main.py, skill.yaml,
REPORT.md). Skills are developed in agentic-skills-exploration's labs (chefbot_labs) and released here when they are
to run on the robot (often as a port for the real robot, e.g. *_fr3); hand-written ones (bringup) live here too.
load() also takes any skill folder by path, so a lab's skill can be tried on the robot before it is released.

A skill's main.py defines run(robot, **args) against the robot API its manifest names (`interface`: chefbot-v1,
interface/skill_api.py), and runs sandboxed (runtime/episode.py). Its manifest says which robot it was built on, which
models it calls and where each runs (`api`: a paid API, key from .env; `vision_models`: our vision server), and where
it came from.

    skill = skills.load("pick_up/gemini_v1")             # or skills.load("/path/to/agentic-skills-exploration/chefbot_labs/skills/...")
    skill.check(robot="tidybot_fr3", vision_url=...)    # raises if anything it needs is missing
    skills.catalog(env="real", interface="chefbot-v1")   # the skills that could run there, with that API

Every skill is one kind of thing, whatever runs it: `interface` names the robot API it is written against
(interface.INTERFACES: chefbot-v1, a sandboxed process speaking interface/skill_api.py; robot-api-v1, a function
in the agent's process given a RobotAPI, never sandboxed), `envs` where it could run (ENVS), and `sandboxed` says
which of the two it is. A third kind, `host-v1`, is a function in the agent's process on the real robot's host (the
laptop side: the base driven on the map by the robot layer, the cameras, mocap; scripts/real_kitchen.py), offered in
chefbot-v1 sessions beside the sandboxed skills it shares the host with. A manifest that says otherwise, or has a key
the loader does not know, is refused.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from chefbot.interface import INTERFACES
from chefbot.interface.robot_api import ALL_CAPABILITIES

ROOT = Path(__file__).resolve().parent
ENVS = ("symbolic", "robocasa", "mujoco_kitchen", "real")  # where a skill could run (chefbot/environments)
HOST_INTERFACES = ("host-v1",)  # run in the agent's process on a session's host (scripts/real_kitchen.py RealKitchen)
OFFERED_WITH = {"robot-api-v1": ("robot-api-v1",), "chefbot-v1": ("chefbot-v1", "host-v1")}  # a session's API: its skills'
IMPORTED_INTERFACES = ("wholebody-v2",)  # agentic-skills-exploration's robot API: kept as lab starting points, never run
KEYS = {"name", "version", "entry", "interface", "envs", "sandboxed", "robot", "needs", "description", "arm_only", "cameras",
        "args", "speed", "models", "assumes", "privileged", "provenance", "imported_from"}  # fmt: skip
ARG_TYPES = {"string", "number", "integer", "boolean"}


class MissingRequirement(RuntimeError):
    pass


@dataclass
class Skill:
    path: str  # e.g. pick_up/run1_v12
    dir: Path
    manifest: dict = field(repr=False)
    alias: str | None = None  # an agent's name for it (agents/<name>/agent.yaml skills:), else its manifest's
    overrides: dict = field(default_factory=dict, repr=False)  # an agent's description / arg docs for it

    @property
    def name(self) -> str:
        """The name it is called by: the agent's for it, if it is seen through an agent's registry."""
        return self.alias or self.manifest["name"]

    def viewed(self, name: str, description: str | None = None, args: dict | None = None) -> "Skill":
        """This skill as an agent sees it: under `name`, with its own description and args if given (arg -> its doc,
        or {doc?, default?}: a default the LLM may leave out). What runs is the same."""
        unknown = set(args or {}) - set(self.manifest.get("args") or {})
        if unknown:
            raise ValueError(f"skill {self.path} has no args {sorted(unknown)}")
        bad = [a for a, v in (args or {}).items() if not (isinstance(v, str) or (isinstance(v, dict) and v and set(v) <= {"doc", "default"}))]
        if bad:
            raise ValueError(f"skill {self.path}: args {bad} must be a doc, or {{doc?, default?}}")
        over = {k: v for k, v in (("description", description), ("args", dict(args or {}))) if v}
        return Skill(self.path, self.dir, self.manifest, alias=name, overrides=over)

    @property
    def interface(self) -> str:
        return self.manifest.get("interface", "chefbot-v1")

    @property
    def envs(self) -> tuple[str, ...]:
        """Where it could run. Empty for an imported skill: nowhere here."""
        return tuple(self.manifest.get("envs") or ())

    @property
    def sandboxed(self) -> bool:
        return bool(self.manifest.get("sandboxed", self.interface not in ("robot-api-v1",) + HOST_INTERFACES))

    @property
    def runnable(self) -> bool:
        """False for an imported skill whose robot API no session speaks."""
        return self.interface in INTERFACES + HOST_INTERFACES

    @property
    def needs(self) -> frozenset[str]:
        """The robot capabilities it needs (interface/robot_api.py); a robot-api-v1 skill is offered only with them."""
        return frozenset(self.manifest.get("needs") or ())

    @property
    def description(self) -> str | None:
        """What the LLM reads about it."""
        return self.overrides.get("description") or self.manifest.get("description")

    @property
    def schema(self) -> dict:
        """The function-calling schema an agent sees: its name, description and args (an arg without a default is
        required)."""
        properties = {}
        for arg, spec in self.args.items():
            properties[arg] = {"type": spec["type"], "description": spec.get("description") or spec.get("doc", "")}
            properties[arg].update({k: spec[k] for k in ("items", "enum") if k in spec})
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description or "",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": [arg for arg, spec in self.args.items() if "default" not in spec],
                    "additionalProperties": False,
                },
            },
        }

    def function(self):
        """A robot-api-v1 skill's entry, to call in the agent's process: run(ctx, **args)."""
        if self.sandboxed:
            raise ValueError(f"skill {self.path} is sandboxed: it runs as its own process (runtime/episode.py)")
        file, _, func = self.manifest["entry"].partition(":")
        spec = importlib.util.spec_from_file_location(f"chefbot_skill.{self.name}.{self.dir.name}", self.dir / file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, func)

    @property
    def robot(self) -> str:
        return self.manifest["robot"]

    @property
    def cameras(self) -> bool | list[str]:
        """False: the skill never looks (no robot.observe). A list: only these."""
        c = self.manifest.get("cameras", True)
        return [str(n) for n in c] if isinstance(c, list) else bool(c)

    @property
    def args(self) -> dict:
        """The arguments run(robot, **args) takes: name -> {type, default, enum, doc}. None declared: run(robot)."""
        args = {k: dict(v) for k, v in (self.manifest.get("args") or {}).items()}
        for arg, over in (self.overrides.get("args") or {}).items():
            over = {"doc": over} if isinstance(over, str) else over
            if "doc" in over:
                args[arg].update(doc=over["doc"], description=over["doc"])
            if "default" in over:
                args[arg]["default"] = over["default"]
        return args

    @property
    def models(self) -> dict:
        return self.manifest.get("models") or {}

    def spec(self) -> dict:
        """What the runtime needs to run it (worker.py)."""
        return {"path": self.path, "dir": str(self.dir), "interface": self.interface, "access": self.access()}

    def access(self) -> dict:
        """Which model calls the runtime serves this skill: exactly what its manifest lists."""
        runs = {m.get("runs") for m in self.models.values()}
        vision = sorted(n for n, m in self.models.items() if m.get("runs") == "vision_models")
        return {"llm": "api" in runs, "vision": vision, "astra": "codex_cli" in runs, "cameras": self.cameras,
                "arm_only": bool(self.manifest.get("arm_only", False))}  # fmt: skip

    def validate(self) -> None:
        """Raise ValueError if the manifest is not one the loader understands."""
        m, problems = self.manifest, []
        unknown = set(m) - KEYS
        if unknown:
            problems.append(f"unknown keys {sorted(unknown)}")
        family, version = self.dir.parent.name, self.dir.name
        if m.get("name") != family or str(m.get("version")) != version:
            problems.append(f"name/version {m.get('name')}/{m.get('version')} is not its folder's, {family}/{version}")
        kinds = INTERFACES + HOST_INTERFACES + IMPORTED_INTERFACES
        if self.interface not in kinds:
            problems.append(f"interface must be one of {list(kinds)}, not {self.interface!r}")
        if "sandboxed" not in m:
            problems.append("no sandboxed (true for a chefbot-v1 skill, false for a robot-api-v1 one)")
        elif not isinstance(m["sandboxed"], bool):
            problems.append(f"sandboxed must be true or false, not {m['sandboxed']!r}")
        elif self.interface in ("robot-api-v1",) + HOST_INTERFACES and self.sandboxed:
            problems.append(f"a {self.interface} skill runs in the agent's process: it cannot be sandboxed")
        elif self.interface not in ("robot-api-v1",) + HOST_INTERFACES and not self.sandboxed:
            problems.append(f"a {self.interface} skill runs as its own process and is always sandboxed")
        envs = m.get("envs")
        if not self.runnable and envs is not None:
            problems.append(f"a {self.interface} skill runs nowhere here: it has no envs")
        elif self.runnable and (not isinstance(envs, list) or not envs or set(envs) - set(ENVS)):
            problems.append(f"envs must be some of {list(ENVS)}, not {envs!r}")
        file, _, func = str(m.get("entry", "")).partition(":")
        if not func or not (self.dir / file).is_file():
            problems.append(f"entry {m.get('entry')!r} is not file:function with the file in its folder")
        for arg, spec in self.args.items():
            if not isinstance(spec, dict) or spec.get("type") not in ARG_TYPES:
                problems.append(f"arg {arg!r} must have a type, one of {sorted(ARG_TYPES)}")
        bad_needs = set(m.get("needs") or ()) - set(ALL_CAPABILITIES)
        if bad_needs:
            problems.append(f"needs {sorted(bad_needs)} are not robot capabilities")
        if self.interface in ("robot-api-v1",) + HOST_INTERFACES and not (isinstance(m.get("description"), str) and m["description"].strip()):
            problems.append(f"a {self.interface} skill is offered to the LLM as is: it needs a description")
        if self.interface in HOST_INTERFACES and envs != ["real"]:
            problems.append("a host-v1 skill runs on the real robot's host: envs [real]")
        if problems:
            raise ValueError(f"skill {self.path}: " + "; ".join(problems))

    def check(self, robot: str, vision_url: str | None = None) -> None:
        """Raise MissingRequirement if the run can't give the skill what it needs."""
        problems = []
        if self.robot != "any" and robot != self.robot:
            problems.append(f"it was built for robot {self.robot}, not {robot}")
        for name, m in self.models.items():
            if m.get("runs") == "api" and m.get("key") and not os.environ.get(m["key"]):
                problems.append(f"{name} needs {m['key']} (in .env)")
            if m.get("runs") == "vision_models" and not vision_url:
                problems.append(f"{name} needs the vision models (chefbot/runtime/vision_server.py; VISION_URL)")
            if m.get("runs") == "codex_cli" and not shutil.which("codex"):
                problems.append(f"{name} needs the Codex CLI, logged in (`codex login`), on this computer")
        if problems:
            raise MissingRequirement(f"skill {self.path}: " + "; ".join(problems))


def load(path: str) -> Skill:
    d = (ROOT / path) if not Path(path).is_absolute() else Path(path)
    if not (d / "skill.yaml").exists():
        raise ValueError(f"no skill at {d} (expected skill.yaml)")
    rel = str(d.relative_to(ROOT)) if d.is_relative_to(ROOT) else str(d)
    skill = Skill(rel, d, yaml.safe_load((d / "skill.yaml").read_text()) or {})
    skill.validate()
    return skill


def names() -> list[str]:
    return sorted(str(p.parent.relative_to(ROOT)) for p in ROOT.glob("*/*/skill.yaml"))


def catalog(env: str | None = None, interface: str | None = None) -> list[Skill]:
    """The registered skills that could run in `env` with `interface` (either left out: any)."""
    if env is not None and env not in ENVS:
        raise ValueError(f"no env {env!r}; envs: {', '.join(ENVS)}")
    if interface is not None and interface not in INTERFACES:
        raise ValueError(f"no interface {interface!r}; interfaces: {', '.join(INTERFACES)}")
    found = [load(path) for path in names()]
    return [s for s in found if (env is None or env in s.envs) and (interface is None or s.interface in OFFERED_WITH[interface])]
