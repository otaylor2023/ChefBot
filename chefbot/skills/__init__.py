"""Registered lab skills: programs the robot runs, one folder each (skills/<name>/<version>/: main.py, skill.yaml,
REPORT.md), put here by register.py from a lab run.

A skill's main.py defines run(robot, **args) against the robot API its manifest names (`interface`: chefbot-v1,
autoresearch/skill_api.py), and runs sandboxed. Its manifest says which robot it was built on, which models it calls
and where each runs (`api`: a paid API, key from .env; `vision_models`: our vision server), and where it came from.

    skill = skills.load("pick_up/run1_v12")
    skill.check(robot="panda_franka", vision_url=...)   # raises if anything it needs is missing
    skills.catalog(env="robocasa", interface="chefbot-v1")   # the skills that could run there, with that API

Every skill is one kind of thing, whatever runs it: `interface` names the robot API it is written against
(interface.INTERFACES: chefbot-v1, a sandboxed process speaking interface/skill_api.py; robot-api-v1, a function
in the agent's process given a RobotAPI, never sandboxed), `envs` where it could run (ENVS), and `sandboxed` says
which of the two it is. A manifest that says otherwise, or has a key the loader does not know, is refused.

(agentic-skills-exploration's skills/, without privileged skills or Astra.)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from chefbot.interface import INTERFACES

ROOT = Path(__file__).resolve().parent
ENVS = ("symbolic", "robocasa", "mujoco_kitchen", "real")  # where a skill could run (chefbot/environments)
IMPORTED_INTERFACES = ("wholebody-v2",)  # agentic-skills-exploration's robot API: kept as lab starting points, never run
KEYS = {"name", "version", "entry", "interface", "envs", "sandboxed", "robot", "arm_only", "cameras", "args", "speed",
        "models", "assumes", "privileged", "provenance", "imported_from"}  # fmt: skip
ARG_TYPES = {"string", "number", "integer", "boolean"}


class MissingRequirement(RuntimeError):
    pass


@dataclass
class Skill:
    path: str  # e.g. pick_up/run1_v12
    dir: Path
    manifest: dict = field(repr=False)

    @property
    def name(self) -> str:
        return self.manifest["name"]

    @property
    def interface(self) -> str:
        return self.manifest.get("interface", "chefbot-v1")

    @property
    def envs(self) -> tuple[str, ...]:
        """Where it could run. Empty for an imported skill: nowhere here."""
        return tuple(self.manifest.get("envs") or ())

    @property
    def sandboxed(self) -> bool:
        return bool(self.manifest.get("sandboxed", self.interface != "robot-api-v1"))

    @property
    def runnable(self) -> bool:
        """False for an imported skill whose robot API no session speaks."""
        return self.interface in INTERFACES

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
        return dict(self.manifest.get("args") or {})

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
        return {"llm": "api" in runs, "vision": vision, "cameras": self.cameras, "arm_only": bool(self.manifest.get("arm_only", False))}

    def validate(self) -> None:
        """Raise ValueError if the manifest is not one the loader understands."""
        m, problems = self.manifest, []
        unknown = set(m) - KEYS
        if unknown:
            problems.append(f"unknown keys {sorted(unknown)}")
        family, version = self.dir.parent.name, self.dir.name
        if m.get("name") != family or str(m.get("version")) != version:
            problems.append(f"name/version {m.get('name')}/{m.get('version')} is not its folder's, {family}/{version}")
        if self.interface not in INTERFACES + IMPORTED_INTERFACES:
            problems.append(f"interface must be one of {list(INTERFACES + IMPORTED_INTERFACES)}, not {self.interface!r}")
        if "sandboxed" not in m:
            problems.append("no sandboxed (true for a chefbot-v1 skill, false for a robot-api-v1 one)")
        elif not isinstance(m["sandboxed"], bool):
            problems.append(f"sandboxed must be true or false, not {m['sandboxed']!r}")
        elif self.interface == "robot-api-v1" and self.sandboxed:
            problems.append("a robot-api-v1 skill runs in the agent's process: it cannot be sandboxed")
        elif self.interface != "robot-api-v1" and not self.sandboxed:
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
    return [s for s in found if (env is None or env in s.envs) and (interface is None or s.interface == interface)]
