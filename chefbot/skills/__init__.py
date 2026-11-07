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
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


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
        return {"llm": "api" in runs, "vision": vision, "astra": "codex_cli" in runs, "cameras": self.cameras,
                "arm_only": bool(self.manifest.get("arm_only", False))}  # fmt: skip

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
    return Skill(rel, d, yaml.safe_load((d / "skill.yaml").read_text()))


def names() -> list[str]:
    return sorted(str(p.parent.relative_to(ROOT)) for p in ROOT.glob("*/*/skill.yaml"))
