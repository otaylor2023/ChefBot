"""A lab's config (labs/<name>.yaml): what changes between labs. Mechanics live in research.py and lab_server.py.

    goal:     {skill: pick_up, task: pick}                  what the lab produces, and the task it is scored on
    robot:    panda_franka
    suites:   {dev: {suite, visible: full}, val: {suite, visible: aggregate, submissions: 5}, test: {suite, visible: hidden}}
    paid_models:   {google: {usd: 10}}                      metered APIs the skill may call, and their budgets
    vision_models: [owlv2, grounding-dino, sam2]            what the vision server offers the skill
    budgets:  {hours: 4, codex_usage_pct: 60}               or codex_usage_add_pct: 10 (the cap: usage at the start + 10)
    brief:    briefs/pick_up.md
    start_from: {skill: pick_up/run1_v12, note: "..."}     optional prior work for the researcher
    interface: chefbot-v1   arm_only: true   example: example_pick   skill_args: {...}      optional

(agentic-skills-exploration's autoresearch/config.py.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

LABS = Path(__file__).resolve().parent / "labs"


@dataclass
class LabConfig:
    name: str
    path: Path
    goal: dict
    robot: str
    suites: dict
    paid_models: dict = field(default_factory=dict)
    vision_models: list = field(default_factory=list)
    budgets: dict = field(default_factory=dict)
    brief: str = "briefs/pick_up.md"
    start_from: dict | None = None
    interface: str = "chefbot-v1"  # the robot API the researcher's skills use (skill_api.py)
    arm_only: bool = False  # the skills may move only the arm (act_arm): the base holds
    example: str | None = None  # templates/<example>: the researcher's skills/example
    raw_args: dict | None = None  # the skill's arguments (skill.yaml `args`), for its manifest when registered

    def access(self) -> dict:
        """The model calls a skill in this lab may make (and whether it is arm only)."""
        return {"llm": "google" in self.paid_models, "vision": list(self.vision_models), "arm_only": self.arm_only}

    @property
    def model_usd(self) -> float:
        """The total budget for metered model calls by the skills (all dev and val episodes)."""
        return float(sum(v.get("usd", 0.0) for v in self.paid_models.values() if isinstance(v, dict)))

    def suite(self, part: str) -> str:
        return self.suites[part]["suite"]

    @property
    def val_submissions(self) -> int:
        return int(self.suites.get("val", {}).get("submissions", 6))


def load(name_or_path: str) -> LabConfig:
    p = Path(name_or_path)
    if not p.suffix:
        p = LABS / f"{name_or_path}.yaml"
    raw = yaml.safe_load(p.read_text())
    return LabConfig(p.stem, p, raw["goal"], raw["robot"], raw["suites"], raw.get("paid_models") or {}, raw.get("vision_models") or [],
                     raw.get("budgets") or {}, raw.get("brief", "briefs/pick_up.md"), raw.get("start_from"), raw.get("interface", "chefbot-v1"),
                     bool(raw.get("arm_only", False)), raw.get("example"), raw.get("skill_args"))  # fmt: skip
