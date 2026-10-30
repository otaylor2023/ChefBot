"""Suites: frozen lists of task instances (setup, task, seeds); suite sets: named suites that belong together (a split).

    suites/<set>/set.yaml       {description, suites: {dev: dev.yaml, val: val.yaml, test: test.yaml}}
    suites/<set>/<part>.yaml    {built_by, items: [{id, setup, task, seeds}, ...]}

`load("robocasa_pick/test")` is one suite; `load_set("robocasa_pick")` all of a set's suites. A suite holds no robot
and no visibility rules: those belong to the eval or lab using it. (agentic-skills-exploration's suites/.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


@dataclass
class Item:
    id: str
    setup: str
    task: str
    seeds: list[int]


@dataclass
class Suite:
    name: str
    built_by: str
    items: list[Item] = field(default_factory=list)

    def item(self, item_id: str) -> Item:
        for it in self.items:
            if it.id == item_id:
                return it
        raise KeyError(f"suite {self.name} has no item {item_id!r}")

    @property
    def episodes(self) -> int:
        return sum(len(it.seeds) for it in self.items)


def load(name: str) -> Suite:
    p = ROOT / f"{name}.yaml"
    if not p.exists():
        raise ValueError(f"no suite {name!r} ({p})")
    raw = yaml.safe_load(p.read_text())
    return Suite(name, raw.get("built_by", ""), [Item(i["id"], i["setup"], i["task"], list(i["seeds"])) for i in raw["items"]])


def is_set(name: str) -> bool:
    return (ROOT / name / "set.yaml").exists()


def load_set(name: str) -> dict[str, Suite]:
    raw = yaml.safe_load((ROOT / name / "set.yaml").read_text())
    return {part: load(f"{name}/{Path(f).stem}") for part, f in raw["suites"].items()}


def names() -> list[str]:
    return sorted(p.parent.name for p in ROOT.glob("*/set.yaml"))
