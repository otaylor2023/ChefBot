"""A task file's `symbolic:` section: scene, start pose, injected failures,
disturbances, and how ground truth decides success.

    symbolic:
      kitchen: default
      start: {x: 3.0, y: 2.0, yaw_deg: -90}
      objects:
        - {name: egg, in: fridge}
        - {name: pan, on: counter_2}
      failures: {grasp: 0.2, place: 0.0, nav: 0.0, detect_miss: 0.1}
      disturbances:
        - {at: start, do: remove, object: egg}                 # after the map is built: the map is stale
        - {at: 200, do: move, object: plate, to: {on: counter_2}}
        - {when_step: s3, status: waiting, do: set_power, target: burner_3, on: true}
        - {after_tool: pick, count: 2, do: close, target: fridge}
      success:
        require: ["egg: on plate", "egg: cook state = set"]   # regexes, each must match a line
        forbid: ["burner_\\\\w+: on", "overcooked|burnt"]       # regexes, none may match
        expect_tools: [request_replan]                         # tools that must have been called
        at_most: {"^egg(_\\d+)?: (inside|on|held)": 1}          # at most this many lines match
        # or alternatives: {any_of: [{require: ...}, {expect_tools: ..., forbid: ...}]}

Disturbances are a test harness: they change the world behind the agent's back.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from chefbot.environments.success import check_success
from chefbot.environments.symbolic.perception import SymbolicDetector, SymbolicObserver
from chefbot.environments.symbolic.robot import SymbolicClock, SymbolicRobot
from chefbot.environments.symbolic.world import Rates, World
from chefbot.interface.types import Pose2D


def load_yaml(path: str | Path):
    """YAML with unquoted `on:` keys kept as "on" (YAML 1.1 reads them as true)."""
    return _on_keys(yaml.safe_load(Path(path).read_text()))


def _on_keys(obj):
    if isinstance(obj, dict):
        return {("on" if k is True else k): _on_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_on_keys(v) for v in obj]
    return obj


@dataclass
class Disturbance:
    spec: dict
    done: bool = False

    @property
    def description(self) -> str:
        return self.spec.get("note") or ", ".join(f"{k}={v}" for k, v in self.spec.items())


@dataclass
class Scenario:
    name: str
    task: dict  # the whole task file (goal, steps, ...)
    world: World
    robot: SymbolicRobot
    detector: SymbolicDetector
    observer: SymbolicObserver
    clock: SymbolicClock
    map: dict  # WorldMap.from_dict format, built before `at: start` disturbances
    disturbances: list[Disturbance] = field(default_factory=list)
    applied: list[tuple[float, str]] = field(default_factory=list)  # (time, description)
    _tool_counts: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path, seed: int = 0, noise: str = "task") -> "Scenario":
        task = load_yaml(path)
        spec = task.get("symbolic") or {}
        world = World.load(spec.get("kitchen", "default"), spec.get("objects") or [], spec.get("timing"))
        start = spec.get("start") or {}
        pose = Pose2D(float(start.get("x", 3.0)), float(start.get("y", 2.0)), math.radians(float(start.get("yaw_deg", -90))))
        rates = Rates.for_run(spec.get("failures"), noise)
        robot = SymbolicRobot(world, pose, rates, np.random.default_rng(seed))
        scenario = cls(
            task.get("name", Path(path).stem), task, world, robot, SymbolicDetector(robot), SymbolicObserver(world),
            SymbolicClock(world), world.map_dict(), [Disturbance(dict(d)) for d in spec.get("disturbances") or []],
        )  # fmt: skip
        for d in scenario.disturbances:
            at = d.spec.get("at")
            if at == "start":
                scenario._apply(d)
            elif isinstance(at, (int, float)):
                world.schedule(float(at), lambda _w, d=d: scenario._apply(d))
        return scenario

    # -- triggers (called by the run script) ------------------------------------

    def on_plan(self, steps: dict[str, str]) -> None:
        """Step statuses now ({id: status}): fire `when_step` disturbances."""
        for d in self.disturbances:
            step = d.spec.get("when_step")
            if not d.done and step is not None and steps.get(step) == d.spec.get("status", "active"):
                self._apply(d)

    def on_tool(self, tool: str) -> None:
        """A tool call finished: fire `after_tool` disturbances."""
        self._tool_counts[tool] = self._tool_counts.get(tool, 0) + 1
        for d in self.disturbances:
            if not d.done and d.spec.get("after_tool") == tool and self._tool_counts[tool] >= int(d.spec.get("count", 1)):
                self._apply(d)

    def safety(self, now: float) -> list[tuple[str, dict]]:
        """EventBus safety monitor: the world's safety messages (empty lit burners, burnt food)."""
        return [(message, {"source": "robot"}) for message in self.world.take_events()]

    # -- success ----------------------------------------------------------------

    def success(self, tools_called: list[str] = ()) -> dict:
        """Ground truth at the end against the task's rules (envs/success.py)."""
        rules = (self.task.get("symbolic") or {}).get("success") or {}
        return check_success(rules, self.world.describe().splitlines(), tools_called)

    # -- disturbances -------------------------------------------------------------

    def _apply(self, d: Disturbance) -> None:
        if d.done:
            return
        d.done = True
        spec, world = d.spec, self.world
        op = spec["do"]
        if op == "move":
            to = spec["to"]
            place = to.get("on") or to.get("in")
            if spec["object"] in world.things:
                world.move(spec["object"], place)
        elif op == "remove":
            if spec["object"] in world.things:
                world.remove(spec["object"])
        elif op == "add":
            world.add(dict(spec["object"]))
        elif op == "set_power":
            world.set_power(spec["target"], bool(spec.get("on", True)))
        elif op in ("open", "close"):
            world.fixtures[spec["target"]].open = op == "open"
        else:
            raise ValueError(f"unknown disturbance {op!r}")
        self.applied.append((world.t, d.description))
