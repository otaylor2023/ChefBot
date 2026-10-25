"""The environments run_agent can run a task in, behind one shape (EnvRun).

  symbolic   the symbolic kitchen (task file `symbolic:` section)
  robocasa   a RoboCasa kitchen (task file `robocasa:` section: layout, style,
             objects to spawn, success rules, timing)

`start_map`: the map the agent starts from, in place of the true one (an
explored_map.yaml from chefbot.scripts.explore). Its fixtures get their
footprints from the kitchen's start map when there is one. With it, a
RoboCasa run writes only the occupancy grid: no true map is made.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from chefbot.agent.world.map import WorldMap

REPO = Path(__file__).resolve().parents[2]


@dataclass
class EnvRun:
    name: str
    task: dict
    robot: Any
    detector: Any
    world_map: WorldMap
    clock: Any
    observer: Any
    safety: Callable[[float], list]
    success: Callable[[list[str]], dict]
    on_limit: Callable[[], None]
    on_plan: Callable[[dict], None] = lambda steps: None
    on_tool: Callable[[str], None] = lambda tool: None
    applied: list = field(default_factory=list)  # disturbances: (time, description)
    info: dict = field(default_factory=dict)  # for start.json
    stack: Any = None  # RoboCasa: for the video recorder
    close: Callable[[], None] = lambda: None


def _given_map(path: Path, fixtures: dict | None = None) -> WorldMap:
    """The map in `path`; `fixtures` (a start map: fixtures with footprints) fills in what it leaves out."""
    import yaml

    from chefbot.explore.rearrange import initial_map

    given = yaml.safe_load(Path(path).read_text())
    return WorldMap.from_dict(initial_map(fixtures, given) if fixtures else given)


def symbolic(task_path: Path, seed: int, noise: str, start_map: Path | None = None) -> EnvRun:
    from chefbot.envs.symbolic.scenario import Scenario

    sc = Scenario.load(task_path, seed, noise)

    def on_limit() -> None:
        for burner in sc.world.burners.values():
            sc.world.set_power(burner.name, False)

    return EnvRun(
        sc.name, sc.task, sc.robot, sc.detector, _given_map(start_map) if start_map else WorldMap.from_dict(sc.map), sc.clock, sc.observer, sc.safety,
        sc.success, on_limit, sc.on_plan, sc.on_tool, sc.applied,
        info={"env": "symbolic", "disturbances": [d.spec for d in sc.disturbances],
              "failures": sc.robot.rates.__dict__, "noise": noise},
    )  # fmt: skip


def robocasa(task_path: Path, seed: int, noise: str, start_map: Path | None = None) -> EnvRun:
    from chefbot.envs.robocasa.build_map import GRID_FILE, SEMANTIC_MAP_FILE, START_MAP_FILE, build_map, write_grid
    from chefbot.envs.robocasa.config import RoboCasaConfig
    from chefbot.envs.robocasa.env import RoboCasaEnv
    from chefbot.envs.robocasa.kitchen_state import RoboCasaClock, RoboCasaKitchen, RoboCasaObserver
    from chefbot.envs.robocasa.serve import RoboCasaStack, wire_stack
    from chefbot.envs.success import check_success
    from chefbot.envs.symbolic.scenario import load_yaml
    from chefbot.envs.symbolic.world import Timing
    from chefbot.robot.config import RobotConfig

    task = load_yaml(task_path)
    spec = task.get("robocasa")
    if not spec:
        raise SystemExit(f"{task_path} has no robocasa: section")
    layout, style = int(spec.get("layout", 1)), int(spec.get("style", spec.get("layout", 1)))
    config = RoboCasaConfig.load(layout=layout, style=style, seed=seed)
    if spec.get("objects"):
        config = dataclasses.replace(config, objects=tuple(dict(o) for o in spec["objects"]))
    env = RoboCasaEnv(config, render=True)
    name = task.get("name", task_path.stem)
    # The map depends on the spawned objects: one per task and kitchen, built once.
    map_dir = REPO / "runs" / "maps" / f"{name}_layout{layout:03d}_style{style:03d}"
    if start_map is not None:  # the agent's own map: the grid alone, no true map
        map_dir = map_dir.with_name(map_dir.name + "_grid")
        if not (map_dir / GRID_FILE).exists():
            write_grid(env, map_dir)
        fixtures = config.map_dir / START_MAP_FILE
        world_map = _given_map(start_map, load_yaml(fixtures) if fixtures.exists() else None)
    elif not (map_dir / GRID_FILE).exists():
        build_map(env, map_dir)
    stack: RoboCasaStack = wire_stack(env, map_dir, RobotConfig.load())
    kitchen = RoboCasaKitchen(env, stack.robot._fixture_actuator, Timing.load(spec.get("timing")))
    rules = spec.get("success") or {}

    def success(tools_called: list[str]) -> dict:
        return check_success(rules, kitchen.describe().splitlines(), tools_called)

    def on_limit() -> None:
        from chefbot.envs.robocasa.appliances import burners, set_burner
        import numpy as np

        with env.lock:
            for burner in burners(env):
                set_burner(env, burner, False, np.random.RandomState(0))

    if start_map is None:
        world_map = WorldMap.load(map_dir / SEMANTIC_MAP_FILE)
    return EnvRun(
        name, task, stack.robot, stack.detector, world_map, RoboCasaClock(kitchen),
        RoboCasaObserver(kitchen), kitchen.safety, success, on_limit,
        info={"env": "robocasa", "layout": layout, "style": style, "start_map": str(start_map) if start_map else None},
        stack=stack, close=env.close,
    )  # fmt: skip


ENVS = {"symbolic": symbolic, "robocasa": robocasa}
