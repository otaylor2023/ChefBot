"""Explore a kitchen before cooking, and score the map the agent ends with.

    python -m chefbot.scripts.explore                                  # symbolic kitchen, LLM agent
    python -m chefbot.scripts.explore --env robocasa --policy scripted # RoboCasa, fixed policy

The agent starts from the fixtures alone (the true map without its objects,
fixtures with their footprints) and explores: go_to each fixture, detect,
open what has a door and look inside. complete() checks coverage from the
agent's own records. At the end its map is written in the semantic map's
format to runs/<run>/explored_map.yaml.

Only then is the true map made (from the simulator, as build_map makes it)
and the explored map scored against it: runs/<run>/true_map.yaml and
score.json. Nothing the agent can reach reads the true map: the robocasa run
writes only the occupancy grid, and complete() is judged on coverage, not on
the world.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from chefbot.agent.config import AgentConfig, LoopConfig
from chefbot.agent.loop.context import system_prompt
from chefbot.agent.loop.runner import AgentRunner, AgentSession
from chefbot.agent.plan.model import Plan
from chefbot.agent.world.map import WorldMap
from chefbot.explore.coverage import Coverage, CoverageChecker, CoverageObserver
from chefbot.explore.output import explored_map, save
from chefbot.explore.score import describe, score

REPO = Path(__file__).resolve().parents[2]
EXPLORE = REPO / "chefbot" / "explore"
TASK = REPO / "chefbot" / "tasks" / "explore_kitchen.yaml"

# What the explorer may call: looking, moving and doors; nothing that moves objects or heats.
TOOLS = ("go_to", "detect", "nearby", "align", "open", "close", "wait", "get_status", "stop", "get_pose",
         "complete", "set_status", "finish")  # fmt: skip


@dataclass
class World:
    name: str
    robot: Any
    detector: Any
    clock: Any
    start: dict  # the map the agent is given
    truth: Callable[[], dict]  # the true map; call only after the run
    close: Callable[[], None] = lambda: None
    info: dict = dataclasses.field(default_factory=dict)


def strip_objects(semantic: dict) -> dict:
    return {"frame": semantic.get("frame", "world"), "fixtures": semantic.get("fixtures") or {}, "objects": {}}


def symbolic_world(task: Path, seed: int) -> World:
    from chefbot.envs.symbolic.scenario import Scenario

    sc = Scenario.load(task, seed, noise="off")
    start = strip_objects(sc.world.map_dict())
    for name, f in sc.world.fixtures.items():
        start["fixtures"][name]["size"] = [round(f.width, 3), 0.6]
    return World(sc.name, sc.robot, sc.detector, sc.clock, start, sc.world.map_dict, info={"env": "symbolic"})


def robocasa_world(task: Path, seed: int, run_dir: Path, detector: str) -> World:
    from chefbot.envs.robocasa.build_map import GRID_FILE, GRID_META_FILE, START_MAP_FILE, build_semantic_map, write_grid
    from chefbot.envs.robocasa.config import RoboCasaConfig
    from chefbot.envs.robocasa.env import RoboCasaEnv
    from chefbot.envs.robocasa.kitchen_state import RoboCasaClock, RoboCasaKitchen
    from chefbot.envs.robocasa.serve import wire_stack
    from chefbot.envs.symbolic.scenario import load_yaml
    from chefbot.envs.symbolic.world import Timing
    from chefbot.robot.config import RobotConfig

    import numpy as np

    spec = load_yaml(task).get("robocasa") or {}
    layout, style = int(spec.get("layout", 1)), int(spec.get("style", spec.get("layout", 1)))
    config = RoboCasaConfig.load(layout=layout, style=style, seed=seed)
    if spec.get("objects"):
        config = dataclasses.replace(config, objects=tuple(dict(o) for o in spec["objects"]))
    start_path = config.map_dir / START_MAP_FILE
    if not start_path.exists():
        raise SystemExit(f"no start map {start_path}: python -m chefbot.envs.robocasa.build_map --start "
                         f"--layout {layout} --style {style}")  # fmt: skip
    env = RoboCasaEnv(config, render=True)
    grid_dir = write_grid(env, run_dir / "grid")  # geometry only: the robot drives on it
    stack = wire_stack(env, grid_dir, RobotConfig.load())
    kitchen = RoboCasaKitchen(env, stack.robot._fixture_actuator, Timing.load(spec.get("timing")))
    agent_detector = stack.detector
    if detector == "vision":
        from chefbot.explore.vision import VisionDetector

        agent_detector = VisionDetector(env, state_of=stack.detector.state_of)

    def truth() -> dict:
        occupied = np.load(grid_dir / GRID_FILE)
        return build_semantic_map(env, occupied, yaml.safe_load((grid_dir / GRID_META_FILE).read_text()))

    return World(task.stem, stack.robot, agent_detector, RoboCasaClock(kitchen), yaml.safe_load(start_path.read_text()),
                 truth, env.close, info={"env": "robocasa", "layout": layout, "style": style, "detector": detector,
                                         "committed_map": str(config.map_dir / "semantic_map.yaml")})  # fmt: skip


def run(env: str = "symbolic", policy: str = "llm", detector: str = "groundtruth", seed: int = 0,
        task: Path = TASK, out: Path = REPO / "runs", config: LoopConfig | None = None, llm=None,
        quiet: bool = False) -> dict:  # fmt: skip
    explore_cfg = yaml.safe_load((EXPLORE / "config.yaml").read_text())
    vocab = yaml.safe_load((EXPLORE / explore_cfg["vocab_file"]).read_text())
    config = config or LoopConfig.load()
    task_spec = yaml.safe_load(task.read_text())
    limits = {k: v for k, v in (task_spec.get("limits") or {}).items() if hasattr(config, k)}
    config = dataclasses.replace(config, **limits)
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = out / f"{stamp}_explore_{env}_{policy}_s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    world = symbolic_world(task, seed) if env == "symbolic" else robocasa_world(task, seed, run_dir, detector)
    (run_dir / "start_map.yaml").write_text(yaml.safe_dump(world.start, sort_keys=False, default_flow_style=None))
    world_map = WorldMap.from_dict(world.start)
    open_labels = frozenset(explore_cfg["open_labels"])
    coverage = Coverage(world_map, open_labels, float(explore_cfg["look_range"]),
                        math.radians(float(explore_cfg["look_half_angle_deg"])))  # fmt: skip
    holder: dict = {}

    def states() -> dict[str, str]:
        return {name: fact.value for name, fact in holder["session"].registry.tools.kitchen.states.items()}

    agent_config = dataclasses.replace(
        AgentConfig.load(), default_vocab=tuple(dict.fromkeys([*vocab["parts"], *vocab["objects"]]))
    )
    names = set(TOOLS)
    prompt = system_prompt(names) + (EXPLORE / "prompt.md").read_text()
    session = AgentSession.build(
        Plan.from_dict(task_spec), world.robot, world.detector, world_map, world.clock, run_dir,
        CoverageChecker(), CoverageObserver(coverage, states), config, agent_config,
        operator=lambda reason: "continue", system_prompt=prompt, tools_only=TOOLS,
    )  # fmt: skip
    holder["session"] = session
    session.log.start(task="explore_kitchen", task_file=str(task), seed=seed, policy=policy,
                      models={"agent": config.model if policy == "llm" else "scripted"},
                      tools=[s["function"]["name"] for s in session.registry.schemas], **world.info)  # fmt: skip

    if llm is None:
        if policy == "scripted":
            from chefbot.explore.scripted import ScriptedExplorer

            llm = ScriptedExplorer(WorldMap.from_dict(world.start), set(open_labels))
        else:
            from chefbot.agent.loop.llm import OpenRouterClient

            llm = OpenRouterClient(config.model, temperature=config.temperature, provider=config.provider)

    def after_action(s: AgentSession, tool: str, args: dict, result, reasoning) -> None:
        if tool == "detect" and result.success:
            coverage.record_detect(s.robot.get_base_pose(), states())
        if not quiet:
            mark = "ok " if result.success else f"ERR {result.error}"
            call = f"{tool}({', '.join(f'{k}={json.dumps(v)}' for k, v in args.items())})"
            print(f"{s.turn:>4} {s.clock.now():>6.0f}s  {call:<48} {mark}  {result.message[:90]}", flush=True)

    try:
        outcome = AgentRunner(session, llm, after_action=after_action).run()
        explored = explored_map(world.start, world_map)
        save(explored, run_dir / "explored_map.yaml")
        # The run is over: only now is the truth made, and only the scorer reads it.
        truth = world.truth()
    finally:
        world.close()
    save(truth, run_dir / "true_map.yaml")
    result = score(explored, truth)
    result["outcome"] = list(outcome)
    result["run_dir"] = str(run_dir)
    if world.info.get("committed_map"):
        committed = yaml.safe_load(Path(world.info["committed_map"]).read_text())
        result["vs_committed_map"] = score(explored, committed)
        result["truth_matches_committed_map"] = score(truth, committed, tolerance=0.0)["perfect"]
    (run_dir / "score.json").write_text(json.dumps(result, indent=2))
    if not quiet:
        print(f"\n{outcome[0]}: {outcome[1]}\n{describe(result)}\nrun: {run_dir}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", choices=["symbolic", "robocasa"], default="symbolic")
    parser.add_argument("--policy", choices=["llm", "scripted"], default="llm")
    parser.add_argument("--detector", choices=["groundtruth", "vision"], default="groundtruth",
                        help="robocasa: what detect() is backed by")  # fmt: skip
    parser.add_argument("--task", type=Path, default=TASK)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model")
    parser.add_argument("--out", type=Path, default=REPO / "runs")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    result = run(args.env, args.policy, args.detector, args.seed, args.task, args.out,
                 LoopConfig.load(model=args.model), quiet=args.quiet)  # fmt: skip
    if args.quiet:
        print(json.dumps({k: v for k, v in result.items() if k != "objects"}))
    return 0 if result["perfect"] else 1


if __name__ == "__main__":
    sys.exit(main())
