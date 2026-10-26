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
from chefbot.agent.world.map import WorldMap, label_of
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
    stack: Any = None  # robocasa: the sim stack, for the video recorder only
    positions: Callable[[], dict] | None = None  # true 3D object positions; call only after the run


def strip_objects(semantic: dict) -> dict:
    return {"frame": semantic.get("frame", "world"), "fixtures": semantic.get("fixtures") or {}, "objects": {}}


def symbolic_world(task: Path, seed: int, noise: str = "off") -> World:
    from chefbot.envs.symbolic.scenario import Scenario

    sc = Scenario.load(task, seed, noise)
    start = strip_objects(sc.world.map_dict())
    for name, f in sc.world.fixtures.items():
        start["fixtures"][name]["size"] = [round(f.width, 3), 0.6]
    return World(sc.name, sc.robot, sc.detector, sc.clock, start, sc.world.map_dict, info={"env": "symbolic"})


def robocasa_world(task: Path, seed: int, run_dir: Path, detector: str, layout: int | None = None,
                   style: int | None = None) -> World:  # fmt: skip
    from chefbot.envs.robocasa.build_map import (GRID_FILE, GRID_META_FILE, START_MAP_FILE, STORAGE_START_MAP_FILE,
                                                 build_semantic_map, object_positions, write_grid)  # fmt: skip
    from chefbot.envs.robocasa.config import RoboCasaConfig
    from chefbot.envs.robocasa.env import RoboCasaEnv
    from chefbot.envs.robocasa.kitchen_state import RoboCasaClock, RoboCasaKitchen
    from chefbot.envs.robocasa.serve import wire_stack
    from chefbot.envs.symbolic.scenario import load_yaml
    from chefbot.envs.symbolic.world import Timing
    from chefbot.robot.config import RobotConfig

    import numpy as np

    spec = load_yaml(task).get("robocasa") or {}
    layout = int(layout or spec.get("layout", 1))
    style = int(style or spec.get("style", spec.get("layout", 1)))
    config = RoboCasaConfig.load(layout=layout, style=style, seed=seed)
    if spec.get("objects"):
        config = dataclasses.replace(config, objects=tuple(dict(o) for o in spec["objects"]))
    storage = bool(spec.get("storage"))  # cabinets and drawers on the map, to be opened too
    start_path = config.map_dir / (STORAGE_START_MAP_FILE if storage else START_MAP_FILE)
    if not start_path.exists():
        raise SystemExit(f"no start map {start_path}: python -m chefbot.envs.robocasa.build_map --start "
                         f"--layout {layout} --style {style}" + (" --storage" if storage else ""))  # fmt: skip
    env = RoboCasaEnv(config, render=True)
    grid_dir = write_grid(env, run_dir / "grid")  # geometry only: the robot drives on it
    stack = wire_stack(env, grid_dir, RobotConfig.load())
    kitchen = RoboCasaKitchen(env, stack.robot._fixture_actuator, Timing.load(spec.get("timing")))
    agent_detector = stack.detector
    start = yaml.safe_load(start_path.read_text())
    if detector in ("vision", "owl"):
        from chefbot.explore.vision import VisionDetector

        model = None  # vision: the VLM
        if detector == "owl":
            from chefbot.explore.owl import OwlV2Model

            model = OwlV2Model()
        footprints = [e for e in WorldMap.from_dict(start).list() if e.kind == "fixture"]
        agent_detector = VisionDetector.for_robocasa(env, stack.detector, log_dir=run_dir / "vision",
                                                     footprints=footprints, vlm=model)  # fmt: skip

    def truth() -> dict:
        occupied = np.load(grid_dir / GRID_FILE)
        return build_semantic_map(env, occupied, yaml.safe_load((grid_dir / GRID_META_FILE).read_text()), storage)

    return World(task.stem, stack.robot, agent_detector, RoboCasaClock(kitchen), start,
                 truth, env.close, info={"env": "robocasa", "layout": layout, "style": style, "detector": detector,
                                         "committed_map": str(config.map_dir / "semantic_map.yaml")},
                 stack=stack, positions=lambda: object_positions(env))  # fmt: skip


def run(env: str = "symbolic", policy: str = "llm", detector: str = "groundtruth", seed: int = 0,
        task: Path = TASK, out: Path = REPO / "runs", config: LoopConfig | None = None, llm=None,
        quiet: bool = False, video_speed: int = 0, layout: int | None = None,
        style: int | None = None) -> dict:  # fmt: skip
    """video_speed > 0 (robocasa): record runs/<run>/explore.mp4, keeping every Nth sim tick."""
    explore_cfg = yaml.safe_load((EXPLORE / "config.yaml").read_text())
    vocab = yaml.safe_load((EXPLORE / explore_cfg["vocab_file"]).read_text())
    config = config or LoopConfig.load()
    task_spec = yaml.safe_load(task.read_text())
    limits = {k: v for k, v in (task_spec.get("limits") or {}).items() if hasattr(config, k)}
    config = dataclasses.replace(config, **limits)
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = out / (f"{stamp}_explore_{env}_{policy}_s{seed}" + (f"_layout{layout:03d}" if layout else ""))
    run_dir.mkdir(parents=True, exist_ok=True)

    world = symbolic_world(task, seed) if env == "symbolic" else robocasa_world(task, seed, run_dir, detector, layout, style)
    (run_dir / "start_map.yaml").write_text(yaml.safe_dump(world.start, sort_keys=False, default_flow_style=None))
    world_map = WorldMap.from_dict(world.start)
    open_labels = frozenset(explore_cfg["open_labels"])
    coverage = Coverage(world_map, open_labels, float(explore_cfg["look_range"]),
                        math.radians(float(explore_cfg["look_half_angle_deg"])),
                        float(explore_cfg["storage_max_bottom"]))  # fmt: skip
    holder: dict = {}

    def states() -> dict[str, str]:
        return {name: fact.value for name, fact in holder["session"].registry.tools.kitchen.states.items()}

    agent_config = dataclasses.replace(
        AgentConfig.load(), default_vocab=tuple(dict.fromkeys([*vocab["parts"], *vocab["objects"]])),
        search_default_vocab=False,  # the whole catalogue (~240 names): not things to search for
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

            vision_cfg = explore_cfg["vision"]
            approach = ({"distance": vision_cfg.get("approach_distance", 0.7), "per_stop": vision_cfg.get("approach_per_stop", 2)}
                        if detector in ("vision", "owl") and vision_cfg.get("approach_small") else None)  # fmt: skip
            llm = ScriptedExplorer(WorldMap.from_dict(world.start), set(open_labels),
                                   storage_max_bottom=float(explore_cfg["storage_max_bottom"]), approach=approach)  # fmt: skip
        else:
            from chefbot.agent.loop.llm import OpenRouterClient

            llm = OpenRouterClient(config.model, temperature=config.temperature, provider=config.provider)

    recorder = None
    if video_speed > 0:
        if world.stack is None:
            raise SystemExit("--video needs --env robocasa")
        from chefbot.scripts.record_rollout import Recorder

        # Renders the sim for the video only; nothing it computes reaches the agent. The map and
        # semantic map panels draw the agent's own records: the start map (fixtures only), the map
        # it is building (as explored_map.yaml will hold it), its coverage and its detect() results.
        recorder = Recorder(world.stack, session.registry, WorldMap.from_dict(world.start), run_dir / "explore.mp4",
                            video_speed, semantic=lambda: explored_map(world.start, world_map)["objects"],
                            remaining=lambda: {line.split(":")[0] for line in coverage.remaining(states())})  # fmt: skip

    def call_text(tool: str, args: dict) -> str:
        return f"{tool}({', '.join(f'{k}={json.dumps(v)}' for k, v in args.items())})"

    def before_action(s: AgentSession, tool: str, args: dict, reasoning) -> None:
        recorder.caption = (f"> {call_text(tool, args)}", "...")

    seen_3d: dict[str, list[float]] = {}  # the agent's map name -> where it last saw it

    def after_action(s: AgentSession, tool: str, args: dict, result, reasoning) -> None:
        if recorder is not None:
            summary = f"{'ok' if result.success else 'FAILED: ' + str(result.error)}  {result.message}".strip()
            recorder.caption = (recorder.caption[0], summary if len(summary) <= 150 else summary[:147] + "...")
            if (result.data or {}).get("status") != "moving":
                recorder.hold(1.5 if tool in ("detect", "nearby") else 0.8)
        if tool == "detect" and result.success:
            for e in s.registry.tools.detections.all():  # where the agent saw each mapped object, in 3D
                if e.map_name is not None:
                    seen_3d[e.map_name] = [round(float(v), 4) for v in e.position_world]
            coverage.record_detect(s.robot.get_base_pose(), states())
        if not quiet:
            mark = "ok " if result.success else f"ERR {result.error}"
            call = call_text(tool, args)
            print(f"{s.turn:>4} {s.clock.now():>6.0f}s  {call:<48} {mark}  {result.message[:90]}", flush=True)

    try:
        if recorder is not None:
            recorder.caption = (f"exploring {world.name} ({world.info.get('detector', env)} detector)", "")
            recorder.hold(1.0)
        outcome = AgentRunner(session, llm, after_action=after_action,
                              before_action=before_action if recorder is not None else None).run()  # fmt: skip
        for label, where in getattr(world.detector, "superseded", lambda: [])():
            for e in world_map.list():  # a far-off misnaming that a closer look corrected
                if e.kind == "object" and label_of(e.name) == label and math.dist(e.location, where[:2]) <= 0.1:
                    world_map.remove(e.name)
        sources: dict = {}
        explored = explored_map(world.start, world_map, sources)
        save(explored, run_dir / "explored_map.yaml")
        # The run is over: only now is the truth made, and only the scorer reads it.
        truth = world.truth()
        true_positions = world.positions() if world.positions else None
    finally:
        if recorder is not None:
            world.robot.stop()
            recorder.close()
        world.close()
    save(truth, run_dir / "true_map.yaml")
    # For the position accuracy report (explore/accuracy.py): 3D, explored and true.
    save({n: seen_3d[m] for n, m in sources.items() if m in seen_3d}, run_dir / "explored_positions.yaml")
    if true_positions is not None:
        save(true_positions, run_dir / "true_positions.yaml")
    result = score(explored, truth)
    result["outcome"] = list(outcome)
    if getattr(llm, "approaches", None) is not None and getattr(llm, "_approach", None) is not None:
        result["approaches"] = dict(llm.approaches)  # go closer: extra stops made, and ones it could not reach
    result["run_dir"] = str(run_dir)
    if world.info.get("committed_map") and Path(world.info["committed_map"]).exists():
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
    parser.add_argument("--detector", choices=["groundtruth", "vision", "owl"], default="groundtruth",
                        help="robocasa: what detect() is backed by")  # fmt: skip
    parser.add_argument("--task", type=Path, default=TASK)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout", type=int, help="robocasa: kitchen layout (default: the task's)")
    parser.add_argument("--style", type=int, help="robocasa: kitchen style (default: the task's)")
    parser.add_argument("--model")
    parser.add_argument("--out", type=Path, default=REPO / "runs")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--video", action="store_true", help="robocasa: record runs/<run>/explore.mp4")
    parser.add_argument("--video-speed", type=int, default=2, help="keep every Nth sim tick (N x playback speed)")
    args = parser.parse_args()
    result = run(args.env, args.policy, args.detector, args.seed, args.task, args.out,
                 LoopConfig.load(model=args.model), quiet=args.quiet,
                 video_speed=args.video_speed if args.video else 0, layout=args.layout, style=args.style)  # fmt: skip
    if args.quiet:
        print(json.dumps({k: v for k, v in result.items() if k != "objects"}))
    return 0 if result["perfect"] else 1


if __name__ == "__main__":
    sys.exit(main())
