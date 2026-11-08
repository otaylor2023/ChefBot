"""The environments run_agent can run a task in, behind one shape (EnvRun).

  symbolic   the symbolic kitchen (task file `symbolic:` section)
  robocasa   a RoboCasa kitchen (task file `robocasa:` section: layout, style,
             objects to spawn, success rules, timing)
  real       the real robot (tidybot_fr3), its skills chefbot-v1 episodes on the real host (a session's skills);
             no map, no detector, no ground truth (environments/real/agent_world.py)
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
    skills: Any = None  # the real robot: its chefbot-v1 skills' runner (runtime/agent_skills.py EpisodeSkills)
    host_skills: Any = None  # the real robot: its host-v1 skills' runner (scripts/real_kitchen.py HostSkills)
    gripper_holds: Any = None  # the real robot: () -> bool, the fingers commanded closed and stopped on something


def symbolic(task_path: Path, seed: int, noise: str) -> EnvRun:
    from chefbot.environments.symbolic.scenario import Scenario

    sc = Scenario.load(task_path, seed, noise)

    def on_limit() -> None:
        for burner in sc.world.burners.values():
            sc.world.set_power(burner.name, False)

    return EnvRun(
        sc.name, sc.task, sc.robot, sc.detector, WorldMap.from_dict(sc.map), sc.clock, sc.observer, sc.safety,
        sc.success, on_limit, sc.on_plan, sc.on_tool, sc.applied,
        info={"env": "symbolic", "disturbances": [d.spec for d in sc.disturbances],
              "failures": sc.robot.rates.__dict__, "noise": noise},
    )  # fmt: skip


def robocasa(task_path: Path, seed: int, noise: str) -> EnvRun:
    from chefbot.environments.robocasa.build_map import GRID_FILE, SEMANTIC_MAP_FILE, build_map
    from chefbot.environments.robocasa.config import RoboCasaConfig
    from chefbot.environments.robocasa.env import RoboCasaEnv
    from chefbot.environments.robocasa.kitchen_state import RoboCasaClock, RoboCasaKitchen, RoboCasaObserver
    from chefbot.environments.robocasa.serve import RoboCasaStack, wire_stack
    from chefbot.environments.success import check_success
    from chefbot.environments.symbolic.scenario import load_yaml
    from chefbot.environments.symbolic.world import Timing
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
    if not (map_dir / GRID_FILE).exists():
        build_map(env, map_dir)
    stack: RoboCasaStack = wire_stack(env, map_dir, RobotConfig.load())
    kitchen = RoboCasaKitchen(env, stack.robot._fixture_actuator, Timing.load(spec.get("timing")))
    rules = spec.get("success") or {}

    def success(tools_called: list[str]) -> dict:
        return check_success(rules, kitchen.describe().splitlines(), tools_called)

    def on_limit() -> None:
        from chefbot.environments.robocasa.appliances import burners, set_burner
        import numpy as np

        with env.lock:
            for burner in burners(env):
                set_burner(env, burner, False, np.random.RandomState(0))

    return EnvRun(
        name, task, stack.robot, stack.detector, WorldMap.load(map_dir / SEMANTIC_MAP_FILE), RoboCasaClock(kitchen),
        RoboCasaObserver(kitchen), kitchen.safety, success, on_limit,
        info={"env": "robocasa", "layout": layout, "style": style}, stack=stack, close=env.close,
    )  # fmt: skip


def real(task_path: Path, seed: int, noise: str, skills: list = (), dry_run: bool = False, confirm: bool = False,
         url: str | None = None, speed: str = "slow", model_usd: float = 0.5, env=None, arm_speed: str | None = None,
         base_speed: str | None = None, objects: Path | None = None,
         map_dir: Path = REPO / "maps" / "src_kitchen_work") -> EnvRun:
    """The real robot through the onboard server (ssh -L 8765:localhost:8765 tidybot01). The arm only, the base held;
    every move checked by the real host's guard, at the robot's real limits. dry_run: nothing is sent to the arm or
    the gripper (the cameras are real). env: a stand-in for RealEnv (tests). objects: an object map to start from
    (scripts/real_kitchen.py explore's objects_memory.json): its objects go on the agent's kitchen map and go_to takes
    their names."""
    import os
    import tempfile

    from chefbot import embodiments
    from chefbot.environments.real.agent_world import (NoDetector, NoGroundTruth, StationaryRobot, WallClock,
                                                       no_safety_events, unjudged)  # fmt: skip
    from chefbot.environments.real.host import RealHost
    from chefbot.environments.symbolic.scenario import load_yaml
    from chefbot.runtime import models as M
    from chefbot.runtime.agent_skills import EpisodeSkills

    task = load_yaml(task_path)
    M.load_env(REPO / ".env")
    robot = embodiments.load("tidybot_fr3")
    missing = []  # what a skill needs and this machine lacks: refused before the robot is touched
    for s in skills:
        try:
            s.check(robot.name, os.environ.get("VISION_URL"))
        except Exception as e:  # skills.MissingRequirement
            missing.append(str(e))
    if missing:
        raise SystemExit("can't run on the real robot: " + "; ".join(missing))
    wanted = [s.cameras for s in skills]
    cameras = None if any(c is True for c in wanted) else sorted({n for c in wanted if isinstance(c, list) for n in c})
    out = Path(tempfile.mkdtemp(prefix="chefbot_real_"))  # run_agent points it into the run's folder
    profile = robot.speed(speed, arm=arm_speed, base=base_speed)  # each default `speed`
    host = RealHost(robot, profile, out, dry_run=dry_run, confirm=confirm, url=url, env=env, cameras=cameras,
                    log=(lambda *a: None) if env is not None else print)  # fmt: skip
    vision = M.Vision(os.environ.get("VISION_URL")) if any(s.access()["vision"] for s in skills) else None
    runner = EpisodeSkills(host, out, M.Ledger(out / "model_ledger.jsonl", model_usd), vision, confirm=confirm)
    host_skills = None
    world_map = WorldMap.from_dict({})
    if any(s.interface == "host-v1" for s in skills):  # the base on the map, finding things: in this process
        from chefbot.scripts.real_kitchen import HostSkills, RealKitchen

        known = None
        if objects is not None:
            from chefbot.perception.kitchen_map import KitchenMap

            known = KitchenMap.load(map_dir / "spec.yaml", objects)
        base_vel = profile.base_max_vel  # the run's base profile for the drives too
        kitchen = RealKitchen(host, map_dir, vision, out, confirm=confirm, speed=base_vel[0], yaw_speed=base_vel[2],
                              log=(lambda *a: None) if env is not None else print, objects=known, world_map=world_map)  # fmt: skip
        kitchen._sync()
        host_skills = HostSkills(kitchen)
    return EnvRun(
        task.get("name", task_path.stem), task, StationaryRobot(host), NoDetector(), world_map, WallClock(),
        NoGroundTruth(), no_safety_events, unjudged, host.halt,
        info={"env": "real", "robot": robot.name, "dry_run": dry_run, "speed": profile.name, "confirm": confirm},
        close=host.close, skills=runner, host_skills=host_skills, gripper_holds=lambda: _holds(host),
    )  # fmt: skip


def _holds(host) -> bool:
    """The real gripper holds something: commanded closed, and its fingers stopped wider than 4 mm apart."""
    s = host.raw_env.get_state()
    return bool(s.gripper_closed_cmd) and float(s.gripper_width) > 0.004


ENVS = {"symbolic": symbolic, "robocasa": robocasa, "real": real}
