"""Scripted navigation trace through the agent tools.

In-process (simulator, robot layer and agent in one process):
    python -m chefbot.scripts.demo_nav --env robocasa

Remote (agent here, robot layer + simulator behind a websocket):
    python -m chefbot.envs.robocasa.serve            # terminal 1
    python -m chefbot.scripts.demo_nav --remote ws://127.0.0.1:8765   # terminal 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chefbot.agent.config import AgentConfig
from chefbot.agent.tools.registry import ToolRegistry
from chefbot.agent.world.map import WorldMap
from chefbot.interface.types import ToolResult

TRACE: list[tuple[str, dict]] = [
    ("go_to", {"place": "stove"}),
    ("detect", {"classes": ["burner", "pan"]}),
    ("nearby", {}),
    ("align", {"id": "burner_1"}),
    ("go_to", {"place": "fridge"}),
    ("detect", {"classes": ["fridge handle"]}),
    ("align", {"id": "fridge handle_1"}),
]


def run_trace(registry: ToolRegistry, verbose: bool = True) -> list[ToolResult]:
    results = []
    for name, args in TRACE:
        result = registry.dispatch(name, args)
        results.append(result)
        if verbose:
            call = ", ".join(f"{k}={json.dumps(v)}" for k, v in args.items())
            print(f"> {name}({call})")
            print(f"  success: {result.success}" + (f"   error: {result.error}" if result.error else ""))
            if result.message:
                print(f"  message: {result.message}")
            print(f"  data: {json.dumps(result.data)}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the scripted navigation trace.")
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--env", choices=["robocasa"], help="run the env and robot layer in this process")
    where.add_argument("--remote", metavar="URL", help="connect to a RobotServer, e.g. ws://127.0.0.1:8765")
    parser.add_argument("--layout", type=int, help="kitchen layout (default: config/envs/robocasa.yaml)")
    parser.add_argument("--style", type=int, help="kitchen style (default: config/envs/robocasa.yaml)")
    parser.add_argument("--map", type=Path, help="semantic_map.yaml (default: the map folder for the layout/style)")
    args = parser.parse_args()

    close = []
    if args.env == "robocasa":
        # Imported here so --remote runs without any simulator installed.
        from chefbot.envs.robocasa.config import RoboCasaConfig
        from chefbot.envs.robocasa.serve import build_stack

        stack = build_stack(RoboCasaConfig.load(layout=args.layout, style=args.style))
        robot, detector = stack.robot, stack.detector
        map_path = args.map or stack.semantic_map_path
        close = [stack.robot.stop, stack.env.close]
    else:
        from chefbot.envs.robocasa.config import RoboCasaConfig
        from chefbot.interface.transport.client import RemoteDetector, RemoteRobot

        robot, detector = RemoteRobot(args.remote), RemoteDetector(args.remote)
        config = RoboCasaConfig.load(layout=args.layout, style=args.style)
        map_path = args.map or config.map_dir / "semantic_map.yaml"
        close = [robot.close, detector.close]

    try:
        registry = ToolRegistry(robot, detector, WorldMap.load(map_path), AgentConfig.load())
        results = run_trace(registry)
    finally:
        for fn in close:
            fn()
    failed = [name for (name, _), result in zip(TRACE, results) if not result.success]
    print("\nall steps succeeded" if not failed else f"\nfailed steps: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
