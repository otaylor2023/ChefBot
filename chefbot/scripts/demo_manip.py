"""Scripted manipulation trace: fetch the egg from the fridge and put it on a counter.

In-process, saving a video and a JSON trace to rollouts/:
    python -m chefbot.scripts.demo_manip --env robocasa

Against a RobotServer (no video: the simulator is in the other process):
    python -m chefbot.environments.robocasa.serve                                # terminal 1
    python -m chefbot.scripts.demo_manip --remote ws://127.0.0.1:8765    # terminal 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from chefbot.agent.config import AgentConfig
from chefbot.agent.tools.registry import ToolRegistry
from chefbot.agent.world.map import WorldMap
from chefbot.interface.types import ToolResult

Run = Callable[[str, dict], ToolResult]


def manip_trace(run: Run, world_map: WorldMap | None = None, counter: str = "counter_1") -> bool:
    """go_to fridge -> open -> pick the egg -> close -> go_to counter -> place.
    Ids come from the detections, as they would for an agent. Stops at the first failure."""

    def first(result: ToolResult, label: str) -> str | None:
        found = [d["id"] for d in result.data.get("detections", []) if d["label"] == label]
        return found[0] if found else None

    def step(name: str, **args) -> ToolResult | None:
        result = run(name, args)
        return result if result.success else None

    if not step("go_to", place="fridge"):
        return False
    seen = step("detect", classes=["fridge handle"])
    handle = first(seen, "fridge handle") if seen else None
    if handle is None or not step("align", id=handle) or not step("open", id=handle):
        return False
    seen = step("detect", classes=["egg"])
    egg = first(seen, "egg") if seen else None
    if egg is None or not step("align", id=egg) or not step("pick", id=egg):
        return False
    # The detect above also saw the fridge itself, so it can be closed by name.
    if not step("close", id="fridge") or not step("go_to", place=counter):
        return False
    seen = step("detect", classes=["counter"])
    if seen is None or not step("align", id=counter) or not step("place", id=counter):
        return False
    return step("holding") is not None


def print_step(name: str, args: dict, result: ToolResult) -> None:
    call = ", ".join(f"{k}={json.dumps(v)}" for k, v in args.items())
    print(f"> {name}({call})")
    print(f"  success: {result.success}" + (f"   error: {result.error}" if result.error else ""))
    if result.message:
        print(f"  message: {result.message}")
    print(f"  data: {json.dumps(result.data)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the scripted manipulation trace.")
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--env", choices=["robocasa"], help="run the env and robot layer in this process")
    where.add_argument("--remote", metavar="URL", help="connect to a RobotServer, e.g. ws://127.0.0.1:8765")
    parser.add_argument("--layout", type=int, help="kitchen layout (default: config/envs/robocasa.yaml)")
    parser.add_argument("--style", type=int, help="kitchen style (default: config/envs/robocasa.yaml)")
    parser.add_argument("--out", type=Path, default=Path("rollouts"), help="folder for the video and JSON trace")
    parser.add_argument("--no-video", action="store_true", help="skip rendering (faster)")
    args = parser.parse_args()

    if args.env == "robocasa" and not args.no_video:
        # Imported here so --remote runs without any simulator installed.
        from chefbot.scripts.record_rollout import record

        video, log = record("manip", args.layout, args.style, args.out)
        ok = bool(log) and all(step["success"] for step in log) and log[-1]["tool"] == "holding"
        print(f"\nwrote {video}")
    else:
        from chefbot.environments.robocasa.config import RoboCasaConfig

        config = RoboCasaConfig.load(layout=args.layout, style=args.style)
        close = []
        if args.env == "robocasa":
            from chefbot.environments.robocasa.serve import build_stack

            stack = build_stack(config)
            robot, detector, map_path = stack.robot, stack.detector, stack.semantic_map_path
            close = [stack.robot.stop, stack.env.close]
        else:
            from chefbot.interface.transport.client import RemoteDetector, RemoteRobot

            robot, detector = RemoteRobot(args.remote), RemoteDetector(args.remote)
            map_path = config.map_dir / "semantic_map.yaml"
            close = [robot.close, detector.close]
        log = []
        try:
            world_map = WorldMap.load(map_path)
            registry = ToolRegistry(robot, detector, world_map, AgentConfig.load())

            def run(name: str, tool_args: dict) -> ToolResult:
                result = registry.dispatch(name, tool_args)
                print_step(name, tool_args, result)
                log.append({"tool": name, "args": tool_args, "success": result.success, "error": result.error,
                            "message": result.message, "data": result.data})  # fmt: skip
                return result

            ok = manip_trace(run, world_map)
        finally:
            for fn in close:
                fn()
        args.out.mkdir(parents=True, exist_ok=True)
        trace_path = args.out / f"manip_layout{config.layout:03d}_style{config.style:03d}.json"
        trace_path.write_text(json.dumps(log, indent=2))
        print(f"\nwrote {trace_path}")
    print("the egg is on the counter" if ok else "the trace did not finish")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
