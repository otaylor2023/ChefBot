"""Composition root for the RoboCasa backend.

Builds Robot(RoboCasaEnv) + GroundTruthDetector, in-process or behind a
RobotServer. The control loop always runs here, next to the simulator.

    python -m chefbot.envs.robocasa.serve --layout 1 --style 1
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from chefbot.envs.robocasa.build_map import GRID_FILE, SEMANTIC_MAP_FILE, ensure_map
from chefbot.envs.robocasa.config import RoboCasaConfig
from chefbot.envs.robocasa.detector import GroundTruthDetector
from chefbot.envs.robocasa.env import RoboCasaEnv
from chefbot.interface.transport.server import RobotServer
from chefbot.robot.config import RobotConfig
from chefbot.robot.nav.grid import OccupancyGrid
from chefbot.robot.robot import Robot


@dataclass
class RoboCasaStack:
    env: RoboCasaEnv
    robot: Robot
    detector: GroundTruthDetector
    map_dir: Path

    @property
    def semantic_map_path(self) -> Path:
        return self.map_dir / SEMANTIC_MAP_FILE


def build_stack(
    config: RoboCasaConfig | None = None, robot_config: RobotConfig | None = None, render: bool = False
) -> RoboCasaStack:
    """Simulator, robot layer and detector for one kitchen. Builds the map files if missing.
    `render=True` enables the cameras (get_camera_rgbd)."""
    env = RoboCasaEnv(config, render=render)
    map_dir = ensure_map(env)
    robot = Robot(env, OccupancyGrid.load(map_dir / GRID_FILE), robot_config)
    return RoboCasaStack(env, robot, GroundTruthDetector(env), map_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a RoboCasa kitchen's robot and detector over a websocket.")
    parser.add_argument("--layout", type=int)
    parser.add_argument("--style", type=int)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    config = RoboCasaConfig.load(
        layout=args.layout, style=args.style, server_host=args.host, server_port=args.port
    )
    stack = build_stack(config)
    server = RobotServer(stack.robot, stack.detector, config.server_host, config.server_port)
    print(f"serving layout {config.layout} / style {config.style} at {server.url}", flush=True)
    print(f"semantic map: {stack.semantic_map_path}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stack.robot.stop_base()
        server.shutdown()
        stack.env.close()


if __name__ == "__main__":
    main()
