"""Shared fixtures for the RoboCasa integration tests.

One simulator per layout for the whole session; each test gets tools wired
either in-process or through RemoteRobot / RemoteDetector over a websocket.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

pytest.importorskip("robosuite")
pytest.importorskip("robocasa")

from chefbot.agent.config import AgentConfig  # noqa: E402
from chefbot.agent.tools.registry import ToolRegistry  # noqa: E402
from chefbot.agent.world.map import WorldMap  # noqa: E402
from chefbot.envs.robocasa.build_map import build_map  # noqa: E402
from chefbot.envs.robocasa.config import RoboCasaConfig  # noqa: E402
from chefbot.envs.robocasa.detector import GroundTruthDetector  # noqa: E402
from chefbot.envs.robocasa.env import RoboCasaEnv  # noqa: E402
from chefbot.interface.detector import Detector  # noqa: E402
from chefbot.interface.robot_api import RobotAPI  # noqa: E402
from chefbot.interface.transport.client import RemoteDetector, RemoteRobot  # noqa: E402
from chefbot.interface.transport.server import RobotServer  # noqa: E402
from chefbot.interface.types import Pose2D  # noqa: E402
from chefbot.robot.config import RobotConfig  # noqa: E402
from chefbot.robot.nav.grid import OccupancyGrid  # noqa: E402
from chefbot.robot.robot import Robot  # noqa: E402

LAYOUTS = [1, 2]  # 1: single wall of counters; 2: wall plus an island

pytestmark = pytest.mark.robocasa


@dataclass
class Stack:
    layout: int
    env: RoboCasaEnv
    robot: Robot
    detector: GroundTruthDetector
    grid: OccupancyGrid
    world_map: WorldMap
    robot_config: RobotConfig

    def free_starts(self, count: int, seed: int) -> list[Pose2D]:
        """Random collision-free base poses with some room around them."""
        inflated = self.grid.inflate(self.robot_config.inflation_radius + 0.1)
        rows, cols = np.nonzero(~inflated.occupied)
        rng = np.random.default_rng(seed)
        picks = rng.choice(len(rows), size=count, replace=False)
        return [
            Pose2D(*inflated.cell_to_world((int(rows[i]), int(cols[i]))), float(rng.uniform(-np.pi, np.pi)))
            for i in picks
        ]


@pytest.fixture(scope="session")
def stacks(tmp_path_factory):
    built: dict[int, Stack] = {}

    def get(layout: int) -> Stack:
        if layout not in built:
            env = RoboCasaEnv(RoboCasaConfig.load(layout=layout, style=layout))
            map_dir = build_map(env, tmp_path_factory.mktemp(f"map_layout{layout}"))
            grid = OccupancyGrid.load(map_dir / "grid.npy")
            robot_config = RobotConfig.load()
            built[layout] = Stack(
                layout,
                env,
                Robot(env, grid, robot_config),
                GroundTruthDetector(env),
                grid,
                WorldMap.load(map_dir / "semantic_map.yaml"),
                robot_config,
            )
        return built[layout]

    yield get
    for stack in built.values():
        stack.robot.stop_base()
        stack.env.close()


@pytest.fixture(params=LAYOUTS, ids=lambda n: f"layout{n}")
def stack(request, stacks) -> Stack:
    return stacks(request.param)


@dataclass
class Session:
    """Agent-side view of a stack: tools over one of the two transports."""

    stack: Stack
    robot: RobotAPI
    detector: Detector
    registry: ToolRegistry

    def call(self, name: str, **args):
        return self.registry.dispatch(name, args)


@pytest.fixture(params=["inprocess", "remote"])
def session(request, stack) -> Session:
    stack.robot.stop_base()
    server = None
    if request.param == "remote":
        server = RobotServer(stack.robot, stack.detector, port=0)
        server.start()
        robot, detector = RemoteRobot(server.url), RemoteDetector(server.url)
    else:
        robot, detector = stack.robot, stack.detector
    registry = ToolRegistry(robot, detector, stack.world_map, AgentConfig.load())
    yield Session(stack, robot, detector, registry)
    stack.robot.stop_base()
    if server is not None:
        robot.close()
        detector.close()
        server.shutdown()
