"""Tools follow the robot's capabilities and the run's limits; the prompt follows the tools."""

from chefbot.agent.loop.context import system_prompt
from chefbot.agent.tools.registry import ToolRegistry
from chefbot.agent.world.map import WorldMap
from chefbot.interface.robot_api import ALL_CAPABILITIES, ARTICULATE, BASE, GRASP
from chefbot.tests.agent.fakes import FakeRobot
from chefbot.tests.agent.test_manip_tools import CFG, MAP, OBJECTS, Scene


class Robot(FakeRobot):
    def __init__(self, caps):
        super().__init__()
        self.caps = frozenset(caps)

    def capabilities(self):
        return self.caps


def names(caps, **kw):
    robot = Robot(caps)
    registry = ToolRegistry(robot, Scene(robot, list(OBJECTS)), WorldMap.from_dict(MAP), CFG, **kw)
    return {s["function"]["name"] for s in registry.schemas}, registry


def test_tools_follow_the_robots_capabilities():
    basic, _ = names({BASE, GRASP, ARTICULATE})  # e.g. RoboCasa today
    assert {"go_to", "pick", "open", "detect", "wait"} <= basic
    assert not {"turn_on", "squeeze", "stir", "pour", "lower_into"} & basic
    full, _ = names(ALL_CAPABILITIES)
    assert {"turn_on", "turn_off", "lower_into", "lift_out", "squeeze", "release", "stir", "pour"} <= full
    stationary, _ = names({GRASP})
    assert "go_to" not in stationary and "pick" in stationary


def test_runs_can_withhold_tools():
    tools, registry = names(ALL_CAPABILITIES, exclude=("pour", "stir"))
    assert "pour" not in tools and "stir" not in tools and "squeeze" in tools
    assert registry.dispatch("pour", {"id": "x"}).error == "unknown_tool"
    only, _ = names(ALL_CAPABILITIES, only=("go_to", "detect", "pick", "place"))
    assert only == {"go_to", "detect", "pick", "place"}


def test_the_prompt_explains_only_the_tools_offered():
    assert "HANDHELD TOOLS" not in system_prompt({"go_to", "pick"})
    assert "HANDHELD TOOLS" in system_prompt({"go_to", "pick", "squeeze"})
    assert system_prompt({"stir", "pour"}).count("HANDHELD TOOLS") == 1
