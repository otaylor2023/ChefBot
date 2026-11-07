"""The agent's robot-moving tools as skills (skills/<name>/robotapi_v1, robot-api-v1): what they declare, and that
the agent's ToolRegistry offers and runs them."""

import pytest

from chefbot import skills
from chefbot.agent.tools.manip import HandheldTools, KitchenTools, ManipTools
from chefbot.agent.tools.nav import NavTools
from chefbot.agent.tools.registry import HANDHELD_SCHEMAS, ROBOT_SKILLS, SKILLS, TOOL_SCHEMAS
from chefbot.interface.robot_api import ALL_CAPABILITIES, BASE, GRASP
from chefbot.interface.types import ToolResult
from chefbot.tests.agent.test_tool_availability import names
from chefbot.tests.skills.test_skill_manifests import _copy

OWNERS = {"nav": NavTools, "manip": ManipTools, "kitchen": KitchenTools, "handheld": HandheldTools}


def test_the_fourteen_are_robot_api_skills_that_could_run_in_the_sims():
    assert len(ROBOT_SKILLS) == 14
    for name in ROBOT_SKILLS:
        s = skills.load(f"{name}/robotapi_v1")
        assert (s.interface, s.sandboxed, s.robot, s.envs) == ("robot-api-v1", False, "any", ("symbolic", "robocasa"))
        assert s.description and s.needs <= ALL_CAPABILITIES


def test_the_registry_offers_their_schemas():
    offered = {s["function"]["name"]: s for s in TOOL_SCHEMAS + HANDHELD_SCHEMAS}
    for name in ROBOT_SKILLS:
        assert offered[name] == skills.load(f"{name}/robotapi_v1").schema


def test_a_skills_needs_decide_whether_it_is_offered():
    assert SKILLS["go_to"].needs == {BASE} and SKILLS["pick_up"].needs == {GRASP}
    stationary, _ = names({GRASP})
    assert "pick_up" in stationary and "go_to" not in stationary and "stir" not in stationary


def test_dispatch_runs_the_skills_main():
    _, registry = names(ALL_CAPABILITIES)
    for name in ROBOT_SKILLS:
        handler = registry._handlers[name]
        assert handler.func.__module__ == f"chefbot_skill.{name}.robotapi_v1"


class Recorder:
    """A stand-in for one of the agent's robot-side tool owners: records the call."""

    def __init__(self, log, owner):
        self.log, self.owner = log, owner

    def __getattr__(self, method):
        def call(*args):
            self.log.append((self.owner, method, args))
            return ToolResult(True, None, "ok")

        return call


class Ctx:
    def __init__(self, log):
        for owner in OWNERS:
            setattr(self, owner, Recorder(log, owner))


@pytest.mark.parametrize("name", ROBOT_SKILLS)
def test_each_main_calls_its_tools_method_on_the_right_owner(name):
    log = []
    s = skills.load(f"{name}/robotapi_v1")
    args = {a: ("x" if spec["type"] == "string" else 3.0 if spec["type"] == "number" else False)
            for a, spec in s.args.items()}  # fmt: skip
    assert s.function()(Ctx(log), **args).success
    [(owner, method, passed)] = log
    method_of = {"pick_up": "pick"}.get(name, name)  # the pick_up skill runs the agent's pick tool
    assert method == method_of and hasattr(OWNERS[owner], method_of)
    assert passed == tuple(args.values())


def test_wait_defaults_to_true_as_the_tools_did():
    log = []
    skills.load("go_to/robotapi_v1").function()(Ctx(log), place="stove")
    assert log == [("nav", "go_to", ("stove", True))]


def test_a_sandboxed_skill_has_no_in_process_function():
    with pytest.raises(ValueError, match="is sandboxed"):
        skills.load("bringup/v1").function()


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda m: (m.update(interface="robot-api-v1", sandboxed=False), m.pop("description")), "needs a description"),
        (lambda m: m.update(needs=["teleport"]), "not robot capabilities"),
    ],
)
def test_a_robot_api_skill_must_say_what_it_needs_and_what_it_does(tmp_path, edit, message):
    with pytest.raises(ValueError, match=message):
        skills.load(str(_copy(tmp_path, edit)))


def test_their_mains_import_nothing():
    # All they reach is ctx: the RobotAPI and the agent's tools on it, never an environment's ground truth.
    import ast

    for name in ROBOT_SKILLS:
        tree = ast.parse((skills.load(f"{name}/robotapi_v1").dir / "main.py").read_text())
        assert not [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))], name
