"""The tools registry (chefbot/tools/<group>/<name>.yaml): loading, checking, the agent's ToolRegistry offering
them, and an agent's `tools:` choosing them."""

from pathlib import Path

import pytest
import yaml

from chefbot import agents
from chefbot import tools as toolbox
from chefbot.agent.tools.manip import ManipTools
from chefbot.agent.tools.nav import NavTools
from chefbot.agent.tools.plan_tools import PlanTools
from chefbot.agent.tools.registry import LOOP_TOOLS
from chefbot.agent.tools.time_tools import TimeTools
from chefbot.interface.robot_api import ALL_CAPABILITIES, BASE, GRASP
from chefbot.tests.agent.test_tool_availability import names
from chefbot.tests.agents.test_agents import _copy, _run

GROUPS = {
    "perception": {"detect", "nearby", "holding", "get_pose"},
    "motion": {"wait", "get_status", "stop"},
    "plan": {"complete", "set_status", "add_step", "skip_step", "request_replan", "finish"},
    "time": {"set_timer", "cancel_timer", "watch"},
}
OWNER_CLASSES = {"nav": NavTools, "manip": ManipTools, "plan": PlanTools, "time": TimeTools}


def test_the_sixteen_tools_that_do_not_move_the_robot():
    assert {name: toolbox.get(name).group for name in toolbox.names()} == {n: g for g, ns in GROUPS.items() for n in ns}


def test_every_entry_is_a_method_of_its_owner():
    for name in toolbox.names():
        owner, method = toolbox.get(name).entry
        assert callable(getattr(OWNER_CLASSES[owner], method, None)), name


def test_looking_and_following_motion_need_the_robot_api_planning_and_time_do_not():
    for name in toolbox.names():
        t = toolbox.get(name)
        expected = ("robot-api-v1",) if t.group in ("perception", "motion") else ("robot-api-v1", "chefbot-v1")
        assert t.interfaces == expected, name


def test_the_registry_offers_them_with_their_files_schemas():
    _, registry = names(ALL_CAPABILITIES)
    offered = {s["function"]["name"]: s for s in registry.schemas}
    for name in GROUPS["perception"] | {"get_status", "stop"}:
        assert offered[name] == toolbox.schema(name)
    assert set(LOOP_TOOLS) == GROUPS["plan"] | GROUPS["time"] | {"wait"}


def test_a_tools_needs_come_from_its_file():
    assert toolbox.get("holding").needs == {GRASP}
    with_hand, _ = names({BASE, GRASP})
    without_hand, _ = names({BASE})
    assert "holding" in with_hand and "holding" not in without_hand
    assert "detect" in without_hand  # needs nothing


def test_the_agent_tools_filter_only_touches_registered_tools():
    tools, registry = names(ALL_CAPABILITIES, agent_tools=("detect", "wait"))
    assert {"detect", "wait"} <= tools and not {"nearby", "holding", "get_pose", "get_status", "stop"} & tools
    assert {"go_to", "pick", "stir"} <= tools  # the robot-moving tools are not chefbot/tools'
    assert registry.dispatch("nearby", {}).error == "unknown_tool"


def _tool_file(tmp_path: Path, edit, name="detect", group="perception") -> Path:
    d = tmp_path / group
    d.mkdir(parents=True, exist_ok=True)
    m = yaml.safe_load((toolbox.ROOT / "perception" / "detect.yaml").read_text())
    edit(m)
    (d / f"{name}.yaml").write_text(yaml.safe_dump(m))
    return d / f"{name}.yaml"


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda m: m.update(colour="blue"), "unknown keys"),
        (lambda m: m.update(name="look"), "is not its file's"),
        (lambda m: m.update(description=""), "no description"),
        (lambda m: m["parameters"]["classes"].update(type="list"), "must have a type"),
        (lambda m: m.update(required=["objects"]), "are not parameters"),
        (lambda m: m.update(entry="detect"), "entry must be owner.method"),
        (lambda m: m.update(entry="camera.detect"), "entry must be owner.method"),
        (lambda m: m.update(needs=["eyes"]), "not robot capabilities"),
        (lambda m: m.update(interfaces=["wholebody-v2"]), "interfaces must be"),
        (lambda m: m.update(interfaces=[]), "interfaces must be"),
    ],
)
def test_a_broken_tool_file_is_refused(tmp_path, edit, message):
    with pytest.raises(ValueError, match=message):
        toolbox.load_file(_tool_file(tmp_path, edit))


def test_a_tool_defined_twice_is_refused(tmp_path):
    _tool_file(tmp_path, lambda m: None, group="perception")
    _tool_file(tmp_path, lambda m: None, group="vision")
    with pytest.raises(ValueError, match="defined twice"):
        toolbox._all(tmp_path)


def test_an_unknown_tool_names_the_known_ones():
    with pytest.raises(ValueError, match="detect"):
        toolbox.get("teleport")


def test_todays_agent_gets_all_sixteen():
    assert set(agents.load("kitchen_planner").tools) == set(toolbox.names())


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda m: m["tools"].append("teleport"), "no tools \\['teleport'\\]"),
        (lambda m: m["tools"].append("detect"), "listed twice"),
        (lambda m: m.update(interfaces=["chefbot-v1"]), "work with none of its interfaces"),
    ],
)
def test_an_agent_listing_a_tool_it_cannot_have_is_refused(tmp_path, edit, message):
    with pytest.raises(ValueError, match=message):
        agents.load(str(_copy(tmp_path, edit)))


def test_an_agent_without_some_tools_is_not_offered_them(tmp_path):
    def drop(m):
        m.update(name="no_timers")
        m["tools"] = [t for t in m["tools"] if t not in ("set_timer", "cancel_timer", "watch")]

    d = _copy(tmp_path, drop).rename(tmp_path / "no_timers")
    llm, start = _run(tmp_path / "runs", agent=str(d))
    offered = {t["function"]["name"] for t in llm.requests[0]["tools"]}
    assert not {"set_timer", "cancel_timer", "watch"} & offered
    assert {"detect", "wait", "complete", "finish", "go_to", "pick"} <= offered
    assert len(offered) == 27


def test_the_full_agent_is_offered_every_tool(tmp_path):
    llm, _ = _run(tmp_path)
    offered = {t["function"]["name"] for t in llm.requests[0]["tools"]}
    assert set(toolbox.names()) <= offered and len(offered) == 30
    for t in llm.requests[0]["tools"]:
        if t["function"]["name"] in toolbox.names():
            assert t == toolbox.schema(t["function"]["name"])


def test_the_tool_files_are_packaged():
    text = (Path(toolbox.__file__).resolve().parents[2] / "pyproject.toml").read_text()
    assert '"tools/*/*.yaml"' in text
    groups = [d for d in toolbox.ROOT.iterdir() if d.is_dir() and d.name != "__pycache__"]
    assert all(p.suffix == ".yaml" for d in groups for p in d.iterdir())  # nothing the package would miss
