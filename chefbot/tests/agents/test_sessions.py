"""Sessions (agents/session.py): one agent, one env, one robot API, and what that offers the agent."""

import shutil
from pathlib import Path

import pytest

from chefbot import agents, skills
from chefbot.agent.tools.registry import ROBOT_SKILLS, ToolRegistry
from chefbot.agent.world.map import WorldMap
from chefbot.agents.session import Session
from chefbot.interface.robot_api import ALL_CAPABILITIES
from chefbot.tests.agent.test_manip_tools import CFG, MAP, OBJECTS, Scene
from chefbot.tests.agent.test_tool_availability import Robot
from chefbot.tests.agents.test_agents import _copy, _run

PLAN_AND_TIME = ("complete", "set_status", "add_step", "skip_step", "request_replan", "finish", "set_timer",
                 "cancel_timer", "watch")  # fmt: skip


def test_the_default_session():
    s = Session.open(agents.load("kitchen_planner"), env="symbolic")
    assert s.interface == "robot-api-v1"
    assert sorted(x.name for x in s.skills) == sorted(ROBOT_SKILLS)
    assert len(s.tools) == 16


def test_run_agent_logs_its_session(tmp_path):
    llm, start = _run(tmp_path)
    assert start["session"]["env"] == "symbolic" and start["session"]["interface"] == "robot-api-v1"
    assert len(start["session"]["skills"]) == 14 and all(p.endswith("/robotapi_v1") for p in start["session"]["skills"])
    assert len(llm.requests[0]["tools"]) == 30


def test_an_interface_the_agent_does_not_run_with_is_refused(tmp_path):
    with pytest.raises(ValueError, match="runs with robot-api-v1, not chefbot-v1"):
        Session.open(agents.load("kitchen_planner"), env="symbolic", interface="chefbot-v1")
    with pytest.raises(ValueError, match="runs with robot-api-v1, not chefbot-v1"):
        _run(tmp_path, interface="chefbot-v1")  # before any env is built


def test_plan_loop_runs_no_chefbot_v1_session_yet(tmp_path):
    d = _copy(tmp_path, lambda m: m.update(interfaces=["robot-api-v1", "chefbot-v1"]))
    with pytest.raises(ValueError, match="harness plan_loop runs robot-api-v1 sessions, not chefbot-v1"):
        Session.open(agents.load(str(d)), env="mujoco_kitchen", interface="chefbot-v1")


def test_a_chefbot_v1_session_would_offer_only_the_tools_that_need_no_robot_api(tmp_path):
    d = _copy(tmp_path, lambda m: m.update(interfaces=["robot-api-v1", "chefbot-v1"]))
    s = Session(agents.load(str(d)), "mujoco_kitchen", "chefbot-v1")  # not opened: the harness refuses it today
    assert s.tools == PLAN_AND_TIME  # no detect, nearby, wait, get_status, stop
    assert s.skills == []  # */robotapi_v1 selects no chefbot-v1 skill


def test_the_agent_chooses_its_skills(tmp_path):
    def few(m):
        m.update(name="mover")
        m["skills"]["include"] = ["go_to/*", "pick/robotapi_v1"]

    d = _copy(tmp_path, few).rename(tmp_path / "mover")
    llm, start = _run(tmp_path / "runs", agent=str(d))
    offered = {t["function"]["name"] for t in llm.requests[0]["tools"]}
    assert offered & set(ROBOT_SKILLS) == {"go_to", "pick"} and len(offered) == 16 + 2
    assert start["session"]["skills"] == ["go_to/robotapi_v1", "pick/robotapi_v1"]


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda m: m["skills"].update(include=["teleport/*"]), "matches no registered skill"),
        (lambda m: m.update(skills=["*/robotapi_v1"]), "skills must be"),
        (lambda m: m["skills"].update(exclude=["pick/*"]), "skills must be"),
    ],
)
def test_a_broken_skill_selection_is_refused(tmp_path, edit, message):
    with pytest.raises(ValueError, match=message):
        agents.load(str(_copy(tmp_path, edit)))


@pytest.fixture
def more_skills(tmp_path, monkeypatch):
    """The registry plus go_to/robotapi_v2 (real only) and wave/robotapi_v1 (a skill no slot names)."""
    root = tmp_path / "skills"
    shutil.copytree(skills.ROOT, root, ignore=shutil.ignore_patterns("__pycache__"))
    v2 = root / "go_to" / "robotapi_v2"
    shutil.copytree(root / "go_to" / "robotapi_v1", v2)
    (v2 / "skill.yaml").write_text((v2 / "skill.yaml").read_text().replace("version: robotapi_v1", "version: robotapi_v2")
                                   .replace("- symbolic\n- robocasa\n", "- real\n"))  # fmt: skip
    wave = root / "wave" / "robotapi_v1"
    wave.mkdir(parents=True)
    (wave / "skill.yaml").write_text(
        "name: wave\nversion: robotapi_v1\nentry: main.py:run\ninterface: robot-api-v1\nenvs: [symbolic]\n"
        "sandboxed: false\nrobot: any\ndescription: Wave at the person.\nargs: {}\n"
    )
    (wave / "main.py").write_text("def run(ctx):\n    return ctx.nav.get_pose()\n")
    monkeypatch.setattr(skills, "ROOT", root)
    return root


def _agent(tmp_path: Path, include: list[str]):
    def edit(m):
        m["skills"]["include"] = include

    return agents.load(str(_copy(tmp_path / "agent", edit)))


def test_a_skill_is_offered_only_in_the_envs_it_could_run_in(tmp_path, more_skills):
    agent = _agent(tmp_path, ["go_to/*"])
    assert [s.path for s in Session.open(agent, env="symbolic").skills] == ["go_to/robotapi_v1"]
    assert [s.path for s in Session.open(agent, env="real").skills] == ["go_to/robotapi_v2"]


def test_two_skills_with_one_name_are_refused(tmp_path, more_skills):
    v2 = more_skills / "go_to" / "robotapi_v2" / "skill.yaml"
    v2.write_text(v2.read_text().replace("- real\n", "- symbolic\n"))
    with pytest.raises(ValueError, match="two skills named go_to"):
        Session.open(_agent(tmp_path, ["go_to/*"]), env="symbolic")


def test_a_skill_no_slot_names_is_offered_last_and_runs(tmp_path, more_skills):
    session = Session.open(_agent(tmp_path, ["*/robotapi_v1"]), env="symbolic")
    robot = Robot(ALL_CAPABILITIES)
    registry = ToolRegistry(robot, Scene(robot, list(OBJECTS)), WorldMap.from_dict(MAP), CFG, robot_skills=session.skills)
    assert registry.schemas[-1]["function"]["name"] == "wave"
    assert registry.dispatch("wave", {}).success


def test_the_registry_runs_no_sandboxed_skill():
    robot = Robot(ALL_CAPABILITIES)
    with pytest.raises(ValueError, match="are not robot-api-v1"):
        ToolRegistry(robot, Scene(robot, list(OBJECTS)), WorldMap.from_dict(MAP), CFG, robot_skills=[skills.load("bringup/v1")])
