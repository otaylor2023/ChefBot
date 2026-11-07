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


def test_a_chefbot_v1_session_offers_the_sandboxed_skills_in_its_registry(tmp_path):
    def edit(m):
        m.update(interfaces=["robot-api-v1", "chefbot-v1"])
        m["skills"]["lift"] = "tongs_lift/mujoco_tongs3_v59_fr3"

    s = Session.open(agents.load(str(_copy(tmp_path, edit))), env="real", interface="chefbot-v1")
    assert [(x.name, x.path) for x in s.skills] == [("lift", "tongs_lift/mujoco_tongs3_v59_fr3")]  # one API per session


def test_its_registry_names_and_describes_its_skills(tmp_path):
    def edit(m):
        m["skills"] = {"fetch": {"path": "pick_up/robotapi_v1", "description": "Fetch it.", "args": {"id": "which"}}}

    (s,) = agents.load(str(_copy(tmp_path, edit))).skills
    f = s.schema["function"]
    assert s.path == "pick_up/robotapi_v1" and f["name"] == "fetch" and f["description"] == "Fetch it."
    assert f["parameters"]["properties"]["id"]["description"] == "which"
    assert skills.load("pick_up/robotapi_v1").schema["function"]["name"] == "pick_up"  # the registered one unchanged


def test_a_chefbot_v1_session_would_offer_only_the_tools_that_need_no_robot_api(tmp_path):
    d = _copy(tmp_path, lambda m: m.update(interfaces=["robot-api-v1", "chefbot-v1"]))
    s = Session.open(agents.load(str(d)), "mujoco_kitchen", "chefbot-v1")
    assert s.tools == PLAN_AND_TIME  # no detect, nearby, wait, get_status, stop
    assert s.skills == []  # its registry has no chefbot-v1 skill


def test_the_agent_chooses_its_skills(tmp_path):
    def few(m):
        m.update(name="mover")
        m["skills"] = {"go_to": "go_to/robotapi_v1", "pick_up": "pick_up/robotapi_v1"}

    d = _copy(tmp_path, few).rename(tmp_path / "mover")
    llm, start = _run(tmp_path / "runs", agent=str(d))
    offered = {t["function"]["name"] for t in llm.requests[0]["tools"]}
    assert offered & set(ROBOT_SKILLS) == {"go_to", "pick_up"} and len(offered) == 16 + 2
    assert start["session"]["skills"] == ["go_to/robotapi_v1", "pick_up/robotapi_v1"]


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda m: m["skills"].update(go=["teleport/v1"]), "a registered skill's path"),
        (lambda m: m["skills"].update(go="teleport/v1"), "no registered skill 'teleport/v1'"),
        (lambda m: m["skills"].update(go="go_to/*"), "a version, exactly"),
        (lambda m: m.update(skills=["go_to/robotapi_v1"]), "skills must map"),
        (lambda m: m["skills"].update({"go to": "go_to/robotapi_v1"}), "not a name the LLM can call"),
        (lambda m: m["skills"].update(go={"path": "go_to/robotapi_v1", "colour": "blue"}), "a registered skill's path"),
        (lambda m: m["skills"].update(go={"path": "go_to/robotapi_v1", "args": {"speed": "x"}}), "has no args"),
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


def _agent(tmp_path: Path, registry: dict):
    def edit(m):
        m["skills"] = registry

    return agents.load(str(_copy(tmp_path / "agent", edit)))


def test_a_skill_is_offered_only_in_the_envs_it_could_run_in(tmp_path, more_skills):
    agent = _agent(tmp_path, {"go_to": "go_to/robotapi_v1", "go_far": "go_to/robotapi_v2"})
    assert [s.path for s in Session.open(agent, env="symbolic").skills] == ["go_to/robotapi_v1"]
    assert [s.path for s in Session.open(agent, env="real").skills] == ["go_to/robotapi_v2"]


def test_a_skill_named_like_a_tool_is_refused(tmp_path):
    with pytest.raises(ValueError, match="skills and tools both named detect"):
        Session.open(_agent(tmp_path, {"detect": "go_to/robotapi_v1"}), env="symbolic")


def test_a_skill_no_slot_names_is_offered_last_and_runs(tmp_path, more_skills):
    registry = {n: f"{n}/robotapi_v1" for n in ROBOT_SKILLS} | {"wave": "wave/robotapi_v1"}
    session = Session.open(_agent(tmp_path, registry), env="symbolic")
    robot = Robot(ALL_CAPABILITIES)
    registry = ToolRegistry(robot, Scene(robot, list(OBJECTS)), WorldMap.from_dict(MAP), CFG, robot_skills=session.skills)
    assert registry.schemas[-1]["function"]["name"] == "wave"
    assert registry.dispatch("wave", {}).success


def test_the_registry_runs_no_sandboxed_skill():
    robot = Robot(ALL_CAPABILITIES)
    with pytest.raises(ValueError, match="are not robot-api-v1"):
        ToolRegistry(robot, Scene(robot, list(OBJECTS)), WorldMap.from_dict(MAP), CFG, robot_skills=[skills.load("bringup/v1")])
