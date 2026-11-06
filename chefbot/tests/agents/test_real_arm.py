"""The real_arm agent on the real robot (run_agent --env real), here on the real-host tests' fake robot: its session,
its skills run as sandboxed episodes on one host kept open for the run, a skill's failure handed back to the agent,
a completed step taken as done unchecked, a dry run that sends nothing."""

import json
from pathlib import Path

import pytest

from chefbot import agents, skills
from chefbot.agent.check.checker import UncheckedJudge
from chefbot.agent.loop.llm import LLMResponse, ToolCall
from chefbot.agents.session import Session
from chefbot.scripts import run_agent
from chefbot.tests.onboard.test_real_host import FakeRobot

REPO = Path(__file__).resolve().parents[3]
JOB = REPO / "chefbot/jobs/real_pick.yaml"
PLAN_TOOLS = ("complete", "set_status", "add_step", "skip_step", "request_replan", "finish")


def test_its_session_on_the_real_robot():
    s = Session.open(agents.load("real_arm"), env="real")
    assert s.interface == "chefbot-v1"
    assert [x.path for x in s.skills] == ["pick_up/gemini_v1", "ready/v1"]  # the wrist camera only
    assert s.tools == PLAN_TOOLS
    assert all(x.description for x in s.skills)  # what the LLM reads about them


def test_in_the_sims_it_gets_only_ready_in_the_fr3s_twin():
    for env in ("symbolic", "robocasa"):
        assert Session.open(agents.load("real_arm"), env=env).skills == []
    assert [x.path for x in Session.open(agents.load("real_arm"), env="mujoco_kitchen").skills] == ["ready/v1"]


class Scripted:
    """An LLM that makes the given calls in order, and keeps what it was offered and told."""

    def __init__(self, calls):
        self.calls, self.i, self.results, self.tools = calls, 0, [], None

    def chat(self, messages, tools=None, **kwargs):
        if self.tools is None:
            self.tools = [t["function"]["name"] for t in tools]
        last = next((m for m in reversed(messages) if m.get("role") == "tool"), None)
        if last is not None and self.i:
            self.results.append(json.loads(last["content"]))
        name, args = self.calls[min(self.i, len(self.calls) - 1)]
        self.i += 1
        return LLMResponse(None, [ToolCall(f"c{self.i}", name, json.dumps(args))])


def with_bringup(tmp_path) -> str:
    """real_arm plus bringup/v1: a skill that runs to its end on the fake robot (no camera, no model)."""
    import shutil

    import yaml

    d = tmp_path / "agent" / "real_arm"
    shutil.copytree(agents.ROOT / "real_arm", d)
    m = yaml.safe_load((d / "agent.yaml").read_text())
    m["skills"]["include"] = ["bringup/v1", "pick_up/gemini_v1", "ready/v1"]
    (d / "agent.yaml").write_text(yaml.safe_dump(m))
    return str(d)


def _run(tmp_path, calls, agent="real_arm", **options):
    llm, fake = Scripted(calls), FakeRobot()
    result = run_agent.run(JOB, 0, tmp_path / "runs", llm=llm, operator=lambda r: "abort", quiet=True, env="real", agent=agent,
                           env_options={"env": fake, "confirm": False, **options})  # fmt: skip
    return llm, fake, result, Path(result["run_dir"])


def test_its_host_asks_the_robot_for_the_wrist_camera_only(tmp_path, monkeypatch):
    asked = []
    monkeypatch.setattr(FakeRobot, "keep_cameras_open", lambda self, names: asked.append(list(names)), raising=False)
    _run(tmp_path, [("finish", {"summary": "nothing", "abandon": True})])
    assert asked == [["wrist"]]


@pytest.fixture(scope="module", autouse=True)
def a_gemini_key():
    """pick_up/gemini_v1 needs one to be offered; no test here reaches Gemini (the fake robot has no camera)."""
    import os

    before = os.environ.get("GEMINI_API_KEY")
    os.environ["GEMINI_API_KEY"] = before or "not-a-real-key"
    yield
    if before is None:
        del os.environ["GEMINI_API_KEY"]


@pytest.fixture(scope="module")
def pick_run(tmp_path_factory, a_gemini_key):
    calls = [("set_status", {"step_id": "s1", "status": "active"}), ("bringup", {"lift_m": 0.03}), ("pick_up", {}),
             ("pick_up", {"object": "apple"}), ("complete", {"step_id": "s1"}), ("finish", {"summary": "scripted"})]  # fmt: skip
    tmp = tmp_path_factory.mktemp("real")
    return _run(tmp, calls, agent=with_bringup(tmp))


def test_the_agent_is_offered_its_skills_and_the_plans_tools(pick_run):
    llm, _, _, run_dir = pick_run
    assert llm.tools == list(PLAN_TOOLS) + ["bringup", "pick_up", "ready"]
    start = json.loads((run_dir / "start.json").read_text())
    assert start["env"] == "real" and start["session"]["interface"] == "chefbot-v1" and start["dry_run"] is False


def test_a_skill_runs_as_one_sandboxed_episode_on_the_real_host(pick_run):
    llm, fake, _, run_dir = pick_run
    bringup = llm.results[1]
    assert bringup["success"] and bringup["data"]["returned"]["ok"] and bringup["data"]["end"] == "skill returned"
    assert fake.sent["arm"] > 0 and fake.sent["base"] == 0  # the arm moved, the base held
    assert (run_dir / "skills" / "01_bringup" / "result.json").exists()


def test_arguments_are_checked_before_anything_runs(pick_run):
    llm, _, _, run_dir = pick_run
    assert llm.results[2]["error"] == "invalid_arguments" and "object" in llm.results[2]["message"]
    assert not (run_dir / "skills" / "02_pick_up_").exists()


def test_a_skill_that_fails_is_handed_back_and_the_run_goes_on(pick_run):
    llm, _, result, run_dir = pick_run
    failed = llm.results[3]  # the fake robot has no wrist camera
    assert not failed["success"] and failed["error"] == "skill_error" and "no camera" in failed["message"]
    assert json.loads((run_dir / "skills" / "02_pick_up" / "result.json").read_text())["n_llm"] == 0  # no model reached
    assert result["outcome"] == "finished"


def test_a_completed_step_is_taken_as_done_unchecked(pick_run):
    llm, _, result, _ = pick_run
    assert llm.results[4]["success"] and UncheckedJudge.REASON in llm.results[4]["message"]
    assert result["success"] is None and result["judged"] is False  # nothing judges the real robot


def test_a_dry_run_sends_nothing_to_the_arm(tmp_path):
    calls = [("bringup", {"lift_m": 0.03}), ("finish", {"summary": "dry", "abandon": True})]
    llm, fake, _, run_dir = _run(tmp_path, calls, agent=with_bringup(tmp_path), dry_run=True)
    assert llm.results[0]["success"]  # the skill saw its moves made
    assert fake.sent["arm"] == 0 and fake.sent["gripper"] == 0 and fake.sent["moves"] == 0
    assert json.loads((run_dir / "start.json").read_text())["dry_run"] is True


def test_a_skill_this_machine_cant_serve_is_refused_before_the_robot_is_touched(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.setattr("chefbot.runtime.models.load_env", lambda path: None)  # not the repo's .env either
    fake = FakeRobot()
    with pytest.raises(SystemExit, match="needs GEMINI_API_KEY"):
        run_agent.run(JOB, 0, tmp_path, llm=Scripted([]), quiet=True, env="real", agent="real_arm",
                      env_options={"env": fake, "confirm": False})  # fmt: skip
    assert fake.sent == {"arm": 0, "base": 0, "gripper": 0, "moves": 0}


def test_the_runner_runs_only_sandboxed_skills():
    from chefbot.runtime.agent_skills import EpisodeSkills

    with pytest.raises(ValueError, match="not sandboxed"):
        EpisodeSkills(None, Path("."), None, None).tool(skills.load("go_to/robotapi_v1"))


def test_the_cli_offers_the_real_robot():
    import subprocess
    import sys

    out = subprocess.run([sys.executable, "-m", "chefbot.scripts.run_agent", "--help"], cwd=REPO, capture_output=True, text=True)
    assert "{symbolic,robocasa,real}" in out.stdout and "--dry-run" in out.stdout and "real_arm" in out.stdout


def test_ready_takes_the_arm_around_the_pole_from_where_the_real_arm_stood(tmp_path):
    # The real arm on 2026-11-05: pick_up's straight move home from here grazed the pole camera's keep-out and was
    # refused. ready gives the controller the pose instead, and every one of its joint targets passes the guard.
    import numpy as np

    from chefbot import embodiments
    from chefbot.environments.real.host import HOME
    from chefbot.robot.control.constraints import for_robot, path_violation

    stood = np.radians([0.1, -76.5, -0.1, -115.2, 0.5, 92.0, 44.2])
    body = embodiments.load("tidybot_fr3").body
    assert path_violation(for_robot(body), body.hand_m, [stood, np.asarray(HOME)]) is not None  # the straight way
    calls = [("ready", {}), ("finish", {"summary": "ready", "abandon": True})]
    llm, fake = Scripted(calls), FakeRobot()
    fake.q = stood.copy()
    run_agent.run(JOB, 0, tmp_path, llm=llm, operator=lambda r: "abort", quiet=True, env="real", agent="real_arm",
                  env_options={"env": fake, "confirm": False})  # fmt: skip
    assert llm.results[0]["success"] and llm.results[0]["data"]["returned"]["reason"] == "at the ready pose"
    assert np.degrees(np.abs(fake.q - np.asarray(HOME)).max()) < 5  # pick_up's own move home is then a few degrees
    assert path_violation(for_robot(body), body.hand_m, [fake.q, np.asarray(HOME)]) is None
