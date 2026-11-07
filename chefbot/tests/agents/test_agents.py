"""Agents as definitions (chefbot/agents/<name>/agent.yaml): loading, checking, and run_agent running one."""

import json
import shutil
from pathlib import Path

import pytest
import yaml

from chefbot import agents
from chefbot.agent import harnesses
from chefbot.agent.config import AgentConfig, LoopConfig
from chefbot.agent.harnesses.plan_loop.context import system_prompt
from chefbot.agent.loop.llm import LLMResponse, ToolCall
from chefbot.scripts import run_agent

REPO = Path(__file__).resolve().parents[3]


def test_todays_agent_is_the_default():
    assert agents.DEFAULT == "kitchen_planner"
    assert agents.names() == ["kitchen_planner", "real_arm"]


def test_the_default_agent_loads():
    a = agents.load("kitchen_planner")
    assert a.harness_name == "plan_loop"
    assert a.harness is harnesses.get("plan_loop")
    assert hasattr(a.harness, "AgentSession") and hasattr(a.harness, "AgentRunner")
    assert a.interfaces == ("robot-api-v1",)
    assert (a.prompts_dir / "system.md").exists()


def test_its_settings_and_loop_are_what_the_config_loaders_default_to():
    # Code that loads AgentConfig / LoopConfig with no path (demos, tests) gets the default agent's.
    a = agents.load(agents.DEFAULT)
    assert a.settings == AgentConfig.load()
    assert a.loop_config() == LoopConfig.load()
    assert a.settings.approach_distance == 1.0 and "burner" in a.settings.default_vocab
    assert a.loop_config(max_turns=7).max_turns == 7


def test_its_prompts_are_the_ones_the_harness_uses_by_default():
    a = agents.load(agents.DEFAULT)
    for tools in (set(), {"go_to", "pick_up"}, {"stir", "squeeze"}):
        assert system_prompt(tools, a.prompts_dir) == system_prompt(tools)


def _copy(tmp_path: Path, edit) -> Path:
    d = tmp_path / "kitchen_planner"
    shutil.copytree(agents.ROOT / "kitchen_planner", d)
    m = yaml.safe_load((d / "agent.yaml").read_text())
    edit(m)
    (d / "agent.yaml").write_text(yaml.safe_dump(m))
    return d


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda m: m.update(colour="blue"), "unknown keys"),
        (lambda m: m.update(name="someone_else"), "is not its folder's"),
        (lambda m: m.update(harness="no_such_loop"), "no harness"),
        (lambda m: m.update(interfaces=["wholebody-v2"]), "interfaces must be"),
        (lambda m: m.update(interfaces=[]), "interfaces must be"),
        (lambda m: m.update(prompts="nowhere"), "no system.md"),
        (lambda m: m["settings"].update(approach_distnce=1.0), "unknown keys in agent.yaml settings"),
        (lambda m: m["loop"].update(time_mode="sometimes"), "time_mode"),
    ],
)
def test_a_broken_agent_is_refused_when_loaded(tmp_path, edit, message):
    with pytest.raises(ValueError, match=message):
        agents.load(str(_copy(tmp_path, edit)))


def test_an_unknown_agent_names_the_known_ones():
    with pytest.raises(ValueError, match="kitchen_planner"):
        agents.load("nobody")


class FirstRequest:
    def __init__(self):
        self.requests = []

    def chat(self, messages, tools=None, **kwargs):
        self.requests.append({"messages": messages, "tools": tools})
        return LLMResponse(None, [ToolCall("c1", "finish", json.dumps({"summary": "test", "abandon": True}))])


class NoJudge:
    def check(self, condition, observation):
        raise AssertionError("not reached")


def _run(tmp_path, **kwargs):
    llm = FirstRequest()
    result = run_agent.run(REPO / "chefbot/jobs/egg_to_pan.yaml", 0, tmp_path, llm=llm, judge=NoJudge(),
                           operator=lambda r: "abort", quiet=True, env="symbolic", noise="off", **kwargs)  # fmt: skip
    start = json.loads((Path(result["run_dir"]) / "start.json").read_text())
    return llm, start


def test_run_agent_runs_the_default_agent_and_logs_which(tmp_path):
    llm, start = _run(tmp_path)
    assert start["agent"] == {"name": "kitchen_planner", "harness": "plan_loop"}
    a = agents.load(agents.DEFAULT)
    sent = llm.requests[0]["messages"][0]["content"][0]["text"]
    assert sent == system_prompt({t["function"]["name"] for t in llm.requests[0]["tools"]}, a.prompts_dir)


def test_run_agent_uses_the_named_agents_prompts_and_settings(tmp_path):
    d = _copy(tmp_path, lambda m: (m.update(name="other"), m["loop"].update(max_turns=9)))
    d = d.rename(d.parent / "other")
    (d / "prompts" / "system.md").write_text("You are the other agent.\n")
    llm, start = _run(tmp_path / "runs", agent=str(d))
    assert start["agent"]["name"] == "other"
    assert start["config"]["max_turns"] == 9
    assert llm.requests[0]["messages"][0]["content"][0]["text"].startswith("You are the other agent.")


def test_the_cli_offers_every_agent():
    import subprocess
    import sys

    out = subprocess.run([sys.executable, "-m", "chefbot.scripts.run_agent", "--help"], cwd=REPO, capture_output=True, text=True)
    assert "--agent {kitchen_planner,real_arm}" in out.stdout
