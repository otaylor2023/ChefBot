"""The planner (agent/plan/planner.py): a request to a checked plan, clarifying questions, refusals, replanning; and
run_agent --plan-from prompt end to end in the symbolic kitchen, with scripted LLMs."""

import json
from pathlib import Path

import pytest

from chefbot.agent.loop.llm import LLMResponse, ToolCall
from chefbot.agent.plan.model import DONE, Plan, Step
from chefbot.agent.plan.planner import Planner, check_plan
from chefbot.scripts import run_agent

REPO = Path(__file__).resolve().parents[3]
EGG = [{"id": "s1", "goal": "Take the egg out of the fridge", "done_when": "the egg is out of the fridge"},
       {"id": "s2", "goal": "Put the egg in the pan", "done_when": "the egg is inside the pan", "depends_on": ["s1"]}]  # fmt: skip


class Script:
    """An LLM that answers with the given tool calls in order, and keeps what it was sent."""

    def __init__(self, calls):
        self.calls, self.i, self.sent = calls, 0, []

    def chat(self, messages, tools=None, **kwargs):
        self.sent.append(json.loads(json.dumps(messages)))  # as it was then: the planner keeps appending to its list
        name, args = self.calls[min(self.i, len(self.calls) - 1)]
        self.i += 1
        return LLMResponse(None, [ToolCall(f"p{self.i}", name, json.dumps(args))], cost_usd=0.001)


def planner(calls, user=None):
    return Planner(Script(calls), "You plan.", user)


def test_a_good_plan_is_accepted():
    r = planner([("submit_plan", {"goal": "egg in pan", "steps": EGG})]).plan("egg in the pan", "fridge: egg", "- pick_up")
    assert r.plan is not None and [s.id for s in r.plan.steps] == ["s1", "s2"] and r.rejected == [] and r.turns == 1


@pytest.mark.parametrize(
    "steps, problem",
    [
        ([], "no steps"),
        ([{"id": "s1", "goal": "g", "done_when": ""}], "no done_when"),
        ([{"id": "s1", "goal": "g", "done_when": "x"}, {"id": "s1", "goal": "h", "done_when": "y"}], "used twice"),
        ([{"id": "s1", "goal": "g", "done_when": "x", "depends_on": ["s9"]}], "not a step"),
        ([{"id": "a", "goal": "g", "done_when": "x", "depends_on": ["b"]}, {"id": "b", "goal": "h", "done_when": "y", "depends_on": ["a"]}], "cycle"),
        ([{"id": "s1", "goal": "g", "done_when": "I placed the egg in the pan"}], "not the robot's action"),
    ],
)
def test_a_bad_plan_is_refused_with_its_problems(steps, problem):
    assert any(problem in p for p in check_plan("goal", steps))


def test_a_refused_plan_goes_back_and_the_fix_is_accepted():
    bad = [{"id": "s1", "goal": "g", "done_when": ""}]
    p = planner([("submit_plan", {"goal": "x", "steps": bad}), ("submit_plan", {"goal": "x", "steps": EGG})])
    r = p.plan("egg", "k", "s")
    assert r.plan is not None and len(r.rejected) == 1 and r.turns == 2
    assert "not accepted" in p.llm.sent[1][-1]["content"]  # the problems went back to the model


def test_a_question_goes_to_the_user_and_the_answer_comes_back():
    asked = []
    p = planner([("ask_user", {"question": "Fried or scrambled?"}), ("submit_plan", {"goal": "x", "steps": EGG})],
                user=lambda q: asked.append(q) or "Fried, please")  # fmt: skip
    r = p.plan("an egg", "k", "s")
    assert asked == ["Fried or scrambled?"] and r.questions == [("Fried or scrambled?", "Fried, please")]
    assert "Fried, please" in p.llm.sent[1][-1]["content"]


def test_with_nobody_to_ask_it_is_told_to_decide():
    p = planner([("ask_user", {"question": "Which?"}), ("submit_plan", {"goal": "x", "steps": EGG})])
    r = p.plan("an egg", "k", "s")
    assert r.plan is not None and "Nobody can answer" in r.questions[0][1]


def test_questions_are_capped():
    p = planner([("ask_user", {"question": f"q{i}"}) for i in range(3)] + [("submit_plan", {"goal": "x", "steps": EGG})])
    r = p.plan("an egg", "k", "s")
    assert len(r.questions) == 2 and r.plan is not None


def test_it_can_say_the_request_cant_be_done():
    r = planner([("cannot_do", {"reason": "there is no egg in this kitchen"})]).plan("an omelette", "k", "s")
    assert r.plan is None and "no egg" in r.refused


def test_a_replan_keeps_the_done_steps():
    current = Plan("egg in pan", [Step("s1", "egg out", "the egg is out of the fridge", status=DONE),
                                  Step("s2", "egg in pan", "the egg is inside the pan", depends_on=["s1"])])  # fmt: skip
    new = [{"id": "s3", "goal": "Put the egg in the pot", "done_when": "the egg is inside the pot", "depends_on": ["s1"]}]
    p = planner([("submit_plan", {"goal": "egg in pot", "steps": new})])
    r = p.replan("egg in the pan", "the pan is broken", current, "k", "s")
    assert [s.id for s in r.plan.steps] == ["s1", "s3"] and r.plan.steps[0].status == DONE
    assert "the pan is broken" in p.llm.sent[0][-1]["content"]


def test_a_replan_may_not_reuse_a_done_steps_id():
    current = Plan("g", [Step("s1", "egg out", "out", status=DONE)])
    r = planner([("submit_plan", {"goal": "g", "steps": [{"id": "s1", "goal": "x", "done_when": "y"}]}),
                 ("cannot_do", {"reason": "stop"})]).replan("r", "why", current, "k", "s")  # fmt: skip
    assert "used twice" in r.rejected[0][0]


class Executor:
    """The executor's LLM: finish at once (the plan is what's under test here), or the given calls."""

    def __init__(self, calls=(("finish", {"summary": "done", "abandon": True}),)):
        self.calls, self.i, self.requests = list(calls), 0, []

    def chat(self, messages, tools=None, **kwargs):
        self.requests.append(messages)
        name, args = self.calls[min(self.i, len(self.calls) - 1)]
        self.i += 1
        return LLMResponse(None, [ToolCall(f"c{self.i}", name, json.dumps(args))])


def _run(tmp_path, planner_calls, executor=None, task="egg_to_pan", **kw):
    executor = executor or Executor()
    result = run_agent.run(REPO / f"chefbot/jobs/{task}.yaml", 0, tmp_path, llm=executor, planner_llm=Script(planner_calls),
                           judge=None, operator=lambda r: "abort", quiet=True, env="symbolic", noise="off",
                           plan_from="prompt", prompt="Take the egg out of the fridge and put it in the pan", **kw)  # fmt: skip
    return result, executor, Path(result["run_dir"])


def test_run_agent_executes_the_planners_plan(tmp_path):
    result, executor, run_dir = _run(tmp_path, [("submit_plan", {"goal": "The egg in the pan", "steps": EGG})])
    state = executor.requests[0][-1]["content"]
    state = state if isinstance(state, str) else "".join(x.get("text", "") for x in state)
    assert "GOAL: The egg in the pan" in state and "Take the egg out of the fridge" in state
    start = json.loads((run_dir / "start.json").read_text())
    assert start["reference_plan"]["goal"] == "The egg in the pan"
    planner_log = json.loads((run_dir / "planner.json").read_text())
    assert planner_log["request"].startswith("Take the egg") and planner_log["calls"][0]["plan"]["goal"] == "The egg in the pan"
    assert result["plan_from"] == "prompt" and result["planner"]["calls"] == 1
    report = (run_dir / "report.html").read_text()
    assert "User request" in report and planner_log["request"] in report and "Take the egg out of the fridge" in report
    conversation = (run_dir / "conversation.html").read_text()
    assert planner_log["request"] in conversation and "submit_plan" in conversation and "turn 1" in conversation


def test_the_planner_sees_the_kitchen_and_the_robots_skills_not_the_executors_tools(tmp_path):
    p = Script([("submit_plan", {"goal": "g", "steps": EGG})])
    run_agent.run(REPO / "chefbot/jobs/egg_to_pan.yaml", 0, tmp_path, llm=Executor(), planner_llm=p, operator=lambda r: "abort",
                  quiet=True, env="symbolic", noise="off", plan_from="prompt", prompt="egg in the pan")  # fmt: skip
    text = p.sent[0][-1]["content"]
    assert "KITCHEN" in text and "fridge" in text and "- pick_up(" in text and "- go_to(" in text
    assert "- complete(" not in text and "- set_timer(" not in text  # the executor's own tools
    assert "- open(id, wait): Open a fixture: a fridge" in text  # a colon doesn't cut the sentence


def test_a_refusal_ends_the_run_before_a_turn(tmp_path):
    result, executor, run_dir = _run(tmp_path, [("cannot_do", {"reason": "no eggs here"})])
    assert result["outcome"] == "refused" and executor.requests == []
    assert result["planner"]["refused"] == "no eggs here"


def test_request_replan_calls_the_planner_and_the_executor_gets_the_new_plan(tmp_path):
    new = [{"id": "s3", "goal": "Put the egg in the pot", "done_when": "the egg is inside the pot"}]
    calls = [("request_replan", {"reason": "the pan is gone"}), ("finish", {"summary": "x", "abandon": True})]
    result, executor, run_dir = _run(tmp_path, [("submit_plan", {"goal": "g", "steps": EGG}),
                                                 ("submit_plan", {"goal": "egg in pot", "steps": new})], Executor(calls))  # fmt: skip
    state = executor.requests[1][-1]["content"]
    state = state if isinstance(state, str) else "".join(x.get("text", "") for x in state)
    assert "Put the egg in the pot" in state and "GOAL: egg in pot" in state
    assert [c["kind"] for c in json.loads((run_dir / "planner.json").read_text())["calls"]] == ["plan", "replan"]


def test_a_replan_the_planner_refuses_ends_the_run(tmp_path):
    calls = [("request_replan", {"reason": "no egg anywhere"}), ("finish", {"summary": "x", "abandon": True})]
    result, _, _ = _run(tmp_path, [("submit_plan", {"goal": "g", "steps": EGG}), ("cannot_do", {"reason": "no egg"})],
                        Executor(calls))  # fmt: skip
    assert result["outcome"] == "aborted"
