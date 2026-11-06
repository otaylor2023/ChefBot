"""The executor's record of a step (the papers' execution record: BUMBLE, the Orchestration Gap): failed attempts and
the check's refusals stay with the step, and complete refused twice is blocked until something in the kitchen
changes. SEARCHING lists only what the agent looked for by name."""

from chefbot.agent.check.checker import CheckResult
from chefbot.agent.plan.model import Plan, Step
from chefbot.agent.plan.store import PlanStore
from chefbot.agent.tools.plan_tools import PlanTools
from chefbot.interface.robot_api import ALL_CAPABILITIES
from chefbot.tests.agent.test_tool_availability import names


def tools(passes):
    store = PlanStore(Plan("g", [Step("s1", "egg in pan", "the egg is inside the pan")]))
    store.set_status("s1", "active")
    verdicts = iter(passes)
    return PlanTools(store, lambda step, cond: CheckResult(next(verdicts), "the egg is still in the fridge"), operator=lambda r: "abort"), store


def test_complete_refused_twice_is_blocked_until_the_kitchen_changes():
    pt, store = tools([False, False, True])
    assert pt.complete("s1").error == "check_failed"
    assert pt.complete("s1").error == "check_failed"
    blocked = pt.complete("s1")
    assert blocked.error == "complete_refused_again" and "the egg is still in the fridge" in blocked.message
    pt.note_action("detect", True)  # looking changes nothing
    assert pt.complete("s1").error == "complete_refused_again"
    pt.note_action("place", False)  # a failed place changes nothing either
    assert pt.complete("s1").error == "complete_refused_again"
    pt.note_action("place", True)
    assert pt.complete("s1").success and store.step("s1").status == "done"


def test_waiting_counts_as_a_change_food_cooks():
    pt, _ = tools([False, False, True])
    pt.complete("s1"), pt.complete("s1")
    pt.note_action("wait", True)
    assert pt.complete("s1").success


def test_one_refusal_does_not_block():
    pt, _ = tools([False, True])
    pt.complete("s1")
    assert pt.complete("s1").success


def _state_lines(run_dir):
    import json

    req = sorted((run_dir / "requests").glob("*.json"))[-1]
    msgs = json.loads(req.read_text())["messages"]
    c = msgs[-1]["content"]
    return c if isinstance(c, str) else "".join(x.get("text", "") for x in c)


def test_the_task_state_shows_failed_attempts_and_refusals(tmp_path):
    import json
    from pathlib import Path

    from chefbot.agent.loop.llm import LLMResponse, ToolCall
    from chefbot.scripts import run_agent

    class Calls:
        def __init__(self, calls):
            self.calls, self.i = calls, 0

        def chat(self, messages, tools=None, **kw):
            name, args = self.calls[min(self.i, len(self.calls) - 1)]
            self.i += 1
            return LLMResponse(None, [ToolCall(f"c{self.i}", name, json.dumps(args))])

    class No:
        def check(self, condition, observation):
            return CheckResult(False, "the egg is not in the pan")

    calls = [("set_status", {"step_id": "s1", "status": "active"}), ("pick_up", {"id": "nothing_1"}),
             ("complete", {"step_id": "s1"}), ("complete", {"step_id": "s1"}), ("complete", {"step_id": "s1"}),
             ("finish", {"summary": "x", "abandon": True})]  # fmt: skip
    repo = Path(run_agent.__file__).resolve().parents[2]
    r = run_agent.run(repo / "chefbot/jobs/egg_to_pan.yaml", 0, tmp_path, llm=Calls(calls), judge=No(), operator=lambda x: "abort",
                      quiet=True, env="symbolic", noise="off")  # fmt: skip
    state = _state_lines(Path(r["run_dir"]))
    assert 'failed so far: pick_up("nothing_1") -> unknown_id' in state
    assert "the check refused complete 2x for this step" in state
    timeline = [json.loads(l) for l in (Path(r["run_dir"]) / "run.jsonl").read_text().splitlines()]
    errors = [a["result"]["error"] for a in timeline if a.get("kind") == "action" and a["tool"] == "complete"]
    assert errors == ["check_failed", "check_failed", "complete_refused_again"]


def test_searching_lists_only_what_was_asked_for_by_name():
    _, registry = names(ALL_CAPABILITIES)
    registry.dispatch("detect", {})  # the default vocabulary: pot, bowl, cup, knife, ...
    assert registry.tools.searched == {}
    registry.dispatch("detect", {"classes": ["whisk"]})
    assert "whisk" in registry.tools.searched and "knife" not in registry.tools.searched


def test_finish_with_a_door_the_robot_left_open_is_refused_once():
    store = PlanStore(Plan("g", [Step("s1", "x", "y", status="done")]))
    open_now = ["cabinet_2"]
    pt = PlanTools(store, lambda s, c: CheckResult(True, "ok"), operator=lambda r: "abort", open_doors=lambda: list(open_now))
    first = pt.finish("done")
    assert first.error == "doors_open" and "cabinet_2" in first.message and pt.outcome is None
    assert pt.finish("done").success  # the second time it lets the run end (a door that won't close)
    pt2 = PlanTools(store, lambda s, c: CheckResult(True, "ok"), operator=lambda r: "abort", open_doors=lambda: [])
    assert pt2.finish("done").success
