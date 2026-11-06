"""The agent loop with a scripted fake LLM (AGENT_SPEC §8, §9, §11, §14 unit criteria)."""

import json
import math

import pytest

from chefbot.agent.check.checker import CheckResult, parse_verdict
from chefbot.agent.config import LoopConfig
from chefbot.agent.logging.run_log import read_timeline
from chefbot.agent.loop.llm import LLMResponse, ToolCall
from chefbot.agent.harnesses.plan_loop.perception import CLEARED, FRESH, STALE, FreshnessConfig, Perception
from chefbot.agent.harnesses.plan_loop.runner import AgentRunner, AgentSession
from chefbot.agent.plan.model import Plan
from chefbot.interface.clock import ManualClock
from chefbot.interface.observer import Observation, Observer
from chefbot.interface.types import Pose2D, ToolResult
from chefbot.scripts.view_run import load, report_html, summary, timeline_text
from chefbot.tests.agent.test_manip_tools import CFG, MAP, OBJECTS, Scene
from chefbot.agent.world.map import WorldMap
from chefbot.tests.agent.fakes import FakeRobot

PLAN = Plan.from_dict({
    "goal": "Mug on counter_1",
    "steps": [{"id": "s1", "goal": "Put the mug on counter_1", "done_when": "mug: on counter_1"}],
})  # fmt: skip


class ScriptedLLM:
    """Replies from a list; records every request."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def chat(self, messages, tools=None, **kwargs):
        self.requests.append({"messages": json.loads(json.dumps(messages)), "tools": tools})
        if not self.replies:
            return call("finish", summary="out of script", abandon=True)
        reply = self.replies.pop(0)
        return reply() if callable(reply) else reply


def call(tool, text=None, **args) -> LLMResponse:
    return LLMResponse(text, [ToolCall(f"call_{tool}", tool, json.dumps(args))], cost_usd=0.01)


def calls(*pairs) -> LLMResponse:
    return LLMResponse(None, [ToolCall(f"call_{i}", n, json.dumps(a)) for i, (n, a) in enumerate(pairs)])


class FakeObserver(Observer):
    """Ground truth from the fake robot: what it holds and where the mug is."""

    def __init__(self, robot: FakeRobot):
        self.robot = robot
        self.mug = "on counter_2"

    def observe(self) -> Observation:
        held = self.robot.held.label if self.robot.held else "nothing"
        if self.robot.held and self.robot.held.label == "mug":
            self.mug = "held"
        elif self.mug == "held":
            self.mug = "on counter_1"
        return Observation(text=f"held: {held}\nmug: {self.mug}")


class SubstringChecker:
    def check(self, condition, obs):
        return CheckResult(condition in (obs.text or ""), f"state says: {obs.text!r}")


def session(tmp_path, replies, config=None, **kw):
    robot = FakeRobot(Pose2D(3.0, 1.3, -math.pi / 2))
    detector = Scene(robot, list(OBJECTS))
    clock = ManualClock()
    s = AgentSession.build(
        PLAN, robot, detector, WorldMap.from_dict(MAP), clock, tmp_path / "run", SubstringChecker(),
        FakeObserver(robot), config or LoopConfig(window_n=4), CFG, operator=lambda reason: "abort",
        system_prompt="You are a test robot.",
    )  # fmt: skip
    s.log.start(task="mug_to_counter", reference_plan=PLAN.to_dict(), models={"agent": "scripted"})
    llm = ScriptedLLM(replies)
    return s, AgentRunner(s, llm, **kw), llm, robot


MUG_RUN = [
    call("set_status", step_id="s1", status="active"),
    call("go_to", text="the mug is on the counter", place="counter_1"),
    call("detect", classes=["mug", "counter"]),
    call("align", id="mug_1"),
    call("pick_up", id="mug_1"),
    call("complete", text="too early", step_id="s1"),
    call("align", id="counter_1"),
    call("place", id="counter_1"),
    call("complete", step_id="s1"),
    call("finish", summary="mug moved"),
]


def test_a_scripted_run_finishes_and_logs_every_action(tmp_path):
    s, runner, llm, robot = session(tmp_path, MUG_RUN)
    outcome = runner.run()
    assert outcome == ("finished", "mug moved")
    assert s.store.live.step("s1").status == "done"
    timeline = read_timeline(s.log.dir)
    actions = [r for r in timeline if r["kind"] == "action"]
    assert [a["tool"] for a in actions] == [c.tool_calls[0].name for c in MUG_RUN]
    pick = next(a for a in actions if a["tool"] == "pick_up")
    assert pick["world_diff"] == ["- held: nothing", "- mug: on counter_2", "+ held: mug", "+ mug: held"]
    early = actions[5]
    assert early["result"]["error"] == "check_failed" and "mug: held" in early["result"]["message"]
    checks = [r for r in timeline if r["kind"] == "check"]
    assert [c["passed"] for c in checks] == [False, True] and all(c["source"] == "complete" for c in checks)
    assert any(r["kind"] == "plan_edit" and "-> done" in r["change"] for r in timeline)
    assert timeline[-1]["kind"] == "run_end"
    end = json.loads((s.log.dir / "end.json").read_text())
    assert end["outcome"] == "finished" and end["totals"]["checks"] == 2 and end["totals"]["cost_usd"] == pytest.approx(0.1)
    assert end["world"] == ["held: nothing", "mug: on counter_1"]
    # Ground truth never reaches the agent: no request mentions the observer's text.
    assert not any("counter_2" in json.dumps(r["messages"]) for r in llm.requests)


def test_viewer_renders_the_run(tmp_path):
    s, runner, llm, robot = session(tmp_path, MUG_RUN)
    runner.run()
    start, end, timeline = load(s.log.dir)
    text = summary(start, end, timeline)
    assert text.splitlines()[0].startswith("mug_to_counter · finished · 10 turns")
    assert "1 check failure" in text and "mismatches: 0" in text
    assert "first failed action: turn 6 complete" in text
    lines = timeline_text(timeline)
    assert 'pick_up(id="mug_1") ✓' in lines and "world: + held: mug" in lines
    page = report_html(start, end, timeline)
    assert page.startswith("<!doctype html>") and "Plan over time" in page and "check failed" not in page[:200]


def test_viewer_flags_a_pick_that_held_nothing(tmp_path):
    s, runner, llm, robot = session(tmp_path, MUG_RUN[:5] + [call("finish", summary="x", abandon=True)])
    robot.pick = lambda target: robot._start(robot.pose)  # "succeeds" without holding anything
    runner.run()
    start, end, timeline = load(s.log.dir)
    assert "mismatches: 1 (turn 5: pick_up succeeded, nothing became held)" in summary(start, end, timeline)


def test_context_keeps_a_static_prefix_a_window_and_only_the_newest_task_state(tmp_path):
    s, runner, llm, robot = session(tmp_path, MUG_RUN)
    runner.run()
    requests = [r["messages"] for r in llm.requests]
    assert all(m[0] == requests[0][0] for m in requests)  # system prompt byte-identical
    assert all(r["tools"] == llm.requests[0]["tools"] for r in llm.requests)
    for messages in requests:
        states = [m for m in messages if m["role"] == "user" and m["content"].startswith("TASK STATE")]
        assert len(states) == 1 and messages[-1] is not None and messages[-1]["content"].startswith("TASK STATE")
        assert sum(m["role"] == "assistant" for m in messages) <= 4  # window_n
    last = requests[-1]
    assert "s1  done" in last[-1]["content"] and "log:" in last[-1]["content"]
    assert any(m["role"] == "tool" for m in last)


def test_text_only_replies_are_reprompted_then_count_as_errors(tmp_path):
    chatty = LLMResponse("I think I should go to the counter.")
    s, runner, llm, robot = session(tmp_path, [chatty, call("finish", summary="done", abandon=True)])
    runner.run()
    second = llm.requests[1]["messages"]
    assert second[-3]["role"] == "assistant" and second[-2] == {"role": "user", "content": "Respond with exactly one tool call."}
    kinds = [r["kind"] for r in read_timeline(s.log.dir)]
    assert "reprompt" in kinds

    s, runner, llm, robot = session(tmp_path / "b", [chatty] * 20, LoopConfig(max_consecutive_errors=2, max_reprompts=1))
    outcome = runner.run()
    assert outcome[0] == "limit_reached" and "no_tool_call" in outcome[1]


def test_invalid_arguments_are_errors_not_crashes(tmp_path):
    bad_json = LLMResponse(None, [ToolCall("c1", "go_to", "{place: counter_1")])
    replies = [bad_json, call("go_to", where="counter_1"), call("no_such_tool"), call("finish", summary="x", abandon=True)]
    s, runner, llm, robot = session(tmp_path, replies)
    runner.run()
    errors = [r["result"]["error"] for r in read_timeline(s.log.dir) if r["kind"] == "action"][:3]
    assert errors == ["invalid_arguments", "invalid_arguments", "unknown_tool"]
    assert robot.goals == []  # nothing was dispatched to the robot


def test_only_the_first_of_parallel_calls_runs(tmp_path):
    replies = [calls(("go_to", {"place": "counter_1"}), ("go_to", {"place": "stove"})), call("finish", summary="x", abandon=True)]
    s, runner, llm, robot = session(tmp_path, replies)
    runner.run()
    assert len(robot.goals) == 1
    tool_messages = [m for m in llm.requests[1]["messages"] if m["role"] == "tool"]
    assert [json.loads(m["content"])["error"] for m in tool_messages] == [None, "one_call_per_turn"]


def test_limits_stop_the_robot_and_end_the_run(tmp_path):
    stopped = []
    loop = [call("get_pose")] * 10
    s, runner, llm, robot = session(tmp_path, loop, LoopConfig(max_turns=3), on_limit=lambda: stopped.append(True))
    assert runner.run() == ("limit_reached", "max_turns (3)")
    assert robot.stops == 1 and stopped == [True]
    kinds = [r["kind"] for r in read_timeline(s.log.dir)]
    assert "limit" in kinds and kinds[-1] == "run_end"

    s, runner, llm, robot = session(tmp_path / "b", [call("go_to", place="nowhere")] * 10, LoopConfig(max_consecutive_errors=3))
    assert runner.run()[1].startswith("max_consecutive_errors (go_to: unknown_place x3)")


def test_finish_needs_every_step_finished_and_request_replan_asks_the_operator(tmp_path):
    s, runner, llm, robot = session(tmp_path, [call("finish", summary="done"), call("request_replan", reason="no egg")])
    assert runner.run() == ("aborted", "replan requested: no egg")
    first = [r for r in read_timeline(s.log.dir) if r["kind"] == "action"][0]
    assert first["result"]["error"] == "steps_remaining"


def test_events_reach_the_next_task_state_and_wake_waits(tmp_path):
    replies = [call("set_timer", name="egg", seconds=60, step_id="s1"), call("wait"), call("get_pose"),
               call("finish", summary="x", abandon=True)]  # fmt: skip
    s, runner, llm, robot = session(tmp_path, replies)
    runner.run()
    state_after_set = llm.requests[1]["messages"][-1]["content"]
    assert "timer 'egg' 1:00 left" in state_after_set
    wait_result = json.loads([m for m in llm.requests[2]["messages"] if m["role"] == "tool"][-1]["content"])
    assert wait_result["data"]["woke_for"] == "event" and s.clock.now() == 60.0
    kinds = [r.get("event") for r in read_timeline(s.log.dir) if r["kind"] == "event"]
    assert kinds == []  # delivered inside wait's result, not logged as a separate turn-start event
    assert "EVENTS SINCE LAST TURN\n  none" in llm.requests[2]["messages"][-1]["content"]


def test_perception_freshness():
    p = Perception(FreshnessConfig(stale_after_calls=2))
    pose = Pose2D(0.0, 0.0, 0.0)
    assert "nothing yet" in p.block(pose)
    p.update("detect", {"classes": ["mug"]}, ToolResult(True, data={"detections": [{"id": "mug", "label": "mug"}]}), pose)
    assert p.freshness(pose)[0] == FRESH
    assert p.freshness(Pose2D(0.1, 0.0, 0.0)) == (STALE, "moved since")
    assert p.freshness(Pose2D(0.0, 0.0, math.radians(10))) == (STALE, "moved since")
    assert p.freshness(Pose2D(0.6, 0.0, 0.0))[0] == CLEARED
    assert "call detect or nearby" in p.block(Pose2D(0.6, 0.0, 0.0))
    for _ in range(3):
        p.update("get_pose", {}, ToolResult(True), pose)
    assert p.freshness(pose) == (STALE, "3 calls ago")
    p.update("pick_up", {"id": "mug"}, ToolResult(True), pose)
    assert "mug (held)" in p.block(pose, held="mug") and "visible:      mug (held)" in p.block(pose, held="mug")


def test_judge_verdicts_are_parsed_leniently():
    assert parse_verdict('```json\n{"passed": true, "reason": "egg: inside pan"}\n```').passed
    v = parse_verdict("Sure! {\"passed\": false, \"reason\": \"held\"}")
    assert not v.passed and v.reason == "held"
    assert not parse_verdict("yes").passed


def test_task_state_remembers_beyond_the_window(tmp_path):
    """Fired timers stay on their step, loops are counted over the whole run,
    and map objects that were not where the map says are listed."""
    replies = [call("set_status", step_id="s1", status="active"), call("set_timer", name="mug", seconds=5, step_id="s1"),
               call("wait")] + [call("detect", classes=["toaster"])] * 3 + [call("finish", summary="x", abandon=True)]  # fmt: skip
    s, runner, llm, robot = session(tmp_path, replies, LoopConfig(window_n=1))
    s.registry.tools._map.get("egg")  # the map says the egg is on counter_1
    s.registry.tools._detector.objects = [o for o in s.registry.tools._detector.objects if o[0] != "egg"]
    s.registry.tools.detect(["egg"])  # looked right there: not found
    runner.run()
    state = llm.requests[-1]["messages"][-1]["content"]
    assert "TIMER 'mug' WENT OFF" in state  # still shown, though the wait that delivered it left the window
    assert "REPEATED WITHOUT PROGRESS" in state and 'detect(classes=["toaster"]) found nothing x3' in state
    assert "not where the map says: egg (map: counter_1; looked 1x)" in state


def test_detections_say_what_they_sit_on():
    from chefbot.agent.tools.nav import _described
    from chefbot.agent.world.detections import DetectionEntry
    import numpy as np

    def e(id, label, xyz):
        return DetectionEntry(id, label, np.array(xyz, dtype=float), 1.0, Pose2D(0, 0, 0), 0.0)

    burner, pan, egg, mug = e("burner_2", "burner", [1, 0, 0.93]), e("pan_1", "pan", [1, 0, 0.96]), \
        e("egg_1", "egg", [1.02, 0, 0.99]), e("mug", "mug", [1.5, 0, 0.95])  # fmt: skip
    entries = [burner, pan, egg, mug]
    assert _described(pan, entries) == {"id": "pan_1", "label": "pan", "on": "burner_2"}
    assert _described(egg, entries)["on"] == "pan_1"  # the highest holder underneath
    assert "on" not in _described(mug, entries) and "on" not in _described(burner, entries)


def test_search_memory_lists_places_looked_and_left_and_drops_missing_map_objects(tmp_path):
    s, runner, llm, robot = session(tmp_path, [call("finish", summary="x", abandon=True)])
    nav = s.registry.tools
    nav._detector.objects = [o for o in nav._detector.objects if o[0] != "egg"]
    nav.detect(["egg"])
    nav.detect(["egg"])  # twice at its map spot: the map is wrong about it
    nav.detect(["toaster"])  # not a map label: a thing to find
    assert nav.missing() == {"egg"}
    assert "egg" not in [e["id"] for e in nav.nearby().data["not_visible"]]
    state = s.task_state.render([])
    assert "SEARCHING" in state and "egg: looked near counter_1 · not yet: stove, fridge" in state
    assert "toaster: looked near" in state
    kitchen = state.split("KITCHEN")[1].split("EVENTS SINCE")[0]
    assert "missing: egg (map said counter_1; not found there)" in kitchen
    assert "counter_1" in kitchen and "  counter_1                   egg" not in kitchen
    nav._detector.objects.append(("egg", (3.0, 0.3, 0.95)))
    nav.detect(["egg"])
    assert nav.missing() == set() and "egg" not in nav.searched


def test_a_waiting_step_with_nothing_timing_it_is_flagged(tmp_path):
    s, runner, llm, robot = session(tmp_path, [call("finish", summary="x", abandon=True)])
    s.store.set_status("s1", "active")
    result = s.registry.dispatch("set_status", {"step_id": "s1", "status": "waiting"})
    assert result.success and "set_timer or watch it" in result.message
    assert "NO TIMER OR MONITOR" in s.task_state.render([])
    s.registry.dispatch("set_timer", {"name": "mug", "seconds": 30, "step_id": "s1"})
    assert "NO TIMER OR MONITOR" not in s.task_state.render([])


def test_the_active_step_records_what_was_done_for_it(tmp_path):
    s, runner, llm, robot = session(tmp_path, MUG_RUN[:8] + [call("finish", summary="x", abandon=True)])
    runner.run()
    state = llm.requests[-1]["messages"][-1]["content"]
    assert 'done so far: pick_up("mug_1") -> success; place("counter_1") -> success' in state
    assert s.store.live.step("s1").log is None  # "did:" notes are a record, not the step's log line
