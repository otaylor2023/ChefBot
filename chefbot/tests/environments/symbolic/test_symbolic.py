"""The symbolic kitchen on its own, and whole tasks through the agent loop with a scripted policy."""

import json
from pathlib import Path

import pytest

from chefbot.agent.config import AgentConfig, LoopConfig
from chefbot.agent.logging.run_log import read_timeline
from chefbot.agent.loop.runner import AgentRunner, AgentSession
from chefbot.agent.plan.model import Plan
from chefbot.agent.world.map import WorldMap
from chefbot.environments.symbolic.scenario import Scenario
from chefbot.environments.symbolic.world import Timing, World

T = Timing.load()  # the tests follow timing.yaml rather than repeating its numbers
EGG = dict(T.cooking["egg"])
CRACKED = dict(T.cooking["cracked egg"])
from chefbot.interface.types import Detection, MotionStatus
from chefbot.tests.environments.symbolic.scripted import Policy, TruthChecker, first, nearest

TASKS = Path(__file__).resolve().parents[3] / "jobs"


def kitchen(*objects):
    return World.load("default", [dict(o) for o in objects])


# -- the world ------------------------------------------------------------------


def test_food_cooks_only_in_cookware_on_a_lit_burner():
    w = kitchen({"name": "egg", "in": "fridge"}, {"name": "pan", "on": "burner_1"})
    w.set_power("burner_1", True)
    w.advance(60)
    assert "egg: cook state = raw" in w.describe()  # in the fridge
    w.move("egg", "pan")
    w.advance(EGG["set"] + 10)
    assert "egg: cook state = set" in w.describe()
    w.set_power("burner_1", False)
    w.advance(500)
    assert "egg: cook state = set" in w.describe()
    w.set_power("burner_1", True)
    w.advance(EGG["overcooked"] - EGG["set"])
    assert "egg: cook state = overcooked" in w.describe() and w.take_events() == ["the egg is overcooked"]


def test_an_empty_lit_burner_turns_itself_off():
    w = kitchen({"name": "pan", "on": "counter_2"})
    w.set_power("burner_3", True)
    w.advance(T.empty_burner_s - 1)
    assert w.take_events() == []
    w.advance(2)
    assert "burner_3: off" in w.describe() and "turned it off" in w.take_events()[0]


def test_placement_rules():
    w = kitchen({"name": "egg", "in": "fridge"}, {"name": "pot", "on": "burner_1"},
                {"name": "pan", "on": "counter_2"}, {"name": "bread", "on": "counter_1"})  # fmt: skip
    assert "already has the pot" in w.can_place("pan", "burner_1")
    assert w.can_place("pan", "burner_2") is None
    assert "does not go on the burner" in w.can_place("egg", "burner_3")
    assert "does not go on the toaster" in w.can_place("egg", "toaster")
    assert w.can_place("bread", "toaster") is None
    assert "closed" in w.can_place("bread", "cabinet_1")
    w.move("egg", "pan")
    w.move("pan", None)  # picked up: the egg comes along
    assert "egg: inside pan" in w.describe() and "pan: held" in w.describe()
    assert "cannot put the pan on itself" in w.can_place("pan", "egg")


def test_closed_doors_hide_what_is_inside(tmp_path):
    task = tmp_path / "t.yaml"
    task.write_text("name: t\ngoal: g\nsteps: [{id: s1, goal: g, done_when: d}]\n"
                    "symbolic: {start: {x: 0.5, y: 1.4, yaw_deg: -90}, objects: [{name: egg, in: fridge}]}\n")  # fmt: skip
    sc = Scenario.load(task)
    labels = lambda: {d.label for d in sc.detector.detect(["egg", "fridge", "fridge handle"])}  # noqa: E731
    assert labels() == {"fridge", "fridge handle"}
    sc.world.fixtures["fridge"].open = True
    assert labels() == {"fridge", "fridge handle", "egg"}


def test_motions_take_time_and_the_arm_needs_to_be_close(tmp_path):
    task = tmp_path / "t.yaml"
    task.write_text("name: t\ngoal: g\nsteps: [{id: s1, goal: g, done_when: d}]\n"
                    "symbolic: {start: {x: 4.1, y: 2.0, yaw_deg: -90}, objects: [{name: mug, on: counter_2}]}\n")  # fmt: skip
    sc = Scenario.load(task)
    robot, mug = sc.robot, sc.world.position("mug")
    robot.pick(Detection("mug", mug))
    assert robot.wait().status == MotionStatus.UNREACHABLE and sc.world.t == pytest.approx(T.robot["pick"])
    robot.align_to_point(mug)
    assert robot.get_status().status == MotionStatus.MOVING
    assert robot.wait(1.0).status == MotionStatus.MOVING
    assert robot.wait().status == MotionStatus.SUCCESS
    robot.pick(Detection("mug", mug))
    assert robot.wait().status == MotionStatus.SUCCESS and sc.world.held == "mug"


# -- whole tasks through the loop -------------------------------------------------

FRIED_EGG_TRUTH = {
    "the pan is on a burner": [r"pan: on burner_"],
    "the egg is inside the pan and the fridge is closed": ["egg: inside pan", "fridge: closed"],
    "the egg is inside the pan, the pan is on a burner that is on, and the egg's cook state is set": [
        "egg: inside pan", "egg: cook state = set"],
    "a plate is on the table": ["plate: on table"],
    "every burner is off": [],
    "the egg is on the plate and the plate is on the table": ["egg: on plate", "plate: on table"],
}  # fmt: skip

FRIED_EGG_SCRIPT = [
    ("set_status", {"step_id": "s1", "status": "active"}),
    ("go_to", {"place": "pan"}), ("detect", {"classes": ["pan"]}), ("align", {"id": "pan"}), ("pick", {"id": "pan"}),
    ("go_to", {"place": "stove"}), ("detect", {"classes": ["burner"]}), first("burner", "align"), first("burner", "place"),
    ("complete", {"step_id": "s1"}),
    ("set_status", {"step_id": "s2", "status": "active"}),
    ("go_to", {"place": "fridge"}), ("detect", {"classes": ["fridge handle"]}), first("fridge handle", "align"),
    first("fridge handle", "open"), ("detect", {"classes": ["egg"]}), ("align", {"id": "egg"}), ("pick", {"id": "egg"}),
    ("close", {"id": "fridge"}),
    ("go_to", {"place": "stove"}), ("detect", {"classes": ["pan", "burner"]}), first("pan", "align"), first("pan", "place"),
    ("complete", {"step_id": "s2"}),
    ("set_status", {"step_id": "s3", "status": "active"}),
    ("detect", {"classes": ["pan", "burner"]}), nearest("burner", "pan", "turn_on"),
    ("set_status", {"step_id": "s3", "status": "waiting", "note": "egg in the pan on a lit burner"}),
    ("set_timer", {"name": "egg", "seconds": EGG["set"] + 10, "step_id": "s3"}),
    ("set_status", {"step_id": "s4", "status": "active"}),
    ("go_to", {"place": "plate"}), ("detect", {"classes": ["plate"]}), ("align", {"id": "plate"}), ("pick", {"id": "plate"}),
    ("go_to", {"place": "table"}), ("detect", {"classes": ["table"]}), ("align", {"id": "table"}), ("place", {"id": "table"}),
    ("complete", {"step_id": "s4"}),
    ("wait", {}),
    ("complete", {"step_id": "s3"}),
    ("set_status", {"step_id": "s5", "status": "active"}),
    ("go_to", {"place": "stove"}), ("detect", {"classes": ["pan", "burner"]}), nearest("burner", "pan", "align"),
    nearest("burner", "pan", "turn_off"), ("complete", {"step_id": "s5"}),
    ("set_status", {"step_id": "s6", "status": "active"}),
    ("detect", {"classes": ["egg"]}), first("egg", "align"), first("egg", "pick"),
    ("go_to", {"place": "table"}), ("detect", {"classes": ["plate"]}), first("plate", "align"), first("plate", "place"),
    ("complete", {"step_id": "s6"}),
    ("finish", {"summary": "egg served"}),
]  # fmt: skip


def run_task(task: str, script: list, truth: dict, tmp_path, seed: int = 0):
    sc = Scenario.load(TASKS / f"{task}.yaml", seed)
    ref: dict = {}
    session = AgentSession.build(
        Plan.from_dict(sc.task), sc.robot, sc.detector, WorldMap.from_dict(sc.map), sc.clock, tmp_path / "run",
        TruthChecker(truth), sc.observer, LoopConfig(max_turns=200), AgentConfig.load(), operator=lambda r: "abort",
        system_prompt="scripted",
    )  # fmt: skip
    session.bus.add_safety(sc.safety)
    session.log.start(task=task, reference_plan=sc.task)
    ref["session"] = session
    runner = AgentRunner(
        session, Policy(ref, script),
        before_turn=lambda s: sc.on_plan({x.id: x.status for x in s.store.live.steps}),
        after_action=lambda s, tool, args, result, text: sc.on_tool(tool),
    )  # fmt: skip
    outcome = runner.run()
    return sc, session, outcome


def test_scripted_fried_egg_succeeds_end_to_end(tmp_path):
    sc, session, outcome = run_task("fried_egg", FRIED_EGG_SCRIPT, FRIED_EGG_TRUTH, tmp_path)
    errors = [(r["turn"], r["tool"], r["result"]["message"]) for r in read_timeline(session.log.dir)
              if r["kind"] == "action" and not r["result"]["success"]]  # fmt: skip
    assert errors == []
    assert outcome == ("finished", "egg served")
    assert sc.success() == {"success": True, "missing": [], "forbidden_present": []}
    timer = next(r for r in read_timeline(session.log.dir) if r["kind"] == "action" and r["tool"] == "wait")
    assert timer["data"]["woke_for"] == "event" and timer["data"]["events"][0]["data"]["timer"] == "egg"


def test_disturbances_fire_on_their_triggers(tmp_path):
    sc = Scenario.load(TASKS / "fried_egg_disturbed.yaml")
    sc.on_tool("open")
    assert sc.world.fixtures["fridge"].open is False and len(sc.applied) == 1
    sc.on_plan({"s3": "waiting"})
    assert sc.world.burners["burner_4"].on
    sc.world.advance(31)
    assert [m for m, _ in sc.safety(sc.world.t)] == [
        "burner_4 was on with nothing on it for 30 s; the robot turned it off"
    ]
    sc.on_plan({"s6": "active"})
    assert "plate: on counter_2" in sc.world.describe()


def test_the_missing_egg_is_still_on_the_map(tmp_path):
    sc = Scenario.load(TASKS / "missing_ingredient.yaml")
    assert "egg" in sc.map["objects"] and "egg" not in sc.world.things
    assert sc.success()["missing"] == ["call request_replan"]
    assert sc.success(["request_replan"])["success"]


@pytest.mark.parametrize("task", sorted(p.stem for p in TASKS.glob("*.yaml")))
def test_every_task_file_loads(task):
    sc = Scenario.load(TASKS / f"{task}.yaml")
    plan = Plan.from_dict(sc.task)
    assert plan.steps and sc.map["fixtures"]
    assert json.dumps(sc.map)


def test_the_kitchen_block_tracks_what_the_agent_did_and_saw(tmp_path):
    """Doors and burners as left or seen, objects where they were put, nested (egg in pan on burner)."""
    script = FRIED_EGG_SCRIPT[:31] + [("finish", {"summary": "x", "abandon": True})]
    sc, session, _ = run_task("fried_egg", script, FRIED_EGG_TRUTH, tmp_path)
    state = session.task_state.render([])
    kitchen = state[state.index("KITCHEN"):state.index("EVENTS SINCE")]
    assert "fridge (closed by you" in kitchen
    assert "burner_1 (turned on by you" in kitchen and "pan (you put it here" in kitchen
    assert "· inside: egg (you put it here" in kitchen
    assert "burner_2, burner_3, burner_4   (off, nothing on them)" in kitchen
    assert "counter_1                     plate, mug" in kitchen
    sc.world.fixtures["fridge"].open = True  # someone opens it: the next look at it says so
    session.registry.dispatch("go_to", {"place": "fridge"})
    session.registry.dispatch("detect", {"classes": ["fridge"]})
    assert "fridge (seen open 0:00 ago)" in session.task_state.render([])
    for _ in range(3):
        session.task_state.record("go_to", {"place": "table"}, ToolResultOk)
    assert "went to table 3x since your last progress" in session.task_state.render([])


class ToolResultOk:
    success, error, data = True, None, {}


# -- handheld tools ------------------------------------------------------------------


def gadgets():
    return kitchen({"name": "egg", "in": "fridge"}, {"name": "egg_2", "kind": "egg", "in": "fridge"},
                   {"name": "egg_cracker", "kind": "egg cracker", "on": "counter_1"}, {"name": "bowl", "on": "counter_1"},
                   {"name": "whisk", "on": "counter_2"}, {"name": "spoon", "on": "counter_2"},
                   {"name": "pan", "on": "burner_1"}, {"name": "plate", "on": "table"})  # fmt: skip


def test_the_egg_cracker_cracks_into_what_it_is_over_and_spills_otherwise():
    w = gadgets()
    w.move("egg", "egg_cracker")
    assert "already has an egg" in w.can_place("egg_2", "egg_cracker")
    w.move("egg_cracker", None)
    assert w.lower_into("bowl") is None and w.squeeze().startswith("cracked 1 egg into the bowl")
    assert "cracked_egg_1: inside bowl" in w.describe() and "egg: " not in w.describe()
    w.release()
    w.move("egg_cracker", None)
    w.lift_out()
    w.pour("trash_can")  # the shell
    w.move("egg_cracker", "counter_1")
    w.move("egg_2", "egg_cracker")
    w.move("egg_cracker", None)
    assert "spilled" in w.squeeze() and "egg_2" not in w.things and w.take_events()


def test_the_whisk_beats_while_squeezed_and_splatters_when_lifted_out_running():
    w = gadgets()
    w.add({"name": "cracked_egg_1", "kind": "cracked egg", "in": "bowl"})
    w.move("whisk", None)
    w.lower_into("bowl")
    w.advance(30)
    assert "not beaten (0:00 whisked)" in w.describe()  # not squeezed: off
    w.squeeze()
    w.advance(T.beaten_after + 1)
    assert "cracked_egg_1: beaten" in w.describe() and "whisk: held, lowered into bowl, running" in w.describe()
    assert "splattered" in w.lift_out() and "bowl: splattered" in w.describe()


def test_stirring_scrambles_eggs_in_a_hot_pan(tmp_path):
    task = tmp_path / "t.yaml"
    task.write_text("name: t\ngoal: g\nsteps: [{id: s1, goal: g, done_when: d}]\n"
                    "symbolic: {start: {x: 2.9, y: 0.95, yaw_deg: -90}, objects: [{name: pan, on: burner_1},"
                    " {name: spoon, on: counter_2}, {name: cracked_egg_1, kind: cracked egg, in: pan}]}\n")  # fmt: skip
    sc = Scenario.load(task)
    w, robot = sc.world, sc.robot
    w.set_power("burner_1", True)
    w.move("spoon", None)
    robot.use_tool("lower_into", Detection("pan", w.position("pan")))
    assert robot.wait().status == MotionStatus.SUCCESS
    robot.use_tool("stir", None, T.scrambled_after + 5)
    assert robot.wait().status == MotionStatus.SUCCESS and "scrambled what is in the pan" in robot.get_status().message
    text = w.describe()
    assert "cracked_egg_1: scrambled" in text and not w.stirring
    w.move("spoon", "counter_2")
    w.add({"name": "cracked_egg_2", "kind": "cracked egg", "in": "pan"})
    w.advance(CRACKED["set"] + 5)
    assert "cracked_egg_2: set in one piece, not scrambled" in w.describe()


def test_a_toaster_pops_up_at_the_end_of_its_cycle():
    w = kitchen({"name": "bread", "in": "toaster"})
    w.set_power("toaster", True)
    w.advance(T.toaster_cycle + 1)
    assert "toaster: off" in w.describe() and "bread: cook state = toasted" in w.describe()
    assert "popped up" in w.take_events()[0]


def test_what_is_inside_the_held_tool_is_seen_in_the_gripper(tmp_path):
    task = tmp_path / "t.yaml"
    task.write_text("name: t\ngoal: g\nsteps: [{id: s1, goal: g, done_when: d}]\n"
                    "symbolic: {start: {x: 1.7, y: 1.0, yaw_deg: -90}, objects: [{name: egg_cracker, kind: egg cracker,"
                    " on: counter_1}, {name: egg, in: egg_cracker}]}\n")  # fmt: skip
    sc = Scenario.load(task)
    assert {d.label for d in sc.detector.detect(["egg", "egg cracker"])} == {"egg", "egg cracker"}
    sc.world.move("egg_cracker", None)
    [egg] = sc.detector.detect(["egg", "egg cracker"])  # the cracker is held (not reported); its egg is visible
    assert egg.label == "egg" and egg.in_gripper


def test_the_agent_keeps_track_of_the_egg_in_the_held_cracker(tmp_path):
    script = [
        ("go_to", {"place": "egg_cracker"}), ("detect", {"classes": ["egg cracker"]}), ("align", {"id": "egg_cracker"}),
        ("go_to", {"place": "fridge"}), ("detect", {"classes": ["fridge handle"]}), first("fridge handle", "align"),
        first("fridge handle", "open"), ("detect", {"classes": ["egg"]}), ("align", {"id": "egg"}), ("pick", {"id": "egg"}),
        ("go_to", {"place": "egg_cracker"}), ("detect", {"classes": ["egg cracker"]}), ("align", {"id": "egg_cracker"}),
        ("place", {"id": "egg_cracker"}), ("pick", {"id": "egg_cracker"}), ("detect", {"classes": ["egg", "bowl"]}),
        ("finish", {"summary": "x", "abandon": True}),
    ]  # fmt: skip
    sc, session, _ = run_task("scrambled_eggs", script, {}, tmp_path)
    errors = [(r["turn"], r["tool"], r["result"]["message"]) for r in read_timeline(session.log.dir)
              if r["kind"] == "action" and not r["result"]["success"]]  # fmt: skip
    assert errors == []
    state = session.task_state.render([])
    assert "holding: egg_cracker · inside: egg (you put it here" in state
    assert "egg (in the gripper)" in state  # perception
    kitchen = state[state.index("KITCHEN"):state.index("EVENTS SINCE")]
    assert "counter_1" in kitchen and "  counter_1                     bowl, plate" in kitchen  # no stray egg on the counter


# -- failure injection ------------------------------------------------------------


def test_noise_modes():
    from chefbot.environments.symbolic.world import Rates

    assert Rates.for_run({"grasp": 0.3}, "off") == Rates()
    assert Rates.for_run({"grasp": 0.3}, "task") == Rates(grasp=0.3)
    on = Rates.for_run({"grasp": 0.3}, "on")
    assert on.grasp == 0.3 and on.actuate > 0 and on.crack_early > 0  # the profile, with the task's on top


def test_transient_failures_can_be_retried_and_eggs_can_crack_early(tmp_path):
    task = tmp_path / "t.yaml"
    task.write_text("name: t\ngoal: g\nsteps: [{id: s1, goal: g, done_when: d}]\n"
                    "symbolic: {start: {x: 1.7, y: 0.9, yaw_deg: -90}, failures: {power: 1.0, crack_early: 1.0},"
                    " objects: [{name: egg_cracker, kind: egg cracker, on: counter_1}, {name: egg, on: counter_1}]}\n")  # fmt: skip
    sc = Scenario.load(task)
    robot, w = sc.robot, sc.world
    w.move("egg", None)
    robot.place(Detection("egg cracker", w.position("egg_cracker")))
    result = robot.wait()
    assert result.status == MotionStatus.FAILED and "cracked" in result.message and "egg" not in w.things
    robot.rates.power = 0.0
    sc_clean = Scenario.load(task, noise="off")
    assert sc_clean.robot.rates.crack_early == 0.0


def test_success_alternatives(tmp_path):
    sc = Scenario.load(TASKS / "scrambled_eggs_fragile.yaml")
    assert not sc.success(["request_replan"])["success"]  # both eggs still there: giving up is wrong
    sc.world.remove("egg")
    assert sc.success(["request_replan"])["success"]  # one lost: two-egg scrambled eggs are impossible


def test_cracking_leaves_a_shell_that_must_go_before_the_next_egg():
    w = gadgets()
    w.move("egg", "egg_cracker")
    w.move("egg_cracker", None)
    w.lower_into("bowl")
    assert "the shell is left in the cracker" in w.squeeze()
    w.release()
    w.lift_out()
    w.move("egg_cracker", "counter_1")
    assert "eggshell in it: empty it first" in w.can_place("egg_2", "egg_cracker")
    w.move("egg_cracker", None)
    assert w.pour("trash_can") is None and "eggshell_1: inside trash_can" in w.describe()
    w.move("egg_cracker", "counter_1")
    assert w.can_place("egg_2", "egg_cracker") is None
    w.add({"name": "eggshell_9", "kind": "eggshell", "in": "bowl"})  # a shell in with the eggs
    assert "cracked_egg_1: has eggshell in it" in w.describe()
