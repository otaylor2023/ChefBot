"""After exploring: moving objects around, and the map keeping up (symbolic kitchen, no LLM)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from chefbot.agent.world.map import WorldMap
from chefbot.explore.rearrange import Move, ScriptedRearranger, initial_map
from chefbot.scripts import explore, rearrange
from chefbot.scripts.envs import symbolic


def test_the_map_follows_every_move(tmp_path):
    result = rearrange.run("symbolic", moves=6, out=tmp_path, quiet=True)
    assert result["picks"] == {"tried": 6, "ok": 6, "first_try": 6}
    assert result["places"]["ok"] == 6
    assert result["map_after"]["perfect"], result["map_after"]["objects"]
    assert result["moved_tracked"] == result["moved_objects"] > 0
    # A map that never followed the moves would be wrong about what moved.
    assert not result["map_stale"]["perfect"]


def test_the_map_follows_the_moves_through_failures(tmp_path):
    task = yaml.safe_load(explore.TASK.read_text())
    task["symbolic"]["failures"] = {"grasp": 0.3, "nav": 0.1, "detect_miss": 0.1}
    path = tmp_path / "explore_failures.yaml"
    path.write_text(yaml.safe_dump(task))
    explored = explore.run("symbolic", "scripted", task=path, out=tmp_path / "explore", quiet=True)
    # The explorer ran without failures (its own run); the moves get them.
    result = rearrange.run("symbolic", moves=8, seed=3, task=path, noise="task", out=tmp_path, quiet=True,
                           explored=Path(explored["run_dir"]) / "explored_map.yaml")  # fmt: skip
    assert result["picks"]["first_try"] < result["picks"]["tried"]  # some grasps failed...
    assert result["map_after"]["perfect"], result["map_after"]["objects"]  # ...and the map still has it right


def test_the_truth_is_made_only_after_the_run(tmp_path, monkeypatch):
    made = []

    def world(task, seed, noise="off"):
        w = original(task, seed, noise)
        truth = w.truth
        w.truth = lambda: made.append(sorted(p.name for p in tmp_path.rglob("final_map.yaml"))) or truth()
        return w

    original = explore.symbolic_world
    monkeypatch.setattr(explore, "symbolic_world", world)
    rearrange.run("symbolic", moves=2, out=tmp_path, quiet=True)
    # Called twice (the explore run's scorer, then the rearrange run's), each time after its run ended.
    assert made[-1] == ["final_map.yaml"]


def test_the_agent_starts_from_the_map_it_is_given(tmp_path):
    """run_agent --map: the explored map, not the true one."""
    given = {"frame": "world", "fixtures": {"counter_1": {"location": [1.7, 0.3], "facing": [0, 1]}},
             "objects": {"mug": {"location": [1.9, 0.3], "in": "counter_1"}}}  # fmt: skip
    path = tmp_path / "explored_map.yaml"
    path.write_text(yaml.safe_dump(given))
    run = symbolic(explore.TASK, 0, "off", path)
    assert [e.name for e in run.world_map.list()] == ["counter_1", "mug"]
    assert run.world_map.get("mug").location == (1.9, 0.3)
    assert run.world_map.get("egg") is None  # the true map has an egg in the fridge
    assert symbolic(explore.TASK, 0, "off").world_map.get("egg") is not None


def test_initial_map_keeps_the_footprints_the_explored_map_drops():
    start = {"frame": "world", "fixtures": {"counter_1": {"location": [0, 0], "facing": [0, 1], "size": [2, 0.6]}},
             "objects": {}}  # fmt: skip
    explored = {"frame": "world", "fixtures": {"counter_1": {"location": [0, 0], "facing": [0, 1]}},
                "objects": {"mug": {"location": [0.5, 0], "in": "counter_1"}}}  # fmt: skip
    out = initial_map(start, explored)
    assert out["fixtures"]["counter_1"]["size"] == [2, 0.6] and out["objects"] == explored["objects"]
    assert start["objects"] == {}


def test_things_go_only_on_surfaces_the_map_says_are_wide_enough():
    fixtures = {"counter_1": {"location": [0, 0], "facing": [0, 1], "size": [2, 0.6]},
                "counter_2": {"location": [1.1, 0], "facing": [0, 1], "size": [0.1, 0.6]},  # a filler strip
                "counter_3": {"location": [1.5, 0]},  # a corner piece: no footprint on the map
                "table": {"location": [3, 3], "facing": [1, 0], "size": [0.8, 0.8]},
                "sink": {"location": [2, 0], "facing": [0, 1], "size": [0.8, 0.6]}}  # fmt: skip
    start = WorldMap.from_dict({"frame": "world", "fixtures": fixtures,
                                "objects": {"mug": {"location": [0.5, 0], "in": "counter_1"}}})  # fmt: skip
    assert [s.name for s in ScriptedRearranger(start, 3)._surfaces] == ["counter_1", "table"]


def test_summary_scores_each_object_after_its_last_move():
    truth = {"frame": "world", "fixtures": {}, "objects": {"mug": {"location": [2.0, 0.0], "in": "counter_2"},
                                                           "egg": {"location": [5.0, 0.0], "in": "fridge"}}}  # fmt: skip
    final = {"frame": "world", "fixtures": {}, "objects": {"mug": {"location": [2.05, 0.0], "in": "counter_2"},
                                                           "egg": {"location": [5.0, 0.0], "in": "fridge"}}}  # fmt: skip
    stale = {"frame": "world", "fixtures": {}, "objects": {"mug": {"location": [0.0, 0.0], "in": "counter_1"},
                                                           "egg": {"location": [5.0, 0.0], "in": "fridge"}}}  # fmt: skip
    moves = [Move("mug", "counter_1", "table", True, 1, True, 1),
             Move("mug", "table", "counter_2", True, 2, True, 1, errors=["pick: grasp_failed"]),
             Move("bowl", "counter_1", "table", False, 2, errors=["pick: no_grasp", "pick: no_grasp"])]  # fmt: skip
    result = rearrange.summarize(moves, final, stale, truth)
    assert result["picks"] == {"tried": 3, "ok": 2, "first_try": 1}
    assert result["places"] == {"tried": 2, "ok": 2, "dropped": 0}
    assert [m["map_status"] for m in result["moves"]] == ["moved again", "ok", "not on the map"]
    assert result["moved_tracked"] == result["moved_objects"] == 1
    assert result["map_after"]["perfect"] and not result["map_stale"]["perfect"]
    assert "picks: 2/3" in rearrange.report(result)


@pytest.mark.robocasa
def test_rearrange_in_robocasa(tmp_path):
    """RoboCasa layout 1, ground-truth detector: explore, then move things around."""
    pytest.importorskip("robocasa")
    result = rearrange.run("robocasa", moves=4, out=tmp_path, quiet=True)
    print("\n" + rearrange.report(result))
    # Some picks and places fail here (the cutting board has no grasp, a small counter has no room):
    # what must hold is that whatever was put down is on the map where it truly is.
    assert result["picks"]["ok"] >= 1 and result["places"]["ok"] >= 1
    assert result["moved_tracked"] == result["moved_objects"]


def test_a_fixed_object_list_makes_the_same_moves_whatever_the_map(tmp_path):
    """--objects: every map gets the same objects to the same targets; one it lacks is a move not made."""
    explored = explore.run("symbolic", "scripted", out=tmp_path / "explore", quiet=True)
    full = Path(explored["run_dir"]) / "explored_map.yaml"
    lacking = yaml.safe_load(full.read_text())
    del lacking["objects"]["mug"]
    lacking_path = tmp_path / "lacking.yaml"
    lacking_path.write_text(yaml.safe_dump(lacking))
    runs = [rearrange.run("symbolic", moves=4, explored=m, out=tmp_path / m.stem, quiet=True,
                          objects=("mug", "pan", "bottled water", "pan")) for m in (full, lacking_path)]  # fmt: skip
    a, b = ([(m["object"], m["target"]) for m in r["moves"]] for r in runs)
    assert [m for m in a if m[0] != "mug"] == [m for m in b if m[0] != "mug"] and len(a) == len(b) == 4
    assert runs[0]["moves_done"] == "4/4" and runs[1]["moves_done"] == "3/4"
    assert runs[1]["moves"][0]["errors"] == ["pick: not on the map"]
