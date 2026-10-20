"""Exploration: the scripted explorer in the symbolic kitchen, coverage, output and scoring."""

from __future__ import annotations

import json

import yaml

from chefbot.agent.world.map import MapEntry, WorldMap
from chefbot.explore.coverage import Coverage
from chefbot.explore.output import explored_map
from chefbot.explore.score import score
from chefbot.interface.types import Pose2D
from chefbot.scripts.explore import run

TRUTH = {
    "frame": "world",
    "fixtures": {"counter_1": {"location": [0.0, 0.0], "facing": [0.0, 1.0]}},
    "objects": {"mug": {"location": [0.5, 0.0], "in": "counter_1"}, "egg": {"location": [-0.5, 0.0], "in": "counter_1"}},
}


def test_scripted_explorer_finds_everything(tmp_path):
    result = run("symbolic", "scripted", out=tmp_path, quiet=True)
    assert result["perfect"], result["objects"]
    run_dir = tmp_path / next(p.name for p in tmp_path.iterdir())
    start = yaml.safe_load((run_dir / "start_map.yaml").read_text())
    assert start["objects"] == {}


def test_agent_never_sees_an_object_before_it_is_in_view(tmp_path):
    """Nothing in what the agent is sent names the egg (in the closed fridge) before it opens the fridge."""
    from chefbot.explore.scripted import ScriptedExplorer

    sent = []

    class Spy(ScriptedExplorer):
        def chat(self, messages, tools=None, **kwargs):
            sent.append(json.dumps(messages))
            return super().chat(messages, tools, **kwargs)

    from chefbot.scripts import explore

    world = explore.symbolic_world(explore.TASK, 0)
    spy = Spy(WorldMap.from_dict(world.start), {"fridge", "cabinet", "microwave"})
    run("symbolic", "scripted", out=tmp_path, llm=spy, quiet=True)
    opened = next(i for i, m in enumerate(sent) if '"open"' in m and "fridge" in m)
    assert not any('"egg' in m or "egg_" in m for m in sent[:opened])
    assert any("egg" in m for m in sent[opened:])


def test_footprint_puts_an_object_in_the_fixture_it_stands_on():
    counter = MapEntry("counter_1", (1.5, -0.3), (0.0, -1.0), size=(3.0, 0.6))
    machine = MapEntry("coffee machine", (0.586, -0.267), (0.0, -1.0), size=(0.3, 0.4))
    assert counter.contains(0.32, -0.488) and not machine.contains(0.32, -0.488)
    assert not counter.contains(1.5, 0.2)


def test_coverage_lists_what_is_left():
    m = WorldMap([MapEntry("fridge", (0.0, 0.0), (0.0, 1.0)), MapEntry("counter_1", (2.0, 0.0), (0.0, 1.0), size=(2.0, 0.6))])
    cov = Coverage(m, frozenset({"fridge"}))
    assert len(cov.remaining({})) == 3
    cov.record_detect(Pose2D(0.0, 1.0, -1.5708), {"fridge": "open"})
    left = cov.remaining({"fridge": "open"})
    assert "fridge: left open (close it)" in left and not any("fridge: not" in x for x in left)
    assert any(x.startswith("counter_1") for x in left)


def test_score_reports_wrong_missing_and_extra():
    explored = {"frame": "world", "fixtures": TRUTH["fixtures"],
                "objects": {"mug": {"location": [0.52, 0.0], "in": "sink"}, "plate": {"location": [0, 0], "in": "counter_1"}}}  # fmt: skip
    r = score(explored, TRUTH)
    status = {o["object"]: o["status"] for o in r["objects"]}
    assert status == {"mug": "wrong", "egg": "missing"} and r["extra_finds"] == ["plate"] and not r["perfect"]
    assert score(TRUTH, TRUTH)["perfect"]
    # Real things the true map leaves out are listed, not failures; a repeat label shifts no names.
    extra = {"frame": "world", "fixtures": TRUTH["fixtures"],
             "objects": {**{f"{n}_1": o for n, o in TRUTH["objects"].items()}, "knife block": {"location": [9, 9], "in": "counter_1"}}}  # fmt: skip
    r = score(extra, TRUTH)
    assert r["perfect"] and r["extra_finds"] == ["knife block"]


def test_explored_map_names_like_the_map_builder():
    start = {"frame": "world", "fixtures": {"counter_1": {"location": [0, 0], "facing": [0, 1], "size": [2, 0.6]}}, "objects": {}}
    m = WorldMap.from_dict(start)
    for name, x in (("mug_2", 0.5), ("mug_1", -0.5), ("bottled water_1", 0.0)):
        m.put(MapEntry(name, (x, 0.0), kind="object", inside="counter_1"))
    out = explored_map(start, m)
    assert list(out["objects"]) == ["bottled water", "mug_1", "mug_2"]
    assert out["objects"]["mug_1"]["location"] == [-0.5, 0.0]
    assert "size" not in out["fixtures"]["counter_1"]


def test_video_needs_the_robocasa_sim(tmp_path):
    import pytest

    with pytest.raises(SystemExit, match="robocasa"):
        run("symbolic", "scripted", out=tmp_path, quiet=True, video_speed=1)
