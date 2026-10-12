import numpy as np

from chefbot.agent.world.detections import DetectionEntry, Detections
from chefbot.agent.world.map import WorldMap, label_of
from chefbot.agent.world.nearby import build_nearby
from chefbot.interface.types import Pose2D

RAW = {
    "frame": "world",
    "fixtures": {
        "stove": {"location": [3.4, 1.0], "facing": [0.0, 1.0]},
        "fridge": {"location": [0.6, 2.1], "facing": [1.0, 0.0]},
        "counter_1": {"location": [2.0, 1.0], "facing": [0.0, 1.0]},
        "counter_2": {"location": [2.0, 3.2]},
    },
    "objects": {"egg_carton": {"location": [0.6, 2.1], "in": "fridge"}},
}


def entry(id, xy, label=None, map_name=None):
    return DetectionEntry(id, label or id, np.array([xy[0], xy[1], 0.9]), 0.9, Pose2D(0, 0, 0), 0.0, map_name)


def test_labels_drop_numeric_suffix():
    assert label_of("counter_2") == "counter"
    assert label_of("stove") == "stove"
    assert label_of("fridge handle_1") == "fridge handle"
    assert label_of("egg_carton") == "egg_carton"


def test_map_get_list_and_fields():
    world_map = WorldMap.from_dict(RAW)
    assert [e.name for e in world_map.list()] == ["stove", "fridge", "counter_1", "counter_2", "egg_carton"]
    assert world_map.get("stove").facing == (0.0, 1.0)
    assert world_map.get("counter_2").facing is None
    assert world_map.get("counter_2").label == "counter"
    egg = world_map.get("egg_carton")
    assert (egg.kind, egg.inside) == ("object", "fridge")
    assert world_map.get("oven") is None
    assert world_map.labels() == {"stove", "fridge", "counter", "egg_carton"}


def test_map_in_range_sorted_by_distance():
    world_map = WorldMap.from_dict(RAW)
    near = world_map.in_range(Pose2D(3.0, 1.0, 0.0), 1.5)
    assert [e.name for e in near] == ["stove", "counter_1"]
    assert world_map.in_range(Pose2D(10.0, 10.0, 0.0), 1.0) == []


def test_map_yaml_round_trip(tmp_path):
    world_map = WorldMap.from_dict(RAW)
    world_map.save(tmp_path / "semantic_map.yaml")
    assert WorldMap.load(tmp_path / "semantic_map.yaml").list() == world_map.list()


def test_detections_replace_and_refine():
    detections = Detections()
    detections.replace([entry("burner_1", (1, 1), "burner"), entry("burner_2", (2, 1), "burner")])
    detections.replace([entry("pan_1", (1, 1), "pan"), entry("pan_2", (2, 2), "pan")])
    assert [e.id for e in detections.all()] == ["pan_1", "pan_2"]
    assert detections.get("burner_1") is None
    detections.refine("pan_1", np.array([1.05, 1.0, 0.9]), 0.99, Pose2D(1, 0, 0), 5.0)
    refined, other = detections.get("pan_1"), detections.get("pan_2")
    assert refined.position_world[0] == 1.05 and refined.confidence == 0.99
    assert refined.seen_from == Pose2D(1, 0, 0) and refined.timestamp == 5.0
    assert other.position_world[0] == 2.0 and other.timestamp == 0.0


def test_nearby_view_splits_visible_and_not_visible():
    world_map = WorldMap.from_dict(RAW)
    detections = Detections()
    detections.replace(
        [
            entry("burner_1", (3.4, 0.9), "burner"),
            entry("stove", (3.4, 1.0), map_name="stove"),
            entry("pan_1", (9.0, 9.0), "pan"),  # out of range
        ]
    )
    view = build_nearby(world_map, detections, Pose2D(3.0, 1.8, 0.0), 1.5)
    assert [e.id for e in view.visible] == ["stove", "burner_1"]  # nearest first
    assert [e.name for e in view.not_visible] == ["counter_1"]
    assert view.find_visible("burner_1") is not None and view.find_visible("counter_1") is None
    assert view.find_not_visible("counter_1") is not None
    assert view.describe() == "Visible: stove, burner_1. Nearby but not visible: counter_1."
    assert build_nearby(world_map, Detections(), Pose2D(20, 20, 0), 1.0).describe() == "Nothing nearby."
