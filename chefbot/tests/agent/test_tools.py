import math

import numpy as np
import pytest

from chefbot.agent.config import AgentConfig
from chefbot.agent.tools.registry import TOOL_SCHEMAS, ToolRegistry
from chefbot.agent.world.map import WorldMap
from chefbot.interface.types import MotionStatus, Pose2D
from chefbot.tests.agent.fakes import FakeDetector, FakeRobot

CFG = AgentConfig(
    approach_distance=1.0,
    nearby_radius=2.0,
    max_align_distance=1.5,
    map_match_radius=0.5,
    refine_radius=0.15,
    default_vocab=("burner", "pan"),
)

# Stove and counters along the y=0 wall facing +y; fridge on the x=0 wall facing +x.
MAP = {
    "frame": "world",
    "fixtures": {
        "stove": {"location": [3.0, 0.3], "facing": [0.0, 1.0]},
        "counter_1": {"location": [2.0, 0.3], "facing": [0.0, 1.0]},
        "counter_2": {"location": [4.0, 0.3], "facing": [0.0, 1.0]},
        "island": {"location": [3.0, 2.6]},
        "fridge": {"location": [0.3, 5.0], "facing": [1.0, 0.0]},
    },
    "objects": {"egg_carton": {"location": [0.3, 5.0], "in": "fridge"}},
}
OBJECTS = [
    ("burner", (2.8, 0.2, 0.9)),
    ("burner", (3.2, 0.2, 0.9)),
    ("burner", (3.0, 0.45, 0.9)),
    ("pan", (2.8, 0.2, 0.95)),
    ("stove", (3.0, 0.3, 0.9)),
    ("counter", (2.05, 0.3, 0.9)),
    ("counter", (3.95, 0.3, 0.9)),
    ("island", (3.0, 2.6, 0.9)),
    ("fridge", (0.3, 5.0, 1.0)),
    ("fridge handle", (0.65, 4.8, 1.1)),
]


def make(pose=Pose2D(3.0, 1.3, -math.pi / 2), fov_deg=360.0, cfg=CFG, objects=OBJECTS):
    robot = FakeRobot(pose)
    detector = FakeDetector(robot, objects, fov_deg)
    registry = ToolRegistry(robot, detector, WorldMap.from_dict(MAP), cfg)
    return registry, robot, detector


def ids(result, key="detections"):
    return [item["id"] for item in result.data[key]]


# -- go_to ---------------------------------------------------------------------


def test_go_to_stands_in_front_of_fixture_facing_it():
    registry, robot, _ = make(pose=Pose2D(1.0, 4.0, 0.0))
    result = registry.dispatch("go_to", {"place": "stove"})
    assert result.success and result.error is None
    assert result.data["status"] == "success"
    goal = robot.goals[-1]
    assert (goal.x, goal.y) == pytest.approx((3.0, 1.3))
    assert goal.yaw == pytest.approx(-math.pi / 2)  # facing the stove
    assert result.data["pose"] == {"x": 3.0, "y": 1.3, "yaw_deg": -90.0}


def test_go_to_without_facing_approaches_from_the_robots_side():
    registry, robot, _ = make(pose=Pose2D(3.0, 5.0, 0.0))
    registry.dispatch("go_to", {"place": "island"})
    goal = robot.goals[-1]
    assert (goal.x, goal.y) == pytest.approx((3.0, 3.6))
    assert goal.yaw == pytest.approx(-math.pi / 2)


def test_go_to_object_inside_fixture_uses_the_fixtures_approach():
    registry, robot, _ = make()
    registry.dispatch("go_to", {"place": "egg_carton"})
    goal = robot.goals[-1]
    assert (goal.x, goal.y) == pytest.approx((1.3, 5.0))
    assert abs(goal.yaw) == pytest.approx(math.pi)


def test_go_to_unknown_place_lists_known_places():
    registry, robot, _ = make()
    result = registry.dispatch("go_to", {"place": "oven"})
    assert (result.success, result.error) == (False, "unknown_place")
    for name in MAP["fixtures"]:
        assert name in result.message
    assert robot.goals == []


@pytest.mark.parametrize("status", ["no_path", "blocked", "timeout", "stopped"])
def test_go_to_reports_motion_failures(status):
    registry, robot, _ = make()
    robot.outcome = status
    result = registry.dispatch("go_to", {"place": "fridge"})
    assert (result.success, result.error) == (False, status)
    assert result.data["status"] == status


def test_go_to_no_wait_returns_moving_then_stop_halts_it():
    registry, robot, _ = make()
    robot.hold = True
    result = registry.dispatch("go_to", {"place": "fridge", "wait": False})
    assert result.success and result.data["status"] == "moving"
    assert registry.dispatch("get_status").data["status"] == "moving"
    assert registry.dispatch("wait", {"timeout_s": 0.01}).data["status"] == "moving"
    stopped = registry.dispatch("stop")
    assert stopped.success and robot.stops == 1
    status = registry.dispatch("get_status")
    assert (status.success, status.error) == (False, "stopped")


# -- detect / nearby -------------------------------------------------------------


def test_detect_returns_only_requested_classes_sorted_by_distance():
    registry, robot, detector = make()
    result = registry.dispatch("detect", {"classes": ["burner"]})
    assert result.success
    # Nearest first: the burner at y=0.45 is closest to the robot at (3.0, 1.3).
    assert ids(result) == ["burner_1", "burner_2", "burner_3"]
    assert all(item["label"] == "burner" for item in result.data["detections"])
    assert set(result.data["detections"][0]) == {"id", "label"}  # no distances, no positions
    nearest = registry.tools.detections.get("burner_1").position_world
    assert tuple(nearest) == (3.0, 0.45, 0.9)


def test_detect_queries_requested_classes_plus_in_range_map_labels():
    registry, _, detector = make()
    registry.dispatch("detect", {"classes": ["burner"]})
    # stove, both counters and the island are within nearby_radius; the fridge is not.
    assert detector.calls[-1][0] == "burner"
    assert set(detector.calls[-1]) == {"burner", "stove", "counter", "island"}


def test_detect_defaults_to_default_vocab():
    registry, _, detector = make()
    result = registry.dispatch("detect")
    assert detector.calls[-1][:2] == ["burner", "pan"]
    assert ids(result)[0] == "burner_1"
    assert set(ids(result)) == {"burner_1", "burner_2", "burner_3", "pan_1"}


def test_detect_nothing_found_is_success_with_empty_list():
    registry, _, _ = make()
    result = registry.dispatch("detect", {"classes": ["spatula"]})
    assert result.success and result.data["detections"] == []
    assert "spatula" in result.message


def test_matched_map_entries_are_visible_and_unmatched_are_not():
    # Facing the stove with a 100 degree field of view: the island is behind the robot.
    registry, _, _ = make(fov_deg=100.0)
    detected = registry.dispatch("detect", {"classes": ["burner"]})
    assert ids(detected) == ["burner_1", "burner_2", "burner_3"]  # map entries stay out of detect
    view = registry.dispatch("nearby")
    assert view.success
    assert set(ids(view, "visible")) == {"stove", "counter_1", "counter_2", "burner_1", "burner_2", "burner_3"}
    assert ids(view, "not_visible") == ["island"]
    assert {"id": "counter_2", "label": "counter"} in view.data["visible"]
    assert registry.tools.detections.get("stove").map_name == "stove"
    assert registry.tools.detections.get("burner_1").map_name is None


def test_nearby_before_any_detect_shows_map_entries_as_not_visible():
    registry, _, _ = make()
    view = registry.dispatch("nearby")
    assert view.data["visible"] == []
    assert ids(view, "not_visible") == ["stove", "island", "counter_1", "counter_2"]  # nearest first


def test_map_matching_assigns_nearest_entry_within_radius():
    objects = [
        ("counter", (2.3, 0.3, 0.9)),  # 0.3 from counter_1
        ("counter", (4.1, 0.3, 0.9)),  # 0.1 from counter_2
        ("counter", (3.4, 1.0, 0.9)),  # 0.92 from counter_2, 1.57 from counter_1: no match
    ]
    registry, _, _ = make(objects=objects)
    registry.dispatch("detect", {"classes": ["burner"]})
    detections = registry.tools.detections
    assert tuple(detections.get("counter_1").position_world[:2]) == (2.3, 0.3)
    assert tuple(detections.get("counter_2").position_world[:2]) == (4.1, 0.3)
    # The unmatched counter detection is dropped, not given an id of its own.
    assert sorted(e.id for e in detections.all()) == ["counter_1", "counter_2"]


def test_map_matching_is_one_to_one_and_prefers_the_closest_pair():
    # Both detections are within range of counter_1 only; the closer one wins it.
    objects = [("counter", (2.4, 0.3, 0.9)), ("counter", (2.1, 0.3, 0.9))]
    registry, _, _ = make(objects=objects)
    registry.dispatch("detect", {"classes": ["burner"]})
    detections = registry.tools.detections
    assert tuple(detections.get("counter_1").position_world[:2]) == (2.1, 0.3)
    assert detections.get("counter_2") is None
    assert ids(registry.dispatch("nearby"), "not_visible") == ["stove", "island", "counter_2"]


def test_requested_map_label_is_returned_under_its_map_name():
    registry, _, _ = make()
    result = registry.dispatch("detect", {"classes": ["counter"]})
    assert sorted(ids(result)) == ["counter_1", "counter_2"]


def test_detect_overwrites_previous_detections():
    registry, _, _ = make()
    registry.dispatch("detect", {"classes": ["burner"]})
    assert registry.tools.detections.get("burner_1") is not None
    registry.dispatch("detect", {"classes": ["pan"]})
    assert registry.tools.detections.get("burner_1") is None
    assert registry.tools.detections.get("pan_1") is not None
    assert registry.dispatch("align", {"id": "burner_1"}).error == "unknown_id"


def test_nearby_drops_items_as_the_robot_moves_away():
    registry, robot, detector = make()
    registry.dispatch("detect", {"classes": ["burner"]})
    assert "burner_1" in ids(registry.dispatch("nearby"), "visible")
    calls = len(detector.calls)
    robot.pose = Pose2D(3.0, 2.3, -math.pi / 2)  # 1 m back: only the nearest burner is still in range
    view = registry.dispatch("nearby")
    assert ids(view, "visible") == ["island", "burner_1", "stove"]
    assert view.data["not_visible"] == []
    robot.pose = Pose2D(1.0, 5.0, math.pi)  # by the fridge
    view = registry.dispatch("nearby")
    assert view.data["visible"] == []
    assert ids(view, "not_visible") == ["fridge", "egg_carton"]
    assert len(detector.calls) == calls  # nearby never runs detection


# -- align -----------------------------------------------------------------------


def test_align_success_refines_only_its_entry():
    registry, robot, detector = make()
    registry.dispatch("detect", {"classes": ["burner"]})
    detections = registry.tools.detections
    before = {e.id: (e.position_world.copy(), e.timestamp) for e in detections.all()}
    # From up close the detector reports burner_1 a few centimeters off.
    detector.objects[2] = ("burner", (3.03, 0.47, 0.9))
    result = registry.dispatch("align", {"id": "burner_1"})
    assert result.success and result.data["refined"] is True
    assert result.data["status"] == "success" and result.data["id"] == "burner_1"
    assert tuple(robot.align_targets[-1]) == (3.0, 0.45, 0.9)  # aligned to the stored position
    assert detector.calls[-1] == ["burner"]
    assert tuple(detections.get("burner_1").position_world) == (3.03, 0.47, 0.9)
    assert detections.get("burner_1").seen_from == robot.pose
    for id, (position, timestamp) in before.items():
        if id != "burner_1":
            assert np.array_equal(detections.get(id).position_world, position)
            assert detections.get(id).timestamp == timestamp
    assert len(detections.all()) == len(before)


def test_align_succeeds_unrefined_when_redetection_fails():
    registry, _, detector = make()
    registry.dispatch("detect", {"classes": ["burner"]})
    detector.objects[2] = ("burner", (3.0, 0.75, 0.9))  # 0.3 m off: beyond refine_radius
    del detector.objects[:2]
    result = registry.dispatch("align", {"id": "burner_1"})
    assert result.success and result.data["refined"] is False
    assert tuple(registry.tools.detections.get("burner_1").position_world) == (3.0, 0.45, 0.9)


def test_align_works_on_a_detected_map_entry():
    registry, robot, _ = make()
    registry.dispatch("detect", {"classes": ["burner"]})
    result = registry.dispatch("align", {"id": "stove"})
    assert result.success and tuple(robot.align_targets[-1]) == (3.0, 0.3, 0.9)


def test_align_unknown_id_lists_the_nearby_view():
    registry, robot, _ = make(fov_deg=100.0)
    registry.dispatch("detect", {"classes": ["burner"]})
    result = registry.dispatch("align", {"id": "kettle_1"})
    assert (result.success, result.error) == (False, "unknown_id")
    for name in ("burner_1", "burner_2", "burner_3", "stove", "island"):
        assert name in result.message
    assert robot.align_targets == []


def test_align_unknown_id_says_to_detect_when_nothing_is_nearby():
    registry, _, _ = make(pose=Pose2D(20.0, 20.0, 0.0))
    result = registry.dispatch("align", {"id": "burner_1"})
    assert result.error == "unknown_id" and "detect" in result.message


def test_align_not_visible_for_undetected_map_entry():
    registry, robot, _ = make(fov_deg=100.0)
    registry.dispatch("detect", {"classes": ["burner"]})
    result = registry.dispatch("align", {"id": "island"})
    assert (result.success, result.error) == (False, "not_visible")
    assert 'go_to("island") first' in result.message
    assert robot.align_targets == []


def test_align_too_far_backstop():
    # In the nearby view (within 2.0 m) but beyond max_align_distance (1.5 m).
    registry, robot, _ = make(pose=Pose2D(3.0, 2.0, -math.pi / 2))
    registry.dispatch("detect", {"classes": ["burner"]})
    assert "burner_2" in ids(registry.dispatch("nearby"), "visible")
    result = registry.dispatch("align", {"id": "burner_2"})
    assert (result.success, result.error) == (False, "too_far")
    assert "go_to" in result.message and robot.align_targets == []


@pytest.mark.parametrize("status", ["unreachable", "no_path", "blocked", "timeout", "stopped"])
def test_align_reports_motion_failures_without_refining(status):
    registry, robot, detector = make()
    registry.dispatch("detect", {"classes": ["burner"]})
    calls = len(detector.calls)
    robot.outcome = status
    result = registry.dispatch("align", {"id": "burner_1"})
    assert (result.success, result.error) == (False, status)
    assert "refined" not in result.data and len(detector.calls) == calls


def test_align_no_wait_returns_moving_without_refining():
    registry, robot, detector = make()
    registry.dispatch("detect", {"classes": ["burner"]})
    calls = len(detector.calls)
    robot.hold = True
    result = registry.dispatch("align", {"id": "burner_1", "wait": False})
    assert result.success and result.data["status"] == "moving"
    assert "refined" not in result.data and len(detector.calls) == calls
    robot.finish()
    assert registry.dispatch("wait").data["status"] == "success"


# -- wait / get_status / stop / get_pose --------------------------------------------


def test_wait_and_status_are_idle_before_any_motion():
    registry, _, _ = make()
    for tool in ("wait", "get_status"):
        result = registry.dispatch(tool)
        assert result.success and result.data["status"] == "idle"


def test_stop_is_valid_anytime():
    registry, robot, _ = make()
    assert registry.dispatch("stop").success and robot.stops == 1


def test_get_pose_reports_degrees():
    registry, _, _ = make(pose=Pose2D(1.23456, -2.0, math.pi / 2))
    result = registry.dispatch("get_pose")
    assert result.success
    assert result.data == {"pose": {"x": 1.235, "y": -2.0, "yaw_deg": 90.0}}


# -- registry --------------------------------------------------------------------


def test_schemas_cover_every_tool_and_state_the_workflow():
    names = [s["function"]["name"] for s in TOOL_SCHEMAS]
    assert names == [
        "go_to", "detect", "nearby", "align", "wait", "get_status", "stop", "get_pose",
        "pick", "place", "open", "close", "turn_on", "turn_off", "holding",
    ]  # fmt: skip
    for schema in TOOL_SCHEMAS:
        function = schema["function"]
        assert schema["type"] == "function" and function["description"]
        assert function["parameters"]["type"] == "object"
        assert set(function["parameters"]["required"]) <= set(function["parameters"]["properties"])
    described = {s["function"]["name"]: s["function"]["description"] for s in TOOL_SCHEMAS}
    workflow = "go_to a fixture -> detect what you need -> align to it -> pick it -> go_to where it goes"
    assert workflow in described["go_to"] and "-> detect -> align to the spot -> place" in described["go_to"]
    assert "visible" in described["align"] and "nearby()" in described["align"]
    for tool in ("open", "close"):
        assert "scripted" in described["open"] and "open" in described[tool]
    assert "base and the arm" in described["stop"]
    assert "wait=false" in described["stop"]


def test_dispatch_rejects_unknown_tools_and_bad_arguments():
    registry, robot, _ = make()
    assert registry.dispatch("fly_to", {"place": "stove"}).error == "unknown_tool"
    assert registry.dispatch("go_to", {}).error == "invalid_arguments"
    assert registry.dispatch("go_to", {"place": "stove", "speed": 2}).error == "invalid_arguments"
    assert registry.dispatch("go_to", {"place": 3}).error == "invalid_arguments"
    assert registry.dispatch("detect", {"classes": "burner"}).error == "invalid_arguments"
    assert registry.dispatch("wait", {"timeout_s": "soon"}).error == "invalid_arguments"
    assert robot.goals == []


def test_dispatch_treats_null_optional_arguments_as_defaults():
    registry, robot, detector = make()
    assert registry.dispatch("detect", {"classes": None}).success
    assert detector.calls[-1][:2] == ["burner", "pan"]
    assert registry.dispatch("wait", {"timeout_s": None}).success
    assert registry.dispatch("go_to", {"place": "stove", "wait": None}).data["status"] == "success"


def test_config_loads_from_yaml():
    cfg = AgentConfig.load()
    assert cfg.approach_distance > 0 and cfg.max_align_distance <= cfg.nearby_radius
    assert "burner" in cfg.default_vocab
