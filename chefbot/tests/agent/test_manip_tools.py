"""Manipulation tools on FakeRobot: error paths and held-object bookkeeping (MANIP_SPEC §11, agent)."""

import math

import pytest

import numpy as np

from chefbot.agent.config import AgentConfig
from chefbot.agent.tools.registry import ToolRegistry
from chefbot.agent.world.map import WorldMap
from chefbot.interface.types import Detection, MotionStatus, Pose2D
from chefbot.tests.agent.fakes import FakeDetector, FakeRobot

CFG = AgentConfig(
    approach_distance=1.0,
    nearby_radius=2.0,
    max_align_distance=1.5,
    map_match_radius=0.5,
    refine_radius=0.15,
    default_vocab=("egg", "mug"),
)
# Counter along the y=0 wall facing +y, fridge next to it; the egg starts on the counter.
MAP = {
    "frame": "world",
    "fixtures": {
        "counter_1": {"location": [3.0, 0.3], "facing": [0.0, 1.0]},
        "stove": {"location": [4.4, 0.3], "facing": [0.0, 1.0]},
        "fridge": {"location": [1.6, 0.3], "facing": [0.0, 1.0]},
    },
    "objects": {"egg": {"location": [3.0, 0.3], "in": "counter_1"}},
}
EGG, MUG = (3.0, 0.3, 0.95), (3.3, 0.3, 0.97)
OBJECTS = [
    ("egg", EGG),
    ("mug", MUG),
    ("counter", (3.0, 0.3, 0.92)),
    ("stove", (4.4, 0.3, 0.92)),
    ("burner", (4.4, 0.3, 0.93)),
    ("fridge", (1.6, 0.5, 1.0)),
    ("fridge handle", (1.9, 0.6, 1.1)),
]


class Scene(FakeDetector):
    """A FakeDetector that, like the real one, does not report what the robot holds."""

    def detect(self, classes):
        held = self.robot.held.label if self.robot.held else None
        return [d for d in super().detect(classes) if d.label != held]


def make(pose=Pose2D(3.0, 1.3, -math.pi / 2)):
    robot = FakeRobot(pose)
    detector = Scene(robot, list(OBJECTS))
    return ToolRegistry(robot, detector, WorldMap.from_dict(MAP), CFG), robot, detector


def ids(result, key="visible"):
    return [item["id"] for item in result.data[key]]


def seen(registry):
    registry.dispatch("detect")
    return registry


# -- pick ----------------------------------------------------------------------


def test_pick_hands_the_robot_the_detection_and_records_what_is_held():
    registry, robot, _ = make()
    seen(registry)
    result = registry.dispatch("pick", {"id": "egg"})
    assert result.success and result.data["id"] == "egg" and result.data["status"] == "success"
    target = robot.picks[-1]
    assert isinstance(target, Detection) and target.label == "egg"
    assert tuple(target.position_world) == EGG
    assert registry.dispatch("holding").data["holding"] == {"id": "egg", "label": "egg"}


def test_picked_object_leaves_the_scene_and_nearby_says_holding():
    registry, robot, _ = make()
    seen(registry)
    assert "egg" in ids(registry.dispatch("nearby"))
    registry.dispatch("pick", {"id": "egg"})
    view = registry.dispatch("nearby")
    assert "egg" not in ids(view) and "egg" not in ids(view, "not_visible")
    assert view.message.endswith("Holding: egg.")
    assert view.data["holding"] == {"id": "egg", "label": "egg"}
    assert registry.dispatch("align", {"id": "egg"}).error == "unknown_id"
    # The map no longer claims the egg is on the counter, and a new look does not find it either.
    registry.dispatch("detect", {"classes": ["egg"]})
    again = registry.dispatch("nearby")
    assert "egg" not in ids(again) + ids(again, "not_visible")


def test_pick_error_paths():
    registry, robot, _ = make()
    assert registry.dispatch("pick", {"id": "egg"}).error == "not_visible"  # on the map, not yet seen
    assert registry.dispatch("pick", {"id": "spoon_1"}).error == "unknown_id"
    seen(registry)
    for status in ("no_grasp", "unreachable", "grasp_failed", "collision", "stopped", "timeout"):
        robot.outcome = status
        result = registry.dispatch("pick", {"id": "egg"})
        assert not result.success and result.error == status
        assert registry.dispatch("holding").data["holding"] is None
        assert "egg" in ids(registry.dispatch("nearby"))  # still there to try again
    robot.outcome = MotionStatus.SUCCESS
    assert registry.dispatch("pick", {"id": "egg"}).success
    held = registry.dispatch("pick", {"id": "mug_1"})
    assert held.error == "already_holding" and "egg" in held.message
    assert len(robot.picks) == 7  # the last one never reached the robot


def test_pick_without_waiting_records_the_object_once_the_motion_ends():
    registry, robot, _ = make()
    seen(registry)
    robot.hold = True
    started = registry.dispatch("pick", {"id": "egg", "wait": False})
    assert started.success and started.data["status"] == "moving"
    assert registry.dispatch("holding").data["holding"] is None
    assert registry.dispatch("get_status").data["status"] == "moving"
    robot.finish()
    assert registry.dispatch("wait").data["status"] == "success"
    assert registry.dispatch("holding").data["holding"] == {"id": "egg", "label": "egg"}


# -- place ---------------------------------------------------------------------


def test_place_puts_it_on_the_named_surface_and_it_can_be_seen_again():
    registry, robot, detector = make()
    seen(registry)
    registry.dispatch("pick", {"id": "mug_1"})
    result = registry.dispatch("place", {"id": "counter_1"})
    assert result.success and result.data == {
        "id": "counter_1", "object": "mug_1", "status": "success", "pose": result.data["pose"],
    }  # fmt: skip
    assert robot.places[-1].label == "counter"
    assert registry.dispatch("holding").message == "Holding nothing."
    assert "Holding" not in registry.dispatch("nearby").message
    found = registry.dispatch("detect", {"classes": ["mug"]})
    assert [d["id"] for d in found.data["detections"]] == ["mug_1"]


def test_an_object_put_down_somewhere_new_keeps_its_name_on_the_map():
    registry, robot, detector = make()
    seen(registry)
    registry.dispatch("pick", {"id": "egg"})
    assert registry.tools._map.get("egg") is None  # in the gripper: not a place to go to
    registry.dispatch("place", {"id": "stove"})
    assert registry.tools._map.get("egg").inside == "stove"  # back on the map, where it was put
    detector.objects[0] = ("egg", (4.4, 0.3, 0.95))  # now on the stove, far from where the map had it
    went = registry.dispatch("go_to", {"place": "egg"})
    assert went.success and "you put the egg on stove" in went.message
    found = registry.dispatch("detect", {"classes": ["egg"]})
    assert [d["id"] for d in found.data["detections"]] == ["egg"]


def test_place_error_paths():
    registry, robot, _ = make()
    seen(registry)
    empty = registry.dispatch("place", {"id": "counter_1"})
    assert empty.error == "not_holding" and robot.places == []
    registry.dispatch("pick", {"id": "egg"})
    assert registry.dispatch("place", {"id": "island"}).error == "unknown_id"
    registry.dispatch("go_to", {"place": "stove"})  # the fridge is now nearby-but-unseen... not here
    for status in ("no_placement", "unreachable", "collision", "stopped", "timeout"):
        robot.outcome = status
        result = registry.dispatch("place", {"id": "counter_1"})
        assert not result.success and result.error == status
        assert registry.dispatch("holding").data["holding"] == {"id": "egg", "label": "egg"}


def test_dropping_the_object_on_the_way_clears_holding():
    registry, robot, _ = make()
    seen(registry)
    registry.dispatch("pick", {"id": "egg"})
    robot.outcome = MotionStatus.GRASP_FAILED
    robot.held = None
    result = registry.dispatch("go_to", {"place": "stove"})
    assert result.error == "grasp_failed"
    assert registry.dispatch("holding").data["holding"] is None


# -- open / close --------------------------------------------------------------


def test_open_and_close_by_fixture_or_handle_id_are_scripted():
    registry, robot, _ = make(pose=Pose2D(1.8, 1.4, -math.pi / 2))
    registry.dispatch("detect", {"classes": ["fridge handle"]})
    opened = registry.dispatch("open", {"id": "fridge handle_1"})
    assert opened.success and opened.data["scripted"] is True and opened.data["id"] == "fridge handle_1"
    closed = registry.dispatch("close", {"id": "fridge"})
    assert closed.success and closed.data["scripted"] is True
    (handle, first), (fridge, second) = robot.actuations
    assert (handle.label, first, fridge.label, second) == ("fridge handle", True, "fridge", False)


def test_open_error_paths():
    registry, robot, _ = make(pose=Pose2D(2.4, 1.4, -math.pi / 2))
    assert registry.dispatch("open", {"id": "fridge"}).error == "not_visible"
    assert registry.dispatch("close", {"id": "pantry"}).error == "unknown_id"
    seen(registry)
    wrong_kind = registry.dispatch("open", {"id": "egg"})
    assert wrong_kind.error == "unsupported" and "not an openable fixture" in wrong_kind.message
    robot.outcome = MotionStatus.UNREACHABLE
    assert registry.dispatch("open", {"id": "fridge"}).error == "unreachable"


def test_any_visible_id_can_be_passed_to_any_tool():
    # No typing in v0.1: a wrong-kind call reaches the robot and fails through the normal path.
    registry, robot, _ = make()
    seen(registry)
    robot.outcome = MotionStatus.NO_GRASP
    assert registry.dispatch("pick", {"id": "counter_1"}).error == "no_grasp"
    assert robot.picks[-1].label == "counter"


# -- registry ------------------------------------------------------------------


def test_manip_tools_validate_arguments():
    registry, _, _ = make()
    assert registry.dispatch("pick", {}).error == "invalid_arguments"
    assert registry.dispatch("place", {"id": 3}).error == "invalid_arguments"
    assert registry.dispatch("open", {"id": "fridge", "force": True}).error == "invalid_arguments"
    assert registry.dispatch("holding", {"id": "egg"}).error == "invalid_arguments"


def test_stop_reports_base_and_arm_and_keeps_the_object():
    registry, robot, _ = make()
    seen(registry)
    registry.dispatch("pick", {"id": "egg"})
    stopped = registry.dispatch("stop")
    assert stopped.success and "arm" in stopped.message and robot.stops == 1
    assert registry.dispatch("holding").data["holding"] == {"id": "egg", "label": "egg"}


def test_big_surfaces_are_worked_on_at_their_near_point():
    registry, robot, detector = make()
    near = np.array([3.0, 0.5, 0.92])
    original = detector.detect
    detector.detect = lambda classes: [
        Detection(d.label, d.position_world, d.confidence, near if d.label == "counter" else None)
        for d in original(classes)
    ]
    seen(registry)
    registry.dispatch("align", {"id": "counter_1"})
    assert tuple(robot.align_targets[-1]) == tuple(near)
    registry.dispatch("pick", {"id": "egg"})
    registry.dispatch("place", {"id": "counter_1"})
    assert tuple(robot.places[-1].position_world) == tuple(near)


def test_an_object_seen_again_from_afar_keeps_its_one_map_entry():
    registry, robot, detector = make(Pose2D(0.0, 1.3, 0.0))  # 4.5 m from the stove: beyond the map query
    detector.objects.append(("pan", (4.4, 0.35, 0.95)))
    registry.dispatch("detect", {"classes": ["pan"]})
    registry.dispatch("detect", {"classes": ["pan"]})
    pans = [e.name for e in registry.tools._map.list() if e.kind == "object" and e.label == "pan"]
    assert pans == ["pan_1"]


def test_go_to_part_or_along_stands_at_that_part_of_a_long_fixture():
    from chefbot.agent.world.map import MapEntry

    registry, robot, _ = make()
    registry.tools._map.put(MapEntry("island", (3.0, -2.0), facing=(0.0, 1.0), size=(4.0, 1.0)))
    middle = registry.dispatch("go_to", {"place": "island"})
    assert middle.success and robot.goals[-1].x == pytest.approx(3.0)
    registry.dispatch("go_to", {"place": "island", "part": "left"})  # facing it, i.e. looking down -y: left is +x
    assert robot.goals[-1].x == pytest.approx(3.0 + 1.6) and robot.goals[-1].y == pytest.approx(-1.0)
    registry.dispatch("go_to", {"place": "island", "along": 0.5})
    assert robot.goals[-1].x == pytest.approx(3.0 - 0.8)
    assert not registry.dispatch("go_to", {"place": "island", "part": "top"}).success
