"""Object memory: which one of like things, what each was used for, queries by category."""

import math

import numpy as np

from chefbot.agent.config import AgentConfig
from chefbot.agent.tools.registry import ToolRegistry
from chefbot.agent.world.map import WorldMap
from chefbot.agent.world.objects import ObjectMemory, category_of
from chefbot.interface.robot_api import ALL_CAPABILITIES
from chefbot.interface.types import Detection, Pose2D
from chefbot.tests.agent.fakes import FakeRobot

CFG = AgentConfig(approach_distance=1.0, nearby_radius=2.0, max_align_distance=1.5, map_match_radius=0.5,
                  refine_radius=0.15, default_vocab=("fork", "bowl"))  # fmt: skip
MAP = {
    "frame": "world",
    "fixtures": {"counter_1": {"location": [3.0, 0.3], "facing": [0.0, 1.0], "size": [2.0, 0.6]}},
    "objects": {
        "fork_1": {"location": [2.9, 0.3], "in": "counter_1"},
        "fork_2": {"location": [3.1, 0.3], "in": "counter_1"},
        "bowl": {"location": [3.4, 0.3], "in": "counter_1"},
    },
}
RED, BLUE = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])


class Robot(FakeRobot):
    def capabilities(self):
        return ALL_CAPABILITIES

    def use_tool(self, action, target=None, seconds=None):
        self._start(self.pose)


class Scene:
    """Sees `things`: (label, position, signature), all of them, from anywhere."""

    def __init__(self, robot, things):
        self.robot, self.things = robot, things

    def detect(self, classes):
        held = self.robot.held.label if self.robot.held else None
        return [Detection(label, np.array(p, dtype=float), 0.9, signature=sig)
                for label, p, sig in self.things if label in classes and label != held]  # fmt: skip


def make(things):
    robot = Robot(Pose2D(3.0, 1.3, -math.pi / 2))
    scene = Scene(robot, things)
    return ToolRegistry(robot, scene, WorldMap.from_dict(MAP), CFG), robot, scene


def test_categories():
    assert category_of("fork") == "cutlery"
    assert category_of("pan") == "cookware"
    assert category_of("milk") == "liquid container"
    assert category_of("toy robot") == "other"


def test_like_things_keep_their_names_by_how_they_look_when_swapped():
    reg, robot, scene = make([("fork", (2.9, 0.3, 0.9), RED), ("fork", (3.1, 0.3, 0.9), BLUE),
                              ("bowl", (3.4, 0.3, 0.9), None)])  # fmt: skip
    reg.dispatch("detect", {"classes": ["fork"]})
    mem = reg.tools.objects
    assert mem.get("fork_1").signature @ RED > 0.99 and mem.get("fork_2").signature @ BLUE > 0.99
    # Someone swaps them: by place alone the names would stay put; by look they follow.
    scene.things[0], scene.things[1] = ("fork", (2.9, 0.3, 0.9), BLUE), ("fork", (3.1, 0.3, 0.9), RED)
    reg.dispatch("detect", {"classes": ["fork"]})
    assert reg.tools.detections.get("fork_1").position_world[0] == 3.1
    assert reg.tools.detections.get("fork_2").position_world[0] == 2.9


def test_without_signatures_place_decides():
    reg, _, _ = make([("fork", (2.9, 0.3, 0.9), None), ("fork", (3.1, 0.3, 0.9), None)])
    reg.dispatch("detect", {"classes": ["fork"]})
    assert reg.tools.detections.get("fork_1").position_world[0] == 2.9


def test_uses_states_and_queries():
    reg, robot, scene = make([("fork", (2.9, 0.3, 0.9), RED), ("fork", (3.1, 0.3, 0.9), BLUE),
                              ("bowl", (3.4, 0.3, 0.9), None), ("counter", (3.0, 0.3, 0.88), None)])  # fmt: skip
    mem = reg.tools.objects
    mem.context = lambda: {"step": "s2", "recipe": "scrambled eggs"}
    reg.dispatch("detect", {"classes": ["fork", "bowl"]})
    assert reg.dispatch("align", {"id": "fork_1"}).success
    assert reg.dispatch("pick", {"id": "fork_1"}).success
    scene.things = [t for t in scene.things if t[2] is not RED]
    reg.dispatch("detect", {"classes": ["bowl"]})
    assert reg.dispatch("lower_into", {"id": "bowl"}).success
    assert reg.dispatch("stir", {"seconds": 20}).success

    fork = mem.get("fork_1")
    assert [u.action for u in fork.uses] == ["picked", "lowered_into", "stirred"]
    assert fork.uses[-1].other == "bowl" and fork.uses[-1].step == "s2" and fork.uses[-1].recipe == "scrambled eggs"
    assert fork.state("clean") == "dirty" and mem.get("fork_2").state("clean") == "clean"
    assert [u.action for u in mem.get("bowl").uses] == ["had_lowered_in", "stirred_in"]

    stirred = reg.dispatch("find_objects", {"category": "cutlery", "used": "stirred"})
    assert [o["id"] for o in stirred.data["objects"]] == ["fork_1"]
    assert stirred.data["objects"][0]["uses"] == ["lowered into bowl (s2)", "stirred bowl (s2): 20 s"]
    clean = reg.dispatch("find_objects", {"label": "fork", "state": "clean"})
    assert [o["id"] for o in clean.data["objects"]] == ["fork_2"]
    assert mem.usage("cutlery") == {"fork_1": ["lowered into bowl (s2)", "stirred bowl (s2): 20 s"], "fork_2": []}
    overview = reg.dispatch("find_objects", {})
    assert overview.data["categories"] == {"cutlery": ["fork_1", "fork_2"], "tableware": ["bowl"]}
    assert reg.dispatch("find_objects", {"state": "sparkly"}).error == "bad_state"


def test_set_aside_for_a_step_and_free():
    reg, _, _ = make([])
    assert reg.dispatch("note_object", {"id": "fork_2", "for_step": "s5", "ingredient": "eggs"}).success
    assert [o["id"] for o in reg.dispatch("find_objects", {"label": "fork", "free": True}).data["objects"]] == ["fork_1"]
    assert [o["id"] for o in reg.dispatch("find_objects", {"step": "s5"}).data["objects"]] == ["fork_2"]
    assert reg.tools.objects.get("fork_2").tag() == "for s5 (eggs)"
    assert reg.dispatch("note_object", {"id": "bowl", "state": "full"}).success
    assert reg.tools.objects.get("bowl").state("fill") == "full"
    assert reg.dispatch("note_object", {"id": "fork_2", "done": True}).success
    assert not reg.tools.objects.get("fork_2").plans
    assert reg.dispatch("note_object", {"id": "spork"}).error == "unknown_object"


def test_pour_moves_fill_and_contents():
    mem = ObjectMemory()
    mem.put_in("egg", "bowl")
    assert mem.get("bowl").state("fill") == "some" and mem.get("bowl").contents == ["egg"]
    mem.ensure("pan")
    mem.did("poured_into", "bowl", "pan")
    assert mem.get("bowl").state("fill") == "empty" and mem.get("bowl").contents == []
    assert mem.get("pan").state("fill") == "full" and mem.get("pan").contents == ["egg"]
    assert mem.get("pan").uses[-1].action == "poured_from"


def test_distinguishing_attributes_and_round_trip(tmp_path):
    mem = ObjectMemory(lambda: 12.0)
    mem.seen("fork_1", "fork", (1, 2), "counter_1", RED, {"color": "red", "size": "small"})
    mem.seen("fork_2", "fork", (1, 3), "counter_1", BLUE, {"color": "blue", "size": "small"})
    mem.did("picked", "fork_1", "counter_1")
    assert mem.distinguishing("fork_1") == {"color": "red"}
    assert mem.appearance_cost("fork_1", RED, {"color": "red"}) < 1e-9
    assert mem.appearance_cost("fork_1", BLUE, {"color": "blue"}) > 1.4
    mem.save(tmp_path / "m.yaml")
    back = ObjectMemory.load(tmp_path / "m.yaml")
    assert back.to_dict() == mem.to_dict()
    assert back.get("fork_1").last_where == "gripper"
