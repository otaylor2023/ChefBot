"""VisionDetector: names from the VLM, positions from depth, fixtures left to fixture geometry."""

from __future__ import annotations

import numpy as np

from chefbot.explore.vision import VisionDetector, box_point, normalize, parse_objects
from chefbot.interface.detector import Detector
from chefbot.interface.types import RGBD, Detection

K = np.array([[400.0, 0, 320], [0, 400.0, 240], [0, 0, 1]])


def scene() -> RGBD:
    """A wall 2 m away, and a box-shaped object 1.5 m away covering pixels x 300-340, y 200-260."""
    depth = np.full((480, 640), 2.0, dtype=np.float32)
    depth[200:260, 300:340] = 1.5
    pose = np.eye(4)
    pose[:3, 3] = [1.0, 2.0, 1.2]
    return RGBD(np.zeros((480, 640, 3), dtype=np.uint8), depth, K, pose)


def test_box_point_takes_the_object_not_the_wall_behind():
    point = box_point(scene(), (290, 190, 350, 270))  # a loose box around it
    expected = np.array([1.0 + (319.5 - 320) / 400 * 1.5, 2.0 + (229.5 - 240) / 400 * 1.5, 1.2 + 1.5])
    assert np.allclose(point, expected, atol=0.02)


def test_normalize_maps_vlm_words_to_catalogue_labels():
    vocab = {"bottled water", "egg", "mug"}
    assert normalize("Water Bottle", {"water bottle": "bottled water"}, vocab) == "bottled water"
    assert normalize("eggs", {}, vocab) == "egg"
    assert normalize("rubber duck", {}, vocab) == "rubber duck"  # open vocabulary: kept as said


def test_parse_objects_tolerates_a_fence():
    text = 'Here:\n```json\n{"objects": [{"label": "mug", "box": [1, 2, 3, 4]}, {"label": "x"}]}\n```'
    assert parse_objects(text) == [{"label": "mug", "box": [1.0, 2.0, 3.0, 4.0]}]


class FakeVLM:
    def __init__(self):
        self.calls = 0

    def objects(self, rgb):
        self.calls += 1
        return [{"label": "coffee mug", "box": [300, 200, 340, 260]}, {"label": "fridge", "box": [0, 0, 10, 10]}]


class Fixtures(Detector):
    def __init__(self):
        self.asked = []

    def detect(self, classes):
        self.asked.append(list(classes))
        return [Detection("fridge handle", np.zeros(3))] if "fridge handle" in classes else []


def test_vision_detector_splits_fixtures_and_objects_and_caches_a_view():
    vlm, fixtures = FakeVLM(), Fixtures()
    det = VisionDetector(scene, fixtures, {"fridge", "fridge handle"}, vlm, {"synonyms": {"coffee mug": "mug"}})
    only_parts = det.detect(["fridge handle"])
    assert [d.label for d in only_parts] == ["fridge handle"] and vlm.calls == 0
    found = det.detect(["fridge handle", "mug", "egg"])
    assert sorted(d.label for d in found) == ["fridge handle", "mug"]  # the VLM's "fridge" is not an object
    det.detect(["egg"])
    assert vlm.calls == 1  # same image: no second VLM call
