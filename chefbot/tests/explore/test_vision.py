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
    det = VisionDetector(scene, fixtures, {"fridge", "fridge handle"}, vlm, {"synonyms": {"coffee mug": "mug"}, "height_range": (0, 5)})
    only_parts = det.detect(["fridge handle"])
    assert [d.label for d in only_parts] == ["fridge handle"] and vlm.calls == 0
    found = det.detect(["fridge handle", "mug", "egg"])
    assert sorted(d.label for d in found) == ["fridge handle", "mug"]  # the VLM's "fridge" is not an object
    det.detect(["egg"])
    assert vlm.calls == 1  # same image: no second VLM call


class ScriptedVLM:
    def __init__(self, replies):
        self.replies = list(replies)

    def objects(self, rgb):
        return self.replies.pop(0)


def views(*shifts):
    """The same scene seen several times; each view differs a little so it is not a cached repeat."""
    out = []
    for i, _ in enumerate(shifts):
        rgbd = scene()
        rgbd.rgb[0, 0, 0] = 0
        rgbd.rgb[::8, ::8][i, 0, 0] = 1 + i
        out.append(rgbd)
    return iter(out)


def test_boxes_cut_off_by_the_image_edge_are_dropped():
    vlm = ScriptedVLM([[{"label": "mug", "box": [0, 200, 30, 260]}, {"label": "bowl", "box": [300, 200, 340, 260]}]])
    det = VisionDetector(scene, Fixtures(), set(), vlm, {"height_range": (0, 5), "same_view_diff": 0})
    assert [d.label for d in det.detect(["mug"])] == ["bowl"]


def test_sightings_of_one_object_are_averaged_and_off_fixture_points_dropped():
    from chefbot.agent.world.map import MapEntry

    shots = views(0, 1)
    counter = MapEntry("counter_1", (0.99, 1.95), kind="fixture", facing=(0.0, -1.0), size=(1.0, 0.6))
    vlm = ScriptedVLM([[{"label": "mug", "box": [300, 200, 340, 260]}],
                       [{"label": "mug", "box": [300, 200, 340, 240]}, {"label": "jug", "box": [600, 20, 620, 40]}]])
    det = VisionDetector(lambda: next(shots), Fixtures(), set(), vlm, {"height_range": (0, 5), "same_view_diff": 0}, footprints=[counter])
    first = det.detect(["mug"])[0].position_world
    second = det.detect(["mug"])
    assert [d.label for d in second] == ["mug"]  # the jug back-projects outside every fixture
    one = box_point(scene(), (300, 200, 340, 240))
    assert np.allclose(second[0].position_world, (first + one) / 2)


def test_a_far_misnaming_is_superseded_by_a_closer_look():
    near = scene()
    near.depth[200:260, 300:340] = 0.9  # the same thing, seen from 0.6 m closer
    near.cam_to_world = near.cam_to_world.copy()
    near.cam_to_world[2, 3] += 0.6
    near.rgb[0, 0, 0] = 9  # a different image, not a cached repeat
    shots = iter([scene(), near])
    vlm = ScriptedVLM([[{"label": "cutting board", "box": [300, 200, 340, 260]}],
                       [{"label": "knife block", "box": [300, 200, 340, 260]}]])
    det = VisionDetector(lambda: next(shots), Fixtures(), set(), vlm, {"height_range": (0, 5), "same_view_diff": 0})
    det.detect(["mug"])
    det.detect(["mug"])
    assert [label for label, _ in det.superseded()] == ["cutting board"]


def test_the_name_seen_most_often_on_a_spot_wins_a_closer_view():
    far = scene()
    near = scene()
    near.depth[200:260, 300:340] = 0.9  # 0.6 m closer
    near.cam_to_world = near.cam_to_world.copy()
    near.cam_to_world[2, 3] += 0.6
    near.rgb[0, 0, 0] = 9
    views = [far, near]
    for i in range(2):  # two more far views, each a different image
        v = scene()
        v.rgb[0, 0, 0] = 20 + i
        views.append(v)
    shots = iter(views)
    vlm = ScriptedVLM([[{"label": "spoon", "box": [300, 200, 340, 260], "score": 0.3}],
                       [{"label": "fork", "box": [300, 200, 340, 260], "score": 0.4}],
                       [{"label": "spoon", "box": [300, 200, 340, 260], "score": 0.3}],
                       [{"label": "spoon", "box": [300, 200, 340, 260], "score": 0.3}]])  # fmt: skip
    det = VisionDetector(lambda: next(shots), Fixtures(), set(), vlm, {"height_range": (0, 5), "label_vote": True, "same_view_diff": 0})
    for _ in range(4):
        det.detect(["mug"])
    assert [label for label, _ in det.superseded()] == ["fork"]  # 3 spoon sightings to 1, though fork was closer
    det_old = VisionDetector(lambda: next(iter([])), Fixtures(), set(), ScriptedVLM([]), {"label_vote": False})
    det_old._tracks = det._tracks
    assert [label for label, _ in det_old.superseded()] == ["spoon"]  # the old rule: the closer view wins


class CropVLM:
    """Says "pear" on the full image; on an enlarged crop it sees the lemon."""

    def __init__(self):
        self.sizes = []

    def objects(self, rgb):
        self.sizes.append(rgb.shape[:2])
        if rgb.shape[:2] == (480, 640):
            return [{"label": "pear", "box": [310, 220, 330, 240], "score": 0.25},
                    {"label": "cutting board", "box": [100, 100, 200, 160], "score": 0.6}]  # fmt: skip
        h, w = rgb.shape[:2]
        return [{"label": "lemon", "box": [w * 0.4, h * 0.4, w * 0.6, h * 0.6], "score": 0.5}]


def test_a_small_box_is_looked_at_again_enlarged():
    vlm = CropVLM()
    det = VisionDetector(scene, Fixtures(), set(), vlm, {"height_range": (0, 5), "refine_small": True,
                                                         "refine_small_px": 1000, "refine_side": 384})  # fmt: skip
    found = det.detect(["mug"])
    assert sorted(d.label for d in found) == ["cutting board", "lemon"]  # the 20x20 box, relabeled; the big one kept
    assert det.refined == 1 and max(vlm.sizes[1]) == 384


def test_the_vlm_stops_at_its_budget():
    from types import SimpleNamespace

    from chefbot.explore.vision import VisionLanguageModel

    class Client:
        def chat(self, messages, model=None):
            return SimpleNamespace(text='{"objects": [{"label": "mug", "box": [1, 2, 3, 4]}]}', cost_usd=0.3)

    vlm = VisionLanguageModel("m", client=Client(), max_cost_usd=0.5)
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    assert vlm.objects(image) and vlm.objects(image) and vlm.objects(image) == []
    assert (vlm.calls, vlm.skipped_over_budget, round(vlm.cost_usd, 2)) == (2, 1, 0.6)


def test_a_nearly_identical_view_reuses_the_answer_and_a_small_spot_is_zoomed_once():
    vlm = CropVLM()
    shots = []
    for i in range(3):
        v = scene()
        v.rgb[0, 0, 0] = i  # physics jitter: one pixel
        shots.append(v)
    shots[2].rgb[:, :320] = 200  # a different view
    it = iter(shots)
    det = VisionDetector(lambda: next(it), Fixtures(), set(), vlm, {"height_range": (0, 5), "refine_small": True})
    for _ in range(3):
        det.detect(["mug"])
    assert len([s for s in vlm.sizes if s == (480, 640)]) == 2  # the jittered view reused the first answer
    assert det.refined == 1  # the lemon's spot was looked at again only once


def test_with_approach_small_an_object_seen_only_small_is_reported_unsure():
    vlm = ScriptedVLM([[{"label": "egg", "box": [300, 200, 320, 220]}, {"label": "mug", "box": [300, 200, 340, 260]}]])
    det = VisionDetector(scene, Fixtures(), set(), vlm, {"height_range": (0, 5), "approach_small": True,
                                                         "approach_small_px": 1000, "conflict_radius": 0.0})  # fmt: skip
    found = {d.label: d.confidence for d in det.detect(["mug"])}
    assert found["egg"] < 0.5 and found["mug"] > 0.5  # 20x20 px against 40x60 px


def test_with_use_aliases_the_detector_reports_the_group_name():
    vlm = ScriptedVLM([[{"label": "coffee cup", "box": [300, 200, 340, 260]}]])
    det = VisionDetector(scene, Fixtures(), set(), vlm, {"height_range": (0, 5), "use_aliases": True, "synonyms": {}})
    assert [d.label for d in det.detect(["mug"])] == ["mug"]


def test_with_aliases_two_things_of_one_group_side_by_side_stay_two():
    from chefbot.explore.aliases import is_group

    assert is_group("mug") and not is_group("apple")
    def two_things():
        rgbd = scene()
        rgbd.depth[200:260, 360:400] = 1.5  # a second thing, about 0.2 m to the side
        return rgbd

    shots = [two_things(), two_things()]
    shots[1].rgb[0, 0, 0] = 9  # a second, different picture
    it = iter(shots)
    vlm = ScriptedVLM([[{"label": "cup", "box": [300, 200, 340, 260]}],
                       [{"label": "mug", "box": [360, 200, 400, 260]}]])  # fmt: skip
    det = VisionDetector(lambda: next(it), Fixtures(), set(), vlm,
                         {"height_range": (0, 5), "use_aliases": True, "synonyms": {}, "same_view_diff": 0})  # fmt: skip
    det.detect(["mug"])
    second = det.detect(["mug"])
    assert len(det._tracks) == 2 and [d.label for d in second] == ["mug"]  # one group name, two things
    assert np.linalg.norm(det._tracks[0]["sum"] - det._tracks[1]["sum"]) > 0.12
