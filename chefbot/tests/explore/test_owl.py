"""OWLv2 box selection (explore/owl.py): no weights needed."""

from __future__ import annotations

import numpy as np

from chefbot.explore.owl import pick_boxes


def test_pick_boxes_keeps_objects_above_threshold_one_box_each_and_drops_fixture_words():
    labels = ["mug", "pan", "burner"]  # the last is a fixture word
    scores = np.array([
        [0.9, 0.1, 0.0],  # a mug
        [0.5, 0.2, 0.0],  # the same mug, a slightly shifted box: suppressed
        [0.0, 0.1, 0.8],  # best match a fixture word: not an object
        [0.1, 0.15, 0.0],  # below threshold
        [0.0, 0.6, 0.0],  # a pan elsewhere
    ])
    boxes = np.array([[0.25, 0.25, 0.1, 0.1], [0.26, 0.25, 0.1, 0.1], [0.5, 0.5, 0.2, 0.2],
                      [0.7, 0.2, 0.1, 0.1], [0.75, 0.6, 0.2, 0.1]])  # fmt: skip
    found = pick_boxes(scores, boxes, labels, n_objects=2, threshold=0.3, nms_iou=0.3, h=480, w=640)
    assert [(f["label"], f["score"]) for f in found] == [("mug", 0.9), ("pan", 0.6)]
    assert found[0]["box"] == [128.0, 128.0, 192.0, 192.0]  # fractions of the 640 x 640 padded square
    assert found[1]["box"][3] == 416.0


def test_hybrid_names_small_boxes_with_siglip_and_drops_kitchen_parts():
    from chefbot.explore.owl import OwlV2Model

    class FakeSiglip:
        labels = ["lemon", "pear", "knob"]
        n_objects = 2

        def name(self, rgb, box):
            return ("lemon", 0.2) if box[0] < 100 else ("knob", 0.2)

    model = OwlV2Model.__new__(OwlV2Model)
    model._siglip, model._hybrid_px, model._hybrid = FakeSiglip(), 1000, True
    model._owl_objects = lambda rgb: [{"label": "pear", "box": [10, 10, 30, 30], "score": 0.3},     # small: renamed
                                      {"label": "pear", "box": [200, 10, 220, 30], "score": 0.3},   # small: a knob, dropped
                                      {"label": "cutting board", "box": [0, 100, 200, 200], "score": 0.6}]  # big: kept as is
    found = model.objects(np.zeros((480, 640, 3), dtype=np.uint8))
    assert [(f["label"], f.get("owl_label")) for f in found] == [("lemon", "pear"), ("cutting board", None)]


def test_the_sam3_checker_renames_by_mask_fit_and_drops_unsupported_boxes():
    from chefbot.explore.owl import OwlV2Model

    class FakeSiglip:
        def top(self, rgb, box, k):
            return ["lemon", "pear", "apple"]

    class FakeSam3:
        def objects(self, rgb, prompts):
            assert prompts == sorted(set(prompts)) and "lemon" in prompts
            return [{"label": "lemon", "box": [10, 10, 31, 31], "score": 0.8}]  # a lemon where the first box is

    model = OwlV2Model.__new__(OwlV2Model)
    model._siglip, model._sam3, model.sam3_checks = FakeSiglip(), FakeSam3(), 0
    items = [{"label": "pear", "box": [10, 10, 30, 30], "score": 0.25}, {"label": "sponge", "box": [200, 10, 220, 30], "score": 0.22}]
    model._sam3_margin, model._sam3_drop = 0.15, True
    out = model.check_many(np.zeros((480, 640, 3), dtype=np.uint8), items)
    assert out[0]["label"] == "lemon" and out[0]["checked_from"] == "pear" and out[1] is None and model.sam3_checks == 1
    model._sam3_drop = False
    assert model.check_many(np.zeros((480, 640, 3), dtype=np.uint8), items)[1] == items[1]  # kept as it was


def test_the_sam3_checker_renames_only_on_a_clear_margin():
    from chefbot.explore.owl import OwlV2Model

    class FakeSiglip:
        def top(self, rgb, box, k):
            return ["apple", "cherry"]

    class FakeSam3:
        def objects(self, rgb, prompts):
            return [{"label": "cherry", "box": [10, 10, 30, 30], "score": 0.6}, {"label": "apple", "box": [10, 10, 30, 30], "score": 0.5}]

    model = OwlV2Model.__new__(OwlV2Model)
    model._siglip, model._sam3, model.sam3_checks, model._sam3_drop = FakeSiglip(), FakeSam3(), 0, False
    item = {"label": "apple", "box": [10, 10, 30, 30], "score": 0.3}
    model._sam3_margin = 0.15
    assert model.check_many(np.zeros((480, 640, 3), dtype=np.uint8), [item])[0]["label"] == "apple"  # 0.6 vs 0.5: not clear
    model._sam3_margin = 0.05
    assert model.check_many(np.zeros((480, 640, 3), dtype=np.uint8), [item])[0]["label"] == "cherry"
