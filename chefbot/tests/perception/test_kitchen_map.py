"""The live object map (perception/kitchen_map.py): synthetic frames from straight above the island."""

import math
from pathlib import Path

import numpy as np
import pytest

from chefbot.perception.kitchen_map import KitchenMap, xyz_image

REPO = Path(__file__).resolve().parents[3]
SPEC = REPO / "maps/src_kitchen_work/spec.yaml"
ISLAND_TOP = 0.895
K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
CAM_XY = (1.0, -2.3)  # over the island (centre (1.305, -2.087), 2.22 x 0.96, -2.96 deg)
CAM_Z = 1.9


def T_down(x=CAM_XY[0], y=CAM_XY[1], z=CAM_Z):
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])  # OpenCV axes looking straight down: image x = world x, image y = -world y
    T[:3, 3] = [x, y, z]
    return T


def frame(*objs, h=0.06, r_px=18, label="tomato"):
    """Depth of the bare island top, with a 6 cm tall disc at each world x/y in `objs`; their detection outlines."""
    depth = np.full((480, 640), CAM_Z - ISLAND_TOP, np.float32)
    dets = []
    for obj_xy in objs:
        u = K[0, 2] + K[0, 0] * (obj_xy[0] - CAM_XY[0]) / (CAM_Z - ISLAND_TOP - h)
        v = K[1, 2] - K[1, 1] * (obj_xy[1] - CAM_XY[1]) / (CAM_Z - ISLAND_TOP - h)
        vv, uu = np.mgrid[:480, :640]
        disc = (uu - u) ** 2 + (vv - v) ** 2 <= (obj_xy[3] if len(obj_xy) > 3 else r_px) ** 2
        depth[disc] = CAM_Z - ISLAND_TOP - h
        r_px = obj_xy[3] if len(obj_xy) > 3 else r_px
        poly = [[u + r_px * math.cos(a), v + r_px * math.sin(a)] for a in np.linspace(0, 2 * math.pi, 24, endpoint=False)]
        dets.append({"label": obj_xy[2] if len(obj_xy) > 2 else label, "score": 0.7, "box": [u - r_px, v - r_px, u + r_px, v + r_px], "poly": poly})
    rgb = np.full((480, 640, 3), 120, np.uint8)
    return rgb, depth, dets


def test_xyz_image_deprojects_and_blanks_bad_depth():
    d = np.full((4, 6), 1.0, np.float32)
    d[0, 0] = 0.0
    p = xyz_image(d, np.array([[2.0, 0, 3], [0, 2.0, 2], [0, 0, 1]]))
    assert np.isnan(p[0, 0, 2]) and p[2, 3].tolist() == pytest.approx([0.0, 0.0, 1.0])
    assert p[2, 5, 0] == pytest.approx(1.0)  # (5 - 3) / 2


def test_an_object_needs_two_views_then_resolves_then_goes_unknown_when_its_spot_is_bare():
    km = KitchenMap(SPEC, log=lambda *a: None)
    rgb, depth, dets = frame((1.10, -2.45))
    r = km.add_frame(rgb, depth, K, T_down(), dets, t=1.0)
    assert r["new"] == ["tomato"] and km.objects() == []  # one view: tentative, not offered
    km.add_frame(rgb, depth, K, T_down(), dets, t=2.0)
    (o,) = km.objects()
    assert o["name"] == "tomato" and o["counter"] == "kitchen_island_countertop" and o["fixture"] == "island"
    assert o["xyz"][:2] == pytest.approx([1.10, -2.45], abs=0.01) and o["xyz"][2] == pytest.approx(ISLAND_TOP + 0.06, abs=0.01)
    assert km.resolve("tomato")["name"] == "tomato" and km.views[o["id"]]  # a view kept for the naming
    bare = frame()
    for t in (3.0, 4.0):
        km.add_frame(*bare[:2], K, T_down(), [], t=t)
    assert km.objects()[0]["status"] == "unknown"
    with pytest.raises(LookupError, match="position unknown"):
        km.resolve("tomato")


def test_naming_once_per_object_renames_and_rejects(tmp_path):
    km = KitchenMap(SPEC, log=lambda *a: None)
    rgb, depth, dets = frame((1.10, -2.45), (0.70, -2.40))
    for t in (1.0, 2.0):
        km.add_frame(rgb, depth, K, T_down(), dets, t=t)
    asked = []

    def ask(im, k, model, hint, url, key=None):
        asked.append(k)
        return {"label": "roma tomato" if len(asked) == 1 else "sticker", "movable_object": len(asked) == 1, "model": model}

    r = km.name_new(model="gemini-test", ask=ask, workers=1)
    assert r["named"] == 2 and km.unnamed() == []
    names = {o["name"]: o for o in km.objects()}
    assert names["roma tomato"]["det_label"] == "tomato" and names["sticker"]["status"] == "rejected"
    assert km.resolve("roma_tomato")["name"] == "roma tomato"
    with pytest.raises(LookupError):
        km.resolve("sticker")  # rejected: never offered
    assert km.name_new(model="gemini-test", ask=ask)["named"] == 0  # once each
    p = km.save(tmp_path / "objects.json")
    back = KitchenMap.load(SPEC, p.with_name("objects_memory.json"), log=lambda *a: None)
    assert {o["name"] for o in back.objects()} == set(names)


def layout(tmp_path):
    from chefbot.perception.kitchen_map import Layout

    p = tmp_path / "layout.yaml"
    p.write_text("items:\n  tomato: {count: 1, prompts: [tomato]}\n"
                 "  bowl: {count: 2, prompts: [bowl], sizes: [large bowl, small bowl]}\n")
    return Layout.load(p)


def test_a_layout_caps_counts_sizes_bowls_and_drops_what_is_not_on_its_list(tmp_path):
    L = layout(tmp_path)
    assert L.queries() == ["tomato", "bowl"] and L.names() == ["tomato", "large bowl", "small bowl"]
    km = KitchenMap(SPEC, log=lambda *a: None, layout=L)
    rgb, depth, dets = frame((1.10, -2.45, "tomato"), (0.60, -2.40, "tomato"), (1.30, -2.30, "bowl", 40),
                             (0.85, -2.25, "bowl", 16), (0.40, -2.20, "sponge"))  # fmt: skip
    for t in (1.0, 2.0):
        km.add_frame(rgb, depth, K, T_down(), dets, t=t)
    by = {o["name"]: o for o in km.objects()}
    assert sorted(o["status"] for o in by.values() if o["item"] == "tomato") == ["extra", "seen"]  # one tomato offered
    assert by["large bowl"]["xyz"][0] == pytest.approx(1.30, abs=0.02) and by["small bowl"]["xyz"][0] == pytest.approx(0.85, abs=0.02)
    assert not any(o["label"] == "sponge" for o in by.values())  # not on the list: never mapped
    km.name_new(model="gemini-test", ask=lambda im, k, model, hint, url, key=None, menu=False: {"label": "other", "model": model})
    assert all(o["status"] == "rejected" for o in km.objects())  # the VLM said none of ours


def test_what_lies_over_the_robots_own_plate_is_not_an_object():
    from chefbot.interface.types import Pose2D

    km = KitchenMap(SPEC, log=lambda *a: None)
    rgb, depth, dets = frame((1.10, -2.45))
    for t in (1.0, 2.0):
        km.add_frame(rgb, depth, K, T_down(), dets, t=t, base_pose=Pose2D(1.05, -2.45, 0.0))  # (as if) the base under it
    assert km.objects() == []
    km.add_frame(rgb, depth, K, T_down(), dets, t=3.0, base_pose=Pose2D(1.05, -3.10, math.pi / 2))
    km.add_frame(rgb, depth, K, T_down(), dets, t=4.0, base_pose=Pose2D(1.05, -3.10, math.pi / 2))
    assert [o["name"] for o in km.objects()] == ["tomato"]


def test_views_survive_save_and_load(tmp_path):
    km = KitchenMap(SPEC, log=lambda *a: None)
    rgb, depth, dets = frame((1.10, -2.45))
    for t in (1.0, 2.0):
        km.add_frame(rgb, depth, K, T_down(), dets, t=t)
    p = km.save(tmp_path / "objects.json")
    back = KitchenMap.load(SPEC, p.with_name("objects_memory.json"), log=lambda *a: None)
    assert back.unnamed() == km.unnamed() != [] and back.strip(back.unnamed()[0]) is not None
