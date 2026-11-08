"""The memory map on a synthetic scene: a camera 1.5 m up looking straight down at a 0.9 m counter, one block on it."""

from types import SimpleNamespace

import numpy as np

from chefbot.perception.memory_map import MemoryMap
from chefbot.perception.pole_geometry import Counter, Intrinsics

W, H, F = 160, 120, 120.0
CAM = SimpleNamespace(color=Intrinsics(W, H, np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]]), np.zeros(5)))
T_WC = np.eye(4)
T_WC[:3, :3] = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]  # camera x = world x, y = -world y, z = down
T_WC[:3, 3] = [0, 0, 1.5]
COUNTER = Counter("c", "counter", np.zeros(2), np.array([2.0, 2.0]), 0.0, 0.9)


def scene(block_px=None):
    """Registered depth (H x W x 3, camera) of the counter top, with a 0.1 m tall 14x14 px block at block_px."""
    z = np.full((H, W), 0.6)
    if block_px is not None:
        u, v = block_px
        z[v : v + 14, u : u + 14] = 0.5
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    return np.stack([(uu - W / 2) / F * z, (vv - H / 2) / F * z, z], axis=-1).astype(np.float32)


def det(block_px, label="cup"):
    u, v = block_px
    return {"label": label, "score": 0.8, "box": [u, v, u + 14, v + 14], "poly": [[u, v], [u + 14, v], [u + 14, v + 14], [u, v + 14]]}


def test_seen_then_gone_then_found_elsewhere():
    mm = MemoryMap([COUNTER])
    a, b = (40, 40), (100, 70)
    for t in (1, 2):
        mm.update([det(a)], scene(a), T_WC, t, float(t), CAM)
    (e,) = mm.entries
    assert e.status == "seen" and e.n_views == 2 and abs(e.xyz[2] - 1.0) < 0.01  # the surface the camera sees: the block's top
    mm.update([], scene(a), T_WC, 3, 3.0, CAM)  # still standing there, detector silent: no evidence
    assert e.misses == 0
    mm.update([], scene(), T_WC, 4, 4.0, CAM)  # bare counter at its spot
    mm.update([], scene(), T_WC, 5, 5.0, CAM)
    assert e.status == "unknown" and e.xyz is None
    mm.update([det(b)], scene(b), T_WC, 6, 6.0, CAM)  # the only cup, seen elsewhere: it moved
    assert len(mm.entries) == 1 and e.status == "seen" and "moved" in e.events[-1][2]
    assert np.linalg.norm(np.asarray(e.xyz[:2]) - np.asarray(e.last_xyz[:2])) < 1e-9


def test_occluded_spot_is_no_evidence_and_single_view_false_positive_dropped():
    mm = MemoryMap([COUNTER])
    a = (40, 40)
    mm.update([det(a)], scene(a), T_WC, 1, 1.0, CAM)
    blocked = scene()
    blocked[:, :, :] *= 0.3 / 0.6  # something 0.3 m from the camera covers the view
    mm.update([], blocked, T_WC, 2, 2.0, CAM)
    assert len(mm.entries) == 1 and mm.entries[0].misses == 0
    mm.update([], scene(), T_WC, 3, 3.0, CAM)  # seen once, then its spot is empty
    assert mm.entries == []


def test_two_of_a_label_get_their_own_entries_and_labels_filtered():
    mm = MemoryMap([COUNTER], labels={"rename": {"mug": "cup"}, "ignore": ["table"]})
    a, b = (30, 30), (100, 70)
    z = scene(a)
    z[70:84, 100:114] = scene(b)[70:84, 100:114]
    for t in (1, 2):
        mm.update([det(a), det(b, "mug"), det(a, "table")], z, T_WC, t, float(t), CAM)
    assert sorted(e.name for e in mm.entries) == ["cup", "cup 2"]


def test_one_spot_several_labels_is_one_object_named_by_vote():
    mm = MemoryMap([COUNTER])
    a = (40, 40)
    for t, labs in ((1, [("lemon", 0.5), ("tennis ball", 0.6)]), (2, [("tennis ball", 0.7)]), (3, [("lime", 0.4)])):
        mm.update([dict(det(a, lab), score=sc) for lab, sc in labs], scene(a), T_WC, t, float(t), CAM)
    (e,) = mm.entries
    assert e.label == "tennis ball" and e.n_views == 3


def test_flat_sticker_is_not_an_object():
    mm = MemoryMap([COUNTER])
    a = (40, 40)
    for t in (1, 2):
        mm.update([det(a, "sticker")], scene(), T_WC, t, float(t), CAM)  # a mask on the bare top
    assert mm.entries == []


def test_lost_object_first_seen_under_another_name_is_merged_back():
    mm = MemoryMap([COUNTER])
    a, b = (40, 40), (100, 70)
    for t in (1, 2):
        mm.update([det(a, "tomato")], scene(a), T_WC, t, float(t), CAM)
    for t in (3, 4):
        mm.update([], scene(), T_WC, t, float(t), CAM)  # gone
    mm.update([dict(det(b, "bowl"), score=0.6)], scene(b), T_WC, 5, 5.0, CAM)  # moved, called a bowl first
    mm.update([dict(det(b, "tomato"), score=0.3)], scene(b), T_WC, 6, 6.0, CAM)
    (e,) = mm.entries
    assert e.status == "seen" and "moved" in e.events[-1][2] and e.n_views == 4


def test_long_object_seen_in_halves_is_one_entry_but_two_cups_stay_two():
    mm = MemoryMap([COUNTER])
    # a 60 px long, 10 px wide "spoon": seen whole, then only its left half, then only its right half
    whole = scene()
    whole[60:70, 40:100] = 0.5 * whole[60:70, 40:100] / whole[60:70, 40:100, 2:3]
    def det_poly(u0, u1):
        return {"label": "spoon", "score": 0.8, "box": [u0, 60, u1, 70], "poly": [[u0, 58], [u1, 58], [u1, 72], [u0, 72]]}
    mm.update([det_poly(40, 100)], whole, T_WC, 1, 1.0, CAM)
    mm.update([det_poly(40, 100)], whole, T_WC, 2, 2.0, CAM)
    mm.update([det_poly(40, 64)], whole, T_WC, 3, 3.0, CAM)
    mm.update([det_poly(76, 100)], whole, T_WC, 4, 4.0, CAM)
    assert len(mm.entries) == 1 and mm.entries[0].n_views == 4
    mm.merge_overlaps()
    assert len(mm.entries) == 1
    two = MemoryMap([COUNTER])
    a, b = (30, 30), (100, 70)
    z = scene(a)
    z[70:84, 100:114] = scene(b)[70:84, 100:114]
    for t in (1, 2):
        two.update([det(a), det(b)], z, T_WC, t, float(t), CAM)
    assert two.merge_overlaps() == 0 and len(two.entries) == 2
