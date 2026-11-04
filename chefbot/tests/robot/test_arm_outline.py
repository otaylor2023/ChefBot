"""robot/kinematics/arm_outline: the FR3 over the floor from its joints."""

import math

import numpy as np
import pytest

from chefbot.robot.kinematics import arm_outline as A

READY = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4]  # Franka's ready pose


def test_the_ready_pose_flange_is_where_franka_puts_it():
    flange = A.frames(READY)[7][:3, 3]
    assert flange == pytest.approx([0.307, 0.0, 0.590], abs=0.002)  # Franka's documented ready pose (flange)


def test_the_hand_tip_points_down_in_the_ready_pose():
    f = A.frames(READY)
    assert f[8][2, 3] < f[7][2, 3] - 0.15  # the tip is below the flange: the hand points at the floor


def test_the_arm_box_covers_every_link_and_follows_the_arm():
    box = A.arm_box(READY)
    (cx, cy), (L, W) = box["center"], box["size"]
    pts = A.arm_points(READY)[:, :2]
    assert np.all(pts[:, 0] >= cx - L / 2) and np.all(pts[:, 0] <= cx + L / 2)
    assert np.all(pts[:, 1] >= cy - W / 2) and np.all(pts[:, 1] <= cy + W / 2)
    stretched = READY.copy()
    stretched[1], stretched[3] = 0.6, -0.5  # leaning forward and opened up
    assert A.reach_ahead(stretched) > A.reach_ahead(READY) + 0.2
    turned = READY.copy()
    turned[0] = math.pi / 2  # base joint turned 90 deg: the arm now reaches to the base's left
    box = A.arm_box(turned)
    assert box["center"][1] + box["size"][1] / 2 > 0.35  # out to the left as far as it reached ahead before
    assert box["center"][0] + box["size"][0] / 2 < 0.2  # and no longer far ahead


def test_the_footprint_grows_to_cover_the_arm():
    from chefbot.scripts.real_nav import arm_footprint

    none = arm_footprint({})
    assert none["radius"] == 0.33 and none["arm"] is None
    stretched = READY.copy()
    stretched[1], stretched[3] = 0.6, -0.5
    fp = arm_footprint({"arm_q": stretched, "arm_live": True})
    assert fp["radius"] > 0.33 + 0.2 and fp["arm_live"]


LOCKED = [-0.0327909, -0.829123, -0.0034275, -2.31953, -0.0898839, 2.33443, 0.859095]  # HT-4a/4b's locked arm


def test_an_arm_above_a_counter_passes_over_it_but_not_over_a_wall():
    from chefbot.scripts.real_nav import arm_footprint, low_footprint

    full = arm_footprint({"arm_q": LOCKED})["radius"]
    assert full > 0.5  # the arm reaches ~0.54 m ahead: walls keep that far
    assert low_footprint({"arm_q": LOCKED}, low_height=0.93) == pytest.approx(0.33)  # its tip is ~12 cm above the counter
    assert low_footprint({"arm_q": LOCKED}, low_height=1.05) == pytest.approx(0.46 + 0.03, abs=0.01)  # taller: the hand counts
    assert low_footprint({}, low_height=0.93) == pytest.approx(0.33)  # unknown arm: the full (here: base) footprint


def test_layered_maps_keep_walls_further_than_counters():
    from chefbot.scripts.real_map import build_layers
    from chefbot.scripts.real_nav import layered_grids

    spec = {"resolution": 0.05, "bounds": [[0, 0], [4, 3]],
            "obstacles": [{"name": "counter", "box": {"center": [2.0, 1.5], "size": [1.0, 0.6]}, "height_m": 0.93}]}  # fmt: skip
    tall, low, h, _ = build_layers(spec)
    assert h == 0.93 and low.occupied.any() and not tall.is_free(-0.05, 1.0) and tall.is_free(2.0, 1.5)
    plan, body, info = layered_grids(tall, low, h, {"arm_q": LOCKED}, margin=0.05)
    assert info["low_m"] < info["full_m"]
    # 0.45 m from the counter's edge is free for the base; 0.45 m from a wall is not.
    assert plan.is_free(2.0, 1.5 - 0.3 - 0.45) and not plan.is_free(0.45, 1.5)
