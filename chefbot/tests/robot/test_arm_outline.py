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
