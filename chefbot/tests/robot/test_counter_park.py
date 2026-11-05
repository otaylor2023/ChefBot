"""robot/nav/goal.py: base poses for a counter work target by the pick task's start rule."""

import math

import numpy as np
import pytest

from chefbot.interface.types import Pose2D
from chefbot.robot.nav.goal import counter_park_candidates, park_report

BOX = ((1.49, 2.06), (2.26, 0.96), 0.0)  # maps/src_counter's counter top


def test_a_target_near_the_south_edge_parks_square_in_front_of_it():
    cands = counter_park_candidates(*BOX, (1.80, 2.06 - 0.48 + 0.13))
    assert {i["side"] for _, i in cands} == {"south"}
    pose, info = cands[0]
    assert info["out_m"] == 0.56 and info["trained"] and info["inset_m"] == pytest.approx(0.13)
    assert (pose.x, pose.y) == pytest.approx((1.80, 2.06 - 0.48 - 0.56)) and math.degrees(pose.yaw) == pytest.approx(90.0)
    r = park_report(*BOX, "south", pose, (1.80, 1.71))
    assert r["out_m"] == pytest.approx(0.56) and r["target_lateral_m"] == pytest.approx(0.0) and r["yaw_off_square_deg"] == pytest.approx(0.0)


@pytest.mark.parametrize("target, side, yaw_deg", [((2.62 - 0.12, 2.0), "east", 180.0), ((0.36 + 0.12, 2.1), "west", 0.0),
                                                    ((1.2, 2.54 - 0.15), "north", -90.0)])
def test_every_side_faces_its_edge(target, side, yaw_deg):
    cands = counter_park_candidates(*BOX, target)
    assert {i["side"] for _, i in cands} == {side}
    assert all(abs((math.degrees(p.yaw) - yaw_deg + 180) % 360 - 180) < 1e-9 for p, _ in cands)  # 180 == -180


def test_targets_too_near_an_end_or_too_far_in_get_no_stance():
    assert counter_park_candidates(*BOX, (0.36 + 0.2, 1.58 + 0.13)) == []  # 0.2 m from the west end of the south side
    assert counter_park_candidates(*BOX, (1.49, 2.06)) == []  # the middle of the top: 0.48 m in from every side


def test_a_rotated_counter():
    yaw = math.radians(30)
    c = np.array([0.0, 0.0])
    n = np.array([-math.sin(yaw), math.cos(yaw)])  # its +y side
    target = c + n * (0.48 - 0.12)
    cands = counter_park_candidates((0, 0), (2.26, 0.96), yaw, target)
    pose, info = cands[0]
    assert info["inset_m"] == pytest.approx(0.12)
    assert np.array([pose.x, pose.y]) == pytest.approx(c + n * (0.48 + 0.56))
    assert math.degrees(pose.yaw) == pytest.approx(30 - 90)
