import math

import pytest

from chefbot.perception.goto import park_target, resolve
from chefbot.robot.nav.goal import counter_park_candidates

ISLAND = ((1.305, -2.087), (2.224, 0.961), math.radians(-2.96))


def test_resolve_by_name_label_and_refuses_unknown_or_ambiguous():
    objs = [{"name": "cup", "label": "cup", "xyz": [1, 2, 0.9]}, {"name": "cup 2", "label": "cup", "xyz": [1, 3, 0.9]},
            {"name": "tomato", "label": "tomato", "xyz": None, "last_xyz": [1, 1, 0.9]}]  # fmt: skip
    assert resolve(objs, "cup 2")["xyz"] == [1, 3, 0.9]
    with pytest.raises(LookupError, match="say which"):
        resolve([dict(o, name=o["name"] + "x") for o in objs[:2]], "cup")  # no entry named "cup": two of the label
    with pytest.raises(LookupError, match="unknown"):
        resolve(objs, "tomato")
    with pytest.raises(LookupError, match="no object"):
        resolve(objs, "spatula")


def test_target_near_edge_is_kept_and_has_a_stance():
    # 0.12 m in from the island's south side (box y is the depth axis; south normal ~ -y)
    c, s = math.cos(ISLAND[2]), math.sin(ISLAND[2])
    xy = (ISLAND[0][0] + 0.3 * c + (0.4805 - 0.12) * s, ISLAND[0][1] + 0.3 * s - (0.4805 - 0.12) * c)
    r = park_target(*ISLAND, xy)
    assert r["within_rule"] and r["side"] == "south" and abs(r["inset_m"] - 0.12) < 1e-3
    assert counter_park_candidates(*ISLAND, r["target"])


def test_deep_target_moves_to_nearest_side_within_the_rule():
    xy = (ISLAND[0][0] + 0.2, ISLAND[0][1] - 0.15)  # middle of the island, a bit south
    r = park_target(*ISLAND, xy)
    assert not r["within_rule"] and r["side"] == "south" and 0.05 < r["reach_extra_m"] < 0.2
    assert counter_park_candidates(*ISLAND, r["target"])  # the moved target has a stance


def test_floor_keeps_stove_counter_stances_off_the_wall_side():
    from pathlib import Path

    from chefbot.perception.goto import Floor

    floor = Floor.from_spec(Path(__file__).parents[3] / "maps" / "src_kitchen" / "spec.yaml")
    stove = ((1.936, -4.638), (3.826, 0.632), math.radians(-2.96))
    # 0.1 m from the stove counter's BACK (wall) edge: nearest side is south, but the base can't stand there
    xy = (1.9, -4.95 + 0.10 + 0.012)
    r = park_target(*stove, xy, floor=floor)
    assert r["side"] == "north" and "south" in r["blocked"] and r["stance"] is not None
    assert floor.why_blocked(*r["stance"][:2], math.radians(r["stance"][2])) is None
