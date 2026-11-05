"""robot/nav/speed.py: speed limited by the measured stop and the free space ahead."""

import numpy as np
import pytest

from chefbot.robot.nav.grid import OccupancyGrid
from chefbot.robot.nav.speed import StopModel, free_distance, lookahead_for

HT4D = StopModel(tau_s=0.333, brake_mps2=0.25)  # 78 mm from 0.131 m/s, last fresh command -> rest


def test_the_model_reproduces_the_measured_stop():
    assert HT4D.distance(0.131) == pytest.approx(0.078, abs=0.001)


@pytest.mark.parametrize("free", [0.05, 0.12, 0.3, 1.0])
def test_speed_for_stops_exactly_within_the_free_space(free):
    v = HT4D.speed_for(free)
    assert HT4D.distance(v) == pytest.approx(free)


def test_no_free_space_no_speed():
    assert HT4D.speed_for(0.0) == 0.0 and HT4D.speed_for(-0.1) == 0.0


def test_no_measured_stop_is_refused():
    with pytest.raises(ValueError):
        StopModel.from_limits({"base_max_vel_xy": 0.125})
    assert StopModel.from_limits({"base_stop_tau_s": 0.333, "base_stop_brake_mps2": 0.25}) == HT4D


def grid_with_wall_at_x(x_wall: float) -> OccupancyGrid:
    occ = np.zeros((40, 80), dtype=bool)  # 2 m x 4 m at 5 cm, origin (0, 0)
    occ[:, int(x_wall / 0.05):] = True
    return OccupancyGrid(occ, 0.05, (0.0, 0.0))


def test_free_distance_along_and_beside_an_obstacle():
    g = grid_with_wall_at_x(2.0)
    assert free_distance(g, (1.0, 1.0), (1.0, 0.0)) == pytest.approx(1.0, abs=0.03)  # toward the wall
    assert free_distance(g, (1.0, 1.0), (0.0, 1.0), max_m=0.8) == pytest.approx(0.8)  # along it: no limit
    assert free_distance(g, (2.1, 1.0), (1.0, 0.0)) == 0.0  # already in it


def test_lookahead_grows_with_speed():
    assert lookahead_for(0.05, 0.1, 0.1) == 0.12
    assert lookahead_for(0.3, 0.3, 0.1) == pytest.approx(0.15 + 0.03 + 0.05)
