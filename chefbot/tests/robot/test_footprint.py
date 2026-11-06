"""The robot's footprint from its measured body (interface/workspace.py Body.footprint) and where the base's centre may
be at each heading for it (robot/nav/footprint.py FootprintSpace); real_nav's use of both."""

import math

import numpy as np
import pytest

from chefbot.interface.workspace import BODY, READY, Footprint, arm_points
from chefbot.robot.nav.footprint import FootprintSpace
from chefbot.robot.nav.grid import OccupancyGrid

LOCKED = [-0.0327909, -0.829123, -0.0034275, -2.31953, -0.0898839, 2.33443, 0.859095]  # HT-4a/4b's locked arm


# -- the footprint ------------------------------------------------------------------------


def test_the_plate_is_centred_on_the_base_and_the_arm_sits_ahead_of_it():
    fp = BODY.footprint()
    assert fp.rect == pytest.approx((-0.2795, 0.2795, -0.241, 0.241))  # 22 x 19 in about the base centre
    assert fp.radius == pytest.approx(math.hypot(0.2795, 0.241))  # its corners: ~0.37 m out
    assert BODY.base_xy[0] == pytest.approx(-0.1825)  # the arm is ~18 cm ahead of the base centre
    base = BODY.to_base(arm_points(READY, BODY.hand_m))
    assert base[0, 0] == pytest.approx(0.1825) and base[0, 2] == pytest.approx(BODY.plate_height + 0.333)  # joint 1


def test_the_arm_grows_the_footprint_and_turns_it():
    assert BODY.footprint(LOCKED).reach(0.0) > 0.65  # leaning out over a counter
    turned = list(LOCKED)
    turned[0] = math.pi / 2  # the arm's base joint turned 90 deg: it reaches to the left now
    fp = BODY.footprint(turned)
    assert fp.reach(90.0) > 0.5 and fp.reach(0.0) < 0.3  # from 18 cm ahead of the centre: ~0.52 m left


def test_only_the_arms_low_points_count_against_a_low_obstacle():
    assert BODY.footprint(LOCKED, below=0.93 + 0.05).radius == pytest.approx(BODY.footprint().radius)  # over the counter
    assert BODY.footprint(LOCKED, below=1.10).reach(0.0) > 0.6  # a taller one: the hand hits it


def test_distance_to_the_footprint():
    fp = Footprint((-0.2, 0.2, -0.1, 0.1), np.array([[0.5, 0.0, 0.05]]))
    d = fp.distance([[0.0, 0.0], [0.3, 0.0], [0.0, 0.3], [0.5, 0.1], [0.5, 0.0]])
    assert d == pytest.approx([0.0, 0.1, 0.2, 0.05, 0.0])


# -- where the base's centre may be at each heading ---------------------------------------


def _wall_at_x(x0: float) -> OccupancyGrid:
    """A wall filling x >= x0 on a 4 x 2 m grid at 0.05 m."""
    occ = np.zeros((40, 80), dtype=bool)
    occ[:, int(round(x0 / 0.05)) :] = True
    return OccupancyGrid(occ, 0.05, (0.0, 0.0))


def test_a_circle_is_never_freer_than_inflate():
    g = _wall_at_x(2.0)
    circle = Footprint((0.0, 0.0, 0.0, 0.0), np.zeros((0, 3)))
    grown = FootprintSpace(g, circle, margin=0.33).at(0.7).occupied
    assert not (g.inflate(0.33).occupied & ~grown).any()


def test_the_plate_comes_closer_end_on_than_corner_on():
    space = FootprintSpace(_wall_at_x(2.0), BODY.footprint(), margin=0.09)
    # Facing the wall: the plate's front (0.28 m) + margin (0.09) + a cell's slack fits at 0.45 m.
    assert space.is_free(2.0 - 0.45, 1.0, 0.0)
    # At 45 deg a corner leads (~0.37 m out): 0.45 m is too close with the margin.
    assert not space.is_free(2.0 - 0.45, 1.0, math.radians(45))
    assert space.radius == pytest.approx(BODY.footprint().radius)


def test_a_slice_covers_every_heading_in_its_band():
    g = _wall_at_x(2.0)
    fp = BODY.footprint(LOCKED)
    coarse = FootprintSpace(g, fp, step_deg=10.0)
    for yaw in np.radians(np.arange(-5.0, 5.01, 0.5)):  # the 0 deg band's headings
        exact = FootprintSpace(g, fp, step_deg=0.05).at(yaw).occupied
        assert not (exact & ~coarse.at(0.0).occupied).any()


def test_a_path_is_checked_at_the_heading_along_it():
    space = FootprintSpace(_wall_at_x(2.0), BODY.footprint(), margin=0.0)
    x = 2.0 - 0.32  # the plate's front fits facing the wall, not its side's corners turned 45 deg
    assert space.path_clear([(x, 1.0, 0.0), (x, 1.2, 0.0)])
    assert not space.path_clear([(x, 1.0, 0.0), (x, 1.2, math.radians(90))])


def test_a_centre_limit_is_not_grown():
    g = _wall_at_x(3.9)
    fence = np.zeros_like(g.occupied)
    fence[:, :4] = True  # x < 0.2 m: off limits for the centre only
    space = FootprintSpace(g, BODY.footprint(), centre_limit=OccupancyGrid(fence, 0.05, (0.0, 0.0)))
    assert not space.is_free(0.15, 1.0, 0.0) and space.is_free(0.25, 1.0, 0.0)


# -- real_nav --------------------------------------------------------------------------------


def test_real_nav_footprint_is_the_plate_without_arm_readings():
    from chefbot.scripts.real_nav import robot_footprints

    below, info = robot_footprints({})
    assert below().radius == pytest.approx(BODY.footprint().radius) and "plate only" in info["note"]
    below, info = robot_footprints({"arm_q": LOCKED, "arm_live": True})
    assert below().reach(0.0) > 0.65 and info["arm_live"]
    assert below(0.93).radius == pytest.approx(BODY.footprint().radius)  # over a counter top: the plate only
    below, _ = robot_footprints({"arm_q": LOCKED}, arm="off")
    assert len(below().discs) == 0


def test_maps_with_heights_park_close_to_a_counter_the_arm_passes_over_but_not_to_a_wall():
    from chefbot.scripts.real_map import build
    from chefbot.scripts.real_nav import nav_space

    spec = {"resolution": 0.05, "bounds": [[0, 0], [4, 3]],
            "obstacles": [{"name": "counter", "box": {"center": [2.0, 1.5], "size": [1.0, 0.6]}, "height_m": 0.93}]}  # fmt: skip
    grid, _ = build(spec)
    assert grid.tops() == [0.93, math.inf]
    space, info = nav_space(grid, {"arm_q": LOCKED})
    plan = space.with_margin(0.09)
    facing_counter = math.pi / 2  # from its south side, 0.45 m out, as the parking rule puts it
    assert plan.is_free(2.0, 1.5 - 0.3 - 0.45, facing_counter)
    assert not plan.is_free(0.45, 1.5, math.pi)  # facing the west wall from 0.45 m: the arm (0.7 m ahead) would hit it
    assert info["by_top"]["0.93 m"] < info["by_top"]["wall"]


# -- the map format (maps/README.md) ---------------------------------------------------------


def test_every_map_folder_has_the_standard_files():
    from pathlib import Path

    import yaml

    maps = Path(__file__).resolve().parents[3] / "maps"
    folders = [d for d in maps.iterdir() if d.is_dir()]
    assert folders
    for d in folders:
        for f in ("grid.npy", "grid.yaml", "heights.npy", "semantic_map.yaml"):
            assert (d / f).exists(), f"{d.name}: no {f}"
        grid = OccupancyGrid.load(d / "grid.npy")
        assert grid.heights.shape == grid.occupied.shape and (grid.heights[grid.occupied] > 0).all(), d.name
        sem = yaml.safe_load((d / "semantic_map.yaml").read_text())
        assert sem["frame"] == "world" and "fixtures" in sem and "objects" in sem, d.name
        assert not list(d.glob("grid_low*")) and not list(d.glob("grid_tall*")), d.name


def test_heights_round_trip_and_a_map_without_them_is_all_walls(tmp_path):
    occ = np.zeros((4, 5), dtype=bool)
    occ[1, 2] = occ[2, 3] = True
    g = OccupancyGrid(occ, 0.1, (0.0, 0.0), np.where(occ, 0.9, 0.0))
    g.heights[2, 3] = np.inf
    g.save(tmp_path / "grid.npy")
    back = OccupancyGrid.load(tmp_path / "grid.npy")
    assert back.tops() == [0.9, math.inf] and back.only_top(0.9).occupied.sum() == 1
    (tmp_path / "heights.npy").unlink()
    assert OccupancyGrid.load(tmp_path / "grid.npy").tops() == [math.inf]


def test_the_real_builder_writes_heights_and_a_semantic_map(tmp_path):
    import yaml

    from chefbot.environments.real.build_map import build_map

    spec = {"resolution": 0.05, "bounds": [[0, 0], [3, 2]],
            "obstacles": [{"name": "c", "box": {"center": [1.5, 1.0], "size": [1.0, 0.5]}, "height_m": 0.9, "fixture": "counter"},
                          {"name": "post", "box": {"center": [0.5, 0.5], "size": [0.1, 0.1]}}],
            "places": {"p": {"x": 0.3, "y": 1.5, "yaw_deg": 0}}}  # fmt: skip
    build_map(spec, tmp_path)
    g = OccupancyGrid.load(tmp_path / "grid.npy")
    assert g.heights[g.world_to_cell(1.5, 1.0)] == 0.9 and math.isinf(g.heights[g.world_to_cell(0.5, 0.5)])
    assert math.isinf(g.heights[0, 0])  # outside the bounds: a wall
    sem = yaml.safe_load((tmp_path / "semantic_map.yaml").read_text())
    assert sem["fixtures"]["counter"]["location"] == [1.5, 1.0] and sem["objects"] == {}
    assert yaml.safe_load((tmp_path / "places.yaml").read_text())["p"]["x"] == 0.3
