import math

import numpy as np

from chefbot.robot.nav.grid import OccupancyGrid


def make_grid(rows=20, cols=30, resolution=0.1, origin=(-1.0, 2.0)):
    return OccupancyGrid(np.zeros((rows, cols), dtype=bool), resolution, origin)


def test_world_cell_round_trip():
    grid = make_grid()
    for cell in [(0, 0), (5, 7), (19, 29)]:
        assert grid.world_to_cell(*grid.cell_to_world(cell)) == cell
    # Row follows y, column follows x.
    assert grid.world_to_cell(-1.0 + 0.25, 2.0 + 0.05) == (0, 2)
    x, y = grid.cell_to_world((3, 4))
    assert (x, y) == (-1.0 + 0.45, 2.0 + 0.35)


def test_outside_is_occupied():
    grid = make_grid()
    assert grid.is_free(0.0, 3.0)
    assert not grid.is_free(-1.5, 3.0)
    assert not grid.is_free(0.0, 4.5)


def test_save_load_round_trip(tmp_path):
    grid = make_grid()
    grid.occupied[3, 4] = True
    grid.save(tmp_path / "grid.npy")
    loaded = OccupancyGrid.load(tmp_path / "grid.npy")
    assert np.array_equal(loaded.occupied, grid.occupied)
    assert loaded.resolution == grid.resolution
    assert loaded.origin == grid.origin


def test_inflation_matches_distance_to_obstacle():
    grid = make_grid(rows=41, cols=41, resolution=0.05, origin=(0.0, 0.0))
    grid.occupied[20, 20] = True
    radius = 0.3
    inflated = grid.inflate(radius)
    ox, oy = grid.cell_to_world((20, 20))
    half = grid.resolution / 2
    for row in range(41):
        for col in range(41):
            x, y = grid.cell_to_world((row, col))
            # Distance from this cell's center to the occupied cell's square.
            gap = math.hypot(max(abs(x - ox) - half, 0.0), max(abs(y - oy) - half, 0.0))
            if abs(gap - radius) < 1e-9:
                continue
            assert inflated.occupied[row, col] == (gap < radius), (row, col)
    assert not grid.occupied[20, 21]  # the source grid is untouched


def test_inflation_never_shrinks_and_zero_is_identity():
    grid = make_grid()
    grid.occupied[5:8, 10:12] = True
    assert np.array_equal(grid.inflate(0.0).occupied, grid.occupied)
    assert np.all(grid.inflate(0.25).occupied[grid.occupied])


def test_nearest_free_cell():
    grid = make_grid(rows=20, cols=20, resolution=0.1, origin=(0.0, 0.0))
    grid.occupied[5:15, 5:15] = True
    x, y = grid.cell_to_world((6, 10))  # one cell in from the bottom edge of the block
    assert grid.nearest_free_cell(x, y, 0.5) == (4, 10)
    cx, cy = grid.cell_to_world((10, 10))
    assert grid.nearest_free_cell(cx, cy, 0.2) is None
    assert grid.nearest_free_cell(0.05, 0.05, 0.2) == (0, 0)
