import math

import numpy as np
import pytest

from chefbot.interface.types import Pose2D
from chefbot.robot.nav.astar import AStarPlanner, NoPath, path_length
from chefbot.robot.nav.grid import OccupancyGrid

MAZE = """
####################
#S.....#...........#
#......#...........#
#......#....####...#
#......#....#......#
#...........#......#
#...........#.....G#
####################
"""


def maze_grid(resolution=0.1):
    rows = [line for line in MAZE.strip().splitlines()][::-1]  # row 0 at the bottom
    occupied = np.array([[ch == "#" for ch in line] for line in rows])
    grid = OccupancyGrid(occupied, resolution, (0.0, 0.0))
    find = lambda ch: next(
        grid.cell_to_world((r, c)) for r, line in enumerate(rows) for c, x in enumerate(line) if x == ch
    )
    return grid, find("S"), find("G")


def assert_collision_free(grid, points):
    for a, b in zip(points, points[1:]):
        assert grid.segment_free(a, b), (a, b)


def test_solves_maze_with_collision_free_shortcuts():
    grid, start, goal = maze_grid()
    points = AStarPlanner(grid).plan_xy(start, goal)
    assert points[0] == start and points[-1] == goal
    assert_collision_free(grid, points)
    # Shortcutting leaves a handful of waypoints, and the path is not absurdly long.
    assert len(points) <= 6
    assert math.dist(start, goal) <= path_length(points) < 2.0 * math.dist(start, goal)


def test_no_path_when_enclosed():
    grid, start, goal = maze_grid()
    row, col = grid.world_to_cell(*goal)
    grid.occupied[row - 1 : row + 2, col - 1 : col + 2] = True
    grid.occupied[row, col] = False
    with pytest.raises(NoPath):
        AStarPlanner(grid).plan_xy(start, goal)


def test_no_path_when_endpoint_in_collision():
    grid, start, goal = maze_grid()
    with pytest.raises(NoPath):
        AStarPlanner(grid).plan_xy(start, (0.05, 0.05))
    with pytest.raises(NoPath):
        AStarPlanner(grid).plan_xy((0.05, 0.05), goal)


def test_does_not_cut_corners():
    occupied = np.zeros((3, 3), dtype=bool)
    occupied[0, 1] = occupied[1, 0] = True  # (0,0) only connects diagonally to (1,1)
    grid = OccupancyGrid(occupied, 1.0, (0.0, 0.0))
    with pytest.raises(NoPath):
        AStarPlanner(grid).plan_xy((0.5, 0.5), (2.5, 2.5))


def test_yaw_interpolates_to_goal_yaw():
    grid, start, goal = maze_grid()
    path = AStarPlanner(grid).plan(Pose2D(*start, 0.0), Pose2D(*goal, math.pi / 2))
    yaws = [p.yaw for p in path]
    assert yaws[0] == 0.0
    assert yaws[-1] == pytest.approx(math.pi / 2)
    assert yaws == sorted(yaws)


def test_same_cell_start_and_goal():
    grid, start, _ = maze_grid()
    goal = (start[0] + 0.01, start[1] + 0.02)
    assert AStarPlanner(grid).plan_xy(start, goal) == [start, goal]


def test_cost_to_matches_path_length():
    grid, start, goal = maze_grid()
    planner = AStarPlanner(grid)
    goal_cell = grid.world_to_cell(*goal)
    enclosed = (0, 0)
    costs = planner.cost_to(start, {goal_cell, enclosed})
    assert enclosed not in costs
    # Grid cost is an upper bound on the shortcut path and within the 8-connected detour factor.
    length = path_length(planner.plan_xy(start, goal))
    assert length <= costs[goal_cell] + 1e-9 <= 1.1 * length
