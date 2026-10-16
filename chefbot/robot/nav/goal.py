"""Goal handling: snapping goals out of collision and choosing standoff poses."""

from __future__ import annotations

import math
from typing import Callable, Protocol

import numpy as np

from chefbot.interface.types import Pose2D
from chefbot.robot.nav.astar import AStarPlanner
from chefbot.robot.nav.grid import OccupancyGrid


class Reachability(Protocol):
    def is_reachable(self, base_pose: Pose2D, target_world: np.ndarray) -> bool: ...


def snap_to_free(grid: OccupancyGrid, pose: Pose2D, snap_radius: float) -> Pose2D | None:
    """`pose` if it is collision-free, else the nearest free cell center within
    `snap_radius` (yaw kept), else None."""
    if grid.is_free(pose.x, pose.y):
        return pose
    cell = grid.nearest_free_cell(pose.x, pose.y, snap_radius)
    if cell is None:
        return None
    x, y = grid.cell_to_world(cell)
    return Pose2D(x, y, pose.yaw)


def standoff_candidates(
    target_xy: tuple[float, float], r_min: float, r_max: float, rings: int, angles: int
) -> list[Pose2D]:
    """Base poses on rings around the target, each facing it."""
    radii = np.linspace(r_min, r_max, rings) if rings > 1 else [r_min]
    poses = []
    for radius in radii:
        for k in range(angles):
            angle = 2.0 * math.pi * k / angles
            x = target_xy[0] + radius * math.cos(angle)
            y = target_xy[1] + radius * math.sin(angle)
            poses.append(Pose2D(x, y, math.atan2(target_xy[1] - y, target_xy[0] - x)))
    return poses


def select_standoff(
    planner: AStarPlanner,
    start: Pose2D,
    target_world: np.ndarray,
    candidates: list[Pose2D],
    reachability: Reachability,
    should_abort: Callable[[], bool] = lambda: False,
    closeness: float = 0.0,
) -> Pose2D | None:
    """The collision-free, arm-reachable candidate with the lowest path cost
    from `start` plus `closeness` x its distance to the target (m of path per m
    of distance: closer stances leave the arm more room), or None."""
    grid = planner.grid
    free = [c for c in candidates if grid.is_free(c.x, c.y)]
    costs = planner.cost_to((start.x, start.y), {grid.world_to_cell(c.x, c.y) for c in free})
    target = np.asarray(target_world, dtype=float)
    ranked = sorted(
        (costs[cell] + closeness * math.hypot(c.x - target[0], c.y - target[1]), i)
        for i, c in enumerate(free)
        if (cell := grid.world_to_cell(c.x, c.y)) in costs
    )
    # Path cost is cheap and IK is not, so check reachability in cost order.
    for _, i in ranked:
        if should_abort():
            return None
        if reachability.is_reachable(free[i], target_world):
            return free[i]
    return None
