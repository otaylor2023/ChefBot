"""Planning a base path that is clear for a footprint that is not a circle, at the heading the base will have along it
(robot/nav/footprint.py FootprintSpace): the base turns from its heading to the goal's in proportion to the way gone.

    begin = escape_start(plan.at(start.yaw), body.at(start.yaw), (start.x, start.y))
    points, why = plan_at_headings(plan, body, start, begin, goal, goal.yaw)
    path = headings(start, subdivide(points, 1.0), goal.yaw)      # the poses the base drives through
"""

from __future__ import annotations

import math

from chefbot.interface.types import Pose2D
from chefbot.robot.nav.astar import AStarPlanner, NoPath, with_yaw
from chefbot.robot.nav.grid import OccupancyGrid


def escape_start(plan_grid, body_grid, start, max_m: float = 0.3):
    """Where planning starts: `start` itself if it is clear with the margin, else the nearest cell that is (within
    `max_m`) when the straight move there is clear for the body itself; None if neither."""
    if plan_grid.is_free(start[0], start[1]):
        return tuple(start)
    cell = plan_grid.nearest_free_cell(start[0], start[1], max_m)
    if cell is None:
        return None
    p = plan_grid.cell_to_world(cell)
    return tuple(p) if body_grid.segment_free(tuple(start), tuple(p)) else None


def subdivide(points, max_segment: float) -> list:
    """Split segments longer than `max_segment` into equal straight pieces (the same path, more waypoints)."""
    out = [tuple(points[0])]
    for a, b in zip(points, points[1:]):
        n = max(1, math.ceil(math.dist(a, b) / max_segment))
        out += [(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) for k in range(1, n + 1)]
    return out


def headings(start: Pose2D, points, goal_yaw: float) -> list[Pose2D]:
    """The poses the base drives through: out of the margin first (if points[0] is not the start) at its own heading,
    then turning to goal_yaw in proportion to the way gone along `points`."""
    poses = with_yaw(list(points), start.yaw, goal_yaw)
    if tuple(points[0]) != (start.x, start.y):
        poses = [start] + poses
    return poses


def plan_at_headings(plan, space, start: Pose2D, begin, pre: Pose2D, goal_yaw: float, repairs: int = 200):
    """A* from `begin` to `pre` whose path is clear for the footprint at the heading the base will have at each point
    (turning from start.yaw to goal_yaw in proportion to the way gone): with `plan`'s margin from `begin` on, and for
    the body itself (`space`) from `start` (out of the margin first, if begin is not start). A* starts on the grid
    blocked only where every heading of the turn is; where the path then meets a cell not clear at its heading, that
    cell is blocked and A* runs again. (points, None) or (None, reason)."""
    g = plan.blocked_for_all(plan.sweep(start.yaw, goal_yaw))
    occ = g.occupied.copy()
    escaped = tuple(begin) != (start.x, start.y)
    keep = {g.world_to_cell(*begin), g.world_to_cell(pre.x, pre.y)}
    for _ in range(repairs):
        try:
            found = AStarPlanner(OccupancyGrid(occ, g.resolution, g.origin)).plan_xy(tuple(begin), (pre.x, pre.y))
        except NoPath as exc:
            return None, f"no path: {exc}"
        poses = [(p.x, p.y, p.yaw) for p in headings(start, found, goal_yaw)]
        bad = plan.first_blocked(poses[1:] if escaped else poses) or space.first_blocked(poses)
        if bad is None:
            return found, None
        r0, c0 = g.world_to_cell(bad[0], bad[1])
        # The cell, or (a point on a cell's edge falls in either) the free cells around it; never the start's or goal's.
        near = [(r0, c0)] if g.in_bounds((r0, c0)) and not occ[r0, c0] else [
            (r, c) for r in (r0 - 1, r0, r0 + 1) for c in (c0 - 1, c0, c0 + 1) if g.in_bounds((r, c)) and not occ[r, c]]
        near = [cell for cell in near if cell not in keep]
        if not near:
            break  # no progress (the turn itself, at the start or the goal, is the problem)
        for cell in near:
            occ[cell] = True
    return None, "no path: none clear for the footprint at the headings along it"
