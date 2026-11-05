"""Planning a base path that is clear for a footprint that is not a circle, at the heading the base will have along it
(robot/nav/footprint.py FootprintSpace): the base turns from its heading to the goal's in proportion to the way gone.

    begin = escape_start(plan.at(start.yaw), body.at(start.yaw), (start.x, start.y))
    points, why = plan_at_headings(plan, body, start, begin, goal, goal.yaw)
    path = headings(start, subdivide(points, 1.0), goal.yaw)      # the poses the base drives through
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import numpy as np

from chefbot.interface.types import Pose2D, wrap_angle
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
    out = _subdivide(points, max_segment)
    if getattr(points, "turn", None) is not None:
        out = _Points(out)
        out.turn = points.turn
    return out


def _subdivide(points, max_segment: float) -> list:
    """Split segments longer than `max_segment` into equal straight pieces (the same path, more waypoints)."""
    out = [tuple(points[0])]
    for a, b in zip(points, points[1:]):
        n = max(1, math.ceil(math.dist(a, b) / max_segment))
        out += [(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) for k in range(1, n + 1)]
    return out


@dataclass(frozen=True)
class PathStyle:
    """How paths are shaped (both off by default: the shortest path, the turn spread evenly).
    clearance_m: A* prefers keeping this far from obstacles (a cost that rises closer in), so paths run down the middle
                 of aisles instead of hugging their corners.
    turn_in_room: the heading turns where the path has room (weighted by (clearance - need_m)^3 along it), not evenly.
    need_m:      the clearance a turn needs (about the footprint's corner radius plus the margin)."""

    clearance_m: float = 0.0
    turn_in_room: bool = False
    need_m: float = 0.42


_CLEARANCE: dict = {}


def clearance_m(space) -> np.ndarray:
    """Distance (m) from each of space.grid's cells to the nearest obstacle of any of `space`'s layers (cached)."""
    key = tuple(id(g) for g, _ in space.layers)
    if key in _CLEARANCE:
        return _CLEARANCE[key]
    occ = np.zeros(space.grid.shape, dtype=bool)
    for g, _ in space.layers:
        occ |= g.occupied
    big = 1e9
    d = np.where(occ, 0.0, big)
    h, w = d.shape
    s2 = math.sqrt(2.0)
    for r in range(h):  # chamfer, forward
        for c in range(w):
            if d[r, c] == 0.0:
                continue
            best = d[r, c]
            if r > 0:
                best = min(best, d[r - 1, c] + 1.0)
                if c > 0:
                    best = min(best, d[r - 1, c - 1] + s2)
                if c < w - 1:
                    best = min(best, d[r - 1, c + 1] + s2)
            if c > 0:
                best = min(best, d[r, c - 1] + 1.0)
            d[r, c] = best
    for r in range(h - 1, -1, -1):  # backward
        for c in range(w - 1, -1, -1):
            best = d[r, c]
            if r < h - 1:
                best = min(best, d[r + 1, c] + 1.0)
                if c > 0:
                    best = min(best, d[r + 1, c - 1] + s2)
                if c < w - 1:
                    best = min(best, d[r + 1, c + 1] + s2)
            if c < w - 1:
                best = min(best, d[r, c + 1] + 1.0)
            d[r, c] = best
    out = d * space.grid.resolution  # metres (a plain array on space.grid's cells: OccupancyGrid would make it bool)
    _CLEARANCE[key] = out
    return out


def _penalty(space, style: PathStyle | None):
    if style is None or style.clearance_m <= 0:
        return None
    c = clearance_m(space)
    return 4.0 * np.clip((style.clearance_m - c) / style.clearance_m, 0.0, 1.0) ** 2


def _min_clearance(clr: np.ndarray, grid: OccupancyGrid, a, b) -> float:
    n = max(2, int(math.dist(a, b) / grid.resolution) + 1)
    out = math.inf
    for k in range(n + 1):
        t = k / n
        cell = grid.world_to_cell(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        if grid.in_bounds(cell):
            out = min(out, float(clr[cell]))
    return out


class _Points(list):
    """plan_at_headings' points, with the turn they were planned for (signed rad, or None: the short way)."""

    turn: float | None = None


def _turn(start_yaw: float, goal_yaw: float, turn: float | None) -> float:
    return wrap_angle(goal_yaw - start_yaw) if turn is None else turn


def headings(start: Pose2D, points, goal_yaw: float, style: PathStyle | None = None, space=None) -> list[Pose2D]:
    """The poses the base drives through: out of the margin first (if points[0] is not the start) at its own heading,
    then turning to goal_yaw in proportion to the way gone along `points` (with style.turn_in_room and a `space`: in
    proportion to the room along it, so the turn happens where there is space for it)."""
    turn = getattr(points, "turn", None)
    points = list(points)
    if style is not None and style.turn_in_room and space is not None and len(points) > 2:
        clr = clearance_m(space)
        weights = [max(0.0, _min_clearance(clr, space.grid, a, b) - style.need_m) ** 3 * math.dist(a, b) for a, b in zip(points, points[1:])]
        total = sum(weights)
        if total > 1e-9:
            delta = _turn(start.yaw, goal_yaw, turn)
            poses, done = [Pose2D(points[0][0], points[0][1], start.yaw)], 0.0
            for (x, y), w in zip(points[1:], weights):
                done += w
                poses.append(Pose2D(x, y, wrap_angle(start.yaw + delta * done / total)))
            poses[-1] = Pose2D(points[-1][0], points[-1][1], wrap_angle(goal_yaw))
            if tuple(points[0]) != (start.x, start.y):
                poses = [start] + poses
            return poses
    poses = with_yaw(points, start.yaw, goal_yaw)
    if tuple(points[0]) != (start.x, start.y):
        poses = [start] + poses
    return poses


def plan_at_headings(plan, space, start: Pose2D, begin, pre: Pose2D, goal_yaw: float, repairs: int = 200,
                     style: PathStyle | None = None):  # fmt: skip
    """As _plan_at_headings; with style.turn_in_room, a turn of more than 150 deg that finds no path the short way is
    tried the other way round too (the arm then sweeps the other side)."""
    if style is None or not style.turn_in_room:
        return _plan_at_headings(plan, space, start, begin, pre, goal_yaw, repairs, style, None, "sweep")
    short = wrap_angle(goal_yaw - start.yaw)
    turns = [None] + ([short - math.copysign(2 * math.pi, short)] if abs(short) > math.radians(150) else [])
    best, why = None, None
    for search in ("sweep", "ends"):  # the whole turn's union (clean paths), or just its ends (when that closes aisles)
        for turn in turns:
            found, w = _plan_at_headings(plan, space, start, begin, pre, goal_yaw, repairs, style, turn, search)
            why = why or w
            if found is not None and (best is None or _length(found) < _length(best)):
                best = found
        if best is not None:
            return best, None
    return None, why


def _length(points) -> float:
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


def _plan_at_headings(plan, space, start: Pose2D, begin, pre: Pose2D, goal_yaw: float, repairs: int,
                      style: PathStyle | None, turn: float | None, search: str = "sweep"):  # fmt: skip
    """A* from `begin` to `pre` whose path is clear for the footprint at the heading the base will have at each point
    (turning from start.yaw to goal_yaw in proportion to the way gone): with `plan`'s margin from `begin` on, and for
    the body itself (`space`) from `start` (out of the margin first, if begin is not start). A* starts on the grid
    blocked only where every heading of the turn is; where the path then meets a cell not clear at its heading, that
    cell is blocked and A* runs again. (points, None) or (None, reason)."""
    sweep_space = plan
    if style is not None and style.turn_in_room and getattr(plan, "heading_zones", None):
        # The search grid leaves the heading zones out (a zone blocks every heading but a few, so the whole turn's
        # union would close it); the poses are still checked against them below, the turn placed where there is room.
        from chefbot.robot.nav.footprint import FootprintSpace

        sweep_space = FootprintSpace(plan.grid, None, plan.margin, 360.0 / plan.n, plan.layers, plan.centre_limit)
    if search == "ends":
        # The turn happens where there is room (headings, below), so the search grid holds only the start's and the
        # goal's headings; every pose's own heading is checked against the full space below (and repaired).
        yaws = [start.yaw, goal_yaw]
    elif turn is None:
        yaws = plan.sweep(start.yaw, goal_yaw)
    else:
        n = max(1, math.ceil(abs(turn) / plan.half))
        yaws = [start.yaw + turn * i / n for i in range(n + 1)]
    g = sweep_space.blocked_for_all(yaws)
    occ = g.occupied.copy()
    escaped = tuple(begin) != (start.x, start.y)
    keep = {g.world_to_cell(*begin), g.world_to_cell(pre.x, pre.y)}
    penalty = _penalty(space, style)
    for _ in range(repairs):
        try:
            found = AStarPlanner(OccupancyGrid(occ, g.resolution, g.origin), penalty).plan_xy(tuple(begin), (pre.x, pre.y))
        except NoPath as exc:
            return None, f"no path: {exc}"
        found = _Points(found)
        found.turn = turn
        sub = _Points(subdivide(found, 1.0))
        sub.turn = turn
        poses = [(p.x, p.y, p.yaw) for p in headings(start, sub, goal_yaw, style, space)]
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
