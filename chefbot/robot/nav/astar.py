"""AStarPlanner: 8-connected A* on an (already inflated) OccupancyGrid."""

from __future__ import annotations

import heapq
import math

from chefbot.interface.types import Pose2D, wrap_angle
from chefbot.robot.nav.grid import Cell, OccupancyGrid

Point = tuple[float, float]

_SQRT2 = math.sqrt(2.0)
_NEIGHBORS = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, _SQRT2), (-1, 1, _SQRT2), (1, -1, _SQRT2), (1, 1, _SQRT2),
]  # fmt: skip


class NoPath(Exception):
    pass


class AStarPlanner:
    def __init__(self, grid: OccupancyGrid):
        self.grid = grid

    def _neighbors(self, cell: Cell):
        row, col = cell
        free = self.grid.is_free_cell
        for dr, dc, step in _NEIGHBORS:
            nxt = (row + dr, col + dc)
            if not free(nxt):
                continue
            # No cutting corners between two diagonal obstacles.
            if dr and dc and not (free((row + dr, col)) and free((row, col + dc))):
                continue
            yield nxt, step

    def plan_xy(self, start: Point, goal: Point) -> list[Point]:
        """Shortcut path from `start` to `goal` (both must be in free cells).
        Raises NoPath."""
        grid = self.grid
        start_cell, goal_cell = grid.world_to_cell(*start), grid.world_to_cell(*goal)
        if not grid.is_free_cell(start_cell):
            raise NoPath("start is in collision")
        if not grid.is_free_cell(goal_cell):
            raise NoPath("goal is in collision")

        def heuristic(cell: Cell) -> float:
            return math.hypot(cell[0] - goal_cell[0], cell[1] - goal_cell[1])

        cost = {start_cell: 0.0}
        parent: dict[Cell, Cell] = {}
        frontier = [(heuristic(start_cell), start_cell)]
        closed: set[Cell] = set()
        while frontier:
            _, cell = heapq.heappop(frontier)
            if cell == goal_cell:
                break
            if cell in closed:
                continue
            closed.add(cell)
            for nxt, step in self._neighbors(cell):
                new_cost = cost[cell] + step
                if new_cost < cost.get(nxt, math.inf):
                    cost[nxt] = new_cost
                    parent[nxt] = cell
                    heapq.heappush(frontier, (new_cost + heuristic(nxt), nxt))
        else:
            raise NoPath("no collision-free path to the goal")

        cells = [goal_cell]
        while cells[-1] != start_cell:
            cells.append(parent[cells[-1]])
        cells.reverse()
        points = [start] + [grid.cell_to_world(c) for c in cells[1:-1]] + [goal]
        if len(cells) == 1:
            points = [start, goal]
        return self._shortcut(points)

    def _shortcut(self, points: list[Point]) -> list[Point]:
        """Drop waypoints wherever a straight line of sight exists."""
        out = [points[0]]
        i = 0
        while i < len(points) - 1:
            j = len(points) - 1
            while j > i + 1 and not self.grid.segment_free(points[i], points[j]):
                j -= 1
            out.append(points[j])
            i = j
        return out

    def plan(self, start: Pose2D, goal: Pose2D) -> list[Pose2D]:
        return with_yaw(self.plan_xy((start.x, start.y), (goal.x, goal.y)), start.yaw, goal.yaw)

    def cost_to(self, start: Point, targets: set[Cell]) -> dict[Cell, float]:
        """Path cost in meters from `start` to each reachable cell in `targets`
        (one Dijkstra sweep; unreachable targets are absent)."""
        start_cell = self.grid.world_to_cell(*start)
        if not self.grid.is_free_cell(start_cell):
            return {}
        remaining = set(targets)
        found: dict[Cell, float] = {}
        cost = {start_cell: 0.0}
        frontier = [(0.0, start_cell)]
        closed: set[Cell] = set()
        while frontier and remaining:
            dist, cell = heapq.heappop(frontier)
            if cell in closed:
                continue
            closed.add(cell)
            if cell in remaining:
                remaining.discard(cell)
                found[cell] = dist * self.grid.resolution
            for nxt, step in self._neighbors(cell):
                new_cost = dist + step
                if new_cost < cost.get(nxt, math.inf):
                    cost[nxt] = new_cost
                    heapq.heappush(frontier, (new_cost, nxt))
        return found


def path_length(points: list[Point]) -> float:
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


def with_yaw(points: list[Point], start_yaw: float, goal_yaw: float) -> list[Pose2D]:
    """Attach yaws: the final waypoint carries the goal yaw, intermediate yaws
    interpolate along the shortest rotation by arc length."""
    total = path_length(points)
    delta = wrap_angle(goal_yaw - start_yaw)
    poses = []
    travelled = 0.0
    for i, (x, y) in enumerate(points):
        if i > 0:
            travelled += math.dist(points[i - 1], (x, y))
        fraction = travelled / total if total > 0 else float(i == len(points) - 1)
        poses.append(Pose2D(x, y, wrap_angle(start_yaw + fraction * delta)))
    poses[-1] = Pose2D(points[-1][0], points[-1][1], wrap_angle(goal_yaw))
    return poses
