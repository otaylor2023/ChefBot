"""Where the base's centre may be, at each heading, for a footprint that is not a circle (interface/workspace.py
Footprint: the plate's outline and the arm): a map's obstacles grown by the footprint turned to that heading.

A slice covers a band of headings (`step_deg` wide): it is grown by the footprint at the band's centre, plus how far
any part of it moves turning half the band, plus `margin`, and an obstacle cell counts whole (half its diagonal), so a
slice is conservative for every heading in its band. Slices are made when first asked for, then kept.

    space = FootprintSpace(grid, footprint, margin=0.09)
    space.at(yaw)                       # an OccupancyGrid: free where the base's centre may be at that heading
    space.blocked_for_all(yaws)         # free where it may be at some heading in the list (for A*; check the path after)
    space.path_clear(poses)             # every pose (x, y, yaw), and the way between them, clear at its own heading

Heights: obstacles of different heights need different footprints (a counter top: the plate, and only the arm's points
low enough to hit it; the arm passes over it). FootprintSpace.by_height(grid, footprint_below) groups the map's obstacles
by their top (OccupancyGrid.heights) and grows each group by footprint_below(top) (inf: walls, the whole footprint).
FootprintSpace.layered([(grid, footprint), ...]) is the general form: each grown by its own, blocked where any blocks.
"""

from __future__ import annotations

import math

import numpy as np

from chefbot.interface.types import wrap_angle
from chefbot.robot.nav.grid import OccupancyGrid


class FootprintSpace:
    def __init__(self, grid: OccupancyGrid, footprint, margin: float = 0.0, step_deg: float = 2.0, layers=None,
                 centre_limit: OccupancyGrid | None = None):  # fmt: skip
        """`centre_limit`: cells the base's centre itself may not enter (a fence), not grown by anything."""
        self.layers = list(layers) if layers is not None else [(grid, footprint)]
        self.grid = self.layers[0][0]
        self.margin = float(margin)
        self.centre_limit = centre_limit
        self.n = max(1, round(360.0 / step_deg))
        self.half = math.pi / self.n  # half a band (rad)
        self._slices: dict[int, OccupancyGrid] = {}

    @classmethod
    def layered(cls, layers, margin: float = 0.0, step_deg: float = 2.0, centre_limit=None) -> "FootprintSpace":
        """Each (grid, footprint) grown by its own footprint; a cell is blocked where any layer blocks it. The grids
        share resolution, origin and shape."""
        layers = list(layers)
        return cls(layers[0][0], layers[0][1], margin, step_deg, layers, centre_limit)

    @classmethod
    def by_height(cls, grid: OccupancyGrid, footprint_below, margin: float = 0.0, step_deg: float = 2.0,
                  centre_limit=None) -> "FootprintSpace":  # fmt: skip
        """`grid`'s obstacles grouped by their top, each group grown by footprint_below(top): the footprint of what
        the robot has lower than that (top inf: all of it)."""
        tops = grid.tops() or [math.inf]
        return cls.layered([(grid.only_top(t), footprint_below(t)) for t in tops], margin, step_deg, centre_limit)

    def with_margin(self, margin: float) -> "FootprintSpace":
        return FootprintSpace(self.grid, None, margin, 360.0 / self.n, self.layers, self.centre_limit)

    @property
    def radius(self) -> float:
        """The farthest any layer's footprint reaches from the base's centre (m)."""
        return max(fp.radius for _, fp in self.layers)

    def band(self, yaw: float) -> int:
        return int(round(wrap_angle(yaw) / (2 * self.half))) % self.n

    def at(self, yaw: float) -> OccupancyGrid:
        """Free where the base's centre may be at heading `yaw` (every heading in its band)."""
        b = self.band(yaw)
        if b not in self._slices:
            centre = b * 2 * self.half
            occ = np.zeros(self.grid.shape, dtype=bool)
            for g, fp in self.layers:
                occ |= _grow(g, fp, centre, self.half, self.margin)
            if self.centre_limit is not None:
                occ |= self.centre_limit.occupied
            self._slices[b] = OccupancyGrid(occ, self.grid.resolution, self.grid.origin)
        return self._slices[b]

    def blocked_for_all(self, yaws) -> OccupancyGrid:
        """Blocked only where blocked at every one of the headings (a permissive grid for A*: check the path with
        path_clear after)."""
        occ = np.ones(self.grid.shape, dtype=bool)
        for y in yaws:
            occ &= self.at(y).occupied
        return OccupancyGrid(occ, self.grid.resolution, self.grid.origin)

    def blocked_for_any(self, yaws) -> OccupancyGrid:
        """Blocked where blocked at any of the headings (conservative over all of them)."""
        occ = np.zeros(self.grid.shape, dtype=bool)
        for y in yaws:
            occ |= self.at(y).occupied
        return OccupancyGrid(occ, self.grid.resolution, self.grid.origin)

    def is_free(self, x: float, y: float, yaw: float) -> bool:
        return self.at(yaw).is_free(x, y)

    def path_clear(self, poses) -> bool:
        """Poses (x, y, yaw) in order: each, and the way to the next (sampled every half cell, the heading moving
        the short way in proportion), clear at its own heading."""
        return self.first_blocked(poses) is None

    def first_blocked(self, poses):
        """The first point (x, y, yaw) along the poses (as path_clear samples them) that is not clear, or None."""
        poses = [tuple(map(float, p)) for p in poses]
        if len(poses) == 1:
            return None if self.is_free(*poses[0]) else poses[0]
        step = self.grid.resolution / 2
        for k, ((x0, y0, a0), (x1, y1, a1)) in enumerate(zip(poses, poses[1:])):
            turn = wrap_angle(a1 - a0)
            n = max(1, math.ceil(max(math.hypot(x1 - x0, y1 - y0) / step, abs(turn) / self.half)))
            for i in range(0 if k == 0 else 1, n + 1):
                t = i / n
                p = (x0 + t * (x1 - x0), y0 + t * (y1 - y0), a0 + t * turn)
                if not self.is_free(*p):
                    return p
        return None

    def sweep(self, yaw0: float, yaw1: float) -> list[float]:
        """The band centres a turn from yaw0 to yaw1 the short way passes through."""
        turn = wrap_angle(yaw1 - yaw0)
        n = max(1, math.ceil(abs(turn) / self.half))
        return [yaw0 + turn * i / n for i in range(n + 1)]


def _grow(grid: OccupancyGrid, footprint, yaw: float, half: float, margin: float) -> np.ndarray:
    """Cells where the base's centre at heading yaw (+- half) would put `footprint` within `margin` of an obstacle."""
    res = grid.resolution
    reach = footprint.radius + margin + res
    k = math.ceil(reach / res)
    dc, dr = np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1))
    off = np.column_stack([dc.ravel() * res, dr.ravel() * res])  # obstacle cell centre - base centre (world x, y)
    c, s = math.cos(yaw), math.sin(yaw)
    local = off @ np.array([[c, -s], [s, c]])  # into the base frame: R(-yaw) applied to each row
    rho = np.linalg.norm(off, axis=1)
    # The footprint's distance changes by at most rho * half over the band; the obstacle cell's square by half its
    # diagonal.
    hit = footprint.distance(local) - res * math.sqrt(0.5) - rho * half < margin
    rows, cols = grid.occupied.shape
    padded = np.zeros((rows + 2 * k, cols + 2 * k), dtype=bool)
    padded[k : k + rows, k : k + cols] = grid.occupied
    out = np.zeros_like(grid.occupied)
    for (ddc, ddr) in zip(dc.ravel()[hit], dr.ravel()[hit]):
        out |= padded[k + ddr : k + ddr + rows, k + ddc : k + ddc + cols]
    return out
