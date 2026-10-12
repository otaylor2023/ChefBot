"""OccupancyGrid: a 2D boolean grid in the world frame.

`occupied[row, col]` with row along +y and col along +x. `origin` is the world
position of the outer corner of cell (0, 0). Anything outside the grid is occupied.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml

Cell = tuple[int, int]  # (row, col)


class OccupancyGrid:
    def __init__(self, occupied: np.ndarray, resolution: float, origin: tuple[float, float]):
        self.occupied = np.asarray(occupied, dtype=bool)
        self.resolution = float(resolution)
        self.origin = (float(origin[0]), float(origin[1]))

    @property
    def shape(self) -> tuple[int, int]:
        return self.occupied.shape

    @classmethod
    def load(cls, npy_path: str | Path, yaml_path: str | Path | None = None) -> "OccupancyGrid":
        npy_path = Path(npy_path)
        meta = yaml.safe_load(Path(yaml_path or npy_path.with_suffix(".yaml")).read_text())
        return cls(np.load(npy_path), meta["resolution"], tuple(meta["origin"]))

    def save(self, npy_path: str | Path, yaml_path: str | Path | None = None) -> None:
        npy_path = Path(npy_path)
        np.save(npy_path, self.occupied)
        meta = {"resolution": self.resolution, "origin": list(self.origin), "frame": "world"}
        Path(yaml_path or npy_path.with_suffix(".yaml")).write_text(yaml.safe_dump(meta))

    def world_to_cell(self, x: float, y: float) -> Cell:
        return (
            math.floor((y - self.origin[1]) / self.resolution),
            math.floor((x - self.origin[0]) / self.resolution),
        )

    def cell_to_world(self, cell: Cell) -> tuple[float, float]:
        """World position of the cell center."""
        row, col = cell
        return (
            self.origin[0] + (col + 0.5) * self.resolution,
            self.origin[1] + (row + 0.5) * self.resolution,
        )

    def in_bounds(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.occupied.shape[0] and 0 <= cell[1] < self.occupied.shape[1]

    def is_free_cell(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and not self.occupied[cell]

    def is_free(self, x: float, y: float) -> bool:
        return self.is_free_cell(self.world_to_cell(x, y))

    def segment_free(self, a: tuple[float, float], b: tuple[float, float]) -> bool:
        """True if the straight segment a -> b stays in free cells."""
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        steps = max(1, math.ceil(length / (self.resolution * 0.5)))
        for i in range(steps + 1):
            t = i / steps
            if not self.is_free(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])):
                return False
        return True

    def inflate(self, radius: float) -> "OccupancyGrid":
        """Grow obstacles by `radius`: a cell becomes occupied if its center is
        within `radius` of any part of an occupied cell."""
        if radius <= 0:
            return OccupancyGrid(self.occupied.copy(), self.resolution, self.origin)
        reach = math.ceil(radius / self.resolution)
        rows, cols = self.occupied.shape
        padded = np.zeros((rows + 2 * reach, cols + 2 * reach), dtype=bool)
        padded[reach : reach + rows, reach : reach + cols] = self.occupied
        out = np.zeros_like(self.occupied)
        for dr in range(-reach, reach + 1):
            for dc in range(-reach, reach + 1):
                # Distance from this cell's center to the nearest edge of the offset cell.
                gap = math.hypot(max(abs(dr) - 0.5, 0.0), max(abs(dc) - 0.5, 0.0)) * self.resolution
                if gap < radius:
                    out |= padded[reach + dr : reach + dr + rows, reach + dc : reach + dc + cols]
        return OccupancyGrid(out, self.resolution, self.origin)

    def nearest_free_cell(self, x: float, y: float, max_radius: float) -> Cell | None:
        """The free cell whose center is nearest to (x, y), within `max_radius`."""
        center = self.world_to_cell(x, y)
        if self.is_free_cell(center):
            return center
        reach = math.ceil(max_radius / self.resolution)
        best: Cell | None = None
        best_dist = max_radius
        for row in range(center[0] - reach, center[0] + reach + 1):
            for col in range(center[1] - reach, center[1] + reach + 1):
                if not self.is_free_cell((row, col)):
                    continue
                cx, cy = self.cell_to_world((row, col))
                dist = math.hypot(cx - x, cy - y)
                if dist <= best_dist:
                    best, best_dist = (row, col), dist
        return best
