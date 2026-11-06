"""How fast the base may go where it is: never faster than it can stop in the free space ahead of it.

The stop model is the robot's own measurement (the onboard server's limits, onboard/config.yaml): the distance from
the last fresh command to rest, timeout included, d(v) = v * tau + v^2 / (2 * brake). Free space is measured on the
footprint-grown grid (a cell is free when the whole footprint, arm included, fits there), along the direction the base
is moving, so travel along a counter is not slowed by the counter beside it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from chefbot.robot.nav.grid import OccupancyGrid


@dataclass(frozen=True)
class StopModel:
    tau_s: float  # s at speed before braking starts (command timeout and the loop), from the measured stop
    brake_mps2: float  # m/s^2 while braking

    @classmethod
    def from_limits(cls, limits: dict) -> "StopModel":
        """From the server's hello limits; refuses to guess when the robot reports no measured stop."""
        try:
            return cls(float(limits["base_stop_tau_s"]), float(limits["base_stop_brake_mps2"]))
        except (KeyError, TypeError, ValueError):
            raise ValueError("the robot reports no measured stop (limits base_stop_tau_s, base_stop_brake_mps2)") from None

    def distance(self, v: float) -> float:
        v = max(float(v), 0.0)
        return v * self.tau_s + v * v / (2.0 * self.brake_mps2)

    def speed_for(self, free_m: float) -> float:
        """The fastest speed that stops within `free_m` (0 when there is none)."""
        if free_m <= 0.0:
            return 0.0
        a, t = self.brake_mps2, self.tau_s
        return a * (-t + math.sqrt(t * t + 2.0 * free_m / a))


def free_distance(grid: OccupancyGrid, xy, direction, max_m: float = 3.0) -> float:
    """How far the base centre can go from `xy` along `direction` before the footprint-grown `grid` is occupied
    (stepping a third of a cell); `max_m` if nothing is in the way. 0 if it is already occupied."""
    d = np.asarray(direction, dtype=float)[:2]
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return max_m
    d = d / n
    step = grid.resolution / 3.0
    p0 = np.asarray(xy, dtype=float)[:2]
    s = 0.0
    while s < max_m:
        p = p0 + d * s
        if not grid.is_free(float(p[0]), float(p[1])):
            return s
        s += step
    return max_m


def lookahead_for(speed: float, accel: float, tau_s: float, minimum: float = 0.12) -> float:
    """A carrot far enough ahead that a driver which plans to come to rest at it can still hold `speed` (its stopping
    distance at its own acceleration, plus the command lag), never shorter than `minimum`."""
    return max(minimum, speed * speed / (2.0 * accel) + speed * tau_s + 0.05)
