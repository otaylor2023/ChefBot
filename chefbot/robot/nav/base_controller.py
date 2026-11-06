"""BaseController: follows a path of Pose2D waypoints with world-frame velocity
commands, or, for a base that drives to a pose itself, with a moving pose target."""

from __future__ import annotations

import math

import numpy as np

from chefbot.interface.types import Pose2D, wrap_angle
from chefbot.robot.config import RobotConfig


class BaseController:
    def __init__(self, path: list[Pose2D], config: RobotConfig):
        if len(path) < 2:
            raise ValueError("path needs at least a start and a goal")
        self._cfg = config
        self._path = path
        self._points = np.array([[p.x, p.y] for p in path])
        self._cum = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(self._points, axis=0), axis=1))]
        )
        self._progress = 0.0  # arc length reached so far; never moves backward
        self._deviation = 0.0
        self._cmd = np.zeros(3)

    @property
    def length(self) -> float:
        """The path's length (m)."""
        return float(self._cum[-1])

    @property
    def goal(self) -> Pose2D:
        return self._path[-1]

    @property
    def deviation(self) -> float:
        """Distance from the base to the path at the last `step`."""
        return self._deviation

    def at_goal(self, pose: Pose2D) -> bool:
        return (
            pose.distance_to(self.goal) <= self._cfg.pos_tolerance
            and abs(wrap_angle(self.goal.yaw - pose.yaw)) <= self._cfg.yaw_tolerance
        )

    def _point_at(self, s: float) -> np.ndarray:
        s = min(max(s, 0.0), self._cum[-1])
        return np.array(
            [np.interp(s, self._cum, self._points[:, 0]), np.interp(s, self._cum, self._points[:, 1])]
        )

    def _yaw_at(self, s: float) -> float:
        total = self._cum[-1]
        if s >= total - 1e-9:
            return self.goal.yaw
        i = min(int(np.searchsorted(self._cum, s, side="right")) - 1, len(self._path) - 2)
        span = self._cum[i + 1] - self._cum[i]
        frac = (s - self._cum[i]) / span if span > 0 else 1.0
        a, b = self._path[i].yaw, self._path[i + 1].yaw
        return wrap_angle(a + frac * wrap_angle(b - a))

    def _project(self, pos: np.ndarray) -> None:
        """Update progress and deviation from the closest point on the path ahead."""
        best_dist, best_s = math.inf, self._progress
        for i in range(len(self._points) - 1):
            if self._cum[i + 1] < self._progress:
                continue
            a, b = self._points[i], self._points[i + 1]
            ab = b - a
            span = self._cum[i + 1] - self._cum[i]
            t = float(np.clip(np.dot(pos - a, ab) / (span * span), 0.0, 1.0)) if span > 0 else 0.0
            s = max(self._cum[i] + t * span, self._progress)
            dist = float(np.linalg.norm(pos - self._point_at(s)))
            if dist < best_dist:
                best_dist, best_s = dist, s
        self._progress, self._deviation = best_s, best_dist

    def target(self, pose: Pose2D) -> Pose2D:
        """For a base that drives to a pose itself: the carrot on the path, with the
        path's heading there. It reaches the goal at the end of the path."""
        self._project(pose.xy)
        s = self._progress + self._cfg.lookahead
        carrot = self._point_at(s)
        return Pose2D(float(carrot[0]), float(carrot[1]), self._yaw_at(s))

    def target_at(self, pose: Pose2D, lookahead: float) -> Pose2D:
        """`target` with its own lookahead (a faster base needs its carrot further ahead)."""
        self._project(pose.xy)
        s = self._progress + lookahead
        carrot = self._point_at(s)
        return Pose2D(float(carrot[0]), float(carrot[1]), self._yaw_at(s))

    def step(self, pose: Pose2D, dt: float, lookahead: float | None = None) -> tuple[float, float, float]:
        """World-frame (vx, vy, vyaw) for this tick, toward the carrot `lookahead` ahead (default the config's)."""
        cfg = self._cfg
        pos = pose.xy
        self._project(pos)
        total = self._cum[-1]

        carrot = self._point_at(self._progress + (cfg.lookahead if lookahead is None else lookahead))
        to_carrot = carrot - pos
        dist = float(np.linalg.norm(to_carrot))
        remaining = self._deviation + (total - self._progress)
        speed = min(
            cfg.max_lin_vel, cfg.kp_pos * remaining, math.sqrt(2.0 * cfg.max_lin_acc * remaining)
        )
        lin = to_carrot / dist * speed if dist > 1e-9 else np.zeros(2)

        yaw_err = wrap_angle(self._yaw_at(self._progress) - pose.yaw)
        ang = float(np.clip(cfg.kp_yaw * yaw_err, -cfg.max_ang_vel, cfg.max_ang_vel))

        # Acceleration limits, applied to the change in command.
        d_lin = lin - self._cmd[:2]
        max_d_lin = cfg.max_lin_acc * dt
        norm = float(np.linalg.norm(d_lin))
        if norm > max_d_lin:
            d_lin *= max_d_lin / norm
        max_d_ang = cfg.max_ang_acc * dt
        d_ang = float(np.clip(ang - self._cmd[2], -max_d_ang, max_d_ang))
        self._cmd = np.array([self._cmd[0] + d_lin[0], self._cmd[1] + d_lin[1], self._cmd[2] + d_ang])
        return float(self._cmd[0]), float(self._cmd[1]), float(self._cmd[2])
