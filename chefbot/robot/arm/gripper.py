"""Gripper: open / close and grasp verification from the measured width."""

from __future__ import annotations

import math
import threading

from chefbot.interface.env import Env
from chefbot.robot.config import RobotConfig

_STILL = 0.0005  # m per tick: the fingers have stopped
_STILL_TICKS = 3


class Gripper:
    def __init__(self, env: Env, config: RobotConfig):
        self._env = env
        self._cfg = config

    def open(self, preempt: threading.Event) -> bool:
        """Open and wait for the fingers to stop. False if stopped early."""
        return self._command(False, preempt, 0.0)

    def close(self, preempt: threading.Event) -> bool:
        """Close, wait for the fingers to stop, then let the grasp settle."""
        return self._command(True, preempt, self._cfg.grasp_settle_s)

    def width(self) -> float:
        return float(self._env.get_state().gripper_width)

    def is_holding(self) -> bool:
        """Closed on something: commanded shut, but the pads did not meet."""
        state = self._env.get_state()
        return bool(state.gripper_closed_cmd and state.gripper_width > self._cfg.min_holding_width)

    def is_open(self) -> bool:
        return self.width() >= self._cfg.gripper_open_width - 0.01

    def _command(self, closed: bool, preempt: threading.Event, extra_s: float) -> bool:
        env, dt = self._env, self._cfg.control_dt
        env.set_gripper(closed)
        last, still = env.get_state().gripper_width, 0
        for _ in range(math.ceil(self._cfg.gripper_settle_s / dt)):
            if preempt.is_set():
                return False
            env.wait_for_next_tick(dt)
            width = env.get_state().gripper_width
            still = still + 1 if abs(width - last) < _STILL else 0
            last = width
            # The first ticks after a command the fingers have not started moving yet.
            if still >= _STILL_TICKS and self._arrived(closed, width):
                break
        for _ in range(math.ceil(extra_s / dt)):
            if preempt.is_set():
                return False
            env.wait_for_next_tick(dt)
        return True

    def _arrived(self, closed: bool, width: float) -> bool:
        # Opening is done when wide open; closing is done wherever the fingers stall.
        return closed or width >= self._cfg.gripper_open_width - 0.01
