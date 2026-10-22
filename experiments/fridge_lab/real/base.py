"""The TidyBot++ base for the real driver: tidybot2-arx's base_server.py on the NUC, unchanged.

Its pose is the base center's (x, y, yaw) in wheel odometry, zeroed when `reset()` starts the
controller; a target pose is driven to with the trajectory limits given at connection.
"""

from __future__ import annotations

import math
import time
from multiprocessing.managers import BaseManager

import numpy as np


class TidybotBase:
    def __init__(self, host: str, port: int, authkey: bytes, max_vel, max_accel):
        class _M(BaseManager):
            pass

        _M.register("Base")
        m = _M(address=(host, port), authkey=authkey)
        try:
            m.connect()
        except ConnectionRefusedError as e:
            raise RuntimeError(f"no base server at {host}:{port}: is base_server.py running on the NUC?") from e
        self.r = m.Base(max_vel=tuple(max_vel), max_accel=tuple(max_accel))

    def reset(self) -> None:
        """(Re)start the controller; odometry restarts at zero."""
        self.r.reset()

    def pose(self) -> np.ndarray:
        return np.asarray(self.r.get_state()["base_pose"], dtype=float)

    def drive_to(self, pose) -> None:
        self.r.execute_action({"base_pose": np.asarray(pose, dtype=float)})

    def close(self) -> None:
        self.r.close()


class FakeBase:
    """No hardware: drives straight to each target at the speed limits."""

    def __init__(self, max_vel, max_accel=None):
        self.vmax = np.asarray(max_vel, dtype=float)
        self.x, self.target, self.t = np.zeros(3), np.zeros(3), time.time()

    def _update(self) -> None:
        dt, self.t = time.time() - self.t, time.time()
        err = self.target - self.x
        err[2] = (err[2] + math.pi) % (2 * math.pi) - math.pi
        self.x += np.clip(err, -self.vmax * dt, self.vmax * dt)

    def reset(self) -> None:
        self.x, self.target, self.t = np.zeros(3), np.zeros(3), time.time()

    def pose(self) -> np.ndarray:
        self._update()
        return self.x.copy()

    def drive_to(self, pose) -> None:
        self._update()
        self.target = np.asarray(pose, dtype=float)

    def close(self) -> None:
        pass


def make_base(cfg: dict):
    b = cfg["base"]
    if cfg.get("dry_run"):
        return FakeBase(b["max_vel"])
    n = cfg["nuc"]
    return TidybotBase(n["host"], int(n["base_port"]), str(n["authkey"]).encode(), b["max_vel"], b["max_accel"])
