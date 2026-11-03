"""A stand-in robot for dry runs off the robot: a fake Redis, a simulated base and mocap, and the real onboard
server in front of them. The laptop tools (envs/real's RealEnv, scripts/real_nav, real_trials) cannot tell it
from the robot, so the same code paths run, without the hardware.

    with SimRobot(start=(0.285, 0.276, 3.134)) as sim:      # world pose (x, y, yaw) the base starts at
        env = RealEnv(sim.url, np.eye(4))

    python -m chefbot.onboard.sim [--start X Y YAW_DEG]     # as a process: prints its url, Ctrl-C stops it

The simulated base does what driver.py does with commands: it moves toward hb1::desired_pose while hb1::cmd_seq
keeps changing (0.5 s timeout), brakes on hb1::stop, and publishes odometry and a heartbeat; mocap reports its
pose in a world frame offset from odometry, with optional noise. It moves at a constant speed with no
acceleration limits, wheel steering or slip: a check of the software, not of the robot's dynamics.
"""

from __future__ import annotations

import json
import math
import threading
import time

import numpy as np

from chefbot.onboard import keys as K
from chefbot.onboard.frames import compose


class FakeRedis:
    """get / set / mget / incr on a dict of strings, with every write recorded."""

    def __init__(self, values: dict | None = None):
        self.values = dict(values or {})
        self.writes: list[tuple[str, str]] = []

    def get(self, key):
        return self.values.get(key)

    def mget(self, keys):
        return [self.values.get(k) for k in keys]

    def set(self, key, value):
        self.values[key] = str(value)
        self.writes.append((key, str(value)))

    def incr(self, key):
        self.set(key, int(self.values.get(key, 0)) + 1)

    def written(self, key) -> list[str]:
        return [v for k, v in self.writes if k == key]


class SimBase:
    """The base driver and mocap together (see the module docstring)."""

    def __init__(self, redis, world_from_odom=(2.0, 1.0, 0.5), speed=0.5, yaw_rate=1.0, period_s=0.005,
                 timeout_s=0.5, noise_m=0.0, seed=0):  # fmt: skip
        self.redis, self.w = redis, world_from_odom
        self.speed, self.yaw_rate, self.dt, self.timeout, self.noise = speed, yaw_rate, period_s, timeout_s, noise_m
        self.rng = np.random.default_rng(seed)
        self.odom = np.zeros(3)
        self.stopped = threading.Event()
        self.mocap_lost = threading.Event()
        self.publish()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @property
    def world_pose(self) -> np.ndarray:
        return compose(self.w, self.odom)

    def publish(self):
        r = self.redis.values
        r[K.BASE_POSE] = json.dumps(self.odom.tolist())
        r[K.BASE_VELOCITY] = "[0.0, 0.0, 0.0]"
        r[K.BASE_HEARTBEAT] = str(int(r.get(K.BASE_HEARTBEAT, "0")) + 1)
        if not self.mocap_lost.is_set():
            x, y, yaw = self.world_pose
            n = self.rng.normal(0, self.noise, 2) if self.noise else (0.0, 0.0)
            r[K.MOCAP_POSITION] = json.dumps([x + n[0], y + n[1], 0.44 + 1e-9 * float(r[K.BASE_HEARTBEAT])])  # a new frame
            r[K.MOCAP_ORIENTATION] = json.dumps([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])
            r[K.MOCAP_VALID] = "1"

    def _run(self):
        seq, seq_at = self.redis.values.get(K.BASE_CMD_SEQ), None
        while not self.stopped.wait(self.dt):
            now = time.monotonic()
            if self.redis.values.get(K.BASE_CMD_SEQ) != seq:
                seq, seq_at = self.redis.values.get(K.BASE_CMD_SEQ), now
            fresh = seq_at is not None and now - seq_at <= self.timeout and self.redis.values.get(K.BASE_STOP) != "stop"
            if self.redis.values.get(K.BASE_STOP) == "stop":
                self.redis.values[K.BASE_STOP] = "ok"
                seq_at = None
            if fresh:
                goal = np.array(json.loads(self.redis.values[K.BASE_DESIRED_POSE]))
                step = goal - self.odom
                step[2] = math.atan2(math.sin(step[2]), math.cos(step[2]))
                n = np.linalg.norm(step[:2])
                if n > self.speed * self.dt:
                    step[:2] *= self.speed * self.dt / n
                step[2] = np.clip(step[2], -self.yaw_rate * self.dt, self.yaw_rate * self.dt)
                self.odom = self.odom + step
            self.publish()

    def stop(self):
        self.stopped.set()
        self._thread.join()


class _NoCameras:
    cameras: dict = {}

    def grab(self, name):
        raise KeyError(name)

    def close_idle(self):
        pass

    def close(self):
        pass


class SimRobot:
    """FakeRedis + SimBase + the real onboard server on a free local port. `start`: the base's world pose."""

    def __init__(self, start=(0.0, 0.0, 0.0), speed=0.5, yaw_rate=1.0, noise_m=0.0, seed=0):
        from chefbot.onboard.server import build

        config = {
            "server": {"host": "127.0.0.1", "port": 0, "state_hz": 50, "watchdog_s": 0.3, "camera_idle_s": 10},
            "base": {"driver_max_vel_xy": 0.125, "driver_max_vel_yaw": 0.4, "marker_yaw_offset_deg": 0.0,
                     "marker_xy": [0.0, 0.0], "mocap_stale_s": 0.25},
            "gripper": {"speed": 0.05, "force": 20.0},
            "limits": {"base_max_vel_xy": 0.125, "base_max_vel_yaw": 0.4},
            "cameras": [],
        }  # fmt: skip
        self.redis = FakeRedis({K.MOCAP_VALID: "1"})
        self.base = SimBase(self.redis, world_from_odom=tuple(start), speed=speed, yaw_rate=yaw_rate, noise_m=noise_m, seed=seed)
        self.server = build(config, self.redis)
        self.server.cameras = _NoCameras()
        self.server.start()
        self.url = self.server.url

    def close(self):
        self.base.stop()
        self.server.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def main() -> None:
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="A simulated robot behind the real onboard server (see the module docstring).")
    ap.add_argument("--start", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("X", "Y", "YAW_DEG"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sim = SimRobot(start=(args.start[0], args.start[1], math.radians(args.start[2])), seed=args.seed)
    print(sim.url, flush=True)
    try:
        sys.stdin.read()  # until the parent closes stdin, or Ctrl-C
    except KeyboardInterrupt:
        pass
    finally:
        sim.close()


if __name__ == "__main__":
    main()
