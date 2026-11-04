"""ChefBot's hb1 base driver: the TidyBot++ Vehicle on the robot's Redis, with a command timeout.

    ~/miniforge3/envs/zitibot/bin/python chefbot/onboard/drivers/base/driver.py [--max-vel-xy 0.125] [--max-vel-yaw 0.4]

Same Redis contract as ZitiBot's controllers/tidybot_base/redis_driver.py (7530046), which it replaces for ChefBot:
it publishes hb1::current_pose / ::current_vel (wheel odometry, [x, y, yaw]) and drives to hb1::desired_pose
(odometry frame) within hb1::max_vel_scale x its limits; hb1::stop = "stop" brakes and holds (reset to "ok" once
still); hb1::kill = "kill" stops it. base_controller.py, constants.py and utils.py next to this file are copied
unchanged from that commit (README.md).

What differs, for safety:
- Commands expire. A target counts only if hb1::cmd_seq changed within --command-timeout (0.5 s); the onboard
  server bumps it with every base target (robot_io.base_target). Without fresh commands the base brakes and holds
  where it is, so a dead client, server or Wi-Fi link stops it. (ZitiBot's driver re-sent the last goal forever.)
  Starting up, the value already in Redis is never fresh.
- hb1::heartbeat counts up every loop, braking or not, so readers can tell a live driver from odometry left in
  Redis by an earlier run (Redis keeps it, across reboots).
- State is published every loop, also while braking; kill exits the loop.
- Braking (no fresh command, hb1::stop, a garbled goal) uses --brake-accel (0.25 m/s^2, 0.8 rad/s^2: the gamepad
  teleop's own limits, which Jack drove with; and 3x the jerk limit where the Vehicle has one: this copy, ZitiBot's, has
  none), not the gentle motion limits (0.1 m/s^2): HT-4c
  measured 66 mm / 1.1 s to rest from 0.12 m/s with those; the worst case at 0.15 m/s was ~19 cm, now ~9 cm with the
  0.3 s timeout.
- The planar speed limit holds for the base's speed, not each axis: Vehicle's trajectory limits x and y
  separately, which let a diagonal go sqrt(2) faster (0.167 m/s measured in HT-3b against 0.125). Each tick the
  x / y limits are set along the direction to the goal, so their combination is at most --max-vel-xy.
"""

from __future__ import annotations

import ast
import time

import numpy as np

POSE, VEL, DESIRED, STOP, KILL = "hb1::current_pose", "hb1::current_vel", "hb1::desired_pose", "hb1::stop", "hb1::kill"
MAX_VEL_SCALE, CMD_SEQ, HEARTBEAT = "hb1::max_vel_scale", "hb1::cmd_seq", "hb1::heartbeat"
SCALE_RANGE = (0.1, 1.0)  # never faster than the CLI limits (ZitiBot allowed up to 3x)
STILL = 1e-3  # |dx| below this counts as stopped


def _list(v) -> str:
    return "[" + ", ".join(str(float(x)) for x in np.asarray(v).reshape(-1)) + "]"


def _parse(raw) -> np.ndarray | None:
    try:
        value = ast.literal_eval(raw) if isinstance(raw, str) else None
    except (SyntaxError, ValueError):
        return None
    if isinstance(value, list) and len(value) == 3 and all(isinstance(x, (int, float)) for x in value):
        a = np.asarray(value, dtype=float)
        return a if np.all(np.isfinite(a)) else None
    return None


class DriverLoop:
    """One `step()` per tick. `vehicle`: x, dx, max_vel, otg_inp, set_target_position, set_target_velocity."""

    def __init__(self, vehicle, redis, command_timeout_s: float = 0.3, clock=time.monotonic,
                 brake_accel=(0.25, 0.25, 0.8), brake_jerk_factor: float = 3.0):  # fmt: skip
        self.v, self.r, self.timeout, self.clock = vehicle, redis, float(command_timeout_s), clock
        self.baseline_max_vel = np.array(vehicle.max_vel, dtype=float)
        self.motion_accel = np.array(getattr(vehicle, "max_accel", brake_accel), dtype=float)
        self.brake_accel = np.maximum(np.array(brake_accel, dtype=float), self.motion_accel)  # never softer than motion
        jerk = getattr(vehicle, "max_jerk", None)
        self.motion_jerk = None if jerk is None else np.array(jerk, dtype=float)
        self.brake_jerk = None if jerk is None else self.motion_jerk * brake_jerk_factor
        self.braking = False
        self.scale = 1.0
        self.ticks = 0
        self.mode = "idle"  # idle (no fresh command) | moving | stopping | killed
        self._seq = redis.get(CMD_SEQ)  # whatever is there now is old: never fresh until it changes
        self._seq_at: float | None = None
        self._held = np.array(vehicle.x, dtype=float)  # the pose held while idle
        redis.set(DESIRED, _list(vehicle.x))
        redis.set(STOP, "ok")
        redis.set(KILL, "ok")
        redis.set(MAX_VEL_SCALE, "1.0")
        self._publish()

    def step(self) -> bool:
        """One tick; False once killed."""
        now = self.clock()
        self._apply_scale()
        seq = self.r.get(CMD_SEQ)
        if seq != self._seq:
            self._seq, self._seq_at = seq, now
        fresh = self._seq_at is not None and now - self._seq_at <= self.timeout
        if self.r.get(KILL) == "kill":
            self.mode = "killed"
            self._publish()
            return False
        if self.r.get(STOP) == "stop":
            self._brake()
            self.mode = "stopping"
            if np.linalg.norm(self.v.dx) < STILL:
                self.r.set(STOP, "ok")
                self._seq_at = None  # a stop also ends the last command
        elif fresh:
            goal = _parse(self.r.get(DESIRED))
            if goal is None:
                self._brake()
                self.mode = "idle"
            else:
                self._set_limits(braking=False)
                self._limit_along(goal)
                self.v.set_target_position(goal)
                self.mode = "moving"
        else:
            if self.mode == "moving":
                print(f"[base] no fresh command for {self.timeout:.2f} s: braking", flush=True)
            self._brake()
            self.mode = "idle"
        self._publish()
        return True

    def _set_limits(self, braking: bool) -> None:
        if braking == self.braking:
            return
        self.braking = braking
        self.v.otg_inp.max_acceleration = (self.brake_accel if braking else self.motion_accel).tolist()
        if self.motion_jerk is not None:
            self.v.otg_inp.max_jerk = (self.brake_jerk if braking else self.motion_jerk).tolist()

    def _brake(self) -> None:
        self._set_limits(braking=True)
        self.v.set_target_velocity(np.zeros(3))
        self.r.set(DESIRED, _list(self.v.x))  # a later fresh command replaces it; nothing old is resumed

    def _limit_along(self, goal: np.ndarray) -> None:
        """x / y velocity limits along the direction to `goal`, so the planar speed is at most the planar limit."""
        v_xy, v_yaw = self.baseline_max_vel[0] * self.scale, self.baseline_max_vel[2] * self.scale
        d = np.asarray(goal[:2], dtype=float) - np.asarray(self.v.x[:2], dtype=float)
        n = float(np.linalg.norm(d))
        u = np.abs(d) / n if n > 1e-4 else np.array([1.0, 1.0]) / np.sqrt(2.0)
        floor = 0.05 * v_xy  # a little on the minor axis, so it can still correct across the line
        self.v.otg_inp.max_velocity = [max(v_xy * u[0], floor), max(v_xy * u[1], floor), v_yaw]

    def _apply_scale(self) -> None:
        try:
            raw = float(self.r.get(MAX_VEL_SCALE))
        except (TypeError, ValueError):
            return
        if not np.isfinite(raw):
            return
        scale = float(np.clip(raw, *SCALE_RANGE))
        if abs(scale - self.scale) > 1e-3:
            self.v.otg_inp.max_velocity = (self.baseline_max_vel * scale).tolist()
            self.scale = scale

    def _publish(self) -> None:
        self.ticks += 1
        self.r.set(POSE, _list(self.v.x))
        self.r.set(VEL, _list(self.v.dx))
        self.r.set(HEARTBEAT, str(self.ticks))


def main() -> None:
    import argparse
    import sys
    from pathlib import Path

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-vel-xy", type=float, default=0.125)
    ap.add_argument("--max-vel-yaw", type=float, default=0.4)
    ap.add_argument("--max-accel-xy", type=float, default=0.1)
    ap.add_argument("--max-accel-yaw", type=float, default=0.5)
    ap.add_argument("--command-timeout", type=float, default=0.3, help="s without a new hb1::cmd_seq before braking")
    ap.add_argument("--brake-accel-xy", type=float, default=0.25, help="m/s^2 when braking (no fresh command, stop)")
    ap.add_argument("--brake-accel-yaw", type=float, default=0.8, help="rad/s^2 when braking")
    args = ap.parse_args()
    if not (0 < args.max_vel_xy <= 0.25 and 0 < args.max_vel_yaw <= 0.8 and 0 < args.command_timeout <= 1.0):
        sys.exit("refusing: --max-vel-xy <= 0.25 m/s, --max-vel-yaw <= 0.8 rad/s, --command-timeout <= 1 s")
    if not (0.1 <= args.brake_accel_xy <= 0.5 and 0.3 <= args.brake_accel_yaw <= 1.6):
        sys.exit("refusing: --brake-accel-xy in [0.1, 0.5] m/s^2, --brake-accel-yaw in [0.3, 1.6] rad/s^2")

    import signal

    import redis

    # SIGTERM as Ctrl-C, so the `finally` below stops the Vehicle (Python's default for TERM would skip it).
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # base_controller imports constants / utils by name
    from base_controller import Vehicle

    max_vel = (args.max_vel_xy, args.max_vel_xy, args.max_vel_yaw)
    max_accel = (args.max_accel_xy, args.max_accel_xy, args.max_accel_yaw)
    print(f"[base] limits {max_vel} m/s, m/s, rad/s; accel {max_accel}; command timeout {args.command_timeout} s", flush=True)
    vehicle = Vehicle(max_vel=max_vel, max_accel=max_accel)
    vehicle.start_control()
    time.sleep(1.0)  # Vehicle's own start-up, as ZitiBot's driver waits
    loop = DriverLoop(vehicle, redis.Redis(host="localhost", port=6379, decode_responses=True), args.command_timeout,
                      brake_accel=(args.brake_accel_xy, args.brake_accel_xy, args.brake_accel_yaw))  # fmt: skip
    print(f"[base] braking at {args.brake_accel_xy} m/s^2, {args.brake_accel_yaw} rad/s^2", flush=True)
    print("[base] ready: idle until a fresh command", flush=True)
    try:
        while loop.step():
            time.sleep(0.01)
        print("[base] killed", flush=True)
    except KeyboardInterrupt:
        print("[base] interrupted", flush=True)
    finally:
        vehicle.stop_control()
        print("[base] stopped", flush=True)


if __name__ == "__main__":
    main()
