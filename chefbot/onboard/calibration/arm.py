"""The arm for calibration, straight through the drivers' Redis (OpenSai's joint controller, the Franka driver's
sensors): slow joint moves, each shown and confirmed at the keyboard first, and still-arm readings.

    arm = Arm()
    arm.move_to(q)            # prints the move, waits for Enter, ramps the joint goal at <= max_speed, waits to settle
    arm.still(2.0)            # {"q", "tau_J", "gravity"} averaged over 2 s with the arm still
"""

from __future__ import annotations

import json
import time

import numpy as np

from chefbot.onboard import keys as K

TAU_J = "opensai::sensors::FrankaRobot::joint_torques"  # measured (link-side torque sensors)
GRAVITY = "opensai::sensors::FrankaRobot::model::robot_gravity"  # libfranka's model, with Desk's end-effector load
# The FR3's joint limits (Franka's datasheet), less a margin.
LIMITS = np.array([[-2.7437, 2.7437], [-1.7837, 1.7837], [-2.9007, 2.9007], [-3.0421, -0.1518], [-2.8065, 2.8065],
                   [0.5445, 4.5169], [-3.0159, 3.0159]])  # fmt: skip
MARGIN = 0.1
RAMP_HZ = 50.0
MAX_LAG = 0.25  # rad behind the ramp: the arm is not following (blocked, or the controller is off): stop and hold


class Stop(Exception):
    """The operator stopped the session."""


def _array(raw) -> np.ndarray | None:
    try:
        return np.asarray(json.loads(raw.decode() if isinstance(raw, bytes) else raw), dtype=float)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


class Arm:
    def __init__(self, host: str = "127.0.0.1", port: int = 6379, max_speed: float = 0.1):
        import redis

        self.r = redis.Redis(host=host, port=port)
        self.max_speed = max_speed  # rad/s, every joint, on average (the ramp's peak is pi/2 of it): ZitiBot's joint
        # controller (zitibot_panda.xml: kp 100, kv 20, velocity saturation) trails faster ramps by over 0.15 rad
        if self.q() is None:
            raise RuntimeError(f"no {K.JOINT_POSITIONS} on Redis: is the Franka driver running?")

    def q(self) -> np.ndarray | None:
        return _array(self.r.get(K.JOINT_POSITIONS))

    def dq(self) -> np.ndarray | None:
        return _array(self.r.get(K.JOINT_VELOCITIES))

    def fresh(self, timeout: float = 1.0) -> bool:
        """The driver is publishing (the joint positions change or the velocities are nonzero within timeout)."""
        a, t0 = self.r.get(K.JOINT_POSITIONS), time.time()
        while time.time() - t0 < timeout:
            time.sleep(0.05)
            if self.r.get(K.JOINT_POSITIONS) != a:
                return True
        return False

    def move_to(self, target, label: str = "", confirm: bool = True, settle_s: float = 1.0) -> None:
        target = np.asarray(target, dtype=float)
        lo, hi = LIMITS[:, 0] + MARGIN, LIMITS[:, 1] - MARGIN
        if np.any(target < lo) or np.any(target > hi):
            bad = [i + 1 for i in np.flatnonzero((target < lo) | (target > hi))]
            raise ValueError(f"target outside the joint limits (less {MARGIN} rad) at joints {bad}: {np.round(target, 3)}")
        q0 = self.q()
        delta = target - q0
        T = max(float(np.abs(delta).max()) / self.max_speed, 0.5)
        if confirm:
            print(f"\n{label}\n  joints now    {np.round(q0, 3).tolist()}\n  joints target {np.round(target, 3).tolist()}"
                  f"\n  largest change {np.degrees(np.abs(delta).max()):.1f} deg (joint {int(np.abs(delta).argmax()) + 1}), {T:.1f} s")  # fmt: skip
            if input("  Enter to move, q then Enter to stop: ").strip().lower() == "q":
                raise Stop()
        self.r.set(K.JOINT_GOAL_VELOCITY, json.dumps([0.0] * 7))
        self.r.set(K.JOINT_GOAL_ACCELERATION, json.dumps([0.0] * 7))
        self.r.set(K.JOINT_GOAL_POSITION, json.dumps(q0.tolist()))
        self.r.set(K.ACTIVE_CONTROLLER, K.JOINT_CONTROLLER)
        t0 = time.time()
        while True:  # a smooth ramp of the goal (the controller tracks it)
            s = min((time.time() - t0) / T, 1.0)
            w = 0.5 - 0.5 * np.cos(np.pi * s)
            self.r.set(K.JOINT_GOAL_POSITION, json.dumps((q0 + w * delta).tolist()))
            if s >= 1.0:
                break
            time.sleep(1.0 / RAMP_HZ)
            lag = np.abs(self.q() - (q0 + w * delta)).max()
            if lag > MAX_LAG:
                self.r.set(K.JOINT_GOAL_POSITION, json.dumps(self.q().tolist()))  # hold where it is
                raise RuntimeError(f"the arm is not following ({np.degrees(lag):.0f} deg behind): stopped, holding")
        t1 = time.time()
        while time.time() - t1 < 10.0:
            dq = self.dq()
            if dq is not None and np.abs(dq).max() < 0.005 and time.time() - t1 > settle_s:
                break
            time.sleep(0.05)
        err = np.abs(self.q() - target).max()
        if err > 0.02:
            print(f"  (settled {np.degrees(err):.1f} deg from the target)")

    def still(self, seconds: float = 2.0) -> dict:
        """The joints, measured torques and the model's gravity torques, averaged with the arm still."""
        qs, taus, gs = [], [], []
        t0 = time.time()
        while time.time() - t0 < seconds:
            q, tau, g = self.q(), _array(self.r.get(TAU_J)), _array(self.r.get(GRAVITY))
            if q is not None and tau is not None and g is not None:
                qs.append(q), taus.append(tau), gs.append(g)
            time.sleep(0.01)
        if not qs:
            raise RuntimeError(f"no {TAU_J} / {GRAVITY} on Redis")
        return {"q": np.mean(qs, 0).tolist(), "tau_J": np.mean(taus, 0).tolist(), "gravity": np.mean(gs, 0).tolist(),
                "tau_sd": np.std(taus, 0).tolist(), "n": len(qs)}  # fmt: skip
