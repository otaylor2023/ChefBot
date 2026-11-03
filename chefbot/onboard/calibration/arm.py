"""The arm for calibration, straight through the drivers' Redis (OpenSai's joint controller, the Franka driver's
sensors): slow joint moves, each shown and confirmed at the keyboard first, and still-arm readings.

    arm = Arm()
    arm.move_to(q)            # prints the move, waits for Enter (confirm=False: doesn't), then leads the arm there
    arm.hold()                # the goal where the arm is
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
# OpenSai's joint task (chefbot_fr3.xml) moves smoothly to a goal itself (jerk-limited, up to 0.5 rad/s): the goal
# goes ahead in large steps only so that a stall shows early.
STEP = 0.3  # rad: how far ahead of the arm the goal goes (the largest-moving joint's share)
CAUGHT_UP = 0.1  # rad from the step's goal: on to the next
SAG = 0.25  # rad: how far short of a goal the joint controller may settle (OpenSai's shoulder tracking is weak: 7 deg seen)
STALL_S = 4.0  # no progress for this long: settled there, or stuck
STEP_TIMEOUT_S = 30.0


# Keep out of the pole behind the arm on its left (the arm's own frame: x forward, y left, z up from its mounting
# plate) and the pole camera on its clamp: no joint, link midpoint or fingertip within these. The pole's center
# measured 5 in back and 8 in left of the arm base's (2026-11-03); the camera where pole.py calibrated it
# (environments/real/config.yaml: 1.04 m up, so 0.70 m above the plate). Radii are for the arm's centerline: the
# pole's ~3 cm half-width, the links' ~8 cm, and ~4 cm to spare. (It replaced a quarter-space, everything behind
# and left of the arm base, that also walled off the plate beside the arm.)
POLE_XY = (-0.127, 0.203)
POLE_RADIUS = 0.15
POLE_TOP = 0.85  # m above the plate: the pole and its clamp
POLE_CAMERA = (-0.25, 0.26, 0.70)
POLE_CAMERA_RADIUS = 0.15
TCP_Z = 0.13065  # m along the flange's z: the fingertips


def arm_points(q) -> np.ndarray:
    """Points along the arm (joint origins, link midpoints, the fingertips) in the arm base frame."""
    from chefbot.onboard.calibration import fr3

    fs = fr3.frames(q)
    p = [f[:3, 3] for f in fs] + [fs[-1][:3, 3] + fs[-1][:3, 2] * TCP_Z]
    return np.array(p + [(a + b) / 2 for a, b in zip(p, p[1:])])


def _segment_distance(p, a, b) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = np.clip((p - a) @ (b - a) / ((b - a) @ (b - a)), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * (b - a)), axis=1)


def _inside(p, margin: float) -> np.ndarray:
    """Points (N x 3, arm base frame) within the keep-out grown by margin: the pole's cylinder, and the camera on its
    clamp (a capsule from the pole's top to the camera)."""
    p = np.atleast_2d(p)
    radial = np.linalg.norm(p[:, :2] - np.asarray(POLE_XY), axis=1)
    pole = (radial < POLE_RADIUS + margin) & (p[:, 2] < POLE_TOP + margin)
    clamp = _segment_distance(p, (*POLE_XY, POLE_TOP - 0.2), POLE_CAMERA) < POLE_CAMERA_RADIUS + margin
    return pole | clamp


def in_keep_out(q, margin: float = 0.0) -> bool:
    """Any point of the arm in the keep-out; the hand's (flange to fingertips) with the keep-out grown by margin (m:
    for something held that sticks out of the hand)."""
    from chefbot.onboard.calibration import fr3

    T = fr3.flange(q)
    hand = np.array([T[:3, 3] + T[:3, 2] * t for t in np.linspace(0.0, TCP_Z, 4)])
    return bool(np.any(_inside(arm_points(q), 0.0)) or np.any(_inside(hand, margin)))


def path_clear(q0, q1, samples: int = 30, margin: float = 0.0) -> bool:
    """The joint-space straight line from q0 to q1 (how the arm moves) stays out of the keep-out."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    return not any(in_keep_out(q0 + (q1 - q0) * t, margin) for t in np.linspace(0, 1, samples))


class Stop(Exception):
    """The operator stopped the session."""


def _array(raw) -> np.ndarray | None:
    try:
        return np.asarray(json.loads(raw.decode() if isinstance(raw, bytes) else raw), dtype=float)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


class Arm:
    def __init__(self, host: str = "127.0.0.1", port: int = 6379):
        import redis

        self.r = redis.Redis(host=host, port=port)
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

    def move_to(self, target, label: str = "", confirm: bool = True, settle_s: float = 0.4, margin: float = 0.0) -> None:
        target = np.asarray(target, dtype=float)
        lo, hi = LIMITS[:, 0] + MARGIN, LIMITS[:, 1] - MARGIN
        if np.any(target < lo) or np.any(target > hi):
            bad = [i + 1 for i in np.flatnonzero((target < lo) | (target > hi))]
            raise ValueError(f"target outside the joint limits (less {MARGIN} rad) at joints {bad}: {np.round(target, 3)}")
        q0 = self.q()
        if not path_clear(q0, target, margin=margin):
            raise ValueError(f"the move would pass behind the arm on its left (the pole): {np.round(target, 3)}")
        delta = target - q0
        n = max(1, int(np.ceil(np.abs(delta).max() / STEP)))
        print(f"\n{label}\n  joints now    {np.round(q0, 3).tolist()}\n  joints target {np.round(target, 3).tolist()}"
              f"\n  largest change {np.degrees(np.abs(delta).max()):.1f} deg (joint {int(np.abs(delta).argmax()) + 1}), in {n} steps")  # fmt: skip
        if confirm:
            if input("  Enter to move, q then Enter to stop: ").strip().lower() == "q":
                raise Stop()
        self.r.set(K.JOINT_GOAL_VELOCITY, json.dumps([0.0] * 7))
        self.r.set(K.JOINT_GOAL_ACCELERATION, json.dumps([0.0] * 7))
        self.r.set(K.JOINT_GOAL_POSITION, json.dumps(q0.tolist()))
        self.r.set(K.ACTIVE_CONTROLLER, K.JOINT_CONTROLLER)
        for k in range(1, n + 1):  # the goal a small step ahead, the next one once the arm has (nearly) caught up
            w = q0 + delta * (k / n)
            self.r.set(K.JOINT_GOAL_POSITION, json.dumps(w.tolist()))
            best, t_best, t0 = np.inf, time.time(), time.time()
            while True:
                time.sleep(0.02)
                d = float(np.abs(self.q() - w).max())
                if d < CAUGHT_UP:
                    break
                if d < best - 0.003:
                    best, t_best = d, time.time()
                if time.time() - t_best > STALL_S:  # no progress: the controller's own sag is fine, anything more is not
                    if d < SAG:
                        break
                    self.r.set(K.JOINT_GOAL_POSITION, json.dumps(self.q().tolist()))  # hold where it is
                    raise RuntimeError(f"the arm stopped {np.degrees(d):.0f} deg short of step {k}/{n} (blocked, or the controller is off): holding")
                if time.time() - t0 > STEP_TIMEOUT_S:
                    self.r.set(K.JOINT_GOAL_POSITION, json.dumps(self.q().tolist()))
                    raise RuntimeError(f"step {k}/{n} took over {STEP_TIMEOUT_S:.0f} s: holding")
        t1 = time.time()
        while time.time() - t1 < 10.0:
            dq = self.dq()
            if dq is not None and np.abs(dq).max() < 0.005 and time.time() - t1 > settle_s:
                break
            time.sleep(0.05)
        err = np.abs(self.q() - target).max()
        if err > 0.02:
            print(f"  (settled {np.degrees(err):.1f} deg from the target)")

    def hold(self) -> None:
        """The joint goal where the arm is now."""
        q = self.q()
        if q is not None:
            self.r.set(K.JOINT_GOAL_POSITION, json.dumps(q.tolist()))

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
