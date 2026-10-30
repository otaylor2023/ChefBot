"""The lab's test probes in an elevator scene, and a scenario runner that drives them while the building runs.

    finger   a fingertip on a mocap mount, pushed along its axis by a motor: presses buttons with a set force, and
             follows a button in a moving car
    cart     a 40 kg box on four 5 cm ball wheels (low friction, which wins over the floor.s), driven toward a goal by horizontal
             forces: feels sill gaps and steps, rides in the car
    block    a box placed in a doorway (mocap): the doors must not crush it

Everything is parked on a pad far below the building when not in use.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from chefbot.environments.mujoco.sim import SceneSim

PARK = np.array([0.0, 0.0, -80.0])
PROBES = f"""
<geom name="lab_park" type="box" pos="{PARK[0]} {PARK[1]} {PARK[2] - 0.5}" size="6 6 0.5" rgba="0.3 0.3 0.3 1"/>
<body name="lab_finger_base" mocap="true" pos="{PARK[0] + 3} {PARK[1]} {PARK[2] + 1}">
  <body name="lab_finger" gravcomp="1">
    <joint name="lab_finger" type="slide" axis="0 0 1" range="0 0.06" damping="4"/>
    <geom name="lab_finger" type="capsule" fromto="0 0 0 0 0 0.03" size="0.007" mass="0.05" rgba="0.9 0.25 0.2 1"/>
  </body>
</body>
<body name="lab_cart" pos="{PARK[0]} {PARK[1]} {PARK[2]}">
  <freejoint name="lab_cart"/>
  <geom name="lab_cart" type="box" size="0.25 0.2 0.12" pos="0 0 0.17" mass="40" rgba="0.2 0.4 0.85 1"/>
  <geom name="lab_cart_w1" type="sphere" size="0.05" pos="0.2 0.15 0.05" mass="1" friction="0.03 0.005 0.0001" priority="1"/>
  <geom name="lab_cart_w2" type="sphere" size="0.05" pos="0.2 -0.15 0.05" mass="1" friction="0.03 0.005 0.0001" priority="1"/>
  <geom name="lab_cart_w3" type="sphere" size="0.05" pos="-0.2 0.15 0.05" mass="1" friction="0.03 0.005 0.0001" priority="1"/>
  <geom name="lab_cart_w4" type="sphere" size="0.05" pos="-0.2 -0.15 0.05" mass="1" friction="0.03 0.005 0.0001" priority="1"/>
</body>
<body name="lab_block" mocap="true" pos="{PARK[0] - 3} {PARK[1]} {PARK[2] + 0.4}">
  <geom name="lab_block" type="box" size="0.15 0.06 0.4" rgba="0.95 0.75 0.1 1"/>
</body>
"""
PROBE_ACTUATOR = '<motor name="lab_finger" joint="lab_finger" ctrlrange="-20 20"/>'
FLIP_X = np.array([0.0, 1.0, 0.0, 0.0])  # 180 degrees about x: the finger's +z into the button


def _quat_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


class Scenario:
    """One building running with the probes. Every helper advances world time; `trace` records what happened."""

    def __init__(self, folder: Path, ann: dict, params: dict | None, seed: int, sandbox=(), python=None, tick_s: float = 0.02):
        self.ann = ann
        self.obj = ann["objects"]
        self.sim = SceneSim(folder, params=params, seed=seed, extra=PROBES, extra_actuators=PROBE_ACTUATOR, sandbox=sandbox, python=python)
        m = self.sim.model
        self.m, self.d = m, self.sim.data
        self.tick_s = tick_s
        self.finger_mocap = m.body("lab_finger_base").mocapid[0]
        self.block_mocap = m.body("lab_block").mocapid[0]
        self.finger_j = m.joint("lab_finger")
        self.cart = m.body("lab_cart").id
        self.cart_q = int(m.jnt_qposadr[m.joint("lab_cart").id])
        self.cart_v = int(m.jnt_dofadr[m.joint("lab_cart").id])
        self.cart_geoms = {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == self.cart}
        self.block_geom = m.geom("lab_block").id
        self.finger_geom = m.geom("lab_finger").id
        self.finger_act = m.actuator("lab_finger").id
        self.press = None  # (button, force, follow) while pressing
        self.goal = None  # cart goal (x, y)
        self.violations: list[str] = []
        self.physical: list[dict] = []
        self._down: dict[str, bool] = {}
        self.rows: list[tuple] = []  # (t, {car: z}, {car: v}, {car: door open fraction}) every tick
        self.door_contact_max = 0.0  # the doors' push on a probe, averaged over 0.2 s (what it takes to hold them back)
        self.door_contact_peak = 0.0  # the largest single-tick push (an impact)
        self._door_push: list[float] = []
        self.cart_dz_max = 0.0
        self._cart_floor_z = None
        self.on_tick = None  # called after every tick (e.g. to film)

    # -- geometry ----------------------------------------------------------------------------

    def site(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        i = self.m.site(name).id
        return self.d.site_xpos[i].copy(), self.d.site_xmat[i].reshape(3, 3).copy()

    def joint_q(self, name: str) -> float:
        return float(self.d.qpos[self.m.jnt_qposadr[self.m.joint(name).id]])

    def joint_v(self, name: str) -> float:
        return float(self.d.qvel[self.m.jnt_dofadr[self.m.joint(name).id]])

    def open_fraction(self, doors: dict) -> float:
        fr = [(self.joint_q(j) - c) / (o - c) for j, c, o in zip(doors["joints"], doors["closed"], doors["open"]) if abs(o - c) > 1e-9]
        return float(np.mean(fr)) if fr else 0.0

    def floor_z(self, floor: str) -> float:
        return float(self.site(self.obj[floor]["site"])[0][2])

    def car_floor_z(self, car: str) -> float:
        return float(self.site(self.obj[car]["floor_site"])[0][2])

    def cars(self) -> list[str]:
        return [k for k, v in self.obj.items() if v["kind"] == "elevator"]

    def at_floor(self, floor: str, tol: float) -> str | None:
        """The car leveled at `floor` with its doors there open, if any."""
        for c in self.cars():
            lnd = self.obj[c]["landings"].get(floor)
            if lnd and abs(self.car_floor_z(c) - self.floor_z(floor)) <= tol and self.open_fraction(lnd["doors"]) >= 0.9 \
                    and self.open_fraction(self.obj[c]["doors"]) >= 0.9:  # fmt: skip
                return c
        return None

    # -- the tick: probes, invariants, records -----------------------------------------------------

    def _before(self, sim: SceneSim) -> None:
        m, d = self.m, self.d
        if self.press is not None:
            b, force, _ = self.press
            p, R = self.site(self.obj[b]["site"])
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, R.reshape(9))
            d.mocap_pos[self.finger_mocap] = p + R[:, 2] * 0.045
            d.mocap_quat[self.finger_mocap] = _quat_mul(q, FLIP_X)
            d.ctrl[self.finger_act] = force
        else:
            d.ctrl[self.finger_act] = -2.0
        if self.goal is not None:
            self._drive()

    def _drive(self) -> None:
        d = self.d
        pos = d.qpos[self.cart_q:self.cart_q + 3]
        vel = d.qvel[self.cart_v:self.cart_v + 3]
        to = np.array(self.goal) - pos[:2]
        dist = float(np.linalg.norm(to))
        want = to / max(dist, 1e-6) * min(0.4, 1.2 * dist)
        f = np.clip(400.0 * (want - vel[:2]), -150, 150)
        d.xfrc_applied[self.cart, :2] = f
        w = d.qvel[self.cart_v + 3:self.cart_v + 6]
        d.xfrc_applied[self.cart, 3:] = -20.0 * w  # no spinning or tipping on purpose

    def _record(self) -> None:
        sim, d, m = self.sim, self.d, self.m
        t = sim.t
        zs = {c: self.joint_q(self.obj[c]["joint"]) for c in self.cars()}
        vs = {c: self.joint_v(self.obj[c]["joint"]) for c in self.cars()}
        fr = {c: self.open_fraction(self.obj[c]["doors"]) for c in self.cars()}
        self.rows.append((t, zs, vs, fr))
        for b, spec in self.obj.items():
            if spec["kind"] != "button":
                continue
            down = self.joint_q(spec["joint"]) >= spec["actuation_travel_m"] - 1e-5
            if down and not self._down.get(b):
                self.physical.append({"t": round(t, 3), "button": b})
            self._down[b] = down
        # Invariants: a landing door opens only with a car leveled there; a car door only near a floor and slow.
        for c in self.cars():
            car = self.obj[c]
            zc = self.car_floor_z(c)
            near = min((abs(zc - self.floor_z(f)) for f in car["landings"]), default=1.0)
            if self.open_fraction(car["doors"]) > 0.1 and (near > 0.075 or abs(vs[c]) > 0.3):
                self.violations.append(f"t={t:.2f}: {c}'s doors open {self.open_fraction(car['doors']):.0%} with the car {near * 1000:.0f} mm from a floor at {vs[c]:.2f} m/s")
            for f, lnd in car["landings"].items():
                if self.open_fraction(lnd["doors"]) > 0.1 and abs(zc - self.floor_z(f)) > 0.075:
                    other = any(abs(self.car_floor_z(o) - self.floor_z(f)) <= 0.075 for o in self.cars() if f in self.obj[o]["landings"])
                    if not other:
                        self.violations.append(f"t={t:.2f}: {f}'s landing doors for {c} open with no car there")
        # Forces on the block and the cart from doors.
        door_bodies = set()
        for c in self.cars():
            car = self.obj[c]
            for j in car["doors"]["joints"] + [j for l in car["landings"].values() for j in l["doors"]["joints"]]:
                door_bodies.add(int(m.jnt_bodyid[m.joint(j).id]))
        f6 = np.zeros(6)
        push = 0.0
        probe = {self.block_geom} | self.cart_geoms
        for i in range(d.ncon):
            con = d.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            if (g1 in probe and int(m.geom_bodyid[g2]) in door_bodies) or (g2 in probe and int(m.geom_bodyid[g1]) in door_bodies):
                mujoco.mj_contactForce(m, d, i, f6)
                push = max(push, float(abs(f6[0])))
        self._door_push = (self._door_push + [push])[-max(1, int(round(0.2 / self.tick_s))):]
        self.door_contact_peak = max(self.door_contact_peak, push)
        self.door_contact_max = max(self.door_contact_max, float(np.mean(self._door_push)))
        if self.goal is not None:
            z = float(d.qpos[self.cart_q + 2])
            ref = self._cart_floor_z if self._cart_floor_z is not None else z
            self.cart_dz_max = max(self.cart_dz_max, abs(z - ref))

    def run(self, seconds: float, until=None) -> float | None:
        """Advance up to `seconds`; with `until`, stop when it returns True and return the time it took."""
        t0 = self.sim.t
        while self.sim.t - t0 < seconds - 1e-9:
            self.sim.run(self.tick_s, before_physics=self._before)
            self._record()
            if self.on_tick is not None:
                self.on_tick(self)
            if until is not None and until():
                return self.sim.t - t0
        return None if until is not None else self.sim.t - t0

    # -- probe actions -----------------------------------------------------------------------------

    def press_button(self, b: str, force: float = 6.0, hold_s: float = 0.4) -> None:
        self.press = (b, force, True)
        self.run(hold_s)
        self.press = None
        self.run(0.3)  # retract
        self.d.mocap_pos[self.finger_mocap] = PARK + [3, 0, 1]

    def place_cart(self, xy, yaw: float, floor_z: float) -> None:
        d = self.d
        d.qpos[self.cart_q:self.cart_q + 3] = [xy[0], xy[1], floor_z + 0.002]
        d.qpos[self.cart_q + 3:self.cart_q + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        d.qvel[self.cart_v:self.cart_v + 6] = 0
        mujoco.mj_forward(self.m, d)
        self._cart_floor_z = floor_z

    def drive_cart(self, xy, timeout_s: float = 20.0, floor_z: float | None = None) -> float | None:
        self.goal = tuple(xy)
        if floor_z is not None:
            self._cart_floor_z = floor_z
        took = self.run(timeout_s, until=lambda: np.linalg.norm(self.d.qpos[self.cart_q:self.cart_q + 2] - np.array(xy)) < 0.08)
        self.goal = None
        self.d.xfrc_applied[self.cart] = 0
        return took

    def cart_xyz(self) -> np.ndarray:
        return self.d.qpos[self.cart_q:self.cart_q + 3].copy()

    def place_block(self, pos) -> None:
        self.d.mocap_pos[self.block_mocap] = pos

    def park_block(self) -> None:
        self.d.mocap_pos[self.block_mocap] = PARK + [-3, 0, 0.4]

    def events(self, since: float = 0.0, kind: str | None = None) -> list[dict]:
        return [e for e in self.sim.timeline if e["t"] >= since - 1e-9 and (kind is None or e.get("type") == kind)]

    def close(self) -> None:
        self.sim.close()
