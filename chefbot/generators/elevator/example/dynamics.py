"""The example building's controller: one car, simple collective dispatch (serve the nearest call ahead, then turn),
a jerk-limited motion profile, door timing with reopening on the light curtain or a door motor straining, lamps, a
floor display, and one fault: a missed press (the next press within the recovery count always registers).
Crude on purpose; real controllers do much more (see the contract)."""

import math

import numpy as np

CAR = "car_A"


class SCurve:
    """A jerk-limited trip of `dist` m (signed) from z0 starting at t0: speed ramps up at the jerk limit to the
    acceleration limit, cruises at the rated speed (lower for short trips), and ramps down the same way."""

    def __init__(self, dist: float, V: float, A: float, J: float, t0: float, z0: float):
        self.sign, D, self.t0, self.z0 = (1.0 if dist >= 0 else -1.0), abs(dist), t0, z0

        def accel(vp):  # (time, distance) to reach speed vp from rest
            if vp >= A * A / J:
                T = vp / A + A / J
            else:
                T = 2 * math.sqrt(vp / J)
            return T, vp * T / 2

        vp = V
        if 2 * accel(V)[1] > D:
            lo, hi = 0.0, V
            for _ in range(60):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if 2 * accel(mid)[1] <= D else (lo, mid)
            vp = lo
        Ta, da = accel(vp)
        Tc = (D - 2 * da) / vp if vp > 0 else 0.0
        self.t_end = t0 + 2 * Ta + Tc
        # Tabulate speed and position every millisecond.
        Ae = min(A, math.sqrt(vp * J)) if vp > 0 else 0.0
        tj, ta = (Ae / J if J > 0 else 0.0), max(0.0, Ta - 2 * (Ae / J if J > 0 else 0.0))

        def v_acc(u):
            if u <= tj:
                return J * u * u / 2
            if u <= tj + ta:
                return J * tj * tj / 2 + Ae * (u - tj)
            r = max(0.0, Ta - u)
            return vp - J * r * r / 2

        n = max(2, int((2 * Ta + Tc) / 0.001) + 1)
        self.ts = np.linspace(0.0, 2 * Ta + Tc, n)
        self.vs = np.array([v_acc(u) if u < Ta else vp if u < Ta + Tc else v_acc(2 * Ta + Tc - u) for u in self.ts])
        self.zs = np.concatenate([[0.0], np.cumsum((self.vs[1:] + self.vs[:-1]) / 2 * np.diff(self.ts))])
        if self.zs[-1] > 0:
            self.zs *= D / self.zs[-1]  # land exactly on the floor

    def at(self, t: float) -> tuple[float, float]:
        u = min(max(t - self.t0, 0.0), self.ts[-1])
        return self.z0 + self.sign * float(np.interp(u, self.ts, self.zs)), self.sign * float(np.interp(u, self.ts, self.vs))


class Dynamics:
    def __init__(self, annotations: dict, params: dict, seed: int):
        self.p = params
        self.rng = np.random.default_rng(seed)
        obj = annotations["objects"]
        self.car = obj[CAR]
        self.floor_ids = sorted((k for k, v in obj.items() if v["kind"] == "floor"), key=lambda k: int(k.split("_")[1]))
        self.floor_z = dict(zip(self.floor_ids, params["floor_z"]))
        self.labels = {k: obj[k]["label"] for k in self.floor_ids}
        self.buttons = {k: v for k, v in obj.items() if v["kind"] == "button"}
        self.lamps = {v["button"]: v["display"] for v in obj.values() if v["kind"] == "display" and v.get("role") == "button_lamp"}
        fault = next(f for f in annotations.get("faults", []) if f["name"] == "missed_press")
        self.miss_rate, self.miss_recover = float(params.get(fault["rate_param"], 0.0)), int(fault["recovery"]["presses"])
        self.was_down = {b: False for b in self.buttons}
        self.misses = {b: 0 for b in self.buttons}
        self.calls: dict[str, set] = {}  # floor -> buttons asking for it
        self.lamp_on: set = set()
        self.z = self.floor_z[self.floor_ids[0]]  # the motion profile's position (the car follows it)
        self.v = 0.0
        self.target, self.trip, self.integral = None, None, 0.0
        self.phase, self.door, self.timer = "idle", 0.0, 0.0  # door: 0 closed .. 1 open
        self.hold_open = False
        self.t_prev = None

    # -- buttons ------------------------------------------------------------------------------

    def _press(self, b: str, events: list) -> None:
        if self.misses[b] < self.miss_recover - 1 and self.rng.random() < self.miss_rate:
            self.misses[b] += 1
            events.append({"type": "fault", "name": "missed_press", "detail": b})
            return
        self.misses[b] = 0
        m = self.buttons[b]["meaning"]
        events.append({"type": "registered", "button": b})
        if m["type"] in ("hall_call", "car_floor"):
            self.calls.setdefault(m["floor"], set()).add(b)
            self.lamp_on.add(b)
            events.append({"type": "assigned", "button": b, "elevator": CAR})
        elif m["type"] == "door_open" and self.phase in ("closing", "open", "idle"):
            self.phase, self.timer = ("opening" if self.phase != "open" else "open"), 0.0
        elif m["type"] == "door_close" and self.phase == "open":
            self.timer = self.p["dwell_s"]

    # -- the car ------------------------------------------------------------------------------

    def _floor_at(self, z: float) -> str:
        return min(self.floor_ids, key=lambda f: abs(self.floor_z[f] - z))

    def _choose(self) -> str | None:
        if not self.calls:
            return None
        here = self._floor_at(self.z)
        ahead = [f for f in self.calls if (self.floor_z[f] - self.z) * (1 if self.v >= 0 else -1) >= -1e-3]
        pool = ahead or list(self.calls)
        return min(pool, key=lambda f: abs(self.floor_z[f] - self.z)) if here not in self.calls else here

    def _move(self, t: float, z_meas: float, v_meas: float, events: list) -> None:
        zt = self.floor_z[self.target]
        if self.trip is None:
            self.trip = SCurve(zt - self.z, self.p["rated_speed_mps"], self.p["accel_mps2"], self.p["jerk_mps3"], t, self.z)
        self.z, self.v = self.trip.at(t)
        if t >= self.trip.t_end and abs(zt - z_meas) < 0.5 * self.p["leveling_tol_m"] and abs(v_meas) < 0.01:
            self.trip = None
            events.append({"type": "arrived", "elevator": CAR, "floor": self.target})
            self._serve(self.target)
            self.phase, self.timer = "opening", 0.0

    def _serve(self, floor: str) -> None:
        for b in self.calls.pop(floor, set()):
            self.lamp_on.discard(b)

    # -- doors ------------------------------------------------------------------------------

    def _doors(self, dt: float, sensed: dict, events: list) -> None:
        blocked = any(sensed.get(b) for b in self.car["beams"])
        here = self._floor_at(self.z)
        forces = [sensed.get(f"car_A_door_force_{s}", 0.0) for s in "lr"] + [sensed.get(f"L{here.split('_')[1]}_A_door_force_{s}", 0.0) for s in "lr"]
        straining = max(abs(f) for f in forces) > 0.9 * self.p["door_force_limit_N"]  # the operator feels car and landing doors
        held = any(sensed.get(b, {}).get("q", 0) >= self.buttons[b]["actuation_travel_m"] for b in self.buttons
                   if self.buttons[b]["meaning"]["type"] == "door_open")  # fmt: skip
        if self.phase == "opening":
            self.door = min(1.0, self.door + dt / self.p["door_open_s"])
            if self.door >= 1.0:
                self.phase, self.timer = "open", 0.0
                events.append({"type": "doors_open", "elevator": CAR})
        elif self.phase == "open":
            self.timer += dt
            if blocked or held:
                self.timer = min(self.timer, self.p["dwell_s"] - 1.0)
            if self.timer >= self.p["dwell_s"]:
                self.phase, self.timer = "closing", 0.0
        elif self.phase == "closing":
            if blocked or straining or held:
                self.phase = "opening"
                events.append({"type": "reopen", "elevator": CAR, "why": "beam" if blocked else "force" if straining else "button"})
            else:
                self.door = max(0.0, self.door - dt / self.p["door_close_s"])
                if self.door <= 0.0:
                    self.timer += dt
                    if self._doors_shut(sensed):  # the door-closed contact
                        self.phase, self.timer = "idle", 0.0
                        events.append({"type": "doors_closed", "elevator": CAR})
                    elif self.timer > 0.3:  # commanded shut but not shut: something is in the way
                        self.phase, self.timer = "opening", 0.0
                        events.append({"type": "reopen", "elevator": CAR, "why": "not shut"})

    def _doors_shut(self, sensed: dict) -> bool:
        car = self.car
        j, c = car["doors"]["joints"][1], car["doors"]["closed"][1]
        return abs(sensed.get("car_A_door", {}).get("q", 1.0) - c) < 0.01 if j == "car_A_door_r" else True

    # -- the tick -----------------------------------------------------------------------------

    def step(self, t: float, sensed: dict) -> dict:
        dt = 0.0 if self.t_prev is None else t - self.t_prev
        self.t_prev = t
        events = []
        for b, spec in self.buttons.items():
            down = sensed.get(b, {}).get("q", 0.0) >= spec["actuation_travel_m"]
            if down and not self.was_down[b]:
                self._press(b, events)
            self.was_down[b] = down
        if self.phase == "idle":
            nxt = self._choose()
            if nxt is not None:
                if abs(self.floor_z[nxt] - self.z) < 0.001:
                    self._serve(nxt)
                    self.phase, self.timer = "opening", 0.0
                else:
                    self.target, self.phase = nxt, "moving"
                    events.append({"type": "departing", "elevator": CAR, "to": nxt})
        elif self.phase == "moving":
            self._move(t, sensed[CAR]["q"], sensed[CAR]["v"], events)
        else:
            self._doors(dt, sensed, events)

        car = self.car
        # The drive follows the profile's speed, corrected toward its position (and holds the floor when parked).
        err = self.z - sensed[CAR]["q"]
        self.integral = float(np.clip(self.integral + err * dt, -0.05, 0.05))  # holds the floor against load
        ctrl = {"car_A_motor": self.v + 3.0 * err + 4.0 * self.integral}
        here = self._floor_at(self.z)
        for j, c, o in zip(car["doors"]["joints"], car["doors"]["closed"], car["doors"]["open"]):
            ctrl[j] = c + self.door * (o - c)
        for floor, landing in car["landings"].items():
            frac = self.door if floor == here and self.phase != "moving" else 0.0
            for j, c, o in zip(landing["doors"]["joints"], landing["doors"]["closed"], landing["doors"]["open"]):
                ctrl[j] = c + frac * (o - c)
        displays = {"pos_A": {"floor": self.labels[here]}}
        displays.update({lamp: {"lamp": "on" if b in self.lamp_on else "off"} for b, lamp in self.lamps.items()})
        return {"ctrl": ctrl, "displays": displays, "events": events}
