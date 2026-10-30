"""The checks for a generated elevator building (family.yaml `checks`): scenarios with the lab's probes (probes.py)
while the building and its controller run. Each scenario is a fresh run of the scene.

    structure    the scene loads; the annotations name what the contract requires and it exists in the model;
                 the required parameters are there; the controller starts
    trip         faults off: a light touch on a hall call does nothing; a press registers (and lights its lamp);
                 a car comes within max_wait_s, levels, opens; the cart drives in over the sill (gap and step
                 within what's declared and the code's limits); a floor button registers; the car rides within
                 its declared speed, acceleration and jerk, the cart riding along; it levels at the destination,
                 its position display shows that floor; the cart drives out
    doors        door opening, dwell and closing times vs declared; and with the dwell overridden, the new dwell
    obstruction  a block in the doorway as the doors close: they reopen, and never push it harder than the limit
    hold_open    holding the door-open button holds the doors open past the dwell
    calls        calls from several floors (and a double press): every one is served
    faults       each declared fault alone at rate 1: it fires, its recovery works, nothing unsafe happens
    speed        the scenarios ran faster than real time
    (every scenario) safety: doors open only at a leveled car; no fault events when every fault rate is 0

run(asset_dir, fam, params) -> {"ok", "failed", "checks": {...}, "measured": {...}, "check_s"}.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from chefbot.environments.mujoco import scene as SC
from chefbot.environments.mujoco.dynamics import DynamicsError
from chefbot.generators.elevator.probes import Scenario

CODE_MAX_SILL_GAP_M = 0.032  # ASME A17.1 / ADA: about 1 1/4 in between car and landing sills
CODE_MAX_LEVELING_M = 0.013  # about 1/2 in
CODE_MAX_DOOR_FORCE_N = 135.0  # about 30 lbf to stop closing doors
REQUIRED_RUN = ("max_wait_s", "door_open_s", "door_close_s", "dwell_s", "rated_speed_mps", "accel_mps2", "jerk_mps3",
                "leveling_tol_m", "door_force_limit_N")  # fmt: skip
REQUIRED_BUILD = ("floors", "cars", "sill_gap_m", "building_type")


def _fail(why: str, **kw) -> dict:
    return {"ok": False, "why": why, **kw}


def _ok(**kw) -> dict:
    return {"ok": True, **kw}


def _whys(out: dict, whys: list[str]) -> dict:
    if whys:
        out.update(ok=False, why="; ".join(whys))
    return out


class Ctx:
    """What every scenario needs: the scene, its annotations, parameters, and how to run its controller."""

    def __init__(self, folder: Path, fam: dict, params: dict, sandbox=(), python=None):
        self.folder, self.fam, self.sandbox, self.python = Path(folder), fam, sandbox, python
        self.ann = SC.load_annotations(self.folder)
        self.obj = self.ann["objects"]
        self.run_params = {**self.ann["params"], **(params.get("run") or {})}
        self.build = params.get("build") or {}
        self.faults = self.ann.get("faults") or []
        self.no_faults = {f["rate_param"]: 0.0 for f in self.faults if f.get("rate_param")}
        self.floors = sorted((k for k, v in self.obj.items() if v["kind"] == "floor"), key=lambda k: self._floor_z(k))
        self.cars = [k for k, v in self.obj.items() if v["kind"] == "elevator"]
        self.sim_s, self.wall_s = 0.0, 0.0

    def _floor_z(self, k):
        import mujoco

        m = SC.load_model(self.folder)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        return float(d.site_xpos[m.site(self.obj[k]["site"]).id][2])

    def scenario(self, params: dict | None = None, seed: int = 0) -> Scenario:
        return Scenario(self.folder, self.ann, {**self.no_faults, **(params or {})}, seed, self.sandbox, self.python)

    def done(self, sc: Scenario, t0: float) -> list[str]:
        self.sim_s += sc.sim.t
        self.wall_s += time.time() - t0
        problems = list(sc.violations[:3])
        faults = sc.events(kind="fault")
        rates = sc.sim.params
        if faults and all(float(rates.get(f["rate_param"], 0)) == 0 for f in self.faults):
            problems.append(f"fault events with every fault rate 0: {faults[:2]}")
        sc.close()
        return problems

    # -- buttons ------------------------------------------------------------------------------

    def buttons(self, **meaning) -> list[str]:
        return [k for k, v in self.obj.items() if v["kind"] == "button" and all(v["meaning"].get(a) == b for a, b in meaning.items())]

    def call_button(self, at: str, to: str) -> str | None:
        """How to call a car at floor `at` to go to `to`: a hall call in that direction, or a destination key."""
        up = self.floors.index(to) > self.floors.index(at)
        b = self.buttons(type="hall_call", floor=at, dir="up" if up else "down")
        if b:
            return b[0]
        b = self.buttons(type="dispatch_key", at=at, to=to)
        return b[0] if b else None

    def lamp_for(self, button: str) -> str | None:
        return next((v["display"] for v in self.obj.values() if v["kind"] == "display" and v.get("role") == "button_lamp" and v.get("button") == button), None)


def _registered(sc: Scenario, button: str, since: float) -> bool:
    return any(e.get("button") == button for e in sc.events(since, "registered"))


def _press_until_registered(sc: Scenario, button: str, presses: int = 1, hold_s: float = 0.4) -> bool:
    for _ in range(presses):
        t = sc.sim.t
        sc.press_button(button, hold_s=hold_s)
        if _registered(sc, button, t):
            return True
        sc.run(1.0)
    return False


def _motion(rows: list[tuple], car: str, t0: float, t1: float) -> dict:
    """Peak speed, acceleration and jerk of a car between t0 and t1 (smoothed over 60 / 120 ms)."""
    sel = [(r[0], r[1][car]) for r in rows if t0 <= r[0] <= t1]
    if len(sel) < 10:
        return {}
    t, z = np.array(sel).T
    dt = float(np.median(np.diff(t)))
    v = np.gradient(z, dt)
    k3 = max(1, int(round(0.06 / dt)))
    a = np.convolve(np.gradient(v, dt), np.ones(k3) / k3, "same")
    k6 = max(1, int(round(0.12 / dt)))
    j = np.convolve(np.gradient(a, dt), np.ones(k6) / k6, "same")
    trim = k6
    return {"max_speed_mps": round(float(np.abs(v).max()), 3), "max_accel_mps2": round(float(np.abs(a[trim:-trim]).max()) if len(a) > 2 * trim else 0.0, 3),
            "max_jerk_mps3": round(float(np.abs(j[2 * trim:-2 * trim]).max()) if len(j) > 4 * trim else 0.0, 3), "ride_s": round(t1 - t0, 2)}  # fmt: skip


def _door_times(rows: list[tuple], car: str, t0: float, t1: float) -> dict:
    """Opening and closing durations (5% to 95%) and dwell (fully open until closing starts) in [t0, t1]."""
    sel = [(r[0], r[3][car]) for r in rows if t0 <= r[0] <= t1]
    out: dict = {}
    t_lo = t_open = None
    for (ta, fa), (tb, fb) in zip(sel, sel[1:]):
        if fa <= 0.05 < fb:
            t_lo = ta
        if fa < 0.95 <= fb:
            if t_lo is not None and "door_open_s" not in out:
                out["door_open_s"] = round(tb - t_lo, 2)
            t_open = tb
        if fa >= 0.95 > fb:
            if t_open is not None and "dwell_s" not in out:
                out["dwell_s"] = round(ta - t_open, 2)
            t_close0 = ta
            out.setdefault("_close_start", t_close0)
        if fa > 0.05 >= fb and "_close_start" in out and "door_close_s" not in out:
            out["door_close_s"] = round(tb - out["_close_start"], 2)
    out.pop("_close_start", None)
    return out


def _approach(sc: Scenario, landing_site: str, back_m: float = 1.6):
    p, R = sc.site(landing_site)
    return p - R[:, 0] * back_m, p[2]


# -- scenarios ------------------------------------------------------------------------------------


def structure(ctx: Ctx, params: dict) -> dict:
    whys = []
    if len(ctx.floors) < 2:
        whys.append(f"{len(ctx.floors)} floors annotated (at least 2)")
    if len(ctx.floors) > ctx.fam["generation"]["max_floors"]:
        whys.append(f"{len(ctx.floors)} floors (at most {ctx.fam['generation']['max_floors']})")
    if not ctx.cars:
        whys.append("no elevator annotated")
    missing_run = [k for k in REQUIRED_RUN if k not in ctx.run_params]
    missing_run += [f["rate_param"] for f in ctx.faults if f.get("rate_param") not in ctx.run_params]
    if missing_run:
        whys.append(f"missing run parameters {missing_run}")
    missing_build = [k for k in REQUIRED_BUILD if k not in ctx.build]
    if missing_build:
        whys.append(f"missing build parameters {missing_build}")
    try:
        sc = ctx.scenario()
    except (SC.SceneError, DynamicsError, KeyError, ValueError) as e:
        return _fail(f"does not run: {e}"[:600])
    m = sc.m
    for c in ctx.cars:
        car = ctx.obj[c]
        for key in ("joint", "floor_site", "sill_site", "doors", "landings", "beams"):
            if key not in car:
                whys.append(f"{c}: no {key}")
        for f, lnd in car.get("landings", {}).items():
            if f not in ctx.obj:
                whys.append(f"{c}: landing at unknown floor {f}")
        if not any(v["kind"] == "display" and v.get("role") == "car_position" and v.get("elevator") == c for v in ctx.obj.values()):
            whys.append(f"{c}: no car_position display")
    for k, v in ctx.obj.items():
        try:
            if v["kind"] == "button":
                m.joint(v["joint"]), m.site(v["site"])
            elif v["kind"] in ("floor",):
                m.site(v["site"])
            elif v["kind"] == "elevator":
                m.joint(v["joint"]), m.site(v["floor_site"]), m.site(v["sill_site"])
                for lnd in v["landings"].values():
                    m.site(lnd["sill_site"])
                    for j in lnd["doors"]["joints"]:
                        m.joint(j)
                for j in v["doors"]["joints"]:
                    m.joint(j)
        except KeyError as e:
            whys.append(f"{k}: the model has no {e}")
    t0 = time.time()
    sc.run(1.0)
    whys += ctx.done(sc, t0)
    return _whys(_ok(floors=len(ctx.floors), cars=len(ctx.cars), buttons=len(ctx.buttons())), whys)


def trip(ctx: Ctx, a: str, b: str) -> tuple[dict, dict]:
    p = ctx.run_params
    measured: dict = {}
    whys: list[str] = []
    t_wall = time.time()
    sc = ctx.scenario()
    call = ctx.call_button(a, b)
    if call is None:
        return _fail(f"no way to call a car at {a} to {b} (a hall_call or dispatch_key)"), measured
    lands = [c for c in ctx.cars if a in ctx.obj[c]["landings"] and b in ctx.obj[c]["landings"]]
    if not lands:
        return _fail(f"no car serves both {a} and {b}"), measured
    start, za = _approach(sc, ctx.obj[lands[0]]["landings"][a]["sill_site"])
    sc.place_cart(start[:2], 0.0, za)
    sc.run(0.5)
    # A light touch must not count.
    t = sc.sim.t
    sc.press_button(call, force=0.4, hold_s=0.5)
    if _registered(sc, call, t) or any(e["button"] == call for e in sc.physical):
        whys.append("a 0.4 N touch registered (or moved the button to its contact)")
    t_press = sc.sim.t
    sc.press_button(call, force=6.0)
    if not _registered(sc, call, t_press):
        whys.append(f"a firm press of {call} did not register")
    lamp = ctx.lamp_for(call)
    if lamp and _registered(sc, call, t_press):
        served_at_once = sc.at_floor(a, 0.02) is not None or any(r[3][c] > 0.05 for r in sc.rows if r[0] >= t_press for c in r[3])
        if sc.sim.display_state.get(lamp, {}).get("lamp") != "on" and not served_at_once:
            whys.append(f"{call}'s lamp did not light (and no car opened there at once)")
    tol = float(p["leveling_tol_m"]) + 0.001
    car = None
    waited = sc.run(float(p["max_wait_s"]) + 10.0, until=lambda: sc.at_floor(a, tol) is not None)
    if waited is None:
        whys.append(f"no car leveled and opened at {a} within {p['max_wait_s'] + 10:.0f} s")
        whys += ctx.done(sc, t_wall)
        return _whys(_ok(), whys), measured
    car = sc.at_floor(a, tol)
    wait_s = sc.sim.t - t_press
    measured["wait_s"] = round(wait_s, 2)
    if wait_s > float(p["max_wait_s"]) + 2.0:
        whys.append(f"waited {wait_s:.1f} s (max_wait_s {p['max_wait_s']})")
    # The sills: the gap and the step as the robot will meet them.
    lnd = ctx.obj[car]["landings"][a]
    pl, Rl = sc.site(lnd["sill_site"])
    pc, _ = sc.site(ctx.obj[car]["sill_site"])
    gap = abs(float((pc - pl) @ Rl[:, 0]))
    step = float(pc[2] - pl[2])
    measured.update(sill_gap_m=round(gap, 4), sill_step_m=round(step, 4))
    declared_gap = float(ctx.build.get("sill_gap_m", gap))
    if gap > min(declared_gap + 0.005, CODE_MAX_SILL_GAP_M):
        whys.append(f"the sill gap is {gap * 1000:.0f} mm (declared {declared_gap * 1000:.0f}, code at most {CODE_MAX_SILL_GAP_M * 1000:.0f})")
    if abs(step) > min(tol, CODE_MAX_LEVELING_M):
        whys.append(f"the car stopped {step * 1000:+.1f} mm from the floor (leveling_tol_m {p['leveling_tol_m']})")
    # In.
    took = sc.drive_cart(sc.site(ctx.obj[car]["floor_site"])[0][:2], timeout_s=25.0)
    if took is None:
        whys.append(f"the cart could not drive into {car} (stuck at {sc.cart_xyz().round(2).tolist()})")
    measured["bump_m"] = round(sc.cart_dz_max, 4)
    # Ask for b (unless destination dispatch already did).
    floor_btn = ctx.buttons(type="car_floor", elevator=car, floor=b)
    t_ride = sc.sim.t
    if floor_btn and not (ctx.obj[call]["meaning"]["type"] == "dispatch_key"):
        if not _press_until_registered(sc, floor_btn[0]):
            whys.append(f"{floor_btn[0]} did not register")
    elif not floor_btn and ctx.obj[call]["meaning"]["type"] != "dispatch_key":
        whys.append(f"{car} has no button for {b}")
    t_dep = sc.sim.t
    arrived = sc.run(float(p["max_wait_s"]) + float(p["dwell_s"]) + 30.0, until=lambda: sc.at_floor(b, tol) == car)
    if arrived is None:
        whys.append(f"{car} did not level and open at {b}")
    else:
        rows = sc.rows
        moving = [r[0] for r in rows if r[0] >= t_ride and abs(r[2][car]) > 0.01]
        if moving:
            mot = _motion(rows, car, moving[0] - 0.3, moving[-1] + 0.3)
            measured.update(mot)
            if mot.get("max_speed_mps", 0) > float(p["rated_speed_mps"]) * 1.1:
                whys.append(f"speed {mot['max_speed_mps']} m/s (rated {p['rated_speed_mps']})")
            if mot.get("max_accel_mps2", 0) > float(p["accel_mps2"]) * 1.2 + 0.1:
                whys.append(f"acceleration {mot['max_accel_mps2']} m/s2 (declared {p['accel_mps2']})")
            if mot.get("max_jerk_mps3", 0) > float(p["jerk_mps3"]) * 1.5 + 0.3:
                whys.append(f"jerk {mot['max_jerk_mps3']} m/s3 (declared {p['jerk_mps3']})")
        rel = sc.cart_xyz() - sc.site(ctx.obj[car]["floor_site"])[0]
        if abs(rel[2]) > 0.05 or np.linalg.norm(rel[:2]) > 0.5:
            whys.append(f"the cart did not ride along (it is {rel.round(2).tolist()} from the car floor)")
        level = sc.car_floor_z(car) - sc.floor_z(b)
        measured["level_error_m"] = round(float(level), 4)
        disp = next((v["display"] for v in ctx.obj.values() if v["kind"] == "display" and v.get("role") == "car_position" and v.get("elevator") == car), None)
        if disp and sc.sim.display_state.get(disp, {}).get("floor") != ctx.obj[b]["label"]:
            whys.append(f"{disp} shows {sc.sim.display_state.get(disp, {}).get('floor')!r} at {b} ({ctx.obj[b]['label']!r})")
        out, zb = _approach(sc, ctx.obj[car]["landings"][b]["sill_site"])
        if sc.drive_cart(out[:2], timeout_s=25.0, floor_z=zb) is None:
            whys.append(f"the cart could not drive out at {b}")
        measured["bump_m"] = round(sc.cart_dz_max, 4)
    measured["door_force_on_cart_N"] = round(sc.door_contact_max, 1)
    if sc.door_contact_max > CODE_MAX_DOOR_FORCE_N:
        whys.append(f"doors pushed the cart with {sc.door_contact_max:.0f} N")
    whys += ctx.done(sc, t_wall)
    return _whys(_ok(**{k: v for k, v in measured.items()}), whys), measured


def summon(ctx: Ctx, sc: Scenario, at: str) -> tuple[str | None, float]:
    """Call a car to `at` (any destination) and wait until it is leveled and open there."""
    other = ctx.floors[1] if at == ctx.floors[0] else ctx.floors[0]
    b = ctx.call_button(at, other)
    if b is None:
        return None, 0.0
    t = sc.sim.t
    _press_until_registered(sc, b)
    tol = float(ctx.run_params["leveling_tol_m"]) + 0.001
    sc.run(float(ctx.run_params["max_wait_s"]) + 10.0, until=lambda: sc.at_floor(at, tol) is not None)
    return sc.at_floor(at, tol), t


def doors(ctx: Ctx) -> tuple[dict, dict]:
    p = ctx.run_params
    whys, measured = [], {}
    for label, over in (("declared", {}), ("override", {"dwell_s": float(p["dwell_s"]) + 3.0})):
        t_wall = time.time()
        sc = ctx.scenario(over)
        at = ctx.floors[-1] if len(ctx.floors) > 1 else ctx.floors[0]
        car, t0 = summon(ctx, sc, at)
        if car is None:
            whys.append(f"no car came to {at}")
            whys += ctx.done(sc, t_wall)
            continue
        sc.run(float(p["dwell_s"]) + float(p["door_close_s"]) + 8.0)
        times = _door_times(sc.rows, car, t0, sc.sim.t)
        whys += ctx.done(sc, t_wall)
        want = {**p, **over}
        if label == "declared":
            measured.update(times)
        for key, rel, floor in (("door_open_s", 0.25, 0.5), ("door_close_s", 0.25, 0.5), ("dwell_s", 0.2, 0.6)):
            if key not in times:
                whys.append(f"{label}: could not measure {key} (doors never fully opened and closed)")
            elif abs(times[key] - float(want[key])) > max(rel * float(want[key]), floor):
                whys.append(f"{label}: {key} {times[key]} s (declared {float(want[key]):.2f})")
    return _whys(_ok(**measured), whys), measured


def obstruction(ctx: Ctx) -> dict:
    p = ctx.run_params
    t_wall = time.time()
    sc = ctx.scenario()
    at = ctx.floors[0]
    car, _ = summon(ctx, sc, at)
    if car is None:
        ctx.done(sc, t_wall)
        return _fail(f"no car came to {at}")
    closing = sc.run(float(p["dwell_s"]) + 10.0, until=lambda: sc.open_fraction(ctx.obj[car]["doors"]) < 0.85)
    if closing is None:
        ctx.done(sc, t_wall)
        return _fail("the doors never started closing")
    pl, Rl = sc.site(ctx.obj[car]["landings"][at]["sill_site"])
    sc.place_block(pl + Rl[:, 0] * 0.0 + np.array([0, 0, 0.4]))
    sc.door_contact_max = 0.0
    lowest = 1.0
    t_block = sc.sim.t
    reopened = sc.run(6.0, until=lambda: sc.open_fraction(ctx.obj[car]["doors"]) >= 0.9 and sc.sim.t - t_block > 0.5)
    fr = [r[3][car] for r in sc.rows if r[0] >= t_block]
    lowest = min(fr) if fr else 1.0
    force, peak = sc.door_contact_max, sc.door_contact_peak
    sc.park_block()
    whys = ctx.done(sc, t_wall)
    out = _ok(door_force_N=round(force, 1), door_impact_peak_N=round(peak, 1), lowest_open_fraction=round(lowest, 2))
    limit = min(float(p["door_force_limit_N"]) * 1.15, CODE_MAX_DOOR_FORCE_N * 1.1)
    if reopened is None:
        whys.append("with a block in the doorway the doors did not reopen within 6 s")
    if lowest < 0.05:
        whys.append("the doors closed on the block")
    if force > limit:
        whys.append(f"the doors pushed the block with {force:.0f} N (declared limit {p['door_force_limit_N']}, code {CODE_MAX_DOOR_FORCE_N:.0f})")
    return _whys(out, whys)


def hold_open(ctx: Ctx) -> dict:
    p = ctx.run_params
    t_wall = time.time()
    sc = ctx.scenario()
    at = ctx.floors[0]
    car, _ = summon(ctx, sc, at)
    if car is None:
        ctx.done(sc, t_wall)
        return _fail(f"no car came to {at}")
    btn = ctx.buttons(type="door_open", elevator=car)
    if not btn:
        ctx.done(sc, t_wall)
        return _ok(skipped=f"{car} has no door_open button")
    hold = float(p["dwell_s"]) + 4.0
    t0 = sc.sim.t
    sc.press_button(btn[0], hold_s=hold)
    fr = [r[3][car] for r in sc.rows if t0 + 0.5 <= r[0] <= t0 + hold]
    whys = ctx.done(sc, t_wall)
    if fr and min(fr) < 0.85:
        whys.append(f"holding {btn[0]} for {hold:.0f} s, the doors closed to {min(fr):.0%} open")
    return _whys(_ok(button=btn[0]), whys)


def calls(ctx: Ctx) -> dict:
    p = ctx.run_params
    t_wall = time.time()
    sc = ctx.scenario()
    picks = ctx.floors if len(ctx.floors) <= 3 else [ctx.floors[0], ctx.floors[len(ctx.floors) // 2], ctx.floors[-1]]
    asked = {}
    for i, f in enumerate(picks):
        to = picks[(i + 1) % len(picks)]
        b = ctx.call_button(f, to)
        if b:
            asked[f] = b
            sc.press_button(b)
            if i == 0:
                sc.press_button(b)  # a double press
    tol = float(p["leveling_tol_m"]) + 0.001
    served: set = set()

    def all_served():
        for f in asked:
            if sc.at_floor(f, tol):
                served.add(f)
        return len(served) == len(asked)

    budget = len(asked) * (float(p["max_wait_s"]) + float(p["dwell_s"]) + float(p["door_close_s"])) + 20
    sc.run(budget, until=all_served)
    whys = ctx.done(sc, t_wall)
    missing = sorted(set(asked) - served)
    if missing:
        whys.append(f"calls at {missing} were not served within {budget:.0f} s")
    return _whys(_ok(calls=len(asked), served=len(served)), whys)


def faults(ctx: Ctx) -> dict:
    out, whys = {}, []
    for f in ctx.faults:
        name, rate = f["name"], f.get("rate_param")
        if not rate:
            whys.append(f"{name}: no rate_param")
            continue
        t_wall = time.time()
        sc = ctx.scenario({rate: 1.0})
        at, to = ctx.floors[0], ctx.floors[-1]
        b = ctx.call_button(at, to)
        rec = f.get("recovery") or {}
        if "hold_s" in rec:
            t0 = sc.sim.t
            sc.press_button(b, hold_s=float(rec["hold_s"]) + 0.5)
        else:
            _press_until_registered(sc, b, presses=int(rec.get("presses", 1)))
        extra = float(rec.get("wait_s", 0.0))
        tol = float(ctx.run_params["leveling_tol_m"]) + 0.001
        came = sc.run(float(ctx.run_params["max_wait_s"]) + extra + 15.0, until=lambda: sc.at_floor(at, tol) is not None)
        fired = [e for e in sc.events(kind="fault") if e.get("name") == name]
        problems = ctx.done(sc, t_wall)
        r = {"fired": len(fired), "recovered": came is not None}
        if not fired:
            problems.append("did not fire at rate 1")
        if came is None:
            problems.append(f"after its recovery ({rec}), no car came")
        if problems:
            whys.append(f"{name}: " + "; ".join(problems))
        out[name] = r
    return _whys(_ok(faults=out), whys)


def run(asset_dir: Path, fam: dict, params: dict | None = None, out_dir: Path | None = None, sandbox=(), python=None) -> dict:
    t0 = time.time()
    asset_dir = Path(asset_dir)
    out_dir = Path(out_dir or asset_dir / "lab")
    out_dir.mkdir(parents=True, exist_ok=True)
    params = params if params is not None else json.loads((asset_dir / "params.json").read_text())
    checks: dict = {}
    measured: dict = {}
    try:
        ctx = Ctx(asset_dir, fam, params, sandbox, python)
    except (SC.SceneError, KeyError, ValueError) as e:
        result = {"ok": False, "failed": ["structure"], "checks": {"structure": _fail(str(e)[:600])}, "measured": {}, "check_s": round(time.time() - t0, 2)}
        (out_dir / "checks.json").write_text(json.dumps(result, indent=1))
        return result
    checks["structure"] = structure(ctx, params)
    if checks["structure"]["ok"]:
        for name, fn in (("trip", lambda: trip(ctx, ctx.floors[0], ctx.floors[-1])), ("doors", lambda: doors(ctx))):
            try:
                checks[name], m = fn()
                measured.update(m)
            except (DynamicsError, SC.SceneError) as e:
                checks[name] = _fail(f"{type(e).__name__}: {e}"[:600])
        for name, fn in (("obstruction", obstruction), ("hold_open", hold_open), ("calls", calls), ("faults", faults)):
            try:
                checks[name] = fn(ctx)
            except (DynamicsError, SC.SceneError) as e:
                checks[name] = _fail(f"{type(e).__name__}: {e}"[:600])
        if isinstance(checks.get("obstruction", {}).get("door_force_N"), (int, float)):
            measured["door_force_N"] = checks["obstruction"]["door_force_N"]
        rt = ctx.sim_s / max(ctx.wall_s, 1e-6)
        measured["realtime_x"] = round(rt, 1)
        checks["speed"] = _ok(realtime_x=round(rt, 1)) if rt >= 1.0 else _fail(f"simulated {rt:.2f}x real time (must be faster than real time)")
    failed = [k for k, v in checks.items() if not v.get("ok")]
    result = {"ok": not failed, "failed": failed, "passed_fraction": round(1 - len(failed) / max(len(checks), 1), 3), "checks": checks,
              "measured": measured, "check_s": round(time.time() - t0, 2)}  # fmt: skip
    (out_dir / "checks.json").write_text(json.dumps(result, indent=1, default=float))
    return result
