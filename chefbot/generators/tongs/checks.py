"""The physics checks for a generated tongs (family.yaml `checks`), in the lab's scenes (scene.py), in the lab frame:
x along the arms (toward the tips), y the opening (tip_a -> tip_b), z the hinge axis (up).

    structure   compiles; the bodies and sites; collision budget; origin at the bounding box centre
    rest        held still, no gravity: the tip gap; the arms don't interpenetrate
    squeeze     forces at grip_a/grip_b ramped 0 -> max: when the tips start to move and when they meet; they stop
                each other (no passing through, little penetration)
    release     the force removed: the spring reopens them
    hold        a test object pinched between the tips (with a hand's force), then gravity: it stays put
    counter     dropped onto a counter lying on its side: settles, stays on, doesn't sink in
    hand_fit    the handles fit the Franka Hand (77 mm) and it can squeeze them shut
    declared    length, mass, rest gap and closing force as measured vs as declared

`run(asset_dir, family)` returns {"ok", "checks": {name: {"ok", "why", ...}}, "measured": {...}} and leaves
traces for the renders in the asset's lab/ folder (squeeze.npz, settled.npz).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import mujoco
import numpy as np

from chefbot.autoresearch.generator_lab import scene as S

FIXTURE_POS = np.array([0.0, 0.0, 1.4])  # the stand: well above the counter
DT_LOG = 0.01


class Rig:
    """The object in one scene, with its sites, arm bodies and collision geoms looked up."""

    def __init__(self, asset_dir: Path, frame: S.Frame, *, fixed: bool, pos, gravity: bool, extra: str = ""):
        self.m, self.d = S.load(S.compose(asset_dir, fixed=fixed, pos=pos, R=frame.R, gravity=gravity, extra=extra))
        m = self.m
        self.site = {s: m.site(s).id for s in ("grip_a", "grip_b", "tip_a", "tip_b")}
        self.arm = {a: m.body(a).id for a in ("arm_a", "arm_b")}
        self.obj = m.body("object").id
        self.arm_geoms = {a: set(S.collision_geoms(m, S.subtree(m, b))) for a, b in self.arm.items()}
        self.obj_geoms = set(S.collision_geoms(m, S.subtree(m, self.obj)))

    def p(self, s: str) -> np.ndarray:
        return self.d.site_xpos[self.site[s]].copy()

    def gap(self) -> float:
        return float(np.linalg.norm(self.p("tip_b") - self.p("tip_a")))

    def side(self) -> float:
        """(tip_b - tip_a) along y: positive while the tips are on their own sides."""
        return float((self.p("tip_b") - self.p("tip_a"))[1])

    def contacts(self, a: set[int], b: set[int]) -> list[tuple[float, np.ndarray]]:
        d = self.d
        out = []
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in a and g2 in b) or (g1 in b and g2 in a):
                out.append((float(c.dist), c.pos.copy()))
        return out

    def arm_contacts(self) -> list[tuple[float, np.ndarray]]:
        return self.contacts(self.arm_geoms["arm_a"], self.arm_geoms["arm_b"])

    def squeeze(self, force: float) -> None:
        """Push grip_a and grip_b toward each other with `force` N each."""
        d = self.d
        d.xfrc_applied[:] = 0
        ga, gb = self.p("grip_a"), self.p("grip_b")
        n = (gb - ga) / max(np.linalg.norm(gb - ga), 1e-9)
        for arm, p, f in (("arm_a", ga, force * n), ("arm_b", gb, -force * n)):
            b = self.arm[arm]
            d.xfrc_applied[b, :3] += f
            d.xfrc_applied[b, 3:] += np.cross(p - d.xipos[b], f)

    def step(self, seconds: float) -> None:
        mujoco.mj_step(self.m, self.d, nstep=max(1, int(round(seconds / self.m.opt.timestep))))

    def unstable(self) -> bool:
        return bool(not np.all(np.isfinite(self.d.qpos)) or np.abs(self.d.qvel).max() > 1e3)


def _fail(why: str, **kw) -> dict:
    return {"ok": False, "why": why, **kw}


def structure(asset_dir: Path, fam: dict) -> tuple[dict, S.Frame | None]:
    lim = fam["generation"]
    try:
        frame = S.natural_frame(asset_dir)
        rig = Rig(asset_dir, frame, fixed=True, pos=(0, 0, 0), gravity=False)
    except (S.AssetError, KeyError) as e:
        return _fail(str(e).strip('"')), None
    m, d = rig.m, rig.d
    a, b = rig.arm["arm_a"], rig.arm["arm_b"]
    if b in S.subtree(m, a) or a in S.subtree(m, b):
        return _fail("arm_a and arm_b are nested: each arm must be its own child of 'object'"), frame
    geoms = sorted(rig.obj_geoms)
    meshes = {int(m.geom_dataid[g]) for g in geoms if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH}
    verts = int(sum(m.mesh_vertnum[i] for i in meshes))
    lo, hi = S.world_box(m, d, [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in S.subtree(m, rig.obj)])
    centre = (lo + hi) / 2  # in the lab frame, the object's origin at 0
    out = {"ok": True, "collision_geoms": len(geoms), "collision_verts": verts, "mass_kg": round(float(m.body_subtreemass[rig.obj]), 4),
           "length_m": round(float(hi[0] - lo[0]), 4), "size_m": [round(float(v), 4) for v in hi - lo],
           "origin_offset_m": round(float(np.linalg.norm(centre)), 4)}  # fmt: skip
    joints = [j for j in range(m.njnt) if int(m.jnt_bodyid[j]) in S.subtree(m, rig.obj)]
    out["joints"] = [m.joint(j).name or f"#{j}" for j in joints]
    whys = []
    if len(geoms) > lim["max_collision_geoms"]:
        whys.append(f"{len(geoms)} collision geoms (limit {lim['max_collision_geoms']})")
    if verts > lim["max_collision_verts"]:
        whys.append(f"{verts} collision-mesh vertices (limit {lim['max_collision_verts']})")
    if out["origin_offset_m"] > 0.01:
        whys.append(f"the origin is {out['origin_offset_m'] * 100:.1f} cm from the bounding box centre (limit 1 cm)")
    if not rig.arm_geoms["arm_a"] or not rig.arm_geoms["arm_b"]:
        whys.append("an arm has no collision geoms")
    if not any(m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0 and m.geom_group[g] in (1, 2) for g in range(m.ngeom)
               if int(m.geom_bodyid[g]) in S.subtree(m, rig.obj)):  # fmt: skip
        whys.append("no visual geoms (group 1 or 2, contype=conaffinity=0)")
    if whys:
        out.update(ok=False, why="; ".join(whys))
    return out, frame


def rest_and_squeeze(asset_dir: Path, frame: S.Frame, fam: dict, out_dir: Path | None) -> dict:
    c = fam["checks"]
    rig = Rig(asset_dir, frame, fixed=True, pos=FIXTURE_POS, gravity=False)
    m, d = rig.m, rig.d
    rig.step(1.0)
    res: dict = {}
    over = [-dist for dist, _ in rig.arm_contacts() if dist < 0]
    rest_gap = rig.gap()
    moving = float(np.abs(d.qvel).max()) if m.nv else 0.0
    res["rest"] = {"ok": True, "rest_tip_gap_m": round(rest_gap, 4), "grip_spread_m": round(float(np.linalg.norm(rig.p("grip_b") - rig.p("grip_a"))), 4),
                   "arm_overlap_m": round(max(over, default=0.0), 4), "still": moving < 0.01}  # fmt: skip
    whys = []
    if max(over, default=0.0) > c["rest_overlap_m"]:
        whys.append(f"the arms interpenetrate {max(over) * 1000:.1f} mm at rest")
    if moving >= 0.01:
        whys.append(f"still moving after 1 s held with no gravity (|qvel| {moving:.3f})")
    if rig.side() <= 0:
        whys.append("tip_b is not on the far side of tip_a at rest")
    if whys:
        res["rest"].update(ok=False, why="; ".join(whys))

    # Squeeze: ramp 0 -> max_force over ramp_s, then hold max_force 0.5 s.
    fmax, ramp = float(c["max_force_N"]), float(c["ramp_s"])
    log, t, start_f, close_f, passed, worst_pen = [], 0.0, None, None, False, 0.0
    tips_mid_rest = (rig.p("tip_a") + rig.p("tip_b")) / 2
    near_tips = 0.25 * float(np.linalg.norm(tips_mid_rest - (rig.p("grip_a") + rig.p("grip_b")) / 2)) + 0.02
    traj = {"t": [], "qpos": [], "force": [], "grip_a": [], "grip_b": []}
    while t < ramp + 0.5:
        f = fmax * min(t / ramp, 1.0)
        rig.squeeze(f)
        rig.step(DT_LOG)
        t += DT_LOG
        if rig.unstable():
            res["squeeze"] = _fail(f"the simulation blew up at {f:.1f} N")
            return res
        g = rig.gap()
        cons = rig.arm_contacts()
        mid = (rig.p("tip_a") + rig.p("tip_b")) / 2
        tip_touch = any(np.linalg.norm(p - mid) < near_tips for _, p in cons)
        worst_pen = max([worst_pen] + [-dist for dist, _ in cons if dist < 0])
        if start_f is None and g < rest_gap - 0.001:
            start_f = f
        if close_f is None and (g <= c["contact_gap_m"] or tip_touch):
            close_f = f
        if rig.side() < -0.001:
            passed = True
        log.append((round(f, 2), round(g, 5), tip_touch))
        for k, v in (("t", t), ("qpos", d.qpos.copy()), ("force", f), ("grip_a", rig.p("grip_a")), ("grip_b", rig.p("grip_b"))):
            traj[k].append(v)
    closed_gap = min(g for _, g, _ in log)
    sq = {"ok": True, "start_force_N": None if start_f is None else round(start_f, 2), "closing_force_N": None if close_f is None else round(close_f, 2),
          "closed_tip_gap_m": round(closed_gap, 4), "max_penetration_m": round(worst_pen, 4), "passes_through": passed,
          "curve": log[:: max(1, len(log) // 40)]}  # fmt: skip
    whys = []
    if close_f is None:
        whys.append(f"the tips never met, even at {fmax:g} N (closest {closed_gap * 1000:.1f} mm)")
    if passed:
        whys.append("the tips passed through each other")
    if worst_pen > c["max_closed_penetration_m"]:
        whys.append(f"the arms interpenetrate {worst_pen * 1000:.1f} mm under {fmax:g} N")
    if whys:
        sq.update(ok=False, why="; ".join(whys))
    res["squeeze"] = sq

    # Release.
    d.xfrc_applied[:] = 0
    t, reopen = 0.0, None
    while t < 1.5:
        rig.step(DT_LOG)
        t += DT_LOG
        for k, v in (("t", ramp + 0.5 + t), ("qpos", d.qpos.copy()), ("force", 0.0), ("grip_a", rig.p("grip_a")), ("grip_b", rig.p("grip_b"))):
            traj[k].append(v)
        if reopen is None and abs(rig.gap() - rest_gap) <= c["reopen_tol_m"]:
            reopen = t
    rel = {"ok": reopen is not None, "reopen_s": None if reopen is None else round(reopen, 2), "tip_gap_after_m": round(rig.gap(), 4)}
    if reopen is None:
        rel["why"] = f"1.5 s after release the tips are {rig.gap() * 1000:.1f} mm apart (rest {rest_gap * 1000:.1f} mm)"
    res["release"] = rel
    if out_dir is not None:
        np.savez_compressed(out_dir / "squeeze.npz", **{k: np.asarray(v) for k, v in traj.items()}, R=frame.R, pos=FIXTURE_POS)
    return res


def hold(asset_dir: Path, frame: S.Frame, fam: dict, rest_gap: float, closing_force: float | None) -> dict:
    c = fam["checks"]
    out = {}
    if closing_force is None:
        return {"ok": False, "why": "the tips never meet, so nothing can be held"}
    force = min(max(1.5 * closing_force, closing_force + 5.0), float(c["hand_max_force_N"]))
    probe = Rig(asset_dir, frame, fixed=True, pos=FIXTURE_POS, gravity=False)
    mid = (probe.p("tip_a") + probe.p("tip_b")) / 2
    for name, spec in c["hold_objects"].items():
        size = [float(v) for v in spec["size"]]
        width = 2 * (size[1] if spec["type"] == "box" else size[0])  # across the opening (y); capsules and cylinders stand upright
        if width > rest_gap - 0.004:
            out[name] = {"ok": True, "skipped": f"{width * 1000:.0f} mm wide; the tongs open {rest_gap * 1000:.0f} mm"}
            continue
        body = (f'<body name="lab_test" pos="{mid[0]:.5f} {mid[1]:.5f} {mid[2]:.5f}"><freejoint/>'
                f'<geom name="lab_test" type="{spec["type"]}" size="{" ".join(map(str, size))}" mass="{spec["mass"]}" '
                f'friction="{spec["friction"]} 0.005 0.0001" material="lab_test"/></body>')  # fmt: skip
        rig = Rig(asset_dir, frame, fixed=True, pos=FIXTURE_POS, gravity=False, extra=body)
        m, d = rig.m, rig.d
        test = m.body("lab_test").id
        tg = {m.geom("lab_test").id}
        for i in range(50):  # close on it: 0.5 s ramp, 0.3 s hold
            rig.squeeze(force * min(1.0, (i + 1) / 30))
            rig.step(0.016)
        both = bool(rig.contacts(tg, rig.arm_geoms["arm_a"])) and bool(rig.contacts(tg, rig.arm_geoms["arm_b"]))
        p0 = d.xpos[test].copy()
        m.opt.gravity[:] = (0, 0, -9.81)
        t = 0.0
        while t < c["hold_s"]:
            rig.squeeze(force)
            rig.step(DT_LOG)
            t += DT_LOG
            if rig.unstable():
                break
        slip = float(np.linalg.norm(d.xpos[test] - p0)) if not rig.unstable() else float("inf")
        r = {"ok": both and slip <= c["hold_slip_m"], "slip_m": round(slip, 4), "force_N": round(force, 1), "gripped_by_both_arms": both}
        if not both:
            r["why"] = "closing on it, the tips did not both touch it"
        elif slip > c["hold_slip_m"]:
            r["why"] = f"it slipped {slip * 100:.1f} cm in {c['hold_s']:g} s at {force:.0f} N"
        out[name] = r
    bad = [k for k, v in out.items() if not v["ok"]]
    return {"ok": not bad, **({"why": "; ".join(f"{k}: {out[k]['why']}" for k in bad)} if bad else {}), "objects": out}


def counter(asset_dir: Path, frame: S.Frame, fam: dict, out_dir: Path | None) -> dict:
    c = fam["checks"]
    probe = Rig(asset_dir, frame, fixed=True, pos=(0, 0, 0), gravity=False)
    lo, hi = S.world_box(probe.m, probe.d, sorted(probe.obj_geoms))
    pos = (0.0, 0.0, S.COUNTER_TOP - lo[2] + 0.01)
    rig = Rig(asset_dir, frame, fixed=False, pos=pos, gravity=True)
    m, d = rig.m, rig.d
    counter_g = {m.geom("lab_counter").id}
    t, worst, fast_late = 0.0, 0.0, 0.0
    while t < c["settle_s"]:
        rig.step(DT_LOG)
        t += DT_LOG
        if rig.unstable():
            return _fail("the simulation blew up on the counter")
        if t > c["settle_s"] - 1.0:  # resting (not the landing)
            worst = max([worst] + [-dist for dist, _ in rig.contacts(rig.obj_geoms, counter_g) if dist < 0])
        if t > 1.0:
            fast_late = max(fast_late, float(np.linalg.norm(d.qvel[:3])))
    speed = float(np.linalg.norm(d.qvel[:3]))
    xyz = d.xpos[rig.obj].copy()
    hinge_axis = d.xmat[rig.obj].reshape(3, 3) @ frame.h
    tilt = float(np.degrees(np.arccos(min(1.0, abs(hinge_axis[2])))))
    lo2, _ = S.world_box(m, d, sorted(rig.obj_geoms))
    res = {"ok": True, "final_speed_mps": round(speed, 4), "max_penetration_m": round(worst, 4), "hinge_tilt_deg": round(tilt, 1),
           "lies_on_side": tilt < 30, "drift_m": round(float(np.linalg.norm(xyz[:2] - np.array(pos[:2]))), 4),
           "bottom_z_m": round(float(lo2[2]), 4)}  # fmt: skip
    whys = []
    if speed > c["settle_speed_mps"]:
        whys.append(f"still moving {speed * 100:.1f} cm/s after {c['settle_s']:g} s")
    if worst > c["max_counter_penetration_m"]:
        whys.append(f"sinks {worst * 1000:.1f} mm into the counter")
    if lo2[2] < S.COUNTER_TOP - 0.05:
        whys.append("fell off the counter")
    if whys:
        res.update(ok=False, why="; ".join(whys))
    if out_dir is not None:
        np.savez_compressed(out_dir / "settled.npz", qpos=d.qpos.copy(), R=frame.R, pos=np.array(pos))
    return res


def declared(params: dict, measured: dict, fam: dict) -> dict:
    out, whys = {"ok": True}, []
    for key, tol in fam["checks"]["declared"].items():
        dv, mv = params.get(key), measured.get(key)
        if not isinstance(dv, (int, float)) or mv is None:
            whys.append(f"{key}: declared {dv!r}, measured {mv!r}")
            continue
        err = abs(mv - dv)
        ok = err <= max(tol["rel"] * abs(dv), tol["abs"])
        out[key] = {"declared": dv, "measured": mv, "ok": ok}
        if not ok:
            whys.append(f"{key}: declared {dv:g}, measured {mv:g}")
    missing = [k for k in fam["required_params"] if k not in params]
    if missing:
        whys.append(f"missing parameters {missing}")
    if whys:
        out.update(ok=False, why="; ".join(whys))
    return out


def run(asset_dir: Path, fam: dict, params: dict | None = None, out_dir: Path | None = None, **_) -> dict:
    """All checks on one generated asset. out_dir (default asset_dir/lab) gets checks.json and the render traces."""
    t0 = time.time()
    asset_dir = Path(asset_dir)
    out_dir = Path(out_dir or asset_dir / "lab")
    out_dir.mkdir(parents=True, exist_ok=True)
    params = params if params is not None else json.loads((asset_dir / "params.json").read_text())
    checks: dict = {}
    st, frame = structure(asset_dir, fam)
    checks["structure"] = st
    measured: dict = {k: st.get(k) for k in ("length_m", "mass_kg")}
    if frame is not None:  # it compiled: simulate it even if a budget is exceeded
        try:
            rs = rest_and_squeeze(asset_dir, frame, fam, out_dir)
            checks.update(rs)
            measured["rest_tip_gap_m"] = rs["rest"]["rest_tip_gap_m"]
            measured["closing_force_N"] = rs.get("squeeze", {}).get("closing_force_N")
            checks["hold"] = hold(asset_dir, frame, fam, rs["rest"]["rest_tip_gap_m"], measured["closing_force_N"])
            checks["counter"] = counter(asset_dir, frame, fam, out_dir)
            spread, cf = rs["rest"]["grip_spread_m"], measured["closing_force_N"]
            hf = {"ok": True, "grip_spread_m": spread, "closing_force_N": cf}
            whys = []
            if spread > fam["checks"]["hand_open_m"]:
                whys.append(f"the handles are {spread * 1000:.0f} mm apart at the grip sites (the hand opens {fam['checks']['hand_open_m'] * 1000:.0f})")
            if cf is None or cf > fam["checks"]["hand_max_force_N"]:
                whys.append(f"closing takes {cf} N (the hand squeezes {fam['checks']['hand_max_force_N']} N)")
            if whys:
                hf.update(ok=False, why="; ".join(whys))
            checks["hand_fit"] = hf
        except S.AssetError as e:
            checks["sim"] = _fail(str(e))
    checks["declared"] = declared(params, measured, fam)
    failed = [k for k, v in checks.items() if not v.get("ok")]
    result = {"ok": not failed, "failed": failed, "passed_fraction": round(1 - len(failed) / max(len(checks), 1), 3),
              "checks": checks, "measured": measured, "check_s": round(time.time() - t0, 2)}  # fmt: skip
    (out_dir / "checks.json").write_text(json.dumps(result, indent=1, default=float))
    return result

