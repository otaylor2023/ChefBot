"""Generated objects in RoboCasa: export an asset to RoboCasa's object layout, and check that it behaves the same in a
kitchen (ChefBot's RoboCasa env) as in the lab. Run by hand on a sample now and then, not on every asset:

    python -m chefbot.autoresearch.objects.robocasa ASSET_DIR [ASSET_DIR ...] [--layout 1 --style 1] [--out DIR]

Export: the asset's files, and its model.xml wrapped as RoboCasa expects (an outer body around `object`, a
`reg_bbox` region from the asset's bounds, the top-level defaults made a class, collision geoms in group 0 and visual
ones in group 1, the only groups robosuite keeps; single sizes padded for RoboCasa's scaler), plus a mark that tells ChefBot's
spawner to put the asset's own per-geom friction and contact softness back (RoboCasa overwrites them).

Check, per asset, in a kitchen on a counter: it spawns and settles, its geoms keep their own physics, and, lifted
off the counter with gravity compensated (as on the lab's stand), squeezed at its grip sites it closes at the force
the lab measured and reopens. The squeeze lying on the counter is reported too (its friction changes it), not checked.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from chefbot.autoresearch.objects import scene as S

CATEGORY = "generated"


def export(asset_dir: Path, dst: Path) -> Path:
    """A RoboCasa-ready copy of the asset in `dst` (emptied first)."""
    asset_dir, dst = Path(asset_dir), Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(asset_dir, dst, ignore=shutil.ignore_patterns("lab", "gen_log.txt", "params.json"))
    m, d = S.load(S.compose(asset_dir, fixed=True, pos=(0, 0, 0), gravity=False, counter=False))
    obj_geoms = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in S.subtree(m, m.body("object").id)]
    lo, hi = S.world_box(m, d, obj_geoms)
    root = ET.parse(asset_dir / "model.xml").getroot()
    defaults = root.find("default")
    if defaults is None:
        defaults = ET.SubElement(root, "default")
    obj = root.find("worldbody/body")
    if any(el.tag != "default" for el in defaults) or defaults.attrib:  # top-level defaults: a class of their own
        own = copy.deepcopy(defaults)
        own.tag, own.attrib = "default", {"class": "generated_main"}
        for el in list(defaults):
            defaults.remove(el)
        defaults.append(own)
        obj.set("childclass", "generated_main")
    # robosuite keeps only group 0 (collision) and group 1 (visual) geoms of an object: the contract also allows 3 and 2.
    for g in obj.iter("geom"):
        visual = g.get("contype") == "0" and g.get("conaffinity") == "0"
        g.set("group", "1" if visual else "0")
        size = (g.get("size") or "").split()
        if len(size) == 1:  # RoboCasa's scaler takes 2 or 3 sizes (MuJoCo ignores the extra ones)
            g.set("size", " ".join(size * 3))
    region = ET.SubElement(defaults, "default", {"class": "region"})
    ET.SubElement(region, "geom", {"group": "1", "conaffinity": "0", "contype": "0", "rgba": "0 1 0 0", "mass": "0"})
    c, half = (lo + hi) / 2, (hi - lo) / 2
    ET.SubElement(obj, "geom", {"class": "region", "name": "reg_bbox", "type": "box", "pos": " ".join(f"{v:.6f}" for v in c),
                                "size": " ".join(f"{v:.6f}" for v in half)})  # fmt: skip
    wb = root.find("worldbody")
    wb.remove(obj)
    outer = ET.SubElement(wb, "body")
    outer.append(obj)
    (dst / "model.xml").write_text(ET.tostring(root, encoding="unicode"))
    from chefbot.envs.robocasa.spawn import GENERATED_MARK

    (dst / GENERATED_MARK).write_text(str(asset_dir) + "\n")
    return dst


def register(folder: Path) -> None:
    """Make RoboCasa's object registry know the exported models under folder/<name>/ (category CATEGORY)."""
    from robocasa.models.objects.kitchen_object_utils import ObjCat
    from robocasa.models.objects.kitchen_objects import OBJ_CATEGORIES

    OBJ_CATEGORIES[CATEGORY] = {"objaverse": ObjCat(name=CATEGORY, types=("tool",), graspable=True, model_folders=[str(folder)])}


def check(asset_dir: Path, exported: Path, layout: int, style: int) -> dict:
    from chefbot.envs.robocasa.config import RoboCasaConfig
    from chefbot.envs.robocasa.env import RoboCasaEnv

    lab = json.loads((Path(asset_dir) / "lab" / "checks.json").read_text()) if (Path(asset_dir) / "lab" / "checks.json").exists() else {}
    cfg = RoboCasaConfig.load(layout=layout, style=style, seed=0, robot="PandaOmron", gripper="franka_hand",
                              objects=({"category": CATEGORY, "model": str(exported), "surface": "counter"},))  # fmt: skip
    out: dict = {"asset": str(asset_dir), "kitchen": [layout, style]}
    try:
        env = RoboCasaEnv(cfg, render=False)
    except Exception as e:  # noqa: BLE001
        return {**out, "ok": False, "why": f"did not spawn: {type(e).__name__}: {e}"}
    m, d = env.model, env.data
    obj = env.suite.objects[CATEGORY]
    pre = obj.naming_prefix
    root = m.body(obj.root_body).id
    own = mujoco.MjModel.from_xml_path(str(exported / "model.xml"))
    own_f = {own.geom(g).name: own.geom_friction[g].copy() for g in range(own.ngeom) if own.geom(g).name}
    mismatch = [n for n, f in own_f.items() if not n.endswith("reg_bbox") and m.geom(pre + n) and not np.allclose(m.geom_friction[m.geom(pre + n).id], f)]
    out["friction_kept"] = not mismatch
    free = m.joint(obj.joints[0])
    out["free_joint_first"] = int(m.jnt_type[free.id]) == int(mujoco.mjtJoint.mjJNT_FREE)
    adr = int(m.jnt_qposadr[free.id])
    start = d.qpos[adr : adr + 3].copy()
    env.wait_for_next_tick(2.0)
    out["settled_moved_m"] = round(float(np.linalg.norm(d.qpos[adr : adr + 3] - start)), 4)
    out["speed_mps"] = round(float(np.linalg.norm(d.qvel[int(m.jnt_dofadr[free.id]) : int(m.jnt_dofadr[free.id]) + 3])), 4)

    site = {s: m.site(pre + s).id for s in ("grip_a", "grip_b", "tip_a", "tip_b")}
    arm = {a: m.body(pre + a).id for a in ("arm_a", "arm_b")}

    def p(s):
        return d.site_xpos[site[s]].copy()

    from chefbot.autoresearch.objects.scene import subtree

    arm_geoms = {a: {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in subtree(m, b) and (m.geom_contype[g] or m.geom_conaffinity[g])}
                 for a, b in arm.items()}  # fmt: skip

    def tips_touch() -> bool:
        """As in the lab: the arms touching near the tips (or the tip sites within 3 mm)."""
        mid = (p("tip_a") + p("tip_b")) / 2
        reach = 0.25 * float(np.linalg.norm(mid - (p("grip_a") + p("grip_b")) / 2)) + 0.02
        for i in range(d.ncon):
            g1, g2 = int(d.contact[i].geom1), int(d.contact[i].geom2)
            if {g1, g2} & arm_geoms["arm_a"] and {g1, g2} & arm_geoms["arm_b"] and np.linalg.norm(d.contact[i].pos - mid) < reach:
                return True
        return float(np.linalg.norm(p("tip_b") - p("tip_a"))) <= 0.003

    def squeeze() -> tuple[float | None, float, float]:
        """Ramp the squeeze to fmax over 4 s of control ticks; then release for 1 s. (closing force, rest gap, gap after)"""
        rest = float(np.linalg.norm(p("tip_b") - p("tip_a")))
        close_f, fmax, steps = None, 40.0, 80
        for i in range(steps):
            f = fmax * (i + 1) / steps
            with env.lock:
                d.xfrc_applied[:] = 0
                ga, gb = p("grip_a"), p("grip_b")
                n = gb - ga
                n[2] = 0.0  # across the counter, never into or off it
                n /= max(np.linalg.norm(n), 1e-9)
                for a, pt, fv in (("arm_a", ga, f * n), ("arm_b", gb, -f * n)):
                    b = arm[a]
                    d.xfrc_applied[b, :3] = fv
                    d.xfrc_applied[b, 3:] = np.cross(pt - d.xipos[b], fv)
            env.wait_for_next_tick(1 / cfg.control_freq)
            if tips_touch():
                close_f = f
                break
        with env.lock:
            d.xfrc_applied[:] = 0
        env.wait_for_next_tick(1.0)
        return close_f, rest, float(np.linalg.norm(p("tip_b") - p("tip_a")))

    # Lying on the counter (for information: the counter's friction is part of it).
    f_counter, rest_counter, after_counter = squeeze()
    out["on_counter"] = {"closing_force_N": f_counter, "rest_tip_gap_m": round(rest_counter, 4), "reopened_gap_m": round(after_counter, 4)}
    # Lifted above the counter, gravity compensated: the lab's stand. This is the comparison.
    bodies = subtree(m, root)
    with env.lock:
        m.body_gravcomp[list(bodies)] = 1.0
        d.qpos[adr + 2] += 0.3
        d.qvel[int(m.jnt_dofadr[free.id]) : int(m.jnt_dofadr[free.id]) + 6] = 0
        mujoco.mj_forward(m, d)
    env.wait_for_next_tick(1.0)
    close_f, rest, after = squeeze()
    out.update(rest_tip_gap_m=round(rest, 4), closing_force_N=close_f, reopened_gap_m=round(after, 4))
    lab_f = lab.get("measured", {}).get("closing_force_N")
    out["lab_closing_force_N"] = lab_f
    whys = []
    if not out["free_joint_first"]:
        whys.append("the free joint is not first")
    if mismatch:
        whys.append(f"friction overwritten on {mismatch[:3]}")
    if out["speed_mps"] > 0.02:
        whys.append(f"still moving {out['speed_mps']} m/s after 2 s")
    if close_f is None:
        whys.append("the tips never met up to 40 N")
    elif lab_f and abs(close_f - lab_f) > max(0.2 * lab_f, 1.5):  # the kitchen's 20 Hz ramp is coarser than the lab's
        whys.append(f"closes at {close_f:.1f} N here, {lab_f} N in the lab")
    if abs(out["reopened_gap_m"] - rest) > 0.004:
        whys.append("did not reopen")
    env.close()
    return {**out, "ok": not whys, **({"why": "; ".join(whys)} if whys else {})}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("assets", nargs="+", type=Path)
    ap.add_argument("--layout", type=int, default=1)
    ap.add_argument("--style", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None, help="where the exports go (default: a temporary folder)")
    a = ap.parse_args()
    folder = a.out or Path(tempfile.mkdtemp(prefix="chefbot_robocasa_"))
    exported = {asset: export(asset, folder / f"{i:03d}_{asset.name}") for i, asset in enumerate(a.assets)}
    register(folder)
    results = []
    for asset, ex in exported.items():
        r = check(asset, ex, a.layout, a.style)
        results.append(r)
        print(json.dumps(r), flush=True)
    print(f"{sum(r['ok'] for r in results)}/{len(results)} behave the same in RoboCasa (exports in {folder})")
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
