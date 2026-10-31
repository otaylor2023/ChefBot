"""A RoboCasa kitchen, with the objects a setup names, exported to plain MuJoCo: the fixtures and objects as RoboCasa
built them, without its robot. Runs where RoboCasa is installed; the export is all the MuJoCo kitchen needs.

    python -m chefbot.environments.mujoco_kitchen.export --layout 11 --style 20 --objects egg:objaverse/egg/egg_3,...
    export_kitchen(layout, style, objects, out) -> out/kitchen.xml, out/meta.json

meta.json: the counters and other map fixtures (name, label, root body, box: origin and edge vectors), the objects
(name, category, model, root body, size, bottom offset, mass), the floor geoms. kitchen_dir() is where an export is
cached ($CHEFBOT_KITCHENS, else ~/.cache/chefbot/kitchens), one folder per (layout, style, objects).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROBOT_BODIES = {"left_eef_target", "right_eef_target", "robot0_base"}
VERSION = 1  # bump when the export changes: cached exports are rebuilt


def kitchens_root() -> Path:
    return Path(os.environ.get("CHEFBOT_KITCHENS", Path.home() / ".cache" / "chefbot" / "kitchens"))


def kitchen_key(layout: int, style: int, objects: list[dict]) -> str:
    spec = json.dumps({"v": VERSION, "objects": [[o["category"], o.get("model")] for o in objects]}, sort_keys=True)
    return f"l{layout:02d}_s{style:02d}_{hashlib.sha1(spec.encode()).hexdigest()[:10]}"


def kitchen_dir(layout: int, style: int, objects: list[dict]) -> Path:
    return kitchens_root() / kitchen_key(layout, style, objects)


def ensure_kitchen(layout: int, style: int, objects: list[dict]) -> Path:
    """The cached export, made now (in a subprocess: RoboCasa and its GL state stay out of the caller) if missing."""
    out = kitchen_dir(layout, style, objects)
    if (out / "meta.json").exists():
        return out
    import subprocess
    import sys

    spec = ",".join(f"{o['category']}:{o.get('model') or ''}" for o in objects)
    subprocess.run([os.environ.get("CHEFBOT_ROBOCASA_PYTHON", sys.executable), "-m", "chefbot.environments.mujoco_kitchen.export",
                    "--layout", str(layout), "--style", str(style), "--objects", spec, "--out", str(out)], check=True)  # fmt: skip
    return out


def _box(box) -> dict:
    return {k: np.asarray(getattr(box, k), float).round(5).tolist() for k in ("origin", "width", "depth", "height")}


def export_kitchen(layout: int, style: int, objects: list[dict], out: Path, seed: int = 0) -> dict:
    from chefbot.environments.robocasa.config import RoboCasaConfig
    from chefbot.environments.robocasa.detector import named_fixtures
    from chefbot.environments.robocasa.env import RoboCasaEnv
    from chefbot.environments.robocasa.spawn import object_names

    cfg = RoboCasaConfig.load(layout=layout, style=style, seed=seed, robot="PandaOmron", gripper="franka_hand",
                              objects=tuple({**o, "surface": "counter"} for o in objects))  # fmt: skip
    env = RoboCasaEnv(cfg, render=False)
    try:
        m, d = env.model, env.data
        root = ET.fromstring(env.suite.model.get_xml())
        wb = root.find("worldbody")
        for b in [b for b in wb if b.tag == "body" and b.get("name") in ROBOT_BODIES]:
            wb.remove(b)
        for tag in ("actuator", "sensor", "keyframe"):
            for el in root.findall(tag):
                root.remove(el)
        names = object_names(objects)
        bodies = {n: env.suite.objects[n].root_body for n in names}
        for b in wb:
            if b.tag == "body" and b.get("name") in bodies.values():
                bid = m.body(b.get("name")).id
                b.set("pos", " ".join(f"{v:.6f}" for v in d.xpos[bid]))
                b.set("quat", " ".join(f"{v:.8f}" for v in d.xquat[bid]))
        out.mkdir(parents=True, exist_ok=True)
        (out / "kitchen.xml").write_text(ET.tostring(root, encoding="unicode"))
        objs = []
        for n, o in zip(names, objects):
            model = env.suite.objects[n]
            bid = m.body(bodies[n]).id
            objs.append({"name": n, "category": o["category"], "model": o.get("model"), "body": bodies[n],
                         "size": np.asarray(model.size, float).round(5).tolist(),
                         "bottom_offset": np.asarray(model.bottom_offset, float).round(5).tolist(),
                         "mass": round(float(m.body_subtreemass[bid]), 5)})  # fmt: skip
        fixtures = [{"name": f.name, "label": f.label, "sim_name": f.sim_name, "root": f.root, "box": _box(f.box)} for f in named_fixtures(env)]
        meta = {"layout": layout, "style": style, "version": VERSION, "objects": objs, "fixtures": fixtures,
                "floor_geoms": sorted(m.geom(int(g)).name for g in env._floor_geoms if m.geom(int(g)).name)}  # fmt: skip
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        return meta
    finally:
        env.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", type=int, required=True)
    ap.add_argument("--style", type=int, required=True)
    ap.add_argument("--objects", required=True, help="category:model,... (model may be empty)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    objects = []
    for item in a.objects.split(","):
        cat, _, model = item.partition(":")
        objects.append({"category": cat, **({"model": model} if model else {})})
    out = a.out or kitchen_dir(a.layout, a.style, objects)
    meta = export_kitchen(a.layout, a.style, objects, out)
    print(out, len(meta["fixtures"]), "fixtures", [o["name"] for o in meta["objects"]])


if __name__ == "__main__":
    main()
