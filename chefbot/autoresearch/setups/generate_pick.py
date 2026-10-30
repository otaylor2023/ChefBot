"""Write the RoboCasa pick setups (setups/robocasa_pick/<split>_<nn>_<object>.yaml) and their suites
(suites/robocasa_pick/{dev,val,test}.yaml).

Kitchens: each split its own. Test: RoboCasa's held-out kitchens (layouts and styles 1-10); dev: RoboCasa's training
layouts and styles 11-50; val: 51-60. Objects: RoboCasa's graspable categories with a model the Franka Hand can close
around from above (its narrow side at most MAX_NARROW_M across: the hand opens to 77 mm), less a few that are made of
pieces or lie flat; shuffled once and split 60/20/20 into dev/val/test categories, so val and test objects are never
seen in dev. Each setup: one kitchen, a target and two other objects of other types from the same split, each a model
pinned at generation (RoboCasa would otherwise draw one per process, not per setup).

    python -m chefbot.autoresearch.setups.generate_pick
"""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

from chefbot.autoresearch import suites
from chefbot.autoresearch.setups import ROOT

OUT = ROOT / "robocasa_pick"
SUITES = suites.ROOT / "robocasa_pick"
N = {"dev": 24, "val": 12, "test": 20}
SEEDS = {"dev": [0, 1, 2, 3, 4], "val": [0, 1], "test": [0, 1]}
KITCHENS = {"dev": range(11, 51), "val": range(51, 61), "test": range(1, 11)}  # layouts and styles, each
MAX_NARROW_M, MIN_HEIGHT_M, MAX_LONG_M = 0.065, 0.02, 0.25
REGISTRIES = {"objaverse": "objaverse", "aigen": "aigen_objs"}  # registry -> its folder under robocasa's objects
EXCLUDE = {  # made of pieces, flat, or awkward on a counter
    "grapes", "aluminum_foil", "straw", "skewers", "kebab_skewer", "kebabs", "sugar_cube", "ice_cube", "candy",
    "marshmallow", "tongs", "whisk", "ladle", "peeler", "pizza_cutter", "rolling_pin", "wooden_spoon", "sponge",
}  # fmt: skip


def model_size(mjcf: str, scale) -> np.ndarray | None:
    """The model's bounding box (x, y, z, m) at its category's scale, from its reg_bbox."""
    for geom in ET.parse(mjcf).getroot().iter("geom"):
        if geom.get("name") == "reg_bbox":
            half = np.array([float(v) for v in geom.get("size").split()])
            return half * 2 * np.asarray(scale if scale is not None else 1.0, dtype=float)
    return None


def graspable_models() -> dict[str, list[str]]:
    """Category -> its models the hand can pick (paths under robocasa's objects/, e.g. aigen_objs/apple/apple_3)."""
    import robocasa
    from robocasa.models.objects.kitchen_objects import OBJ_CATEGORIES

    objects = Path(robocasa.models.assets_root) / "objects"
    out = {}
    for cat, regs in sorted(OBJ_CATEGORIES.items()):
        if cat in EXCLUDE:
            continue
        models = []
        for reg, folder in REGISTRIES.items():
            meta = regs.get(reg)
            if meta is None or not meta.graspable:
                continue
            for mjcf in meta.mjcf_paths:
                size = model_size(mjcf, meta.scale)
                if size is None:
                    continue
                narrow, long = sorted(size[:2])
                if narrow <= MAX_NARROW_M and long <= MAX_LONG_M and size[2] >= MIN_HEIGHT_M:
                    models.append(str(Path(mjcf).parent.relative_to(objects)))
        if models:
            out[cat] = sorted(models)
    return out


def types(cat: str) -> set[str]:
    from robocasa.models.objects.kitchen_objects import OBJ_CATEGORIES

    found = set()
    for meta in OBJ_CATEGORIES[cat].values():
        t = meta.types
        found |= {t} if isinstance(t, str) else set(t or ())
    return found


def main() -> None:
    rng = random.Random(29)
    models = graspable_models()
    cats = sorted(models)
    rng.shuffle(cats)
    a, b = int(0.6 * len(cats)), int(0.8 * len(cats))
    split_cats = {"dev": cats[:a], "val": cats[a:b], "test": cats[b:]}
    OUT.mkdir(parents=True, exist_ok=True)
    SUITES.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.yaml"):
        old.unlink()
    for split, n in N.items():
        pool, kitchens = split_cats[split], list(KITCHENS[split])
        items = []
        for i in range(n):
            target = pool[i % len(pool)]
            others = [c for c in pool if c != target and not types(c) & types(target)] or [c for c in pool if c != target]
            distractors = rng.sample(others, 2)
            layout, style = rng.choice(kitchens), rng.choice(kitchens)
            objects = [{"category": c, "model": rng.choice(models[c])} for c in [target, *distractors]]
            name = f"{split}_{i:02d}_{target}"
            setup = {"environment": "robocasa",
                     "scene": {"layout": layout, "style": style, "seed": i, "objects": objects},
                     "attributes": {"object": target, "object_types": sorted(types(target)), "kitchen": f"l{layout}_s{style}", "split": split},
                     "tasks": {"pick": {"type": "pick_up", "object": target}}}  # fmt: skip
            head = f"# Pick up the {target.replace('_', ' ')} ({split}); generated by setups/generate_pick.py.\n"
            (OUT / f"{name}.yaml").write_text(head + yaml.safe_dump(setup, sort_keys=False, default_flow_style=None))
            items.append({"id": name, "setup": f"robocasa_pick/{name}", "task": "pick", "seeds": SEEDS[split]})
        built = f"setups/generate_pick.py: {len(pool)} {split} categories, kitchens l/s {kitchens[0]}-{kitchens[-1]}"
        (SUITES / f"{split}.yaml").write_text(yaml.safe_dump({"built_by": built, "items": items}, sort_keys=False, default_flow_style=None))
        print(f"{split}: {n} setups, {len(pool)} categories: {', '.join(sorted(pool))}")
    (SUITES / "set.yaml").write_text(yaml.safe_dump({
        "description": "RoboCasa pick: one named object on a counter among two others, arm only, Panda + Franka Hand; val and "
                       "test kitchens and object categories never in dev",
        "suites": {"dev": "dev.yaml", "val": "val.yaml", "test": "test.yaml"}}, sort_keys=False))  # fmt: skip


if __name__ == "__main__":
    main()
