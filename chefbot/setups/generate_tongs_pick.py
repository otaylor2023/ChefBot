"""Write the tongs-pick setups (setups/mujoco_tongs/<split>_<nn>_<food>.yaml) and their suites
(suites/mujoco_tongs/{dev,val,test}.yaml). Run where RoboCasa is installed (it reads the object models' sizes).

Kitchens: RoboCasa's, exported to plain MuJoCo; each split its own (dev: layouts and styles 11-50, val 51-60, test
1-10, as the pick setups). Tongs: the tongs lab's final generator (generators/tongs/final), each split its own seeds,
only seeds whose tongs pass the tongs lab's checks. Foods: whole produce and a few baked goods, each split its own
models (categories are shared: the skill must handle eggs it has never seen), a model only where it fits between the
tongs' tips with room to spare. Where: a third on the counter, the rest in a pan, a bowl or a plate. Each setup: one
kitchen, one pair of tongs, the food, its container, and a distractor food.

    python -m chefbot.setups.generate_tongs_pick
"""

from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

import yaml

from chefbot import suites
from chefbot.setups import ROOT
from chefbot.setups.generate_pick import REGISTRIES, model_size

OUT = ROOT / "mujoco_tongs"
SUITES = suites.ROOT / "mujoco_tongs"
N = {"dev": 24, "val": 12, "test": 16}
SEEDS = {"dev": [0, 1, 2], "val": [0, 1], "test": [0, 1]}
KITCHENS = {"dev": range(11, 51), "val": range(51, 61), "test": range(1, 11)}
TONGS_SEEDS = {"dev": range(1000, 1060), "val": range(2000, 2030), "test": range(3000, 3040)}
TONGS_PER_SPLIT = {"dev": 12, "val": 6, "test": 8}
FOODS = ["egg", "tomato", "peach", "apricot", "strawberry", "kiwi", "banana", "potato", "apple", "lemon", "lime", "orange",
         "onion", "avocado", "mango"]  # fmt: skip
CONTAINERS = ["pan", "bowl", "plate"]
GENERATOR = "tongs/final"
MARGIN_M = 0.012  # the food's narrow side at most the tongs' open tip gap less this


def food_models() -> dict[str, list[tuple[str, float]]]:
    """Category -> [(model path under robocasa's objects/, narrow side m)]."""
    import robocasa
    from robocasa.models.objects.kitchen_objects import OBJ_CATEGORIES

    objects = Path(robocasa.models.assets_root) / "objects"
    out = {}
    for cat in FOODS + CONTAINERS:
        models = []
        for reg in REGISTRIES:
            meta = OBJ_CATEGORIES.get(cat, {}).get(reg)
            if meta is None:
                continue
            for mjcf in meta.mjcf_paths:
                size = model_size(mjcf, meta.scale)
                if size is not None:
                    models.append((str(Path(mjcf).parent.relative_to(objects)), float(min(size[:2]))))
        out[cat] = sorted(models)
    return out


def good_tongs(split: str) -> list[tuple[int, dict]]:
    """(seed, params) of tongs that pass the tongs lab's checks, as many as the split takes."""
    from chefbot.environments.mujoco_kitchen.host import GENERATORS, generated
    from chefbot.generators.tongs import checks

    fam = {**yaml.safe_load((GENERATORS / "tongs" / "family.yaml").read_text()), "dir": str(GENERATORS / "tongs")}
    out = []
    for seed in TONGS_SEEDS[split]:
        model = generated(GENERATOR, seed)
        params = json.loads((model.parent / "params.json").read_text())
        result = checks.run(model.parent, fam, params, out_dir=Path(tempfile.mkdtemp(prefix="tongs_check_")))
        print(f"  tongs {seed}: {params.get('construction')}, tips {params['rest_tip_gap_m']:.3f} m, checks {'ok' if result['ok'] else result['failed']}")
        if result["ok"]:
            out.append((seed, params))
        if len(out) >= TONGS_PER_SPLIT[split]:
            break
    return out


def main() -> None:
    rng = random.Random(31)
    models = food_models()
    split_models = {"dev": {}, "val": {}, "test": {}}
    for cat, ms in models.items():
        ms = ms[:]
        rng.shuffle(ms)
        a, b = max(1, int(0.6 * len(ms))), max(1, int(0.8 * len(ms)))
        parts = {"dev": ms[:a], "val": ms[a:b] or ms[:1], "test": ms[b:] or ms[-1:]}
        for split, p in parts.items():
            split_models[split][cat] = p
    OUT.mkdir(parents=True, exist_ok=True)
    SUITES.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.yaml"):
        old.unlink()
    for split, n in N.items():
        print(f"{split}: checking tongs")
        tongs = good_tongs(split)
        kitchens = list(KITCHENS[split])
        items = []
        for i in range(n):
            t_seed, t_params = tongs[i % len(tongs)]
            gap = float(t_params["rest_tip_gap_m"])
            fits = {c: [m for m, w in split_models[split][c] if w <= gap - MARGIN_M] for c in FOODS}
            cats = [c for c in FOODS if fits[c]]
            food = cats[i % len(cats)] if i < len(cats) else rng.choice(cats)
            other = rng.choice([c for c in cats if c != food])
            where = "counter" if i % 3 == 0 else CONTAINERS[(i // 3) % 3]
            objects = [{"category": food, "model": rng.choice(fits[food])}, {"category": other, "model": rng.choice(fits[other])}]
            if where != "counter":
                objects.append({"category": where, "model": rng.choice(split_models[split][where])[0]})
            layout, style = rng.choice(kitchens), rng.choice(kitchens)
            name = f"{split}_{i:02d}_{food}"
            task = {"type": "tongs_pick", "object": food, "distractors": [other]}
            if where != "counter":
                task["container"] = where
            setup = {"environment": "mujoco_kitchen",
                     "scene": {"layout": layout, "style": style, "objects": objects,
                               "extras": {"tongs": {"generator": GENERATOR, "seed": t_seed}}},
                     "attributes": {"object": food, "where": where, "tongs": t_params.get("construction"),
                                    "tongs_style": t_params.get("style"), "kitchen": f"l{layout}_s{style}", "split": split},
                     "tasks": {"lift": task}}  # fmt: skip
            head = f"# Pick up the tongs and lift the {food} ({'on the counter' if where == 'counter' else 'in the ' + where}; {split}); generated by setups/generate_tongs_pick.py.\n"
            (OUT / f"{name}.yaml").write_text(head + yaml.safe_dump(setup, sort_keys=False, default_flow_style=None))
            items.append({"id": name, "setup": f"mujoco_tongs/{name}", "task": "lift", "seeds": SEEDS[split]})
        built = f"setups/generate_tongs_pick.py: kitchens l/s {kitchens[0]}-{kitchens[-1]}, {len(tongs)} tongs"
        (SUITES / f"{split}.yaml").write_text(yaml.safe_dump({"built_by": built, "items": items}, sort_keys=False, default_flow_style=None))
        print(f"{split}: {n} setups")
    (SUITES / "set.yaml").write_text(yaml.safe_dump({
        "description": "Tongs pick in plain MuJoCo kitchens: pick up generated tongs, lift a named food with them (on the counter "
                       "or in a pan, bowl or plate); TidyBot++ FR3 + Franka Hand; val and test kitchens, tongs and food models never in dev",
        "suites": {"dev": "dev.yaml", "val": "val.yaml", "test": "test.yaml"}}, sort_keys=False))  # fmt: skip


if __name__ == "__main__":
    main()
