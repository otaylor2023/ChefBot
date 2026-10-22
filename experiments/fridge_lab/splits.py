"""Dev / val / test scenes from the oracle survey (results/survey.json) -> scenes.json.

A scene is one fridge door in one kitchen: {id, split, model, type, layout, style, door,
oracle_deg}. Only doors the oracle opened to MIN_DEG are kept.
  test: RoboCasa's test fridges (15 models, test layouts/styles); run once at the end, by us.
  val:  VAL_PER_TYPE train fridge models of each type, held out from the researcher (scores only).
  dev:  the other train fridge models; the researcher runs these freely.
Models never cross splits. Seeds: dev any; val VAL_SEEDS; test TEST_SEEDS.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

from experiments.fridge_lab.survey import OUT

MIN_DEG = 60.0
VAL_PER_TYPE = {"fridge_french_door": 4, "fridge_side_by_side": 3, "fridge_bottom_freezer": 3}
VAL_SEEDS = (100, 101)
TEST_SEEDS = (1000, 1001, 1002)
SCENES = OUT.parent / "scenes.json"


def build() -> dict:
    survey = json.loads((OUT / "survey.json").read_text())
    rows = []
    for r in survey:
        ok = [t for t in r["tried"] if any(d["oracle_deg"] >= MIN_DEG for d in t.get("doors", []))]
        if not ok:
            continue
        t = ok[-1]
        for d in t["doors"]:
            if d["oracle_deg"] >= MIN_DEG:
                rows.append({"split": r["split"], "model": r["model"], "type": r["type"], "layout": t["layout"], "style": t["style"],
                             "door": d["door"], "oracle_deg": round(d["oracle_deg"], 1)})  # fmt: skip
    rng = random.Random(329)
    val_models = set()
    for typ, n in VAL_PER_TYPE.items():
        models = sorted({x["model"] for x in rows if x["split"] == "train" and x["type"] == typ})
        val_models |= set(rng.sample(models, n))
    for x in rows:
        x["split"] = "test" if x["split"] == "test" else ("val" if x["model"] in val_models else "dev")
    counters: dict[str, int] = {}
    for x in sorted(rows, key=lambda x: (x["split"], x["type"], x["model"], x["door"])):
        counters[x["split"]] = counters.get(x["split"], 0) + 1
        x["id"] = f"{x['split']}{counters[x['split']]:02d}"
    rows.sort(key=lambda x: x["id"])
    return {"min_deg": MIN_DEG, "val_seeds": VAL_SEEDS, "test_seeds": TEST_SEEDS, "scenes": rows}


def build_mixed(oracle_evals: list[Path], seed: int = 2026) -> dict:
    """A split for a robot that can't do every door (the YAM), mixing RoboCasa's train and test kitchens.

    Keeps the doors this robot's oracle opened to MIN_DEG (oracle_evals: its evaluate.py --oracle results on
    the current scenes), then splits by fridge model within each (fridge type, RoboCasa kitchen split) group,
    about 50% dev / 20% val / 30% test, so every split has its share of RoboCasa's tighter test kitchens."""
    old = {x["id"]: x for x in load_default()["scenes"]}
    best: dict[str, float] = {}
    for f in oracle_evals:
        r = json.loads(Path(f).read_text())
        for e in r["episodes"] if isinstance(r, dict) else r:
            best[e["scene"]] = max(best.get(e["scene"], 0.0), e["final_deg"])
    rows = [{**x, "robocasa": "test" if x["split"] == "test" else "train", "oracle_deg": round(best[i], 1)}
            for i, x in old.items() if best.get(i, 0.0) >= MIN_DEG]  # fmt: skip
    rng = random.Random(seed)
    split_of = {}
    for g in sorted({(x["type"], x["robocasa"]) for x in rows}):
        models = sorted({x["model"] for x in rows if (x["type"], x["robocasa"]) == g})
        rng.shuffle(models)
        n_test, n_val = round(0.3 * len(models) + 1e-9), max(1, round(0.2 * len(models)))
        for i, m in enumerate(models):
            split_of[m] = "test" if i < n_test else "val" if i < n_test + n_val else "dev"
    counters: dict[str, int] = {}
    for x in sorted(rows, key=lambda x: (split_of[x["model"]], x["type"], x["model"], x["door"])):
        x["split"] = split_of[x["model"]]
        counters[x["split"]] = counters.get(x["split"], 0) + 1
        x["source_id"], x["id"] = x["id"], f"{x['split']}{counters[x['split']]:02d}"
    rows.sort(key=lambda x: x["id"])
    return {"min_deg": MIN_DEG, "val_seeds": VAL_SEEDS, "test_seeds": TEST_SEEDS, "seed": seed, "scenes": rows}


def load_default() -> dict:
    return json.loads(SCENES.read_text())


def load() -> dict:
    """The lab's scenes: FRIDGE_SCENES if set (e.g. scenes_yam.json), else scenes.json."""
    return json.loads(Path(os.environ.get("FRIDGE_SCENES") or SCENES).read_text())


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "mixed":  # splits.py mixed OUT.json ORACLE_RESULTS.json...
        s = build_mixed([Path(a) for a in sys.argv[3:]])
        Path(sys.argv[2]).write_text(json.dumps(s, indent=1))
    else:
        s = build()
        SCENES.write_text(json.dumps(s, indent=1))
    from collections import Counter

    print(Counter((x["split"], x["type"]) for x in s["scenes"]))
    print({sp: len({x["model"] for x in s["scenes"] if x["split"] == sp}) for sp in ("dev", "val", "test")}, "models")
    print({sp: sum(x["split"] == sp for x in s["scenes"]) for sp in ("dev", "val", "test")}, "scenes (doors)")
