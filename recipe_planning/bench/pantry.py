#!/usr/bin/env python
"""PantryBench: does the recipe chooser find the recipe a kitchen was stocked for, and keep its hard constraints?

  python -m bench.pantry --corpus ../../recipe_corpus [--n 80] [--seed 0] [--out pantry_results.json]

Each scenario takes a real corpus recipe (the target), stocks a pantry with its needed ingredients as a person would
type them ("lemon", not "¼ cup fresh lemon juice"), removes `drop` of them, and adds distractor items from other
recipes. Measures, per drop level:
  recall@1, recall@3   target returned first / in the top 3 (other recipes may fit the pantry equally well, so @3 is
                       the main number)
  missing_exact        the target's reported missing list is exactly the dropped items (nothing hidden, nothing extra)
And one constraint scenario per diet over the whole corpus: no returned recipe may contain a forbidden ingredient
(constraint_violations_returned must be 0). No LLM, no network.
"""
from __future__ import annotations
import argparse
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from recipe_ir.actions import load_skills  # noqa: E402
from recipe_ir.choose import (ANIMAL, MEAT, PARTS, canon, Prefs, assess, choose, is_staple, load_corpus,  # noqa: E402
                              parse_ingredients)


def typed(name: str) -> str:
    """How a person names the item in their kitchen: drop part words ('lemon juice' -> 'lemon')."""
    w = name.split()
    while len(w) > 1 and w[-1] in PARTS:
        w = w[:-1]
    return " ".join(w)


REVERSE_SYN = {"scallion": "green onions", "chickpea": "garbanzo beans", "cilantro": "coriander", "cornstarch": "cornflour",
               "zucchini": "courgette", "eggplant": "aubergine", "shrimp": "prawns", "flour": "plain flour"}


def as_typed(item: str, style: str) -> str:
    """exact: the normalized name; terse: just the head noun ('red onion' -> 'onion'); synonym: another common name
    where one exists, plural like a shopping list."""
    if style == "terse":
        return item.split()[-1]
    if style == "synonym":
        c = canon(item)
        if c in REVERSE_SYN:
            return REVERSE_SYN[c]
        return item + "s" if not item.endswith("s") else item
    return item


def scenarios(corpus: dict[str, str], n: int, seed: int, drops=(0, 1), n_distractors: int = 15,
              styles=("exact", "terse", "synonym")):
    rng = random.Random(seed)
    prefs = Prefs()
    pool = []
    for name, text in corpus.items():
        need = [i["name"] for i in parse_ingredients(text) if not i["optional"] and not is_staple(i["name"], prefs.staples)]
        if 3 <= len(need) <= 10:
            pool.append((name, need))
    every = sorted({typed(x) for _, need in pool for x in need})
    picked = rng.sample(pool, min(n, len(pool)))
    for name, need in picked:
        items = sorted({typed(x) for x in need})
        for style in styles:
            for drop in drops:
                gone = rng.sample(items, drop) if drop else []
                distract = rng.sample([x for x in every if x not in items], n_distractors)
                pantry = [as_typed(x, style) for x in items if x not in gone] + distract
                rng.shuffle(pantry)
                yield {"target": name, "style": style, "drop": drop, "dropped": gone, "pantry": pantry}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skills", default="chefbot_sim")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    corpus = load_corpus(Path(a.corpus))
    skills = load_skills(a.skills)
    rows = []
    for sc in scenarios(corpus, a.n, a.seed):
        prefs = Prefs(max_missing=max(2, sc["drop"]))
        top = choose(corpus, sc["pantry"], prefs, skills, k=10)
        names = [t["name"] for t in top]
        rank = names.index(sc["target"]) + 1 if sc["target"] in names else None
        tgt = assess(sc["target"], corpus[sc["target"]], sc["pantry"], prefs, skills)
        missing_typed = sorted({typed(m) for m in (
            next(i["name"] for i in parse_ingredients(corpus[sc["target"]]) if i["raw"] == raw) for raw in tgt["missing"])})
        dropped = set(sc["dropped"])
        rows.append({**sc, "rank": rank, "missing_reported": tgt["missing"],
                     "missing_exact": missing_typed == sorted(dropped),
                     "hidden_missing": sorted(dropped - set(missing_typed)),      # unsafe: a gap the agent hid
                     "extra_missing": sorted(set(missing_typed) - dropped)})      # conservative: asked for more
    summary = {}
    pct = lambda xs: round(100 * sum(xs) / len(xs), 1)
    for style, drop in sorted({(r["style"], r["drop"]) for r in rows}):
        rs = [r for r in rows if r["drop"] == drop and r["style"] == style]
        summary[f"{style}, drop={drop}"] = {"scenarios": len(rs), "recall@1": pct([r["rank"] == 1 for r in rs]),
                                   "recall@3": pct([bool(r["rank"]) and r["rank"] <= 3 for r in rs]),
                                   "missing_exact": pct([r["missing_exact"] for r in rs]),
                                   "hid_a_missing_item": pct([bool(r["hidden_missing"]) for r in rs]),
                                   "reported_extra": pct([bool(r["extra_missing"]) for r in rs])}
    # hard constraints over the whole corpus: a returned recipe never contains what the diet forbids
    checks = {}
    for diet, pat in (("vegetarian", MEAT), ("vegan", ANIMAL)):
        pantry = ["egg", "tomato", "onion", "garlic", "rice", "lemon", "chickpea", "spinach", "cheese", "chicken"]
        top = choose(corpus, pantry, Prefs(diet=diet, max_missing=3), skills, k=50)
        bad = [t["name"] for t in top if any(re.search(pat, i["raw"].lower())
                                              for i in parse_ingredients(corpus[t["name"]]) if not i["optional"])]
        checks[diet] = {"returned": len(top), "constraint_violations_returned": len(bad), "examples": bad[:3]}
    summary["constraints"] = checks
    print(json.dumps(summary, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
