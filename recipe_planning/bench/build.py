#!/usr/bin/env python
"""Build RecipeBench instances from the frozen eval set.

  python -m bench.build --frozen ../../recipe_eval_v1 --silver-runs ../../ChefBot_recipe_ref/recipe_planning/runs

Writes <frozen>/bench/references/<name>.json and <frozen>/bench/instances_{full,lite}.jsonl. Each instance is one
recipe text (a frozen base or a deterministic perturbation of it) with the expectation its test checks. Instances
are self-contained (full text inline, sha256), so a run never depends on the live web or on the corpus folder.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from recipe_ir.adapters import from_text  # noqa: E402
from recipe_ir.recipe_pipeline import validate  # noqa: E402
from recipe_ir.schema import Recipe  # noqa: E402
from bench.perturb import PERTURBATIONS, drop_cooking  # noqa: E402

PROTEIN = ["egg", "chicken", "beef", "pork", "steak", "sausage", "bacon", "turkey", "lamb", "shrimp", "scallop",
           "salmon", "fish", "cod", "tuna", "prawn", "duck", "ham", "meat"]


def head_word(name: str) -> str:
    words = re.findall(r"[a-z]+", name.lower())
    for w in words:
        for p in PROTEIN:
            if w.startswith(p):
                return p
    return (words[-1].rstrip("s") if words else name.lower())


def cooking_quotes(v) -> dict[str, list[str]]:
    """unsafe-raw ingredient -> quotes of the stated steps that cook it (from the reference's state trace)."""
    out: dict[str, list[str]] = {}
    steps = {s.id: s for s in v.recipe.steps}
    unsafe = {i.id for i in v.recipe.ingredients if i.needs_cooking}
    for t in (v.engine.trace if v.engine else []):
        s = steps.get(t["step"])
        if s is None or t.get("status") != "ok" or s.verb not in ("cook", "bake") or not s.evidence.quote:
            continue
        if s.evidence.prov != "stated":
            continue
        for ing in set(t.get("components", [])) & unsafe:
            out.setdefault(ing, []).append(s.evidence.quote)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--silver-runs", required=True, help="folder(s) searched for the reference model's run result.json files")
    ap.add_argument("--lite-robot", type=int, default=5, help="robot-plausible bases in the lite split")
    ap.add_argument("--lite-broad", type=int, default=4, help="broad bases in the lite split")
    a = ap.parse_args(argv)
    frozen = Path(a.frozen)
    man = json.loads((frozen / "manifest.json").read_text())
    out = frozen / "bench"
    (out / "references").mkdir(parents=True, exist_ok=True)

    bases = []   # (name, split, text, reference recipe dict, kind)
    for g in man["sets"]["gold"]:
        name = g["name"]
        text = (HERE / "recipes" / f"{name}.txt").read_text(encoding="utf-8")
        gold = json.loads((HERE / "recipes" / "gold" / f"{name}.gold.json").read_text())
        bases.append((name, "gold", text, gold, "gold"))
    results = {}
    for rj in Path(a.silver_runs).glob("**/runs/*/result.json"):
        name = re.sub(r"-r\d+$", "", rj.parent.name.split("__")[0])
        results.setdefault(name, rj)
    for split in ("robot_plausible", "broad"):
        for item in man["sets"][split]:
            name = item["name"]
            text = (frozen / "recipes" / f"{name}.txt").read_text(encoding="utf-8")
            assert hashlib.sha256(text.encode()).hexdigest() == item["sha256"], f"{name}: frozen text changed"
            rj = results.get(name)
            if rj is None:
                print(f"  {name}: no silver run, skipped", file=sys.stderr)
                continue
            res = json.loads(rj.read_text())
            if (rj.parent / "source.txt").read_text(encoding="utf-8") != text:
                print(f"  {name}: silver run saw a different text, skipped", file=sys.stderr)
                continue
            if not res.get("recipe"):
                continue
            bases.append((name, split, text, res["recipe"], "silver"))

    instances, info = [], {}
    for name, split, text, ref, kind in bases:
        v = validate(Recipe.model_validate(ref), from_text(text))
        cq = cooking_quotes(v) if v.recipe else {}
        words = {i.id: head_word(i.name) for i in v.recipe.ingredients} if v.recipe else {}
        (out / "references" / f"{name}.json").write_text(json.dumps(
            {"name": name, "kind": kind, "status": v.status, "steps": len(v.recipe.steps) if v.recipe else 0,
             "recipe": ref, "source": text, "source_sha256": hashlib.sha256(text.encode()).hexdigest()}, indent=1))
        if kind == "silver" and v.status == "extraction_errors":
            print(f"  {name}: silver reference has extraction errors; base only, no perturbations", file=sys.stderr)
        variants = [("base", (text, {"kind": "base", "gold": kind == "gold", "status": v.status}))]
        if not (kind == "silver" and v.status == "extraction_errors"):
            for pname, fn in PERTURBATIONS.items():
                if (res := fn(text, seed=0)) is not None:
                    variants.append((pname, res))
            if v.status == "valid" and (res := drop_cooking(text, {"cooking_quotes": cq, "words": words})) is not None:
                variants.append(("planted_gap", res))
        info[name] = {"split": split, "kind": kind, "steps": len(v.recipe.steps) if v.recipe else 0,
                      "variants": [p for p, _ in variants]}
        for pname, (t, expect) in variants:
            instances.append({"id": f"{name}__{pname}", "base": name, "perturbation": pname, "split": split,
                              "reference": name, "reference_kind": kind, "expect": expect,
                              "sha256": hashlib.sha256(t.encode()).hexdigest(), "chars": len(t), "text": t})

    # lite: every gold base, then bases that carry the most perturbation types, smallest first (cheap to run)
    order = sorted((n for n in info if info[n]["kind"] == "silver"),
                   key=lambda n: (-len(info[n]["variants"]), info[n]["steps"]))
    lite_bases = [n for n in info if info[n]["kind"] == "gold"]
    lite_bases += [n for n in order if info[n]["split"] == "robot_plausible"][: a.lite_robot]
    lite_bases += [n for n in order if info[n]["split"] == "broad"][: a.lite_broad]
    write = lambda p, rows: p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    write(out / "instances_full.jsonl", instances)
    write(out / "instances_lite.jsonl", [i for i in instances if i["base"] in lite_bases])
    summary = {"bases": len(info), "instances_full": len(instances),
               "instances_lite": sum(i["base"] in lite_bases for i in instances), "lite_bases": lite_bases,
               "by_perturbation": {p: sum(i["perturbation"] == p for i in instances)
                                   for p in ["base", *PERTURBATIONS, "planted_gap"]}, "bases_info": info}
    (out / "build_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "bases_info"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
