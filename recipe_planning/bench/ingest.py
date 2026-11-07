#!/usr/bin/env python
"""Tier 0 of RecipeBench: web page -> recipe text, no LLM.

  python -m bench.ingest --html ../../recipe_corpus/html --out bench_runs/ingest.json

For every saved page that has schema.org Recipe JSON-LD, the JSON-LD text is the reference. The test hides the
JSON-LD and runs the plain-text fallback (what a page without metadata gets):
  ingredient recall   share of reference ingredient lines found in the fallback text
  instruction recall  share of reference instruction sentences found
  junk ratio          fallback length / reference length (ads, comments, navigation)
A page passes when both recalls are >= 0.9. This measures how hard "random pages" are before any model runs.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from recipe_ir.adapters import from_html  # noqa: E402

LD = re.compile(r'<script[^>]*application/ld\+json[^>]*>.*?</script>', re.S | re.I)
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def score(page: str) -> dict | None:
    ref = from_html(page)
    if ref.structured_hints.get("source") != "json-ld":
        return None
    text = ref.text
    ing = [l[2:] for l in text.split("\nInstructions\n")[0].splitlines() if l.startswith("- ")]
    ins = [re.sub(r"^\d+\.\s*", "", s) for s in text.split("\nInstructions\n", 1)[-1].splitlines() if s.strip()]
    sents = [x for s in ins for x in re.split(r"(?<=[.!?])\s+", s) if len(x) > 12]
    fb = norm(from_html(LD.sub(" ", page)).text)
    found = lambda xs: sum(norm(x) in fb for x in xs) / len(xs) if xs else 1.0
    r_ing, r_ins = round(found(ing), 3), round(found(sents), 3)
    return {"ingredient_recall": r_ing, "instruction_recall": r_ins,
            "junk_ratio": round(len(fb) / max(1, len(norm(text))), 2), "passed": r_ing >= 0.9 and r_ins >= 0.9}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--html", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows = {}
    for p in sorted(Path(a.html).glob("*.html")):
        s = score(p.read_text(encoding="utf-8", errors="replace"))
        if s:
            rows[p.stem] = s
    n = len(rows)
    agg = {"pages": n, "passed_pct": round(100 * sum(r["passed"] for r in rows.values()) / n, 1) if n else None,
           "mean_ingredient_recall": round(sum(r["ingredient_recall"] for r in rows.values()) / n, 3) if n else None,
           "mean_instruction_recall": round(sum(r["instruction_recall"] for r in rows.values()) / n, 3) if n else None,
           "median_junk_ratio": sorted(r["junk_ratio"] for r in rows.values())[n // 2] if n else None}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"summary": agg, "pages": rows}, indent=1))
    print(json.dumps(agg, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
