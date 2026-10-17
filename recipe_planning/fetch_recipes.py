#!/usr/bin/env python
"""Turn a list of recipe URLs into eval inputs.

  cp urls.example.txt urls.txt   # add 20-30 URLs
  python fetch_recipes.py urls.txt                 # -> recipes/web/<slug>.txt (+ <slug>.meta.json with the URL)
  python eval_llm.py --recipes recipes/web --model gpt-6.1-sol --reasoning-effort low --runs 3

Only pages with schema.org Recipe JSON-LD are saved (most recipe sites have it). Pages without it are reported and
skipped, because their visible text includes ads/comments and would make coverage meaningless.
recipes/web/ is git-ignored: recipe text belongs to its authors, so keep it local and cite the URLs in your write-up.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import time
from pathlib import Path
from recipe_ir.adapters import from_url, load_source


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:60] or "recipe"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("urls", help="text file: one URL (or local .html path) per line; # comments allowed")
    ap.add_argument("--out", default="recipes/web")
    ap.add_argument("--delay", type=float, default=1.5, help="seconds between requests (be polite)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = [l.split("#", 1)[0].strip() for l in Path(a.urls).read_text().splitlines()]
    ok = skipped = 0
    for i, url in enumerate(l for l in lines if l):
        try:
            doc = from_url(url) if url.startswith(("http://", "https://")) else load_source(url)
        except Exception as e:
            print(f"  FAIL {url}: {type(e).__name__}: {e}", file=sys.stderr)
            skipped += 1
            continue
        if doc.structured_hints.get("source") != "json-ld":
            print(f"  SKIP {url}: no schema.org Recipe JSON-LD", file=sys.stderr)
            skipped += 1
            continue
        name = slug(doc.text.split("\n", 1)[0])
        path = out / f"{name}.txt"
        n = 2
        while path.exists() and json.loads((out / f"{path.stem}.meta.json").read_text()).get("url") != url:
            path = out / f"{name}_{n}.txt"
            n += 1
        path.write_text(doc.text, encoding="utf-8")
        (out / f"{path.stem}.meta.json").write_text(json.dumps({"url": url, **doc.structured_hints}, indent=1))

        print(f"  saved {path}  ({len(doc.text)} chars)")
        ok += 1
        if a.delay and url.startswith("http"):
            time.sleep(a.delay)
    print(f"\n{ok} saved, {skipped} skipped -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
