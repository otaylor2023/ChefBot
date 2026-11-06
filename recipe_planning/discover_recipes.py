#!/usr/bin/env python
"""Grow the recipe corpus from recipe sites' own sitemaps.

  python discover_recipes.py --sites sites.txt --per-site 25 --out ../../recipe_corpus
  python discover_recipes.py --sites sites.txt --per-site 25 --out ../../recipe_corpus --dry-run   # list URLs only

Per site: read robots.txt (its Sitemap: lines, and what it disallows), walk the sitemap index to recipe/post
sitemaps, keep recipe-looking URLs that robots.txt allows, shuffle (seeded), and fetch until --per-site recipes
with schema.org Recipe JSON-LD are saved. Pages without JSON-LD are skipped (their text includes ads and comments).
Already-saved URLs are skipped, so reruns only add. Then writes <out>/index.jsonl (difficulty features, the
robot_plausible label) for eval sampling.

Recipe text belongs to its authors: keep the corpus local (outside git), cite URLs, be polite (--delay).
"""
from __future__ import annotations
import argparse
import gzip
import json
import random
import re
import sys
import time
import urllib.request
import urllib.robotparser
from pathlib import Path
from recipe_ir.adapters import from_html
from recipe_ir.corpus import write_index
from fetch_recipes import slug

UA = "Mozilla/5.0 (ChefBot recipe-ir research prototype)"
SKIP = re.compile(r"/(tag|tags|category|categories|author|page|about|contact|privacy|shop|product|collections?|"
                  r"videos?|search|feed|wp-content|cart|account|roundup|gift|guide|how-to|articles?|blog/page)(/|$)|"
                  r"\.(jpg|jpeg|png|gif|webp|pdf|xml)$", re.I)
CHILD_PREFER = re.compile(r"recipe|post", re.I)
CHILD_AVOID = re.compile(r"page|tag|categor|author|image|video|product|attachment|web-?stor", re.I)


def get(url: str, timeout: float = 25) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    return gzip.decompress(data) if url.endswith(".gz") or data[:2] == b"\x1f\x8b" else data


def locs(xml: bytes) -> list[str]:
    return [x.strip() for x in re.findall(r"<loc>\s*(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?\s*</loc>", xml.decode("utf-8", "replace"), re.S)]


def is_index(xml: bytes) -> bool:
    return b"<sitemapindex" in xml[:2000]


def site_root(site: str) -> str:
    site = site.strip().rstrip("/")
    return site if site.startswith("http") else "https://" + site


def candidate_urls(root: str, rp: urllib.robotparser.RobotFileParser, max_children: int, log) -> list[str]:
    maps = list(rp.site_maps() or []) or [root + p for p in ("/sitemap_index.xml", "/sitemap.xml", "/wp-sitemap.xml")]
    pages: list[str] = []
    for sm in maps:
        try:
            xml = get(sm)
        except Exception as e:
            log(f"    sitemap {sm}: {type(e).__name__}")
            continue
        if not is_index(xml):
            pages += locs(xml)
            continue
        children = locs(xml)
        preferred = [c for c in children if CHILD_PREFER.search(c) and not CHILD_AVOID.search(c)]
        children = (preferred or [c for c in children if not CHILD_AVOID.search(c)])[:max_children]
        for c in children:
            try:
                sub = get(c)
            except Exception as e:
                log(f"    child {c}: {type(e).__name__}")
                continue
            pages += locs(sub) if not is_index(sub) else []
            time.sleep(0.5)
        if pages:
            break
    host = root.split("/")[2]
    out = [u for u in dict.fromkeys(pages) if host in u and not SKIP.search(u) and u.rstrip("/") != root
           and rp.can_fetch(UA, u)]
    return out


def saved_urls(out: Path) -> set[str]:
    urls = set()
    for m in out.glob("*.meta.json"):
        try:
            urls.add(json.loads(m.read_text()).get("url"))
        except json.JSONDecodeError:
            pass
    return urls


def save(doc, url: str, out: Path) -> Path:
    name = slug(doc.text.split("\n", 1)[0])
    path, n = out / f"{name}.txt", 2
    while path.exists():
        path, n = out / f"{name}_{n}.txt", n + 1
    path.write_text(doc.text, encoding="utf-8")
    (out / f"{path.stem}.meta.json").write_text(json.dumps({"url": url, **doc.structured_hints}, indent=1))
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", required=True, help="file: one site (domain or https://root) per line; # comments")
    ap.add_argument("--out", required=True, help="corpus folder (keep it outside git)")
    ap.add_argument("--per-site", type=int, default=20)
    ap.add_argument("--max-tries", type=int, default=3, help="give up on a site after per-site * this many fetches")
    ap.add_argument("--max-children", type=int, default=4, help="child sitemaps to read per site")
    ap.add_argument("--delay", type=float, default=2.0, help="seconds between page requests to one site")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, file=sys.stderr, flush=True)
    sites = [l.split("#", 1)[0].strip() for l in Path(a.sites).read_text().splitlines()]
    have = saved_urls(out)
    report = {}
    for site in (s for s in sites if s):
        root = site_root(site)
        rp = urllib.robotparser.RobotFileParser(root + "/robots.txt")
        try:
            rp.parse(get(root + "/robots.txt").decode("utf-8", "replace").splitlines())
        except Exception as e:
            log(f"{root}: robots.txt unreadable ({type(e).__name__}); skipping site")
            report[root] = {"saved": 0, "note": "robots.txt unreadable"}
            continue
        cands = candidate_urls(root, rp, a.max_children, log)
        random.Random(a.seed).shuffle(cands)
        todo = [u for u in cands if u not in have]
        log(f"{root}: {len(cands)} candidate URLs, {len(cands) - len(todo)} already saved")
        if a.dry_run:
            print("\n".join(todo[:a.per_site]))
            report[root] = {"candidates": len(cands)}
            continue
        saved = tried = skipped = failed = 0
        for url in todo:
            if saved >= a.per_site or tried >= a.per_site * a.max_tries:
                break
            tried += 1
            try:
                page = get(url).decode("utf-8", "replace")
                doc = from_html(page)
            except Exception as e:
                failed += 1
                log(f"  FAIL {url}: {type(e).__name__}")
                if failed >= 5 and saved == 0:
                    log(f"  {root}: 5 failures and nothing saved (blocked?); next site")
                    break
                time.sleep(a.delay)
                continue
            if doc.structured_hints.get("source") != "json-ld" or "\nInstructions\n1." not in doc.text:
                skipped += 1
            else:
                doc.structured_hints["url"] = url
                p = save(doc, url, out)
                have.add(url)
                saved += 1
                log(f"  saved {p.name} ({len(doc.text)} chars)")
            time.sleep(a.delay)
        report[root] = {"candidates": len(cands), "tried": tried, "saved": saved, "no_jsonld": skipped, "failed": failed}
    if not a.dry_run:
        idx = write_index(out)
        log(f"\nindex: {idx}")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
