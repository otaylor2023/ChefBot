"""Recipe corpus: difficulty features and the two evaluation sets (GOAL.md: broad, robot-plausible).

Features come from the saved recipe text only (no LLM), so they can stratify samples before any model runs and
explain afterwards which recipes were hard.
"""
from __future__ import annotations
import json
import random
import re
from pathlib import Path
from typing import Any
from .adapters import from_text

_OVEN = re.compile(r"\b(oven|bake[sd]?|baking|roast(ed|ing)?|broil)\b", re.I)
_DOUGH = re.compile(r"\b(dough|knead|yeast|proof|rise|rising|ferment)\w*", re.I)
_STOVE = re.compile(r"\b(skillet|pan|pot|saucepan|wok|burner|stove|simmer|boil|saut[eé]|fry|fried|scramble)\w*", re.I)
_CUT = re.compile(r"\b(chop|dice|slice|mince|grate|julienne|cube)\w*", re.I)
_OPTIONAL = re.compile(r"\b(optional|if desired|if using|to taste)\b", re.I)
_ALTERNATIVE = re.compile(r"\b(alternatively|instead|or use|you can also|substitute)\b", re.I)
_EQUIPMENT = re.compile(r"\b(blender|food processor|stand mixer|mixer|instant pot|pressure cooker|slow cooker|grill|air fryer|deep fry|thermometer)\b", re.I)


def features(text: str) -> dict[str, Any]:
    doc = from_text(text)
    secs = {s.kind: text[s.start:s.end] for s in doc.sections}
    ing = [l for l in secs.get("ingredients", "").splitlines() if l.strip().startswith("-")]
    instr = secs.get("instructions", text)
    steps = [l for l in instr.splitlines() if re.match(r"\s*\d+\.", l)]
    sentences = [x for x in re.split(r"(?<=[.!?])\s+", instr) if len(x.strip()) > 3]
    f = {
        "chars": len(text), "instruction_chars": len(instr),
        "n_ingredients": len(ing), "n_numbered_steps": len(steps), "n_sentences": len(sentences),
        "oven": bool(_OVEN.search(instr)), "dough": bool(_DOUGH.search(text)), "stovetop": bool(_STOVE.search(instr)),
        "cutting": bool(_CUT.search(text)), "optional": bool(_OPTIONAL.search(text)),
        "alternatives": bool(_ALTERNATIVE.search(instr)), "special_equipment": bool(_EQUIPMENT.search(text)),
    }
    f["size"] = "small" if f["n_sentences"] <= 8 else ("medium" if f["n_sentences"] <= 20 else "large")
    # robot-plausible: what a single-arm kitchen robot might attempt first. A heuristic label on the text, not a
    # capability check (that is actions.feasibility, after extraction).
    f["robot_plausible"] = (not f["oven"] and not f["dough"] and not f["special_equipment"]
                            and f["n_ingredients"] <= 10 and f["n_sentences"] <= 15)
    return f


def build_index(recipes_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for p in sorted(recipes_dir.glob("*.txt")):
        meta_p = recipes_dir / f"{p.stem}.meta.json"
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        url = meta.get("url", "")
        site = re.sub(r"^www\.", "", url.split("/")[2]) if url.count("/") >= 2 else "local"
        rows.append({"name": p.stem, "site": site, "url": url, **features(p.read_text(encoding="utf-8"))})
    return rows


def write_index(recipes_dir: Path) -> Path:
    out = recipes_dir / "index.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in build_index(recipes_dir)))
    return out


def load_index(recipes_dir: Path) -> list[dict[str, Any]]:
    p = recipes_dir / "index.jsonl"
    if not p.exists():
        write_index(recipes_dir)
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    present = {q.stem for q in recipes_dir.glob("*.txt")}
    return [r for r in rows if r["name"] in present]


def select(rows: list[dict], *, subset: str = "all", n: int | None = None, seed: int = 0,
           max_per_site: int | None = None) -> list[str]:
    """subset: all | robot_plausible | broad (= not robot_plausible). Stratified by size and site so a small sample
    still spans easy and hard recipes."""
    if subset == "robot_plausible":
        rows = [r for r in rows if r["robot_plausible"]]
    elif subset == "broad":
        rows = [r for r in rows if not r["robot_plausible"]]
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)
    if max_per_site:
        seen: dict[str, int] = {}
        kept = []
        for r in rows:
            if seen.get(r["site"], 0) < max_per_site:
                kept.append(r)
                seen[r["site"]] = seen.get(r["site"], 0) + 1
        rows = kept
    if n is None or n >= len(rows):
        return sorted(r["name"] for r in rows)
    buckets: dict[str, list] = {}
    for r in rows:
        buckets.setdefault(r["size"], []).append(r)
    picked, order = [], ["small", "medium", "large"]
    while len(picked) < n and any(buckets.values()):
        for b in order:
            if buckets.get(b) and len(picked) < n:
                picked.append(buckets[b].pop())
    return sorted(r["name"] for r in picked)
