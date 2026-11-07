"""Compare VLM naming set-ups on saved view strips (name_objects.py --strips) against a reference file
({id: {label, dup_of?}}): accuracy and seconds per object, for each model / prompt variant.

    python -m chefbot.perception.eval_naming STRIPS_DIR REF.json --model qwen2.5vl:7b --hint vocab.txt --tag q25_hint
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from chefbot.perception.name_objects import ask, gemini_key

# list prices, $ per 1M tokens (input, output incl. thinking), Nov 2026 from third-party listings: CHECK ai.google.dev
PRICES = {"gemini-3.1-pro-preview": (2.0, 12.0), "gemini-4-argon": (2.0, 10.0), "gemini-3.8-flash": (0.75, 3.75),
          "gemini-3.1-flash-lite": (0.25, 1.5), "gemini-2.5-flash-lite": (0.10, 0.40), "gemini-2.5-flash": (0.30, 2.50)}  # fmt: skip

# what counts as naming the reference object (any of these words in the answer)
SYNONYMS = {
    "black spoon": ["spoon", "ladle"], "spoon": ["spoon", "ladle"], "tomato": ["tomato"], "tongs": ["tong"],
    "frying pan": ["pan", "skillet", "wok"], "tennis ball": ["tennis ball", "ball"], "dish soap bottle": ["soap", "detergent", "dish"],
    "sponge": ["sponge", "scrub"], "game controller": ["controller", "gamepad", "joystick"], "lego brick": ["lego", "brick", "block"],
    "laptop": ["laptop", "computer", "macbook"], "computer mouse": ["mouse"], "spice jar": ["spice", "jar", "seasoning", "cumin"],
    "cell phone": ["phone"], "whisk": ["whisk", "beater"], "purple cube": ["cube", "block", "box"],
    "tape measure": ["tape measure", "measuring tape"], "coffee cup": ["cup", "coffee"], "water bottle": ["bottle", "flask", "tumbler"],
    "power adapter": ["adapter", "charger", "power"],
}


def correct(ref_label: str, answer: str) -> bool:
    return any(w in answer.lower() for w in SYNONYMS.get(ref_label, [ref_label]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("strips")
    ap.add_argument("ref")
    ap.add_argument("--model", default="qwen2.5vl:7b")
    ap.add_argument("--hint")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:11434")
    ap.add_argument("--ids", help="only these ids (comma-separated): the ones still present after merging")
    ap.add_argument("--env-file", help=".env with GEMINI_API_KEY")
    ap.add_argument("--workers", type=int, default=1, help="requests in parallel (API models)")
    ap.add_argument("--retry-failed", action="store_true", help="keep this tag's earlier answers; ask again only where it errored")
    a = ap.parse_args(argv)
    from chefbot.perception.detect_frames import load_classes
    from PIL import Image

    if a.ref == "-":  # no answer key: label every strip
        ids = sorted((f.stem[1:] for f in Path(a.strips).glob("o*.jpg")), key=int)
        ref = {i: {"label": ""} for i in ids}
    else:
        ref = {k: v for k, v in json.loads(Path(a.ref).read_text()).items() if not k.startswith("_")}
        ids = a.ids.split(",") if a.ids else [k for k, v in ref.items() if "dup_of" not in v]
    hint = load_classes(a.hint) if a.hint else None
    rows, t0 = [], time.time()
    key = gemini_key(a.env_file) if a.model.startswith("gemini") else None
    if not key:
        ask(Image.new("RGB", (64, 64)), 1, a.model, None, a.url)  # load the local model first: not counted
    t0 = time.time()
    from concurrent.futures import ThreadPoolExecutor

    def one(i):
        im = Image.open(Path(a.strips) / f"o{i}.jpg")
        return i, ask(im, max(1, im.width // 256), a.model, hint, a.url, key=key)

    out_file = Path(a.strips).parent / f"eval_{a.tag}.json"
    kept = {}
    if a.retry_failed and out_file.exists():
        for r in json.loads(out_file.read_text())["rows"]:
            if not r.get("error"):
                kept[r["id"]] = {"label": r["answer"], "movable_object": r["movable"], "ms": r["ms"], "tokens": r["tokens"]}
    with ThreadPoolExecutor(max(1, a.workers)) as pool:
        answers = dict(pool.map(one, [i for i in ids if i not in kept]))
    answers |= kept
    for i in ids:
        ans = answers[i]
        ok = bool(ref[i]["label"]) and correct(ref[i]["label"], ans.get("label", ""))
        rows.append({"id": i, "ref": ref[i]["label"], "answer": ans.get("label", ""), "movable": ans.get("movable_object"),
                     "ok": ok, "ms": ans.get("ms"), "tokens": ans.get("tokens"), "error": ans.get("error")})  # fmt: skip
    secs = time.time() - t0
    n_ok = sum(r["ok"] for r in rows)
    tin = sum((r["tokens"] or {}).get("in", 0) for r in rows)
    tout = sum((r["tokens"] or {}).get("out", 0) + (r["tokens"] or {}).get("thinking", 0) for r in rows)
    price = PRICES.get(a.model)
    cost = (tin * price[0] + tout * price[1]) / 1e6 if price else None
    errs = sum(1 for r in rows if r.get("error"))
    print(f"{a.tag}: {n_ok}/{len(rows)} right, {secs / len(rows):.2f} s/object" + (f", tokens in {tin} out {tout}" if tin else "")
          + (f", ${cost:.4f} total, ${cost / len(rows):.5f}/object" if cost is not None else "") + (f", {errs} errors" if errs else ""))
    for r in rows:
        if not r["ok"] and r["ref"]:
            print(f"   o{r['id']}: {r['ref']} -> {r['answer']}")
    Path(a.strips).parent.joinpath(f"eval_{a.tag}.json").write_text(json.dumps({"tag": a.tag, "model": a.model, "hint": bool(hint),
        "right": n_ok, "n": len(rows), "s_per_object": round(secs / len(rows), 2), "tokens_in": tin, "tokens_out": tout,
        "cost_usd": cost, "rows": rows}, indent=1))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
