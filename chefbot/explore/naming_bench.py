"""Offline naming benchmark: which model names the kitchen's objects best, on the saved frames.

    python -m chefbot.explore.naming_bench <sightings.csv> runs/<run> ... --out <folder> [--models owl,gdino,siglip]

Uses the sighting table of rootcause.py (every true object in every saved detector frame, with its
visible box) and the frames themselves (runs/<run>/vision/NNN.png). For each model and each sighting
of a real object: named right, named something else, or not named; by visible pixels. On whole
frames also the false finds per picture (boxes on no real object) and the time per picture.

  owl          OWLv2 as the explorer uses it (explore/owl.py), on every frame
  gdino        an open Grounding DINO successor (LLMDet, or MM Grounding DINO), the catalogue as
               text prompts in chunks
  siglip       SigLIP 2 naming a crop of each real object (its true box, a little enlarged): the
               naming alone, given a good box
  owl+siglip   OWLv2's boxes, each renamed by SigLIP 2 from its crop
  vlm          the VLM's saved answers (only on the frames it saw; no new calls)

Offline and free: models run locally; nothing here reads the scene, only the saved frames and the
table. The true names and boxes come from the table (analysis only).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from chefbot.explore.aliases import canonical

PX_BINS = (0, 100, 300, 1000, 3000, 10**9)
_ON = 0.3  # a box covering this share of an object's visible box is on it


def _overlap(box, bbox) -> float:
    x0, y0, x1, y1 = max(box[0], bbox[0]), max(box[1], bbox[1]), min(box[2], bbox[2]), min(box[3], bbox[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return inter / max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))


def _iou(a, b) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _labels():
    from chefbot.explore.vision import _CONFIG, _VOCAB, FIXTURE_WORDS

    return list(_VOCAB["objects"]), [w.strip() for w in FIXTURE_WORDS.split(",")], dict(_CONFIG["vision"].get("synonyms") or {})


def _normalize(label: str) -> str:
    from chefbot.explore.vision import _CONFIG, _VOCAB, normalize

    return normalize(label, dict(_CONFIG["vision"].get("synonyms") or {}), set(_VOCAB["objects"]))


# -- models ----------------------------------------------------------------------------------


class GroundingDino:
    """An open Grounding DINO successor: boxes and names for the catalogue, prompted in chunks."""

    def __init__(self, model_id: str, threshold: float = 0.3, chunk: int = 40):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.device = "cpu"  # on the Mac GPU (MPS) its boxes come out wrong
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device).eval()
        objects, fixtures, _ = _labels()
        words = objects + [f for f in fixtures if f not in objects]
        self.fixtures = set(fixtures) - set(objects)
        self.chunks = [words[i:i + chunk] for i in range(0, len(words), chunk)]
        self.threshold = threshold

    def objects(self, rgb: np.ndarray) -> list[dict]:
        from PIL import Image

        image = Image.fromarray(rgb)
        found = []
        for words in self.chunks:
            inp = self.proc(images=image, text=[words], return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                out = self.model(**inp)
            res = self.proc.post_process_grounded_object_detection(
                out, threshold=self.threshold, text_threshold=self.threshold, target_sizes=[rgb.shape[:2]])[0]  # fmt: skip
            labels = res.get("text_labels", res.get("labels"))
            for box, score, label in zip(res["boxes"].cpu().numpy(), res["scores"].cpu().numpy(), labels):
                found.append({"label": str(label), "box": [float(v) for v in box], "score": float(score)})
        found.sort(key=lambda f: -f["score"])
        kept = []
        for f in found:  # one box per object, whatever the name
            if all(_iou(f["box"], k["box"]) < 0.5 for k in kept):
                kept.append(f)
        return [k for k in kept if k["label"] not in self.fixtures]


# -- evaluation -------------------------------------------------------------------------------


def evaluate(name: str, frames: dict, sightings: dict, items_of, fixture_words: set, aliases: bool = False) -> dict:
    """items_of(frame_key) -> [{"label", "box"}]; frames: key -> rgb path; sightings: key -> rows."""
    rows, false_finds, times = [], [], []
    for key in frames:
        t = time.time()
        items = items_of(key)
        times.append(time.time() - t)
        items = [dict(i, label=_normalize(i["label"])) for i in items]
        items = [i for i in items if i["label"] not in fixture_words]
        truth = sightings.get(key, [])
        boxes = [[float(r[k]) for k in ("bx0", "by0", "bx1", "by1")] for r in truth]
        for r, bbox in zip(truth, boxes):
            on = [i for i in items if _overlap(i["box"], bbox) >= _ON]
            same = (lambda a, b: canonical(a) == canonical(b)) if aliases else (lambda a, b: a == b)
            right = any(same(i["label"], r["label"]) for i in on)
            outcome = "right" if right else ("wrong" if on else "missed")
            rows.append({"px": int(r["visible_px"]), "outcome": outcome, "label": r["label"],
                         "as": next((i["label"] for i in on if not same(i["label"], r["label"])), "")})  # fmt: skip
        false_finds.append(sum(1 for i in items if all(_overlap(i["box"], b) < _ON and _overlap(b, i["box"]) < _ON for b in boxes)))
    return {"name": name, "rows": rows, "false_per_frame": float(np.mean(false_finds)) if false_finds else 0.0,
            "sec_per_frame": float(np.mean(times)) if times else 0.0, "frames": len(frames)}  # fmt: skip


def table(results: list[dict]) -> str:
    out = ["| model | sightings | named right | wrong name | not named | false finds per picture | s per picture |",
           "|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in results:
        n = len(r["rows"])
        c = lambda o: 100 * sum(x["outcome"] == o for x in r["rows"]) / max(1, n)
        out.append(f"| {r['name']} | {n} | {c('right'):.0f}% | {c('wrong'):.0f}% | {c('missed'):.0f}% | "
                   f"{r['false_per_frame']:.1f} | {r['sec_per_frame']:.2f} |")  # fmt: skip
    out += ["", "Named right, by visible pixels:", "",
            "| model | " + " | ".join(f"{lo}-{hi if hi < 10**8 else 'more'}" for lo, hi in zip(PX_BINS, PX_BINS[1:])) + " |",
            "|---|" + "---|" * (len(PX_BINS) - 1)]  # fmt: skip
    for r in results:
        cells = []
        for lo, hi in zip(PX_BINS, PX_BINS[1:]):
            b = [x for x in r["rows"] if lo <= x["px"] < hi]
            cells.append(f"{100 * sum(x['outcome'] == 'right' for x in b) / len(b):.0f}% ({len(b)})" if b else "")
        out.append(f"| {r['name']} | " + " | ".join(cells) + " |")
    out += ["", "Most frequent wrong names:", ""]
    for r in results:
        wrong = [f"{x['label']} as {x['as']}" for x in r["rows"] if x["outcome"] == "wrong"]
        top = sorted(set(wrong), key=wrong.count, reverse=True)[:6]
        out.append(f"- {r['name']}: " + (", ".join(f"{w} ({wrong.count(w)})" for w in top) or "none"))
    return "\n".join(out)


def main() -> int:
    from PIL import Image

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sightings", type=Path)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", default="owl,gdino,siglip,owl+siglip,vlm")
    parser.add_argument("--gdino", default="iSEE-Laboratory/llmdet_tiny")
    parser.add_argument("--aliases", action="store_true", help="also score with near-synonyms as one name")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    _, fixtures, _ = _labels()
    fixture_words = set(fixtures)
    rows = [r for r in csv.DictReader(open(args.sightings)) if r.get("bx0") not in (None, "")]
    dirs = {d.name: d for d in args.runs}
    sightings: dict = {}
    for r in rows:
        sightings.setdefault((r["run"], int(r["frame"])), []).append(r)
    frames = {}
    for name, d in dirs.items():
        for png in sorted((d / "vision").glob("*.png")):
            frames[(name, int(png.stem))] = png
    cache: dict = {}

    def rgb(key):
        if key not in cache:
            cache.clear()
            cache[key] = np.asarray(Image.open(frames[key]).convert("RGB"))
        return cache[key]

    wanted = args.models.split(",")
    results = []
    if "vlm" in wanted:
        vlm_runs = {n for n, d in dirs.items() if json.loads((d / "start.json").read_text()).get("detector") == "vision"}
        vlm_frames = {k: v for k, v in frames.items() if k[0] in vlm_runs}
        for al in ([False, True] if args.aliases else [False]):
            results.append(evaluate("vlm (saved answers; its frames only)" + (", aliases" if al else ""), vlm_frames, sightings,
                                    lambda k: json.loads(frames[k].with_suffix(".json").read_text())["vlm"], fixture_words, al))  # fmt: skip
    owl = None
    if "owl" in wanted or "owl+siglip" in wanted or "hybrid" in wanted:
        from chefbot.explore.owl import OwlV2Model

        owl = OwlV2Model()
        owl_items = {}

        def owl_of(k):
            if k not in owl_items:
                owl_items[k] = owl.objects(rgb(k))
            return owl_items[k]

        if "owl" in wanted:
            for al in ([False, True] if args.aliases else [False]):
                results.append(evaluate("owl (as the explorer uses it)" + (", aliases" if al else ""), frames, sightings,
                                        owl_of, fixture_words, al))  # fmt: skip
                print(table(results[-1:]), flush=True)
    from chefbot.explore.siglip import Siglip

    siglip = Siglip() if ("siglip" in wanted or "owl+siglip" in wanted or "hybrid" in wanted) else None
    if "hybrid" in wanted:
        hybrid_items = {}

        def hybrid_of(k):  # OWLv2's names for boxes of 1000 px and more, SigLIP 2's under
            if k not in hybrid_items:
                out = []
                for i in owl_of(k):
                    x0, y0, x1, y1 = i["box"]
                    if (x1 - x0) * (y1 - y0) < 1000:
                        name = siglip.name(rgb(k), i["box"])[0]
                        if name not in siglip.labels[: siglip.n_objects]:
                            continue
                        i = dict(i, label=name)
                    out.append(i)
                hybrid_items[k] = out
            return hybrid_items[k]

        for al in ([False, True] if args.aliases else [False]):
            results.append(evaluate("hybrid: owl, siglip under 1000 px" + (", aliases" if al else ""), frames, sightings,
                                    hybrid_of, fixture_words, al))  # fmt: skip
            print(table(results[-1:]), flush=True)
    if "owl+siglip" in wanted:
        results.append(evaluate("owl boxes + siglip names", frames, sightings,
                                lambda k: [dict(i, label=siglip.name(rgb(k), i["box"])[0]) for i in owl_of(k)], fixture_words))  # fmt: skip
        print(table(results[-1:]), flush=True)
    if "siglip" in wanted:
        def oracle(k):  # the true objects' own boxes, each named by SigLIP 2
            return [{"label": siglip.name(rgb(k), [float(r[c]) for c in ("bx0", "by0", "bx1", "by1")])[0],
                     "box": [float(r[c]) for c in ("bx0", "by0", "bx1", "by1")]} for r in sightings.get(k, [])]  # fmt: skip

        results.append(evaluate("siglip naming the true boxes (naming only)", frames, sightings, oracle, fixture_words))
        print(table(results[-1:]), flush=True)
    if "gdino" in wanted:
        gd = GroundingDino(args.gdino)
        results.append(evaluate(f"{args.gdino.split('/')[-1]}", frames, sightings, lambda k: gd.objects(rgb(k)), fixture_words))
        print(table(results[-1:]), flush=True)
    text = table(results)
    (args.out / "naming_bench.md").write_text(text + "\n")
    print("\n" + text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
