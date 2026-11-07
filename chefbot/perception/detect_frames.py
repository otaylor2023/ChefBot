"""Open-vocabulary detection + instance masks on recorded camera frames with YOLOE (Ultralytics): the "what is it"
step of the memory map (docs/memory-map-next.md). Runs in a vision env (ultralytics >= 8.4, torch); imports nothing
of ChefBot, so it runs as a plain script:

    # prompt-free: YOLOE's built-in 4,585-name vocabulary, to see what's there (no class list needed)
    python chefbot/perception/detect_frames.py STREAM/p2_layout3 --model yoloe-26s-seg-pf.pt --out OUT
    # text prompts: our own vocabulary (one name per line; '#' comments)
    python chefbot/perception/detect_frames.py STREAM/p* --model yoloe-26s-seg.pt --classes vocab.txt --out OUT

Writes OUT/<pass>/<cam>/detections.jsonl, one line per frame:
    {"n", "file", "ms": {preprocess, inference, postprocess}, "dets": [{"label", "score", "box": [x0, y0, x1, y1],
     "poly": [[x, y], ...]}]}            (poly: the instance mask's outline in color-image pixels)
plus OUT/<pass>/<cam>/overlay/ (every --overlay-every'th frame, annotated, half size) and OUT/summary.json (labels by
number of frames they appear in, timing). Weights download into --weights-dir (keep it off a small root disk).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path


def load_classes(path: str | None) -> list[str] | None:
    if not path:
        return None
    names = [l.split("#")[0].strip() for l in Path(path).read_text().splitlines()]
    return [n for n in names if n]


def make_model(name: str, classes: list[str] | None, weights_dir: str):
    os.makedirs(weights_dir, exist_ok=True)
    os.chdir(weights_dir)  # YOLOE fetches its weights and the text encoder (mobileclip) into the working directory
    from ultralytics import YOLOE

    model = YOLOE(name)
    if classes:
        if "-pf" in name:
            raise SystemExit("prompt-free (-pf) weights take no --classes: use the -seg weights")
        model.set_classes(classes)
    return model


def predict(model, img, imgsz: int = 1280, conf: float = 0.15, device=None, half: bool = False):
    """One image (a path or an RGB array): the ultralytics result and its detections as plain dicts."""
    src = str(img) if isinstance(img, (str, Path)) else img[..., ::-1]  # ultralytics takes arrays as BGR
    kw = {"half": True} if half else {}  # (passing half=False makes ultralytics warn on every frame)
    r = model.predict(src, imgsz=imgsz, conf=conf, device=device, verbose=False, **kw)[0]
    dets = []
    polys = r.masks.xy if r.masks is not None else [None] * len(r.boxes)
    for b, poly in zip(r.boxes, polys):
        dets.append({"label": r.names[int(b.cls)], "score": round(float(b.conf), 3),
                     "box": [round(float(v), 1) for v in b.xyxy[0].tolist()],
                     "poly": [] if poly is None else [[round(float(x), 1), round(float(y), 1)] for x, y in poly]})  # fmt: skip
    return r, dets


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("passes", nargs="+", help="stream/<pass> dirs")
    ap.add_argument("--cam", default="pole")
    ap.add_argument("--model", default="yoloe-26s-seg-pf.pt")
    ap.add_argument("--classes", help="vocabulary file (text-prompt weights only)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--device", default=None, help="0 (GPU: CUDA or ROCm), mps, cpu; default: ultralytics picks")
    ap.add_argument("--half", action="store_true", help="fp16 (GPU)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights-dir", default=os.environ.get("YOLOE_WEIGHTS", str(Path.home() / ".cache" / "yoloe")))
    ap.add_argument("--overlay-every", type=int, default=5, help="0: none")
    ap.add_argument("--limit", type=int, default=0, help="frames per pass (0: all), for quick trials and benchmarks")
    a = ap.parse_args(argv)

    passes = [Path(p).resolve() for p in a.passes]
    out_root = Path(a.out).resolve()
    model = make_model(a.model, load_classes(a.classes), a.weights_dir)
    seen, times, n_frames = Counter(), [], 0
    for root in passes:
        cam_dir = root / a.cam
        frames = [json.loads(l) for l in (cam_dir / "frames.jsonl").read_text().splitlines() if l.strip()]
        if a.limit:
            frames = frames[: a.limit]
        out = out_root / root.name / a.cam
        (out / "overlay").mkdir(parents=True, exist_ok=True)
        with open(out / "detections.jsonl", "w") as fo:
            for i, f in enumerate(frames):
                img = cam_dir / "color" / f"{int(f['n']):06d}.png"
                if not img.exists():
                    continue
                r, dets = predict(model, img, a.imgsz, a.conf, a.device, a.half)
                for lab in {d["label"] for d in dets}:
                    seen[lab] += 1
                if i > 0:  # the first frame pays for warm-up
                    times.append(r.speed)
                n_frames += 1
                fo.write(json.dumps({"n": f["n"], "file": img.name, "ms": {k: round(v, 2) for k, v in r.speed.items()},
                                     "dets": dets}) + "\n")  # fmt: skip
                if a.overlay_every and i % a.overlay_every == 0:
                    from PIL import Image

                    im = Image.fromarray(r.plot()[..., ::-1])
                    im.resize((im.width // 2, im.height // 2)).save(out / "overlay" / img.name.replace(".png", ".jpg"), quality=85)
        print(f"{root.name}: {len(frames)} frames -> {out / 'detections.jsonl'}", flush=True)

    mean = {k: round(sum(t[k] for t in times) / max(1, len(times)), 2) for k in ("preprocess", "inference", "postprocess")}
    summary = {"model": a.model, "imgsz": a.imgsz, "conf": a.conf, "device": a.device, "half": a.half,
               "frames": n_frames, "mean_ms": mean, "labels_by_frames": dict(seen.most_common()),
               "when": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("model", "imgsz", "device", "frames", "mean_ms")}))
    print("top labels:", list(seen.most_common(40)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
