"""The vision models skills may call (open-vocabulary detection, SAM 2), over localhost HTTP: CUDA, Apple's MPS or
the CPU. Runs in its own env (torch + transformers); standard library + numpy otherwise. Imports nothing of ChefBot,
so it runs as a plain script:

    <vision env>/bin/python chefbot/runtime/vision_server.py --port 8777     # then VISION_URL=http://127.0.0.1:8777

POST /detect  {"image": <.npy path>, "queries": [...], "model": "owlv2" | "grounding-dino" | "yoloe", "threshold": 0.1,
               "imgsz": 1280 (yoloe only)}
              -> {"detections": [{"label", "score", "box": [x0, y0, x1, y1]}]}; yoloe's also carry "poly": its mask's
                 outline [[x, y], ...] (the kitchen map's points come from it), one frame for the whole vocabulary
                 (YOLOE-26 text prompts, ultralytics: weights YOLOE_MODEL in YOLOE_WEIGHTS, default
                 yoloe-26s-seg.pt in ~/.cache/yoloe; ~0.14 s a 1280 frame on an M1's MPS)
POST /segment {"image": <.npy path>, "boxes": [[x0, y0, x1, y1], ...] | null, "points": [[x, y], ...] | null,
               "point_labels": [1, 0, ...] | null, "model": "sam2"}
              -> {"masks": <.npy path of (N, H, W) bool>}
Models load at startup (--lazy: on first use) and stay loaded; requests are served one at a time.
(agentic-skills-exploration's vision_models/serve.py.)
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

YOLOE_WEIGHTS = Path(os.environ.get("YOLOE_WEIGHTS", Path.home() / ".cache" / "yoloe"))
YOLOE_MODEL = os.environ.get("YOLOE_MODEL", "yoloe-26s-seg.pt")
CHECKPOINTS = {"owlv2": "google/owlv2-base-patch16-ensemble", "grounding-dino": "IDEA-Research/grounding-dino-tiny",
               "sam2": os.environ.get("SAM2_CHECKPOINT", "facebook/sam2.1-hiera-small")}  # fmt: skip
_models: dict = {}
_lock = threading.Lock()
OUT = Path(tempfile.mkdtemp(prefix="chefbot_vision_"))


def _device():
    import torch

    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def _load(name: str):
    if name in _models:
        return _models[name]
    import torch
    from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor, Owlv2ForObjectDetection, Owlv2Processor,
                              Sam2Model, Sam2Processor)  # fmt: skip

    ck, dev = CHECKPOINTS[name], _device()
    dtype = torch.float16 if dev == "cuda" else torch.float32
    if name == "owlv2":
        m = (Owlv2Processor.from_pretrained(ck), Owlv2ForObjectDetection.from_pretrained(ck, torch_dtype=dtype).to(dev).eval())
    elif name == "grounding-dino":
        m = (AutoProcessor.from_pretrained(ck), AutoModelForZeroShotObjectDetection.from_pretrained(ck).to(dev).eval())
    else:
        m = (Sam2Processor.from_pretrained(ck), Sam2Model.from_pretrained(ck).to(dev).eval())
    _models[name] = (m, dev, dtype)
    return _models[name]


def _image(path: str):
    from PIL import Image

    return Image.fromarray(np.load(path).astype(np.uint8)).convert("RGB")


class _InDir:
    """YOLOE fetches its weights and its text encoder (mobileclip) into the working directory: there, then back."""

    def __enter__(self):
        self.was = os.getcwd()
        YOLOE_WEIGHTS.mkdir(parents=True, exist_ok=True)
        os.chdir(YOLOE_WEIGHTS)

    def __exit__(self, *exc):
        os.chdir(self.was)


def _yoloe(queries: tuple[str, ...]):
    """The YOLOE model, its text prompts set to `queries` (re-set only when they change: ~0.1 s a vocabulary)."""
    if "yoloe" not in _models:
        with _InDir():
            from ultralytics import YOLOE

            _models["yoloe"] = [YOLOE(YOLOE_MODEL), None]
    m = _models["yoloe"]
    if queries and m[1] != queries:
        with _InDir():
            m[0].set_classes(list(queries))
        m[1] = queries
    return m[0]


def detect_yoloe(image: str, queries: list[str], threshold: float = 0.15, imgsz: int = 1280) -> dict:
    net = _yoloe(tuple(queries))
    rgb = np.load(image).astype(np.uint8)
    r = net.predict(rgb[..., ::-1], imgsz=imgsz, conf=threshold, device=_device(), verbose=False)[0]  # BGR in
    polys = r.masks.xy if r.masks is not None else [None] * len(r.boxes)
    dets = [{"label": r.names[int(b.cls)], "score": round(float(b.conf), 4), "box": [round(float(v), 1) for v in b.xyxy[0].tolist()],
             "poly": [] if poly is None else [[round(float(x), 1), round(float(y), 1)] for x, y in poly]}
            for b, poly in zip(r.boxes, polys)]  # fmt: skip
    dets.sort(key=lambda d: -d["score"])
    return {"detections": dets[:100]}


def detect(image: str, queries: list[str], model: str = "owlv2", threshold: float = 0.1, imgsz: int = 1280) -> dict:
    if model == "yoloe":
        return detect_yoloe(image, queries, threshold, imgsz)
    import torch

    if model not in ("owlv2", "grounding-dino"):
        raise ValueError(f"detect models: owlv2, grounding-dino, yoloe (not {model!r})")
    (proc, net), dev, dtype = _load(model)
    img = _image(image)
    size = torch.tensor([img.size[::-1]])
    with torch.no_grad():
        if model == "owlv2":
            inp = proc(text=[list(queries)], images=img, return_tensors="pt").to(dev)
            inp = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in inp.items()}
            out = net(**inp)
            # OWLv2 pads to a square: boxes come back relative to the padded image
            side = max(img.size)
            r = proc.post_process_grounded_object_detection(out, threshold=threshold, target_sizes=torch.tensor([[side, side]]))[0]
            labels = [queries[int(i)] for i in r["labels"]]
        else:
            inp = proc(images=img, text=". ".join(q.rstrip(".") for q in queries) + ".", return_tensors="pt").to(dev)
            out = net(**inp)
            r = proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=threshold, text_threshold=threshold,
                                                            target_sizes=size)[0]  # fmt: skip
            labels = [str(l) for l in r.get("text_labels", r.get("labels"))]
    dets = [{"label": l, "score": round(float(s), 4), "box": [round(float(v), 1) for v in b]} for l, s, b in zip(labels, r["scores"], r["boxes"])]
    dets.sort(key=lambda d: -d["score"])
    return {"detections": dets[:50]}


def segment(image: str, boxes=None, points=None, point_labels=None, model: str = "sam2") -> dict:
    import torch

    (proc, net), dev, _ = _load("sam2")
    img = _image(image)
    kw = {}
    if boxes:
        kw["input_boxes"] = [[list(map(float, b)) for b in boxes]]
    if points:
        kw["input_points"] = [[[list(map(float, p)) for p in points]]]
        kw["input_labels"] = [[[int(l) for l in (point_labels or [1] * len(points))]]]
    if not kw:
        raise ValueError("segment needs boxes or points")
    inp = proc(images=img, return_tensors="pt", **kw).to(dev)
    with torch.no_grad():
        out = net(**inp, multimask_output=False)
    masks = proc.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"])[0]  # (N, 1, H, W)
    masks = masks.reshape(-1, *masks.shape[-2:]).numpy().astype(bool)
    p = OUT / f"masks_{uuid.uuid4().hex[:10]}.npy"
    np.save(p, masks)
    return {"masks": str(p)}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length") or 0)
        try:
            args = json.loads(self.rfile.read(n) or b"{}")
            op = self.path.strip("/")
            t0 = time.time()
            with _lock:
                out = detect(**args) if op == "detect" else segment(**args) if op == "segment" else {"ok": True, "device": _device()}
            out["server_s"] = round(time.time() - t0, 3)
            code = 200
        except Exception as e:  # noqa: BLE001
            out, code = {"error": f"{type(e).__name__}: {e}"[:500]}, 400
        body = json.dumps(out).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # quiet
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--lazy", action="store_true", help="load each model on first use (the first episode then pays for it)")
    ap.add_argument("--models", nargs="+", default=[*CHECKPOINTS, "yoloe"],
                    help="what to load at startup (default all; an env without transformers: --models yoloe)")
    args = ap.parse_args()
    if not args.lazy:
        for name in args.models:
            _yoloe(()) if name == "yoloe" else _load(name)
    print(f"vision server on 127.0.0.1:{args.port} ({_device()})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
