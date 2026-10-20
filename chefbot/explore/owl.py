"""OWLv2: an off-the-shelf open-vocabulary detector in place of the VLM (--detector owl).

It answers the same question as VisionLanguageModel.objects: the loose objects in
an image, each a label and a pixel box. The rest of VisionDetector (edge filter,
depth back-projection, footprint and height filters, tracking) is unchanged, so
the two can be compared on the same pipeline and scorer.

The text queries are the catalogue's object names, embedded once. The fixture
words (counter, burner, handle, ...) are queried too: a box whose best match is a
fixture word is a part of the kitchen and is dropped, instead of a look-alike
object name winning it. Runs locally (Apple GPU when available).
"""

from __future__ import annotations

import numpy as np

from chefbot.explore.vision import _CONFIG, _VOCAB, FIXTURE_WORDS


class OwlV2Model:
    def __init__(self, config: dict | None = None, names: list[str] | None = None, fixture_words: list[str] | None = None):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        cfg = {**_CONFIG["owl"], **(config or {})}
        self._threshold = float(cfg["threshold"])
        self._nms_iou = float(cfg["nms_iou"])
        self._device = cfg.get("device") or ("mps" if torch.backends.mps.is_available() else "cpu")
        self._torch = torch
        self._processor = Owlv2Processor.from_pretrained(cfg["model"])
        self._model = Owlv2ForObjectDetection.from_pretrained(cfg["model"]).to(self._device).eval()
        objects = list(names if names is not None else _VOCAB["objects"])
        fixtures = list(fixture_words if fixture_words is not None else [w.strip() for w in FIXTURE_WORDS.split(",")])
        if not cfg.get("fixture_queries", True):
            fixtures = []
        self._labels = objects + [w for w in fixtures if w not in objects]
        self._n_objects = len(objects)  # queries past this are fixture words
        self._queries = self._embed([cfg["prompt"].format(name=n) for n in self._labels])
        self.cost_usd = 0.0  # local: no API cost

    def _embed(self, prompts: list[str]):
        torch = self._torch
        out = []
        with torch.no_grad():
            for i in range(0, len(prompts), 64):
                tokens = self._processor.tokenizer(prompts[i : i + 64], padding="max_length", max_length=16,
                                                   truncation=True, return_tensors="pt").to(self._device)  # fmt: skip
                features = self._model.owlv2.get_text_features(**tokens)
                out.append(getattr(features, "pooler_output", features))
        return torch.cat(out)[None]  # (1, queries, dim)

    def objects(self, rgb: np.ndarray) -> list[dict]:
        torch = self._torch
        from PIL import Image

        h, w = rgb.shape[:2]
        pixels = self._processor.image_processor(images=Image.fromarray(rgb), return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            feature_map, _ = self._model.image_embedder(pixel_values=pixels.to(self._device))
            feats = feature_map.reshape(1, -1, feature_map.shape[-1])
            mask = torch.ones(self._queries.shape[:2], dtype=torch.bool, device=self._device)
            logits, _ = self._model.class_predictor(feats, self._queries, mask)
            boxes = self._model.box_predictor(feats, feature_map)
        scores = torch.sigmoid(logits[0]).float().cpu().numpy()  # (patches, queries)
        boxes = boxes[0].float().cpu().numpy()  # cx, cy, bw, bh as fractions of the padded square
        return pick_boxes(scores, boxes, self._labels, self._n_objects, self._threshold, self._nms_iou, h, w)


def pick_boxes(scores: np.ndarray, boxes: np.ndarray, labels: list[str], n_objects: int, threshold: float,
               nms_iou: float, h: int, w: int) -> list[dict]:  # fmt: skip
    """Each patch's best-matching name; kept if it is an object name scoring at least `threshold`,
    then one box per object (the best-scoring one), in pixels of the h x w image."""
    best = scores.argmax(axis=1)
    best_score = scores[np.arange(len(best)), best]
    keep = np.nonzero((best_score >= threshold) & (best < n_objects))[0]  # a fixture word's box: not an object
    side = max(h, w)  # the processor pads to a square at the bottom right, then resizes
    found = []
    for i in keep[np.argsort(-best_score[keep])]:
        cx, cy, bw, bh = (float(v) * side for v in boxes[i])
        box = [max(0.0, cx - bw / 2), max(0.0, cy - bh / 2), min(w - 1.0, cx + bw / 2), min(h - 1.0, cy + bh / 2)]
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
            continue
        if all(_iou(box, other["box"]) < nms_iou for other in found):  # one box per object, whatever its name
            found.append({"label": labels[best[i]], "box": [round(v, 1) for v in box], "score": round(float(best_score[i]), 3)})
    return found


def _iou(a, b) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
