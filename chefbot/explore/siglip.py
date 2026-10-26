"""SigLIP 2 naming an image crop: the word-list name whose text best matches it (local, free)."""

from __future__ import annotations

import numpy as np


def _labels():
    from chefbot.explore.vision import _VOCAB, FIXTURE_WORDS

    return list(_VOCAB["objects"]), [w.strip() for w in FIXTURE_WORDS.split(",")]


class Siglip:
    """Names an image crop: the catalogue word whose text best matches it."""

    def __init__(self, model_id: str = "google/siglip2-so400m-patch14-384"):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        objects, fixtures = _labels()
        self.labels = objects + [f for f in fixtures if f not in objects]
        self.n_objects = len(objects)
        with torch.no_grad():
            feats = []
            for i in range(0, len(self.labels), 64):
                t = self.proc(text=[f"a photo of a {w}." for w in self.labels[i:i + 64]], padding="max_length",
                              max_length=64, return_tensors="pt").to(self.device)  # fmt: skip
                f = self.model.get_text_features(**t)
                feats.append(getattr(f, "pooler_output", f))
            self.text = torch.cat(feats)
            self.text = self.text / self.text.norm(dim=-1, keepdim=True)

    def name(self, rgb: np.ndarray, box, context: float = 1.3) -> tuple[str, float]:
        from PIL import Image

        h, w = rgb.shape[:2]
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        half = max(16.0, max(box[2] - box[0], box[3] - box[1]) * context / 2)
        crop = rgb[int(max(0, cy - half)):int(min(h, cy + half)), int(max(0, cx - half)):int(min(w, cx + half))]
        if crop.size == 0:
            return "", 0.0
        with self.torch.no_grad():
            x = self.proc(images=Image.fromarray(crop), return_tensors="pt").to(self.device)
            f = self.model.get_image_features(**x)
            f = getattr(f, "pooler_output", f)
            f = f / f.norm(dim=-1, keepdim=True)
            sims = (f @ self.text.T)[0]
        i = int(sims.argmax())
        return self.labels[i], float(sims[i])
