"""SAM 3 (Meta, "Segment Anything with Concepts"): text prompts in, masks and boxes out.

Each word-list name is a prompt; the image is encoded once and the prompts run against it in
batches. Local (the facebook/sam3 weights, gated: a Hugging Face login with Meta's license
accepted). Used by the offline naming benchmark (naming_bench.py).
"""

from __future__ import annotations

import numpy as np


class Sam3:
    def __init__(self, model_id: str = "facebook/sam3", threshold: float = 0.4, batch: int = 4, device: str = "cpu"):
        """`device`: cpu by default (as fast as the Mac GPU here, and it leaves the GPU's memory to
        the other models)."""
        import torch
        from transformers import Sam3Model, Sam3Processor

        self.torch = torch
        self.device = device
        self.proc = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id).to(self.device).eval()
        self.threshold = threshold
        self.batch = batch

    def objects(self, rgb: np.ndarray, prompts: list[str]) -> list[dict]:
        """[{"label", "box", "score", "mask"}] over all prompts, one box per thing (best score)."""
        from PIL import Image

        torch = self.torch
        image = Image.fromarray(rgb)
        h, w = rgb.shape[:2]
        found = []
        with torch.no_grad():
            pixels = self.proc(images=image, return_tensors="pt").to(self.device)
            vision = self.model.get_vision_features(pixel_values=pixels["pixel_values"])
            for i in range(0, len(prompts), self.batch):
                words = prompts[i:i + self.batch]
                text = self.proc(text=words, return_tensors="pt", padding=True).to(self.device)
                embeds = _repeat(vision, len(words))
                out = self.model(vision_embeds=embeds, input_ids=text["input_ids"], attention_mask=text["attention_mask"])
                results = self.proc.post_process_instance_segmentation(
                    out, threshold=self.threshold, mask_threshold=0.5, target_sizes=[(h, w)] * len(words))  # fmt: skip
                for word, res in zip(words, results):
                    for box, score, mask in zip(res["boxes"], res["scores"], res["masks"]):
                        found.append({"label": word, "box": [float(v) for v in box.cpu()], "score": float(score),
                                      "mask": mask.cpu().numpy().astype(bool)})  # fmt: skip
        found.sort(key=lambda f: -f["score"])
        kept = []
        for f in found:
            if all(_iou(f["box"], k["box"]) < 0.5 for k in kept):
                kept.append(f)
        return kept


def _repeat(vision, n: int):
    """The image's encoding, repeated for a batch of n prompts."""
    import torch

    def rep(x):
        if isinstance(x, torch.Tensor):
            return x.expand(n, *x.shape[1:]) if x.shape[0] == 1 else x
        if isinstance(x, (list, tuple)):
            return type(x)(rep(v) for v in x)
        return x

    out = type(vision)(**{k: rep(v) for k, v in vision.items()}) if hasattr(vision, "items") else rep(vision)
    return out


def _iou(a, b) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
