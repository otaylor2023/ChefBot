"""VisionDetector: loose objects from the camera image, named by a vision language model.

detect(classes) renders RGB-D from the robot's camera, asks a VLM to name and
box every loose object in the image (open vocabulary: a catalogue of kitchen
nouns is offered as preferred names, anything else is kept as the VLM says
it), and turns each box into a world position from the depth image and the
camera pose. Nothing about the scene's objects is read from the simulator.

Fixtures and their parts (handles, burners, the faucet) are answered by the
scene's fixture geometry (the ground-truth detector restricted to those
labels): the starting map already gives the fixtures, and the doors and
burners are what the robot acts on.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

from chefbot.interface.detector import Detector
from chefbot.interface.types import RGBD, Detection

_EXPLORE = Path(__file__).parent
_CONFIG = yaml.safe_load((_EXPLORE / "config.yaml").read_text())
_VOCAB = yaml.safe_load((_EXPLORE / _CONFIG["vocab_file"]).read_text())

# What the VLM is told to leave out: the kitchen's own fixtures and their parts.
FIXTURE_WORDS = (
    "counter, countertop, cabinet, drawer, shelf, sink, faucet, stove, cooktop, burner, knob, oven, "
    "fridge, refrigerator, freezer, microwave, dishwasher, coffee machine, toaster, toaster oven, "
    "kettle, blender, stand mixer, range hood, handle, window, wall, floor, backsplash"
)

PROMPT = """This is a kitchen seen through a mobile robot's camera ({width}x{height} pixels). The robot's \
own arm may be in the lower middle of the image: ignore it.

List every loose object you can see: food, drinks, bottles, cookware, dishes, cups, utensils, \
containers, boards, small items. Include objects inside an open fridge or cabinet. Do not list the \
kitchen's fixtures or their parts ({fixtures}).

Name each object with a name from this list when one fits, otherwise a short common noun:
{vocab}

For each object give a tight bounding box in pixels, [x0, y0, x1, y1], x to the right and y down \
from the top-left corner. One entry per object; two mugs are two entries.

Reply with JSON only: {{"objects": [{{"label": "mug", "box": [x0, y0, x1, y1]}}]}}"""


def normalize(label: str, synonyms: dict[str, str], vocab: set[str]) -> str:
    """A VLM's name as a catalogue label where one fits: synonyms, then a plain plural."""
    name = re.sub(r"\s+", " ", label.strip().lower().replace("_", " "))
    if name in synonyms:
        return synonyms[name]
    if name not in vocab:
        for singular in (name[:-1] if name.endswith("s") else None, name[:-2] if name.endswith("es") else None):
            if singular and singular in vocab:
                return singular
    return name


def box_point(rgbd: RGBD, box: tuple[float, float, float, float]) -> np.ndarray | None:
    """World point of the object in `box`: the nearest surface in the box's middle half,
    which is the object and not the counter or wall behind it."""
    h, w = rgbd.depth.shape
    x0, y0, x1, y1 = (float(v) for v in box)
    x0, x1 = sorted((max(0.0, min(w - 1.0, x0)), max(0.0, min(w - 1.0, x1))))
    y0, y1 = sorted((max(0.0, min(h - 1.0, y0)), max(0.0, min(h - 1.0, y1))))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 4, (y1 - y0) / 4
    us = np.arange(int(cx - hw), int(cx + hw) + 1)
    vs = np.arange(int(cy - hh), int(cy + hh) + 1)
    uu, vv = np.meshgrid(us, vs)
    d = rgbd.depth[vv, uu]
    valid = np.isfinite(d) & (d > 0.05)
    if not valid.any():
        return None
    near = np.percentile(d[valid], 30)
    keep = valid & (d <= near + 0.03)
    u, v, z = float(uu[keep].mean()), float(vv[keep].mean()), float(np.median(d[keep]))
    k = rgbd.intrinsics
    camera = np.array([(u - k[0, 2]) / k[0, 0] * z, (v - k[1, 2]) / k[1, 1] * z, z, 1.0])
    return (rgbd.cam_to_world @ camera)[:3]


class VisionLanguageModel:
    """Names and boxes the loose objects in an image (OpenRouter chat with an image)."""

    def __init__(self, model: str, client=None):
        if client is None:
            from chefbot.agent.loop.llm import OpenRouterClient

            client = OpenRouterClient(model, temperature=0.0)
        self._client = client
        self._model = model
        self.cost_usd = 0.0

    def objects(self, rgb: np.ndarray) -> list[dict]:
        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(rgb).save(buffer, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        h, w = rgb.shape[:2]
        text = PROMPT.format(width=w, height=h, fixtures=FIXTURE_WORDS, vocab=", ".join(_VOCAB["objects"]))
        messages = [{"role": "user", "content": [{"type": "text", "text": text},
                                                 {"type": "image_url", "image_url": {"url": url}}]}]  # fmt: skip
        reply = self._client.chat(messages, model=self._model)
        self.cost_usd += reply.cost_usd
        return parse_objects(reply.text or "")


def parse_objects(text: str) -> list[dict]:
    """The `objects` list from a reply that should be JSON (tolerates a code fence or prose around it)."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return []
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for item in raw.get("objects") or []:
        box = item.get("box")
        if isinstance(item.get("label"), str) and isinstance(box, list) and len(box) == 4:
            out.append({"label": item["label"], "box": [float(v) for v in box]})
    return out


class VisionDetector(Detector):
    def __init__(
        self,
        render: Callable[[], RGBD],
        fixtures: Detector,
        fixture_labels: set[str],
        vlm: VisionLanguageModel | None = None,
        config: dict | None = None,
        log_dir: Path | None = None,
    ):
        self._render = render  # () -> RGBD from the robot's camera, now
        self._fixtures = fixtures  # answers fixture and part labels
        self._fixture_labels = fixture_labels
        cfg = {**_CONFIG["vision"], **(config or {})}
        self._vlm = vlm or VisionLanguageModel(cfg["model"])
        self._max_range = float(cfg["max_range"])
        self._synonyms = dict(cfg.get("synonyms") or {})
        self._vocab = set(_VOCAB["objects"])
        self._cache: tuple[str, list[Detection]] | None = None
        self._log_dir = log_dir
        self._calls = 0

    @classmethod
    def for_robocasa(cls, env, ground_truth, log_dir: Path | None = None) -> "VisionDetector":
        from chefbot.envs.robocasa.detector import FIXTURE_LABELS, PART_RULES

        camera = _CONFIG["vision"]["camera"]
        labels = set(FIXTURE_LABELS.values()) | {rule[3] for rule in PART_RULES} | set(_VOCAB["parts"])
        return cls(lambda: env.get_camera_rgbd(camera), ground_truth, labels, log_dir=log_dir)

    def detect(self, classes: list[str]) -> list[Detection]:
        wanted = set(classes)
        found = self._fixtures.detect([c for c in classes if c in self._fixture_labels])
        if wanted - self._fixture_labels:  # something that is not a fixture part: look at the image
            found += self._objects()
        return found

    def _objects(self) -> list[Detection]:
        rgbd = self._render()
        key = hashlib.sha1(rgbd.rgb[::8, ::8].tobytes()).hexdigest()  # same view: same answer
        if self._cache is not None and self._cache[0] == key:
            return list(self._cache[1])
        named = self._vlm.objects(rgbd.rgb)
        origin = rgbd.cam_to_world[:3, 3]
        out = []
        for item in named:
            label = normalize(item["label"], self._synonyms, self._vocab)
            if label in self._fixture_labels:
                continue
            point = box_point(rgbd, item["box"])
            if point is None or float(np.linalg.norm(point - origin)) > self._max_range:
                continue
            out.append(Detection(label, point, 0.9))
        self._cache = (key, out)
        self._log(rgbd, named, out)
        return list(out)

    def _log(self, rgbd: RGBD, named: list[dict], out: list[Detection]) -> None:
        """Each image the VLM saw, with what it said and where that put things (runs/<run>/vision/)."""
        if self._log_dir is None:
            return
        from PIL import Image

        self._calls += 1
        self._log_dir.mkdir(parents=True, exist_ok=True)
        stem = self._log_dir / f"{self._calls:03d}"
        Image.fromarray(rgbd.rgb).save(stem.with_suffix(".png"))
        stem.with_suffix(".json").write_text(json.dumps({
            "vlm": named,
            "detections": [{"label": d.label, "position": [round(float(v), 3) for v in d.position_world]} for d in out],
        }, indent=1))  # fmt: skip
