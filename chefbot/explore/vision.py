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
import math
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
    "microwave turntable, kettle, blender, stand mixer, range hood, handle, window, wall, floor, backsplash"
)

PROMPT = """This is a kitchen seen through a mobile robot's camera ({width}x{height} pixels). The robot's \
own arm may be in the lower middle of the image: ignore it.

List every loose object you can see: food, drinks, bottles, cookware, dishes, cups, utensils, \
containers, boards, small items. Include objects inside an open fridge or cabinet. Do not list the \
kitchen's fixtures or their parts ({fixtures}); the glass turntable inside a microwave is part of the \
microwave, not a plate.

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

    def __init__(self, model: str, client=None, max_cost_usd: float | None = None):
        if client is None:
            from chefbot.agent.loop.llm import OpenRouterClient

            client = OpenRouterClient(model, temperature=0.0)
        self._client = client
        self._model = model
        self._max_cost = max_cost_usd  # stop calling the model once this much is spent
        self.cost_usd = 0.0
        self.calls = 0
        self.skipped_over_budget = 0

    def objects(self, rgb: np.ndarray) -> list[dict]:
        from PIL import Image

        if self._max_cost is not None and self.cost_usd >= self._max_cost:
            self.skipped_over_budget += 1
            return []
        self.calls += 1

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
        footprints: list | None = None,
    ):
        self._render = render  # () -> RGBD from the robot's camera, now
        self._fixtures = fixtures  # answers fixture and part labels
        self._fixture_labels = fixture_labels
        cfg = {**_CONFIG["vision"], **(config or {})}
        self._vlm = vlm or VisionLanguageModel(cfg["model"], max_cost_usd=cfg.get("max_cost_usd"))
        self._same_view = float(cfg.get("same_view_diff", 1.0))
        self._refined_spots: list[tuple[np.ndarray, dict]] = []  # where a small box was looked at again, and the answer
        self._max_range = float(cfg["max_range"])
        self._synonyms = dict(cfg.get("synonyms") or {})
        self._vocab = set(_VOCAB["objects"])
        self._edge = float(cfg.get("edge_margin", 8))
        self._merge = float(cfg.get("merge_radius", 0.3))
        self._height = tuple(cfg.get("height_range", (0.2, 2.2)))
        self._margin = float(cfg.get("footprint_margin", 0.1))
        self._conflict = float(cfg.get("conflict_radius", 0.08))
        self._closer = float(cfg.get("closer_by", 0.5))
        self._vote = bool(cfg.get("label_vote", False))
        self._aliases = bool(cfg.get("use_aliases", False))
        # A group name (mug: cup and mug) can stand for two real things side by side: its
        # sightings are one object only this close.
        self._alias_merge = float(cfg.get("alias_merge_radius", 0.12))
        self._refine = bool(cfg.get("refine_small", False))
        self._refine_px = float(cfg.get("refine_small_px", 1000))
        self._refine_context = float(cfg.get("refine_context", 3.0))
        self._refine_side = int(cfg.get("refine_side", 384))
        self.refined = 0  # boxes looked at again, enlarged
        self._approach = bool(cfg.get("approach_small", False))
        self._approach_px = float(cfg.get("approach_small_px", 1000))
        self._footprints = [f for f in (footprints or []) if f.size is not None and f.facing is not None]
        self._tracks: list[dict] = []  # one per object: label, summed position, sightings
        self._cache: tuple[str, list[int]] | None = None
        self._last_thumb: np.ndarray | None = None
        self._log_dir = log_dir
        self._calls = 0
        self.camera_name: Callable[[], str] | None = None  # which camera the last view came from

    @classmethod
    def for_robocasa(cls, env, ground_truth, log_dir: Path | None = None, footprints: list | None = None,
                     vlm=None) -> "VisionDetector":  # fmt: skip
        from chefbot.envs.robocasa.detector import FIXTURE_LABELS, PART_RULES

        cameras = list(_CONFIG["vision"].get("cameras") or [_CONFIG["vision"]["camera"]])
        labels = set(FIXTURE_LABELS.values()) | {rule[3] for rule in PART_RULES} | set(_VOCAB["parts"])
        chosen = [cameras[0]]

        def render() -> RGBD:
            # The camera the robot's own arm blocks least (into an open fridge it sits right in front of one).
            if len(cameras) > 1:
                occluded = {cam: env.camera_self_occlusion(cam) for cam in cameras}
                chosen[0] = min(cameras, key=lambda cam: (round(occluded[cam], 2), cameras.index(cam)))
            return env.get_camera_rgbd(chosen[0])

        det = cls(render, ground_truth, labels, vlm, log_dir=log_dir, footprints=footprints)
        det.camera_name = lambda: chosen[0]
        return det

    def detect(self, classes: list[str]) -> list[Detection]:
        wanted = set(classes)
        found = self._fixtures.detect([c for c in classes if c in self._fixture_labels])
        if wanted - self._fixture_labels:  # something that is not a fixture part: look at the image
            found += self._objects()
        return found

    def _objects(self) -> list[Detection]:
        rgbd = self._render()
        key = hashlib.sha1(rgbd.rgb[::8, ::8].tobytes()).hexdigest()  # same view: same answer
        thumb = rgbd.rgb[::4, ::4].astype(np.int16)
        if self._cache is not None and (self._cache[0] == key or (
                self._last_thumb is not None and float(np.abs(thumb - self._last_thumb).mean()) < self._same_view)):  # fmt: skip
            return [self._report(i) for i in self._cache[1]]  # the same view (give or take physics jitter)
        self._last_thumb = thumb
        named = self._vlm.objects(rgbd.rgb)
        if self._refine:
            named = [self._refined_once(rgbd, item) for item in named]
        origin = rgbd.cam_to_world[:3, 3]
        h, w = rgbd.depth.shape
        seen = []
        for item in named:
            label = normalize(item["label"], self._synonyms, self._vocab)
            if self._aliases:
                from chefbot.explore.aliases import canonical

                label = canonical(label)  # near-synonyms (cup, mug) as one name
            if label in self._fixture_labels:
                continue
            x0, y0, x1, y1 = item["box"]
            if min(x0, y0, x1, y1) < self._edge or max(x0, x1) > w - 1 - self._edge or max(y0, y1) > h - 1 - self._edge:
                continue  # cut off by the image edge: the visible sliver puts it in the wrong place
            point = box_point(rgbd, item["box"])
            if point is None or float(np.linalg.norm(point - origin)) > self._max_range or not self._plausible(point):
                continue
            seen.append(self._track(label, point, float(np.linalg.norm(point - origin)), exclude=seen,
                                    score=float(item.get("score") or 0.0), px=(x1 - x0) * (y1 - y0)))  # fmt: skip
        self._cache = (key, seen)
        out = [self._report(i) for i in seen]
        self._log(rgbd, named, out)
        return out

    def _refined_once(self, rgbd: RGBD, item: dict) -> dict:
        """_refined, once per spot: a small box over a spot already looked at again takes that answer."""
        x0, y0, x1, y1 = item["box"]
        if (x1 - x0) * (y1 - y0) >= self._refine_px:
            return item
        point = box_point(rgbd, item["box"])
        if point is not None:
            for spot, answer in self._refined_spots:
                if float(np.linalg.norm(spot[:2] - point[:2])) <= self._conflict:
                    return {**item, "label": answer["label"], "score": answer.get("score"), "refined_from": item["label"]}
        again = self._refined(rgbd.rgb, item)
        if point is not None:
            self._refined_spots.append((point, again))
        return again

    def _refined(self, rgb: np.ndarray, item: dict) -> dict:
        """A small box (under refine_small_px) looked at again: the detector run on an enlarged crop
        around it, whose answer for the thing in the middle replaces the first (label, box, score)."""
        x0, y0, x1, y1 = item["box"]
        if (x1 - x0) * (y1 - y0) >= self._refine_px:
            return item
        h, w = rgb.shape[:2]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half = max(24.0, max(x1 - x0, y1 - y0) * self._refine_context / 2)
        cx0, cy0 = int(max(0, cx - half)), int(max(0, cy - half))
        cx1, cy1 = int(min(w, cx + half)), int(min(h, cy + half))
        if cx1 - cx0 < 8 or cy1 - cy0 < 8:
            return item
        from PIL import Image

        scale = self._refine_side / max(cx1 - cx0, cy1 - cy0)
        crop = np.asarray(Image.fromarray(rgb[cy0:cy1, cx0:cx1]).resize(
            (round((cx1 - cx0) * scale), round((cy1 - cy0) * scale)), Image.BICUBIC))  # fmt: skip
        self.refined += 1
        again = []
        for other in self._vlm.objects(crop):
            bx0, by0, bx1, by1 = (v / scale for v in other["box"])
            box = [bx0 + cx0, by0 + cy0, bx1 + cx0, by1 + cy0]
            mx, my = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            if x0 <= mx <= x1 and y0 <= my <= y1:  # the same thing: its middle within the first box
                again.append(({**other, "box": box}, math.hypot(mx - cx, my - cy)))
        if not again:
            return item
        best = min(again, key=lambda a: a[1])[0]
        return {**best, "refined_from": item["label"]}

    def _plausible(self, point: np.ndarray) -> bool:
        """On or in a fixture: within a fixture's footprint and at a height things sit at."""
        if not self._height[0] <= float(point[2]) <= self._height[1]:
            return False
        if not self._footprints:
            return True
        for f in self._footprints:
            fx, fy = f.facing
            dx, dy = float(point[0]) - f.location[0], float(point[1]) - f.location[1]
            along, depth = dx * -fy + dy * fx, dx * fx + dy * fy
            if abs(along) <= f.size[0] / 2 + self._margin and abs(depth) <= f.size[1] / 2 + self._margin:
                return True
        return False

    def _track(self, label: str, point: np.ndarray, distance: float, exclude: list[int], score: float = 0.0,
               px: float = 0.0) -> int:  # fmt: skip
        """The object this sighting is of (the nearest of that label within merge_radius), or a new one.
        Its position is the mean of its sightings, so later full views wash out a poor first one."""
        best, best_d = None, self._merge
        if self._aliases:
            from chefbot.explore.aliases import is_group

            if is_group(label):
                best_d = min(best_d, self._alias_merge)
        for i, t in enumerate(self._tracks):
            if t["label"] != label or i in exclude:
                continue
            d = float(np.linalg.norm((t["sum"] / t["n"])[:2] - point[:2]))
            if d <= best_d:
                best, best_d = i, d
        if best is None:
            self._tracks.append({"label": label, "sum": np.zeros(3), "n": 0, "nearest": distance, "score": 0.0, "px": 0.0})
            best = len(self._tracks) - 1
        self._tracks[best]["sum"] += point
        self._tracks[best]["n"] += 1
        self._tracks[best]["nearest"] = min(self._tracks[best]["nearest"], distance)
        self._tracks[best]["score"] += score
        self._tracks[best]["px"] = max(self._tracks[best]["px"], px)  # its largest box so far
        return best

    def superseded(self) -> list[tuple[str, np.ndarray]]:
        """Differently named objects on one spot (within conflict_radius) that are one thing:
        (label, position) of each name that loses. With label_vote, the name seen most often there
        wins (ties: the higher summed score, then the closest view); otherwise a name from a view at least closer_by nearer
        wins (a far-off "cutting board" that up close is a knife block)."""
        if self._vote:
            return self._outvoted()
        out = []
        for t in self._tracks:
            here = t["sum"] / t["n"]
            for o in self._tracks:
                if (o["label"] != t["label"] and t["nearest"] - o["nearest"] >= self._closer
                        and float(np.linalg.norm((o["sum"] / o["n"])[:2] - here[:2])) <= self._conflict):  # fmt: skip
                    out.append((t["label"], here))
                    break
        return out

    def _outvoted(self) -> list[tuple[str, np.ndarray]]:
        spot = [(t["sum"] / t["n"])[:2] for t in self._tracks]
        group = list(range(len(self._tracks)))  # union-find over tracks within conflict_radius

        def root(i: int) -> int:
            while group[i] != i:
                group[i] = group[group[i]]
                i = group[i]
            return i

        for i in range(len(self._tracks)):
            for j in range(i + 1, len(self._tracks)):
                if self._tracks[i]["label"] != self._tracks[j]["label"] and float(np.linalg.norm(spot[i] - spot[j])) <= self._conflict:
                    group[root(i)] = root(j)
        clusters: dict[int, list[int]] = {}
        for i in range(len(self._tracks)):
            clusters.setdefault(root(i), []).append(i)
        out = []
        for members in clusters.values():
            votes: dict[str, list[float]] = {}
            for i in members:
                v = votes.setdefault(self._tracks[i]["label"], [0, 0.0, math.inf])
                v[0] += self._tracks[i]["n"]
                v[1] += self._tracks[i]["score"]
                v[2] = min(v[2], self._tracks[i]["nearest"])
            # Most sightings; then the higher summed score; then the name from the closest view.
            winner = max(votes, key=lambda label: (votes[label][0], votes[label][1], -votes[label][2]))
            out += [(self._tracks[i]["label"], self._tracks[i]["sum"] / self._tracks[i]["n"])
                    for i in members if self._tracks[i]["label"] != winner]  # fmt: skip
        return out

    def _report(self, i: int) -> Detection:
        t = self._tracks[i]
        # approach_small: an object only ever seen small says so (low confidence), so it can be looked at closer.
        small = self._approach and t["px"] < self._approach_px
        return Detection(t["label"], t["sum"] / t["n"], 0.4 if small else 0.9)

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
            "camera": self.camera_name() if self.camera_name else None,
            "zoomed_so_far": self.refined,
            "model_calls_so_far": getattr(self._vlm, "calls", None),
            "cost_usd_so_far": round(float(getattr(self._vlm, "cost_usd", 0.0)), 4),
            "detections": [{"label": d.label, "position": [round(float(v), 3) for v in d.position_world]} for d in out],
        }, indent=1))  # fmt: skip
