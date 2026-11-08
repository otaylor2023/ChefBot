"""The kitchen's object map, live on the laptop: frames from the robot's cameras (env.get_camera_rgbd) with their
world poses, a detector's masks (the vision server's YOLOE), into memory_map.MemoryMap (3D clusters on the counter
tops, label votes, "gone" when the depth reaches the bare top), named once per object by a VLM (name_objects.ask)
from the best views kept as they came. Pure: no robot, no model in this process.

    km = KitchenMap(maps / "src_kitchen_work" / "spec.yaml")
    km.add_frame(rgb, depth, K, T_wc, dets, t=time.time(), camera="pole")     # T_wc: camera (OpenCV) to world
    km.name_new(model="gemini-3.5-flash-lite")                                # each new object once
    km.objects()                    # [{name, label, xyz, counter, fixture, status, n_views, ...}], as objects.json
    km.resolve("tomato")            # the one entry called that (goto.resolve's rules)

Positions are in the map's world frame (Motive's, z up). An object needs `min_views` sightings before it is offered
(one frame's false detection never is); an entry whose spot was seen bare twice has no position ("unknown").
"""

from __future__ import annotations

import io
import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chefbot.perception.memory_map import MemoryMap
from chefbot.perception.pole_geometry import Intrinsics, load_counters, to_world

VIEW_PX = 256  # a kept view: this square around the object's point
KEEP_VIEWS = 6  # per object, the best by score / distance
MIN_VIEWS = 2


@dataclass
class _Cam:
    """What MemoryMap's gone-test needs of a camera: `.color.project` (undistorted: get_camera_rgbd's color)."""

    color: Intrinsics


@dataclass
class Layout:
    """A test layout's closed vocabulary (perception/layouts/*.yaml): what the detector looks for, how many of each."""

    items: dict  # item -> {count, prompts, sizes?}

    @classmethod
    def load(cls, path) -> "Layout":
        import yaml

        raw = yaml.safe_load(Path(path).read_text())["items"]
        return cls({k: {"count": int((v or {}).get("count", 1)), "prompts": list((v or {}).get("prompts") or [k]),
                        "sizes": list((v or {}).get("sizes") or [])} for k, v in raw.items()})  # fmt: skip

    def queries(self) -> list[str]:
        return list(dict.fromkeys(q for v in self.items.values() for q in v["prompts"]))

    def labels(self) -> dict:
        """MemoryMap's labels: each prompt counted as its item; anything else ignored."""
        return {"rename": {q: k for k, v in self.items.items() for q in v["prompts"]}, "keep_only": list(self.items)}

    def names(self) -> list[str]:
        """The names the VLM may give: the items, each size name in place of its item."""
        out = []
        for k, v in self.items.items():
            out += list(dict.fromkeys(v["sizes"])) or [k]
        return out

    def item_of(self, name: str) -> str | None:
        for k, v in self.items.items():
            if name == k or name in v["sizes"]:
                return k
        return None


def xyz_image(depth: np.ndarray, K: np.ndarray, near: float = 0.1, far: float = 3.0) -> np.ndarray:
    """Depth (H x W, metres, registered to the color image) -> camera points H x W x 3, NaN where there is none."""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    z = np.asarray(depth, dtype=np.float32).copy()
    z[(z < near) | (z > far) | ~np.isfinite(z)] = np.nan
    return np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=-1)


def _on_the_robot(base_pose, margin: float = 0.12):
    """A test for world points (N x 3): their median over the base's plate (+ margin), i.e. the robot itself (its pole,
    its bins, the arm over it), seen by its own camera."""
    from chefbot.perception.goto import PLATE_X, PLATE_Y

    x, y, yaw = (float(v) for v in ((base_pose.x, base_pose.y, base_pose.yaw) if hasattr(base_pose, "yaw") else base_pose[:3]))
    c, s = math.cos(yaw), math.sin(yaw)

    def reject(pw) -> bool:
        m = np.median(pw[:, :2], axis=0) - (x, y)
        a, b = c * m[0] + s * m[1], -s * m[0] + c * m[1]
        return PLATE_X[0] - margin <= a <= PLATE_X[1] + margin and PLATE_Y[0] - margin <= b <= PLATE_Y[1] + margin

    return reject


class KitchenMap:
    def __init__(self, spec_path, labels: dict | None = None, min_views: int = MIN_VIEWS, log=print,
                 layout: "Layout | None" = None):  # fmt: skip
        """`layout`: a closed vocabulary (the detector's labels mapped to its items, at most `count` of each offered,
        the VLM naming from its names only)."""
        self.spec_path = Path(spec_path)
        self.counters = [c for c in load_counters(self.spec_path) if c.height < 1.2]  # counter tops, not the fridge
        self.layout = layout
        self.mm = MemoryMap(self.counters, layout.labels() if layout else labels)
        self.min_views = int(min_views)
        self.views: dict[int, list] = {}  # entry id -> [(quality, jpeg bytes, distance m)], best first
        self.vlm: dict[int, dict] = {}  # entry id -> the VLM's answer
        self.n = 0
        self.lock = threading.RLock()  # frames come from the drive's capture thread and the stops
        self.log = log

    # -- frames -------------------------------------------------------------------------------------------------------

    def add_frame(self, rgb, depth, K, T_wc, dets, t: float | None = None, camera: str = "pole", xyz=None,
                  base_pose=None) -> dict:  # fmt: skip
        """One frame into the map: depth registered to the color image (metres), or its points already (`xyz`,
        H x W x 3 in the color camera, NaN where none: a recording's PoleCamera.aligned_xyz). Returns {n,
        observations, new, entries, ms}."""
        t0 = time.time()
        t = time.time() if t is None else float(t)
        K = np.asarray(K, dtype=float)
        xyz = xyz_image(np.asarray(depth, dtype=float), K) if xyz is None else np.asarray(xyz)
        H, W = xyz.shape[:2]
        cam = _Cam(Intrinsics(W, H, K, np.zeros(5)))
        with self.lock:
            self.n += 1
            before = {e.id for e in self.mm.entries}
            self.mm._pass = camera
            self.mm.reject = None if base_pose is None else _on_the_robot(base_pose)
            obs = self.mm.update(dets, xyz, np.asarray(T_wc, dtype=float), self.n, t, cam)
            seen = [e for e in self.mm.entries if e.last_seen == t and e.xyz is not None]
            for e in seen:
                self._keep_view(e, rgb, K, np.asarray(T_wc, dtype=float))
            new = [e.name for e in self.mm.entries if e.id not in before]
        return {"n": self.n, "observations": len(obs), "new": new, "entries": len(self.mm.entries),
                "ms": round((time.time() - t0) * 1000, 1)}  # fmt: skip

    def _keep_view(self, e, rgb, K, T_wc) -> None:
        """A crop around the entry's point in this frame, kept if it is among its best (the naming's views)."""
        from PIL import Image, ImageDraw

        pc = to_world(np.linalg.inv(T_wc), np.asarray(e.xyz, float)[None])[0]
        if pc[2] <= 0.1:
            return
        u, v = K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2]
        H, W = rgb.shape[:2]
        if not (0 <= u < W and 0 <= v < H):
            return
        half = int(np.clip(0.30 * K[0, 0] / pc[2], 48, 220))  # ~60 cm of counter around it, whatever the distance
        x0, y0 = int(u) - half, int(v) - half
        im = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).crop((x0, y0, x0 + 2 * half, y0 + 2 * half))
        im = im.resize((VIEW_PX, VIEW_PX))
        r = 0.18 * VIEW_PX
        ImageDraw.Draw(im).ellipse((VIEW_PX / 2 - r, VIEW_PX / 2 - r, VIEW_PX / 2 + r, VIEW_PX / 2 + r), outline=(255, 0, 255), width=3)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        score = float(e.sightings[-1][2]) if e.sightings else 0.0
        q = score / max(float(pc[2]), 0.3)
        kept = self.views.setdefault(e.id, [])
        kept.append((q, buf.getvalue(), round(float(pc[2]), 2)))
        kept.sort(key=lambda k: -k[0])
        del kept[KEEP_VIEWS:]

    def tidy(self) -> int:
        """After a sweep: one object split in two (a long thing seen end-on, then side-on) made one again
        (MemoryMap.merge_overlaps); the merged-away entry's views and name go with it. Returns how many merged."""
        with self.lock:
            before = {e.id for e in self.mm.entries}
            n = self.mm.merge_overlaps()
            for i in before - {e.id for e in self.mm.entries}:
                self.views.pop(i, None)
                self.vlm.pop(i, None)
        return n

    # -- names --------------------------------------------------------------------------------------------------------

    def strip(self, entry_id: int, k: int = 4):
        """Its best k views side by side (PIL image), or None."""
        from PIL import Image

        kept = self.views.get(entry_id) or []
        if not kept:
            return None
        ims = [Image.open(io.BytesIO(b)) for _, b, _ in kept[:k]]
        out = Image.new("RGB", (VIEW_PX * len(ims), VIEW_PX))
        for i, im in enumerate(ims):
            out.paste(im, (VIEW_PX * i, 0))
        return out

    def unnamed(self) -> list[int]:
        with self.lock:
            return [e.id for e in self.mm.entries if e.n_views >= self.min_views and e.id not in self.vlm and self.views.get(e.id)]

    def name_new(self, model: str = "gemini-3.5-flash-lite", hint: list[str] | None = None, k: int = 4, workers: int = 4,
                 url: str = "http://127.0.0.1:11434", key: str | None = None, ask=None) -> dict:  # fmt: skip
        """Each offered entry not named yet, once, by the VLM (several at once for an API model). Returns
        {named, seconds, answers: {id: answer}}. `ask` (tests): replaces name_objects.ask."""
        from concurrent.futures import ThreadPoolExecutor

        if ask is None:
            from chefbot.perception.name_objects import ask
        t0 = time.time()
        ids = self.unnamed()
        jobs = [(i, self.strip(i, k)) for i in ids]
        jobs = [(i, im) for i, im in jobs if im is not None]

        def one(job):
            i, im = job
            n = im.size[0] // VIEW_PX
            if self.layout is not None:
                return i, ask(im, n, model, self.layout.names(), url, key=key, menu=True)
            return i, ask(im, n, model, hint, url, key=key)

        answers = {}
        with ThreadPoolExecutor(max(1, workers if model.startswith("gemini") else 1)) as pool:
            for i, ans in pool.map(one, jobs):
                answers[i] = ans
                with self.lock:
                    self.vlm[i] = ans
        return {"named": len(answers), "seconds": round(time.time() - t0, 2), "model": model, "answers": answers}

    # -- what it holds ------------------------------------------------------------------------------------------------

    def objects(self, include_tentative: bool = False) -> list[dict]:
        """The entries as objects.json's `objects`: the VLM's name when it gave one (the detector's as det_label),
        numbered per label ("tomato", "tomato 2" in order of first sighting); entries the VLM called not movable are
        status "rejected" and never offered."""
        with self.lock:
            d = self.mm.to_json({})["objects"]
        out = []
        for o in d:
            if o["n_views"] < self.min_views and not include_tentative:
                continue
            o["det_label"] = o["label"]
            ans = self.vlm.get(o["id"])
            if ans:
                o["vlm"] = {k: v for k, v in ans.items() if k in ("label", "movable_object", "confidence", "model", "ms", "error")}
                if ans.get("label"):
                    o["label"] = ans["label"]
                if not ans.get("movable_object", True):
                    o["status"] = "rejected"
            o["tentative"] = o["n_views"] < self.min_views
            for k in ("history", "sightings", "extents"):
                o.pop(k, None)
            out.append(o)
        if self.layout is not None:
            self._apply_layout(out)
        count: dict = {}
        for o in sorted(out, key=lambda o: (o["first_seen"], o["id"])):
            count[o["label"]] = count.get(o["label"], 0) + 1
            o["name"] = o["label"] if count[o["label"]] == 1 else f"{o['label']} {count[o['label']]}"
        return sorted(out, key=lambda o: (o["label"], o["name"]))

    def _apply_layout(self, objs: list[dict]) -> None:
        """At most `count` of each item offered (the most seen; the rest status "extra"); "other" from the VLM ->
        rejected; an item with sizes named by footprint, largest first (a VLM size name stands)."""
        L = self.layout
        for o in objs:
            if o["label"] == "other":
                o["status"] = "rejected"
            item = L.item_of(o["label"])
            if item is None and o["status"] != "rejected":
                o["status"] = "rejected"  # the VLM named something not on the list
            o["item"] = item
        for item, spec in L.items.items():
            mine = [o for o in objs if o["item"] == item and o["status"] == "seen" and not o["tentative"]]
            mine.sort(key=lambda o: -o["n_views"])
            for o in mine[spec["count"]:]:
                o["status"] = "extra"
            kept = mine[: spec["count"]]
            if spec["sizes"]:
                def area(o):
                    e = o.get("extent")
                    return 0.0 if not e else (e[2] - e[0]) * (e[3] - e[1])
                from collections import Counter as _C

                vlm_sizes = _C(o["label"] for o in kept)
                fits = all(n <= spec["sizes"].count(lab) for lab, n in vlm_sizes.items() if lab in spec["sizes"])
                if not fits or any(o["label"] not in spec["sizes"] for o in kept):  # the VLM's sizes don't fit the list: by footprint
                    for o, size in zip(sorted(kept, key=lambda o: -area(o)), spec["sizes"]):
                        o["label"] = size

    def resolve(self, name: str) -> dict:
        """The entry called `name` ("tomato", "tomato 2", "tomato_2"); LookupError if none, several or no position."""
        from chefbot.perception.goto import resolve

        objs = [o for o in self.objects() if o["status"] not in ("rejected", "extra")]
        return resolve(objs, name.replace("_", " ").strip().lower())

    def to_json(self, meta: dict | None = None) -> dict:
        return {"frame": "world", "spec": str(self.spec_path), "frames": self.n, "saved": time.time(), **(meta or {}),
                "objects": self.objects()}  # fmt: skip

    def save(self, path) -> Path:
        """objects.json (atomic), with the full MemoryMap beside it (objects_memory.json) so a later run can go on."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        for p, d in ((path, self.to_json()), (path.with_name(path.stem + "_memory.json"), self._memory())):
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(d, indent=1, default=float))
            tmp.replace(p)
        return path

    def _memory(self) -> dict:
        import base64

        with self.lock:
            d = self.mm.to_json({"frames": self.n})
            d["views"] = {str(i): [[q, base64.b64encode(b).decode(), dist] for q, b, dist in v] for i, v in self.views.items()}
        d["vlm"] = {str(k): v for k, v in self.vlm.items()}
        return d

    @classmethod
    def load(cls, spec_path, memory_path, **kw) -> "KitchenMap":
        """A map saved by save() (its _memory.json): the objects as they were, to go on updating."""
        km = cls(spec_path, **kw)
        d = json.loads(Path(memory_path).read_text())
        km.mm = MemoryMap.from_json(d, km.counters, km.mm.labels)
        import base64

        km.vlm = {int(k): v for k, v in (d.get("vlm") or {}).items()}
        km.views = {int(i): [(q, base64.b64decode(b), dist) for q, b, dist in v] for i, v in (d.get("views") or {}).items()}
        km.n = int(d.get("frames", 0))
        return km


def counter_box(spec_path, name: str) -> tuple:
    """(centre, size, yaw rad, name) of the spec's box obstacle `name`."""
    import yaml

    for o in yaml.safe_load(Path(spec_path).read_text()).get("obstacles") or []:
        if "box" in o and o["name"] == name:
            b = o["box"]
            return tuple(b["center"]), tuple(b["size"]), math.radians(b.get("yaw_deg", 0.0)), o["name"]
    raise LookupError(f"no box obstacle {name!r} in {spec_path}")
