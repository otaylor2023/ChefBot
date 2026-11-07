"""The kitchen memory map: objects on the counters with world positions, built from recorded pole-camera frames and
their detections (detect_frames.py), and kept current frame by frame (docs/memory-map-next.md).

Per frame: each detection's mask picks its depth points -> world (pole_geometry) -> an observation if it stands on a
counter top (its lowest points within REST_M of the top, its middle within STACK_M: stacked things count; things on
the wall or flat stickers don't). An object is a 3D cluster: an observation refreshes the nearest entry within MATCH_M
(3D, so a cup on a plate stays two), whatever its label; the entry's label is the score-weighted vote over its views
(open-vocabulary detectors call one object several things). An entry not matched in a frame that looks at its spot is tested by geometry, not
by the detector's silence: the depth ray through the spot
  - stops well short of it          -> something in front (the arm, another object): no evidence
  - hits something standing there   -> still there (the detector missed it): no evidence
  - reaches the bare counter top    -> gone from there: a miss; MISSES_TO_UNKNOWN in a row -> position unknown
  - passes through the top          -> bad depth (the stove's black glass): no evidence
An unknown entry whose label is unique is relocated by a new sighting elsewhere ("moved"); otherwise a new sighting is
a new entry. An entry seen once and then found missing is dropped (a false detection).

    python -m chefbot.perception.memory_map build STREAM/p2_layout3 --dets DETS --config CAL.yaml \\
        --spec maps/src_kitchen/spec.yaml --out objects.json [--init objects_from_before.json] [--labels labels.yaml]

labels.yaml (optional): {rename: {detector label: our label}, ignore: [labels], keep_only: [labels]} for the
prompt-free vocabulary's names.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

from chefbot.perception.pole_geometry import Counter, Pass, load_counters, load_mount, to_world, world_from_camera

STACK_M = 0.35  # an object's points may reach this far above its counter's top
ON_TOP_M = 0.04  # its lowest points may be this far below the top (depth noise)
REST_M = 0.10  # ... and at most this far above it (resting on the top or on something low on it)
MIN_HEIGHT_M = 0.015  # its highest points at least this above the top (not a sticker)
MATCH_M = 0.08  # an observation refreshes the nearest entry this close (3D) that has been called its label before
MATCH_OTHER_M = 0.05  # ... or, called something else, this close (one object, another name for it)
SAME_FRAME_M = 0.05  # detections this close (3D) in one frame are one object (overlapping boxes, several labels)
MISSES_TO_UNKNOWN = 2
EMPTY_TOP_M = 0.025  # a ray's point this close to the counter top (either side): nothing stands there
BEYOND_M = 0.30  # ... and no further than this past the spot (further: depth seen through a glass or dark top)
OCCLUDED_M = 0.06  # a ray this much shorter than the spot's range: something in front
MIN_POINTS = 25
MAX_EXTENT_M = 0.6  # wider than this: furniture, not an object
MAX_MASK_FRAC = 0.12


@dataclass
class Observation:
    label: str
    score: float
    xyz: np.ndarray
    counter: str
    n: int
    t: float
    votes: dict = field(default_factory=dict)  # label -> score, of the detections merged into this one
    extent: list | None = None  # world x/y box of its points [x0, y0, x1, y1] (5th-95th percentile)


@dataclass
class Entry:
    name: str
    label: str
    counter: str
    xyz: list | None  # None: position unknown (it was looked for where it was and not found)
    status: str = "seen"  # seen | unknown
    n_views: int = 0
    misses: int = 0
    score: float = 0.0
    first_seen: float = 0.0
    last_seen: float = 0.0
    last_xyz: list | None = None  # where it was last seen
    history: list = field(default_factory=list)  # recent x/y/z sightings
    events: list = field(default_factory=list)  # (t, what)
    votes: dict = field(default_factory=dict)  # label -> summed score over its views
    id: int = 0  # stable across updates (names follow the voted label and can change)
    lost_at: float | None = None  # when its position became unknown
    sightings: list = field(default_factory=list)  # [pass, frame n, score, label] per view (the last 60)
    extents: list = field(default_factory=list)  # recent views' x/y boxes; `extent` is their median
    extent: list | None = None

    def point(self):
        return np.asarray(self.xyz if self.xyz is not None else self.last_xyz, dtype=float)


def _inside(xyz, ext, pad: float = 0.02) -> bool:
    return ext is not None and ext[0] - pad <= xyz[0] <= ext[2] + pad and ext[1] - pad <= xyz[1] <= ext[3] + pad


def _overlap(a, b) -> float:
    """Intersection over the smaller of two x/y boxes."""
    if a is None or b is None:
        return 0.0
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return w * h / max(small, 1e-6)


def _mask(poly, w: int, h: int):
    """The polygon's pixels, eroded (the rim picks up the background's depth), as (row slice, col slice, bool crop):
    drawn only inside its box, so a frame's masks cost little."""
    from PIL import Image, ImageDraw, ImageFilter

    if len(poly) < 3:
        return None
    pts = np.asarray(poly, dtype=float)
    x0, y0 = (max(0, int(np.floor(v)) - 3) for v in pts.min(axis=0))
    x1, y1 = min(w, int(np.ceil(pts[:, 0].max())) + 4), min(h, int(np.ceil(pts[:, 1].max())) + 4)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    im = Image.new("L", (x1 - x0, y1 - y0), 0)
    ImageDraw.Draw(im).polygon([(x - x0, y - y0) for x, y in pts], fill=255)
    im = im.filter(ImageFilter.MinFilter(5))
    return slice(y0, y1), slice(x0, x1), np.asarray(im) > 0


class MemoryMap:
    def __init__(self, counters: list[Counter], labels: dict | None = None):
        self.counters = counters
        self.entries: list[Entry] = []
        self.labels = labels or {}
        self.log: list[str] = []

    # -- labels ------------------------------------------------------------------------------------------------------

    def our_label(self, label: str) -> str | None:
        label = self.labels.get("rename", {}).get(label, label)
        if label in set(self.labels.get("ignore", [])):
            return None
        keep = self.labels.get("keep_only")
        return label if not keep or label in keep else None

    # -- frame -> observations -----------------------------------------------------------------------------------------

    def counter_of(self, pts: np.ndarray) -> Counter | None:
        """The counter an object's world points stand on: its median x/y inside the box, its lowest points at the top."""
        xy = np.median(pts[:, :2], axis=0)
        for c in self.counters:
            if c.contains(xy[None], margin=0.03)[0]:
                low, high = np.percentile(pts[:, 2], 5), np.percentile(pts[:, 2], 95)
                if (-ON_TOP_M <= low - c.height <= REST_M and np.median(pts[:, 2]) - c.height <= STACK_M
                        and high - c.height >= MIN_HEIGHT_M):
                    return c
        return None

    def observations(self, dets: list[dict], xyz_cam: np.ndarray, T_wc: np.ndarray, n: int, t: float) -> list[Observation]:
        H, W = xyz_cam.shape[:2]
        obs = []
        for d in dets:
            label = self.our_label(d["label"])
            if label is None:
                continue
            m = _mask(d.get("poly") or [], W, H)
            if m is None or m[2].sum() > MAX_MASK_FRAC * W * H:
                continue
            p = xyz_cam[m[0], m[1]][m[2]]
            p = p[~np.isnan(p[:, 2])]
            if len(p) < MIN_POINTS:
                continue
            pw = to_world(T_wc, p)
            # the object is the near cluster of its mask: drop points far behind its median range (background)
            rng = np.linalg.norm(p, axis=1)
            pw = pw[rng < np.median(rng) + 0.10]
            if len(pw) < MIN_POINTS or np.ptp(pw[:, 0]) > MAX_EXTENT_M or np.ptp(pw[:, 1]) > MAX_EXTENT_M:
                continue
            c = self.counter_of(pw)
            if c is None:
                continue
            ext = [round(float(v), 4) for v in (*np.percentile(pw[:, :2], 5, axis=0), *np.percentile(pw[:, :2], 95, axis=0))]
            obs.append(Observation(label, float(d["score"]), np.median(pw, axis=0), c.name, n, t, {label: float(d["score"])}, ext))
        # one object per spot in a frame (overlapping boxes, often with different labels): their labels vote
        obs.sort(key=lambda o: -o.score)
        kept: list[Observation] = []
        for o in obs:
            near = [k for k in kept if np.linalg.norm(k.xyz - o.xyz) <= SAME_FRAME_M]
            if near:
                near[0].votes[o.label] = max(near[0].votes.get(o.label, 0.0), o.score)
            else:
                kept.append(o)
        return kept

    # -- update -------------------------------------------------------------------------------------------------------

    def _new_name(self, label: str) -> str:
        names = {e.name for e in self.entries}
        if label not in names:
            return label
        k = 2
        while f"{label} {k}" in names:
            k += 1
        return f"{label} {k}"

    def _event(self, e: Entry, t: float, what: str):
        n = getattr(self, "_frame", None)
        e.events.append([round(t, 2), n, what])
        self.log.append(f"{t:.1f} n={n} {e.name}: {what}")

    def _see(self, e: Entry, o: Observation):
        e.history = (e.history + [[round(float(v), 4) for v in o.xyz]])[-15:]
        e.xyz = [round(float(v), 4) for v in np.median(np.asarray(e.history), axis=0)]
        e.last_xyz, e.counter, e.status, e.misses = e.xyz, o.counter, "seen", 0
        if o.extent:
            e.extents = (e.extents + [o.extent])[-15:]
            e.extent = [round(float(v), 4) for v in np.median(np.asarray(e.extents), axis=0)]
        e.n_views += 1
        e.sightings = (e.sightings + [[getattr(self, "_pass", ""), o.n, round(o.score, 3), max(o.votes, key=o.votes.get)]])[-60:]
        e.score = max(e.score, o.score)
        e.last_seen = o.t
        for lab, sc in o.votes.items():
            e.votes[lab] = round(e.votes.get(lab, 0.0) + sc, 3)
        e.label = max(e.votes, key=e.votes.get)

    def update(self, dets: list[dict], xyz_cam: np.ndarray, T_wc: np.ndarray, n: int, t: float, camera) -> list[Observation]:
        """One frame: its detections, its depth registered to color (H x W x 3, camera), camera-to-world pose."""
        self._frame = n
        obs = self.observations(dets, xyz_cam, T_wc, n, t)
        matched: set[int] = set()
        used: set[int] = set()
        pairs = []
        for i, o in enumerate(obs):
            for j, e in enumerate(self.entries):
                if e.xyz is not None:
                    d = float(np.linalg.norm(np.asarray(e.xyz) - o.xyz))
                    top = max(e.votes.values(), default=0.0)  # a label it has often been called (>= a quarter of its top)
                    known = any(e.votes.get(lab, 0.0) >= 0.25 * top for lab in o.votes)
                    if d < (MATCH_M if known else MATCH_OTHER_M):
                        pairs.append((d if known else d + MATCH_M, i, j))  # same-label matches first
                    elif known and _inside(o.xyz, e.extent) and abs(o.xyz[2] - e.xyz[2]) < 0.06:
                        pairs.append((d, i, j))  # a long object (a spoon) seen from another side
        for d, i, j in sorted(pairs):
            if i in used or j in matched:
                continue
            used.add(i)
            matched.add(j)
            self._see(self.entries[j], obs[i])
        for i, o in enumerate(obs):
            if i in used:
                continue
            label = max(o.votes, key=o.votes.get)
            same = [e for e in self.entries if e.label == label]
            lost = [e for e in same if e.status == "unknown"]
            if len(same) == 1 and lost:  # the only one of its label, lost from where it was: it moved here
                e = lost[0]
                self._see(e, o)
                e.history = e.history[-1:]
                e.xyz = e.last_xyz = [round(float(v), 4) for v in o.xyz]
                self._event(e, t, f"found again at {o.counter} ({o.xyz[0]:.2f}, {o.xyz[1]:.2f}): moved")
                matched.add(self.entries.index(e))
                continue
            self._next_id = getattr(self, "_next_id", max((x.id for x in self.entries), default=0)) + 1
            e = Entry("", label, o.counter, None, first_seen=t, id=self._next_id)
            self._see(e, o)
            e.name = self._new_name(e.label)
            self.entries.append(e)
            matched.add(len(self.entries) - 1)
            self._event(e, t, f"new at {o.counter} ({o.xyz[0]:.2f}, {o.xyz[1]:.2f})")
        self._look_for_missing(matched, xyz_cam, T_wc, t, camera)
        self._reconcile(t)
        return obs

    def _reconcile(self, t: float):
        """A lost entry and an entry first seen after it was lost, often called the lost one's label (a quarter of its
        votes or more): the same object, moved (the detector called it something else at first). Only when the lost
        one is the only entry of its label and there is one such newcomer."""
        for e in [x for x in self.entries if x.status == "unknown" and x.lost_at is not None]:
            if sum(1 for x in self.entries if x.label == e.label) != 1:
                continue
            new = [x for x in self.entries if x is not e and x.status == "seen" and x.first_seen > e.lost_at
                   and x.votes.get(e.label, 0.0) >= 0.25 * max(x.votes.values())]  # fmt: skip
            if len(new) != 1:
                continue
            x = new[0]
            for lab, sc in x.votes.items():
                e.votes[lab] = round(e.votes.get(lab, 0.0) + sc, 3)
            e.xyz = e.last_xyz = x.xyz
            e.history, e.counter, e.status, e.misses, e.lost_at = x.history, x.counter, "seen", 0, None
            e.n_views += x.n_views
            e.sightings = (e.sightings + x.sightings)[-60:]
            e.extents, e.extent = x.extents, x.extent
            e.last_seen = max(e.last_seen, x.last_seen)
            self.entries.remove(x)
            self._event(e, t, f"found again at {e.counter} ({e.xyz[0]:.2f}, {e.xyz[1]:.2f}): moved (seen as {x.label} at first)")

    def _look_for_missing(self, matched: set[int], xyz_cam, T_wc, t: float, camera):
        T_cw = np.linalg.inv(T_wc)
        H, W = xyz_cam.shape[:2]
        drop = []
        for j, e in enumerate(self.entries):
            if j in matched or e.xyz is None:
                continue
            pc = to_world(T_cw, np.asarray(e.xyz, float)[None])[0]
            if not 0.3 < pc[2] < 3.0:
                continue
            u, v = camera.color.project(pc[None])[0]
            if not (20 <= u < W - 20 and 20 <= v < H - 20):
                continue
            win = xyz_cam[int(v) - 3 : int(v) + 4, int(u) - 3 : int(u) + 4].reshape(-1, 3)
            win = win[~np.isnan(win[:, 2])]
            if len(win) < 10:
                continue
            hit = np.median(win, axis=0)
            if np.linalg.norm(hit) < np.linalg.norm(pc) - OCCLUDED_M:
                continue  # something in front of the spot
            top = next(c.height for c in self.counters if c.name == e.counter)
            dz = to_world(T_wc, hit[None])[0][2] - top
            if dz > EMPTY_TOP_M:
                continue  # something stands there (the detector missed it, or something else sits there now)
            if dz < -EMPTY_TOP_M or np.linalg.norm(hit) > np.linalg.norm(pc) + BEYOND_M:
                continue  # the ray went through the top: bad depth (black glass, shine), no evidence either way
            e.misses += 1
            if e.n_views <= 1:  # seen once, then not there: a false detection
                drop.append(j)
                self.log.append(f"{t:.1f} n={self._frame} {e.name}: dropped (seen once, its spot is empty)")
            elif e.misses >= MISSES_TO_UNKNOWN and e.status == "seen":
                e.status, e.xyz, e.lost_at = "unknown", None, t
                self._event(e, t, f"not at its spot on {e.counter} ({e.last_xyz[0]:.2f}, {e.last_xyz[1]:.2f}): position unknown")
        for j in reversed(drop):
            del self.entries[j]

    # -- files --------------------------------------------------------------------------------------------------------

    def merge_overlaps(self, min_overlap: float = 0.5) -> int:
        """Entries that are one object split in two (a long thing seen end-on, then side-on): seen, same counter, boxes
        overlapping by at least min_overlap of the smaller, heights within 6 cm, and a label in common (a quarter of
        either's votes). The one with more views keeps the other's views and votes. Returns how many merged."""
        merged = 0
        again = True
        while again:
            again = False
            seen = [e for e in self.entries if e.status == "seen" and e.xyz is not None]
            for a in seen:
                for b in seen:
                    if a is b or a.counter != b.counter or abs(a.xyz[2] - b.xyz[2]) > 0.06:
                        continue
                    if _overlap(a.extent, b.extent) < min_overlap:
                        continue
                    ta, tb = max(a.votes.values()), max(b.votes.values())
                    if not any(a.votes.get(l, 0) >= 0.25 * ta and b.votes.get(l, 0) >= 0.25 * tb for l in set(a.votes) | set(b.votes)):
                        continue
                    keep, drop = (a, b) if a.n_views >= b.n_views else (b, a)
                    for lab, sc in drop.votes.items():
                        keep.votes[lab] = round(keep.votes.get(lab, 0.0) + sc, 3)
                    keep.label = max(keep.votes, key=keep.votes.get)
                    keep.n_views += drop.n_views
                    keep.sightings = (keep.sightings + drop.sightings)[-60:]
                    keep.history = (keep.history + drop.history)[-15:]
                    keep.xyz = keep.last_xyz = [round(float(v), 4) for v in np.median(np.asarray(keep.history), axis=0)]
                    keep.extents = (keep.extents + drop.extents)[-15:]
                    keep.extent = [round(float(v), 4) for v in np.median(np.asarray(keep.extents), axis=0)] if keep.extents else keep.extent
                    self.entries.remove(drop)
                    self.log.append(f"{drop.name} merged into {keep.name} (one object, overlapping)")
                    merged += 1
                    again = True
                    break
                if again:
                    break
        return merged

    def rename(self) -> None:
        """Names from the voted labels: the label, then "label 2", "label 3" ... in order of first sighting."""
        count: dict = {}
        for e in sorted(self.entries, key=lambda e: (e.first_seen, e.id)):
            count[e.label] = count.get(e.label, 0) + 1
            e.name = e.label if count[e.label] == 1 else f"{e.label} {count[e.label]}"

    def to_json(self, meta: dict) -> dict:
        self.rename()
        fixture = {c.name: c.fixture for c in self.counters}
        objs = []
        for e in sorted(self.entries, key=lambda e: (e.label, e.name)):
            d = asdict(e)
            d["fixture"] = fixture.get(e.counter, "")
            objs.append(d)
        return {"frame": "world", **meta, "objects": objs}

    @classmethod
    def from_json(cls, d: dict, counters, labels=None) -> "MemoryMap":
        m = cls(counters, labels)
        keys = set(Entry.__dataclass_fields__)
        m.entries = [Entry(**{k: v for k, v in o.items() if k in keys}) for o in d["objects"]]
        return m


def build(passes, dets_root, config, spec, init=None, labels=None, every: int = 1) -> MemoryMap:
    counters = [c for c in load_counters(spec) if c.height < 1.2]
    T_mount = load_mount(config)
    mm = MemoryMap.from_json(json.loads(Path(init).read_text()), counters, labels) if init else MemoryMap(counters, labels)
    for root in passes:
        p = Pass(root)
        mm._pass = Path(root).name
        det_file = Path(dets_root) / Path(root).name / "pole" / "detections.jsonl"
        dets = {d["n"]: d["dets"] for d in map(json.loads, det_file.read_text().splitlines())}
        for k, f in enumerate(p.frames):
            if k % every or f["n"] not in dets:
                continue
            pose = p.pose(f)
            if pose is None:
                continue
            xyz = p.camera.aligned_xyz(p.depth(f))
            mm.update(dets[f["n"]], xyz, world_from_camera(pose, T_mount), f["n"], float(f["wall"]), p.camera)
        mm.log.append(f"-- end of {Path(root).name}")
    # entries seen in a single frame only are too weak to keep
    mm.entries = [e for e in mm.entries if e.n_views >= 2 or e.status == "unknown"]
    mm.merge_overlaps()
    mm.rename()
    return mm


def probe(root, n: int, xyz, config, out_png=None) -> dict:
    """Why an entry was or wasn't seen: where world point xyz falls in pass `root`'s frame n, and what the depth ray
    there hits (world). With out_png, the frame with the spot circled."""
    p = Pass(root)
    f = next(f for f in p.frames if f["n"] == n)
    T_wc = world_from_camera(p.pose(f), load_mount(config))
    pc = to_world(np.linalg.inv(T_wc), np.asarray(xyz, float)[None])[0]
    u, v = p.camera.color.project(pc[None])[0]
    xyz_cam = p.camera.aligned_xyz(p.depth(f))
    H, W = xyz_cam.shape[:2]
    rep = {"pixel": [round(float(u)), round(float(v))], "range_m": round(float(np.linalg.norm(pc)), 3), "in_view": bool(0 <= u < W and 0 <= v < H and pc[2] > 0)}
    if rep["in_view"]:
        win = xyz_cam[max(0, int(v) - 3) : int(v) + 4, max(0, int(u) - 3) : int(u) + 4].reshape(-1, 3)
        win = win[~np.isnan(win[:, 2])]
        if len(win):
            hit = np.median(win, axis=0)
            rep |= {"hit_range_m": round(float(np.linalg.norm(hit)), 3), "hit_world": [round(float(x), 3) for x in to_world(T_wc, hit[None])[0]]}
    if out_png:
        from PIL import Image, ImageDraw

        im = Image.fromarray(p.color(f))
        d = ImageDraw.Draw(im)
        d.ellipse([u - 25, v - 25, u + 25, v + 25], outline=(255, 0, 255), width=4)
        im.resize((im.width // 2, im.height // 2)).save(out_png)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("passes", nargs="+")
    b.add_argument("--dets", required=True, help="detect_frames.py's --out")
    b.add_argument("--config", required=True)
    b.add_argument("--spec", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--init", help="a map to start from (then: what changed)")
    b.add_argument("--labels", help="labels.yaml: rename / ignore / keep_only")
    b.add_argument("--every", type=int, default=1)
    q = sub.add_parser("probe", help="where a world point falls in a frame, and what the depth there hits")
    q.add_argument("pass_dir")
    q.add_argument("n", type=int)
    q.add_argument("xyz", type=float, nargs=3)
    q.add_argument("--config", required=True)
    q.add_argument("--png")
    a = ap.parse_args(argv)
    if a.cmd == "probe":
        print(json.dumps(probe(a.pass_dir, a.n, a.xyz, a.config, a.png)))
        return 0
    labels = yaml.safe_load(Path(a.labels).read_text()) if a.labels else None
    mm = build(a.passes, a.dets, a.config, a.spec, a.init, labels, a.every)
    meta = {"built_from": [Path(p).name for p in a.passes], "init": a.init, "detections": str(a.dets)}
    Path(a.out).write_text(json.dumps(mm.to_json(meta), indent=1))
    print("\n".join(mm.log))
    print(f"\n{len(mm.entries)} objects -> {a.out}")
    for e in sorted(mm.entries, key=lambda e: (e.counter, e.label)):
        where = f"({e.xyz[0]:.2f}, {e.xyz[1]:.2f}, {e.xyz[2]:.2f})" if e.xyz else "unknown"
        print(f"  {e.name:24s} {e.counter:32s} {where:24s} views {e.n_views:3d}  {e.status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
