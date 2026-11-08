"""Name the memory map's objects with a vision-language model: once per object, not per frame. The detector (YOLOE)
finds objects and the map places them; its labels are poor (a small model forced to pick from a list). Here each
entry's best few views (detection score x frame sharpness / distance, spread over time) are cropped around the map's
point, ringed, put side by side and shown to a VLM served by ollama, which names the object freely (a list of likely
kitchen things is a hint, not a menu) and says whether it is a movable object at all.

    python -m chefbot.perception.name_objects MAP.json STREAM/p1_layout2 [STREAM/...] --config CAL.yaml \\
        --out MAP_named.json [--model qwen2.5vl:7b] [--hint chefbot/perception/vocab_kitchen.txt] [--views 4]

Each entry gets `det_label` (the detector's vote winner), `vlm` ({label, movable_object, confidence, model, ms, views})
and `label` = the VLM's label when it answered (else the detector's). Entries the VLM calls not movable are kept
with status "rejected" (not offered to goto). Entries that overlap and get the same name are merged.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from chefbot.perception.memory_map import MemoryMap, _overlap
from chefbot.perception.pole_geometry import Pass, load_mount, to_world, world_from_camera

PROMPT = (
    "These are {k} views of ONE object on a kitchen counter, taken by a robot's camera as it drove past; a magenta ring "
    "marks the object in each view. Name that object in 1-4 words, as specific as the views allow. Name the whole object the ring is on, the "
    "thing a robot would pick up, not a part of it, a logo or a sticker on it (a laptop, not its keyboard or logo; a game controller, not its "
    "d-pad). {hint}"
    "If the ring is on part of the counter, the stove, the sink, the wall, a sticker or a person rather than a movable "
    "object, set movable_object to false. "
    'Reply only with JSON: {{"label": string, "movable_object": true or false, "confidence": number from 0 to 1}}'
)


def _view_crop(p: Pass, frame: dict, xyz, T_mount, size: int = 256):
    """The color frame cropped around world point xyz (about 0.4 m across at the object), ringed; None if not in view."""
    from PIL import Image, ImageDraw

    pose = p.pose(frame)
    if pose is None:
        return None, None
    T_wc = world_from_camera(pose, T_mount)
    pc = to_world(np.linalg.inv(T_wc), np.asarray(xyz, float)[None])[0]
    if pc[2] <= 0.2:
        return None, None
    u, v = p.camera.color.project(pc[None])[0]
    W, H = p.camera.color.width, p.camera.color.height
    if not (0 <= u < W and 0 <= v < H):
        return None, None
    half = int(np.clip(0.2 * p.camera.color.K[0, 0] / pc[2], 64, 300))
    x0, y0 = int(np.clip(u - half, 0, W - 2 * half)), int(np.clip(v - half, 0, H - 2 * half))
    im = Image.fromarray(p.color(frame)).crop((x0, y0, x0 + 2 * half, y0 + 2 * half)).resize((size, size))
    s = size / (2 * half)
    cx, cy, r = (u - x0) * s, (v - y0) * s, 0.09 * size
    ImageDraw.Draw(im).ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 0, 200), width=3)
    return im, float(pc[2])


def pick_views(entry: dict, passes: dict, k: int = 4) -> list[tuple[Pass, dict, float]]:
    """The entry's sightings ranked by detection score x sharpness / distance, then spread: no two from frames < 3 apart."""
    ranked = []
    for pname, n, score, _label in entry.get("sightings", []):
        p = passes.get(pname)
        if p is None:
            continue
        f = p._by_n.get(n)
        if f is None:
            continue
        ranked.append((float(score) * float(f.get("blur", 300.0)) ** 0.5, p, f))
    ranked.sort(key=lambda r: -r[0])
    out = []
    for q, p, f in ranked:
        if all(o[0] is not p or abs(o[1]["n"] - f["n"]) >= 3 for o in out):
            out.append((p, f, q))
        if len(out) == k:
            break
    return out


def strip(entry: dict, passes: dict, T_mount, k: int = 4):
    from PIL import Image

    xyz = entry["xyz"] or entry.get("last_xyz")
    ims, used = [], []
    for p, f, _q in pick_views(entry, passes, k + 2):
        im, dist = _view_crop(p, f, xyz, T_mount)
        if im is not None:
            ims.append(im)
            used.append([p.root.name, f["n"], round(dist, 2)])
        if len(ims) == k:
            break
    if not ims:
        return None, used
    out = Image.new("RGB", (256 * len(ims), 256))
    for i, im in enumerate(ims):
        out.paste(im, (256 * i, 0))
    return out, used


GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_SCHEMA = {"type": "OBJECT", "properties": {"label": {"type": "STRING"}, "movable_object": {"type": "BOOLEAN"},
                 "confidence": {"type": "NUMBER"}}, "required": ["label", "movable_object", "confidence"]}  # fmt: skip


def gemini_key(env_file: str | None = None) -> str:
    """GEMINI_API_KEY from the environment, else from a .env file (KEY=value lines). Never logged."""
    import os

    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    for f in [env_file] if env_file else []:
        for line in Path(f).read_text().splitlines():
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("no GEMINI_API_KEY (set it, or pass --env-file)")


def _prompt(k: int, hint, menu: bool = False) -> str:
    if menu and hint:  # a closed list: one of these or "other"
        h = (f"The object is one of these: {', '.join(hint)}. Answer with exactly one of those names, or \"other\" if it is "
             "none of them. ")
    else:
        h = f"Common things in this kitchen include: {', '.join(hint)}; use one of those names if it fits, otherwise your own. " if hint else ""
    return PROMPT.format(k=k, hint=h)


def ask_gemini(jpeg: bytes, k: int, model: str, hint, key: str, retries: int = 8, menu: bool = False) -> dict:
    body = {"contents": [{"parts": [{"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(jpeg).decode()}},
                                    {"text": _prompt(k, hint, menu)}]}],
            "generationConfig": {"responseMimeType": "application/json", "responseSchema": GEMINI_SCHEMA}}  # fmt: skip
    last = None
    for attempt in range(retries):
        t0 = time.time()
        try:
            req = urllib.request.Request(GEMINI_URL.format(model=model), json.dumps(body).encode(),
                                         {"Content-Type": "application/json", "x-goog-api-key": key})  # fmt: skip
            r = json.loads(urllib.request.urlopen(req, timeout=120).read())
            parts = r["candidates"][0]["content"]["parts"]
            txt = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            ans = json.loads(txt)
            u = r.get("usageMetadata", {})
            return {"label": str(ans.get("label", "")).strip().lower(), "movable_object": bool(ans.get("movable_object", True)),
                    "confidence": float(ans.get("confidence", 0) or 0), "model": model, "ms": round((time.time() - t0) * 1000),
                    "tokens": {"in": u.get("promptTokenCount", 0), "out": u.get("candidatesTokenCount", 0),
                               "thinking": u.get("thoughtsTokenCount", 0)}}  # fmt: skip
        except urllib.error.HTTPError as exc:
            text = exc.read().decode(errors="replace")
            last = f"HTTP {exc.code}: {text[:200]}"
            if exc.code not in (429, 500, 503):
                break
            m = re.search(r'"retryDelay":\s*"(\d+)', text)  # Google says how long to wait on a rate limit
            time.sleep(min(90, int(m.group(1)) + 1) if m else min(60, 5 * 2**attempt))
            continue
        except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError) as exc:
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(3 * (attempt + 1))
    return {"label": "", "error": last, "model": model}


def ask(image, k: int, model: str, hint: list[str] | None, url: str, retries: int = 3, key: str | None = None,
        menu: bool = False) -> dict:  # fmt: skip
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=88)
    if model.startswith("gemini"):
        return ask_gemini(buf.getvalue(), k, model, hint, key or gemini_key(), menu=menu)
    h = f"Common things in this kitchen include: {', '.join(hint)}; use one of those names if it fits, otherwise your own. " if hint else ""
    body = {"model": model, "prompt": _prompt(k, hint, menu), "images": [base64.b64encode(buf.getvalue()).decode()],
            "stream": False, "format": "json", "options": {"temperature": 0}}  # fmt: skip
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url + "/api/generate", json.dumps(body).encode(), {"Content-Type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=300).read())
            txt = r["response"]
            try:
                ans = json.loads(txt)
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", txt, re.S)
                ans = json.loads(m.group(0)) if m else {"label": txt.strip()}
            return {"label": str(ans.get("label", "")).strip().lower(), "movable_object": bool(ans.get("movable_object", True)),
                    "confidence": float(ans.get("confidence", 0) or 0), "model": model,
                    "ms": round(r.get("total_duration", 0) / 1e6), "load_ms": round(r.get("load_duration", 0) / 1e6)}  # fmt: skip
        except (urllib.error.URLError, OSError, ValueError) as exc:  # the server can drop a request while it (re)loads
            last = exc
            time.sleep(5 * (attempt + 1))
    return {"label": "", "error": f"{type(last).__name__}: {last}", "model": model}


def name_map(m: dict, pass_dirs, config, model="gemini-3.5-flash-lite", hint=None, url="http://127.0.0.1:11434", k=4,
             save_strips: Path | None = None, log=print, strips_only: bool = False, key: str | None = None,
             workers: int = 4) -> dict:  # fmt: skip
    passes = {}
    for d in pass_dirs:
        p = Pass(d)
        p._by_n = {f["n"]: f for f in p.frames}
        passes[p.root.name] = p
    T_mount = load_mount(config)
    t_all = time.time()
    jobs = []
    for o in m["objects"]:  # cut every object's views first (local, fast) ...
        o.setdefault("det_label", o["label"])
        t0 = time.time()
        im, used = strip(o, passes, T_mount, k)
        if im is None:
            o["vlm"] = {"label": "", "error": "no view", "views": used}
            continue
        if save_strips:
            save_strips.mkdir(parents=True, exist_ok=True)
            im.save(save_strips / f"o{o['id']}.jpg", quality=85)
        if strips_only:
            o["views_used"] = used
            continue
        jobs.append((o, im, used, round((time.time() - t0) * 1000)))
    from concurrent.futures import ThreadPoolExecutor

    def one(job):  # ... then ask about them, several at once for an API model
        o, im, used, crop_ms = job
        ans = ask(im, len(used), model, hint, url, key=key)
        ans["views"], ans["crop_ms"] = used, crop_ms
        return o, ans

    with ThreadPoolExecutor(workers if model.startswith("gemini") else 1) as pool:
        for o, ans in pool.map(one, jobs):
            o["vlm"] = ans
            if ans.get("label"):
                o["label"] = ans["label"]
            if not ans.get("movable_object", True):
                o["status"] = "rejected"
            log(f"  {o['name']:22s} det {o['det_label']:16s} -> {ans.get('label') or ans.get('error')!s:26s} "
                f"movable {ans.get('movable_object')} {ans.get('ms')} ms")  # fmt: skip
    m["naming"] = {"model": model, "views": k, "hint": bool(hint), "seconds": round(time.time() - t_all, 1)}
    return m


def merge_same_named(m: dict, counters) -> dict:
    """Overlapping entries the VLM gave one name: one object."""
    mm = MemoryMap.from_json(m, counters)
    for e in mm.entries:  # vote by the VLM's name so merge_overlaps sees the shared label
        e.votes = {e.label: 1.0}
    n = mm.merge_overlaps(min_overlap=0.3)
    by_id = {o["id"]: o for o in m["objects"]}
    for e in mm.entries:  # the survivors carry the merged views and position
        o = by_id[e.id]
        if e.n_views != o["n_views"]:
            gone = [x["id"] for x in m["objects"] if x["id"] not in {y.id for y in mm.entries}]
            o.update(xyz=e.xyz, last_xyz=e.last_xyz, n_views=e.n_views, sightings=e.sightings, extent=e.extent)
            o["merged_from"] = sorted(set(o.get("merged_from", [])) | {g for g in gone if by_id[g]["label"] == o["label"]})
    keep = {e.id for e in mm.entries}
    m["objects"] = [by_id[i] for i in by_id if i in keep]
    m["merged_after_naming"] = n
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map")
    ap.add_argument("passes", nargs="+")
    ap.add_argument("--config", required=True)
    ap.add_argument("--spec", default="maps/src_kitchen/spec.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gemini-3.5-flash-lite",
                    help="gemini-* (API, GEMINI_API_KEY) or an ollama model (e.g. qwen3-vl:8b-instruct, free, local)")
    ap.add_argument("--workers", type=int, default=4, help="API requests in parallel")
    ap.add_argument("--hint", help="vocabulary file: likely names, a hint to the VLM")
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--url", default="http://127.0.0.1:11434")
    ap.add_argument("--strips", help="save each object's views here")
    ap.add_argument("--strips-only", action="store_true", help="only cut the view strips (for eval_naming); no model")
    ap.add_argument("--env-file", help="a .env with GEMINI_API_KEY (gemini-* models)")
    a = ap.parse_args(argv)
    from chefbot.perception.detect_frames import load_classes
    from chefbot.perception.memory_map import load_counters

    m = json.loads(Path(a.map).read_text())
    hint = load_classes(a.hint) if a.hint else None
    key = gemini_key(a.env_file) if a.model.startswith("gemini") and not a.strips_only else None
    m = name_map(m, a.passes, a.config, a.model, hint, a.url, a.views, Path(a.strips) if a.strips else None,
                 strips_only=a.strips_only, key=key, workers=a.workers)  # fmt: skip
    if a.strips_only:
        Path(a.out).write_text(json.dumps(m, indent=1))
        print(f"strips for {len(m['objects'])} objects -> {a.strips}")
        return 0
    m = merge_same_named(m, [c for c in load_counters(a.spec) if c.height < 1.2])
    Path(a.out).write_text(json.dumps(m, indent=1))
    print(f"{len(m['objects'])} objects ({m['merged_after_naming']} merged after naming), {m['naming']['seconds']} s -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
