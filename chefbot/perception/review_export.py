"""Export memory maps for a person to review: per object, crops of the frames it was seen in (the spot circled), its
label votes, world position, counter, and the park stance goto would use. Writes OUT/review.json + OUT/crops/*.jpg.

    python -m chefbot.perception.review_export --config CAL.yaml --spec maps/src_kitchen/spec.yaml --out OUT \\
        --map "Layout 2=STREAM/p1_layout2:DETS/map_p1_layout2.json" --map "Layout 3=...:..." \\
        --map "Layout 3, after the moves=STREAM/p2_layout3,STREAM/p3_moved_unseen:DETS/map_p2_then_p3.json"
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from chefbot.perception.goto import Floor, park_target
from chefbot.perception.pole_geometry import Pass, load_counters, load_mount, to_world, world_from_camera

CROP = 260  # px around the spot, in the 1280x800 color frame
CROPS_PER_OBJECT = 4


def _crop(p: Pass, frame: dict, xyz, T_mount, out: Path) -> bool:
    from PIL import Image, ImageDraw

    pose = p.pose(frame)
    if pose is None:
        return False
    T_wc = world_from_camera(pose, T_mount)
    pc = to_world(np.linalg.inv(T_wc), np.asarray(xyz, float)[None])[0]
    if pc[2] <= 0.2:
        return False
    u, v = p.camera.color.project(pc[None])[0]
    W, H = p.camera.color.width, p.camera.color.height
    if not (0 <= u < W and 0 <= v < H):
        return False
    # the crop scales with distance: ~0.35 m across at the object
    half = int(np.clip(0.175 * p.camera.color.K[0, 0] / pc[2], 60, CROP))
    x0, y0 = int(np.clip(u - half, 0, W - 2 * half)), int(np.clip(v - half, 0, H - 2 * half))
    im = Image.fromarray(p.color(frame)).crop((x0, y0, x0 + 2 * half, y0 + 2 * half)).resize((240, 240))
    s = 240 / (2 * half)
    cx, cy, r = (u - x0) * s, (v - y0) * s, 22
    ImageDraw.Draw(im).ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 0, 200), width=3)
    im.save(out, quality=82)
    return True


def _spread(items: list, k: int) -> list:
    if len(items) <= k:
        return items
    return [items[i] for i in np.linspace(0, len(items) - 1, k).round().astype(int)]


def export(maps: list[tuple[str, list[str], str]], config, spec, out) -> dict:
    out = Path(out)
    (out / "crops").mkdir(parents=True, exist_ok=True)
    T_mount = load_mount(config)
    counters = [c for c in load_counters(spec) if c.height < 1.2]
    by_name = {c.name: c for c in counters}
    floor = Floor.from_spec(spec)
    passes: dict[str, Pass] = {}
    result = {"counters": [{"name": c.name, "fixture": c.fixture, "center": c.center.tolist(), "size": c.size.tolist(),
                            "yaw_deg": math.degrees(c.yaw), "height": c.height} for c in load_counters(spec, fixtures_only=False)],
              "maps": []}  # fmt: skip
    for mi, (title, pass_dirs, map_file) in enumerate(maps):
        for d in pass_dirs:
            passes.setdefault(Path(d).name, Pass(d))
        m = json.loads(Path(map_file).read_text())
        objs = []
        for o in m["objects"]:
            xyz = o["xyz"] or o.get("last_xyz")
            entry = {k: o.get(k) for k in ("id", "name", "label", "counter", "fixture", "xyz", "last_xyz", "status", "n_views", "score", "events")}
            entry["votes"] = sorted(([k, round(v, 2)] for k, v in (o.get("votes") or {}).items()), key=lambda kv: -kv[1])[:5]
            if o["xyz"] is not None and o["counter"] in by_name:
                c = by_name[o["counter"]]
                try:
                    entry["park"] = park_target(c.center, c.size, c.yaw, o["xyz"][:2], floor=floor)
                except ValueError as exc:
                    entry["park"] = {"stance": None, "error": str(exc)}
            crops = []
            seen = [s for s in o.get("sightings", []) if s[0] in passes]
            best = sorted(seen, key=lambda s: -s[2])[:1]
            for s in best + [s for s in _spread(seen, CROPS_PER_OBJECT + 2) if s not in best]:
                if len(crops) >= CROPS_PER_OBJECT or xyz is None:
                    break
                p = passes[s[0]]
                f = next((f for f in p.frames if f["n"] == s[1]), None)
                name = f"m{mi}_o{o['id']}_{s[0][:2]}_{s[1]}.jpg"
                if f is not None and _crop(p, f, xyz, T_mount, out / "crops" / name):
                    crops.append({"file": f"crops/{name}", "pass": s[0], "n": s[1], "score": s[2], "label": s[3]})
            entry["crops"] = crops
            objs.append(entry)
        objs.sort(key=lambda e: (e["counter"] or "", e["label"]))
        result["maps"].append({"key": f"m{mi}", "title": title, "passes": [Path(d).name for d in pass_dirs], "objects": objs})
        print(f"{title}: {len(objs)} objects")
    (out / "review.json").write_text(json.dumps(result))
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--map", action="append", required=True, help='"TITLE=PASS[,PASS]:MAP.json"')
    a = ap.parse_args(argv)
    maps = []
    for spec in a.map:
        title, rest = spec.split("=", 1)
        pass_list, map_file = rest.rsplit(":", 1)
        maps.append((title, pass_list.split(","), map_file))
    export(maps, a.config, a.spec, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
