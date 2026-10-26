"""The exploration report: per run, every true object (found or not, name and fixture given, xy and
3D error), the false finds, a top-down map of true vs found positions, a review video of the
detector's pictures with the mistakes drawn on, and a gallery of the pictures where it failed.

    python -m chefbot.explore.report --out docs/exploration_report RUN_SPEC [RUN_SPEC ...]

RUN_SPEC is `title=runs/<run>[@sightings.csv]`: the sighting table (rootcause.py) gives the true
objects' boxes in each saved picture; without one (the exact detector takes no pictures) the run
gets its table and map only. Offline: it reads the truth (true_map.yaml, true_positions.yaml),
which the agent never did. Writes index.html, summary.md and media next to them.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path

import numpy as np
import yaml

from chefbot.explore.accuracy import _label, load_run, match, object_errors

GREEN, ORANGE, RED, BLUE, GREY = (40, 170, 60), (240, 150, 20), (220, 40, 40), (40, 110, 230), (150, 150, 150)
_ON = 0.3  # a detector box covering this share of a true object's visible box is on it
_SHOW_PX = 100  # true objects with fewer visible pixels than this are not drawn as missed in a picture


def _overlap(box, bbox) -> float:
    x0, y0, x1, y1 = max(box[0], bbox[0]), max(box[1], bbox[1]), min(box[2], bbox[2]), min(box[3], bbox[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0) / max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))


def _same(a: str, b: str) -> bool:
    return _label(a) == _label(b)


# -- per run: the object table --------------------------------------------------------------


def object_table(run_dir: Path) -> dict:
    run = load_run(run_dir)
    rows = object_errors(run)
    m = match(run.explored, run.truth)
    used = {v for v in m.values() if v}
    extras = {n: e for n, e in run.explored.items() if n not in used}
    out = []
    for r in rows:
        t = run.truth[r["object"]]
        row = {"object": r["object"], "true_in": t.get("in"), "found": r["found_as"] is not None}
        if r["found_as"]:
            e = run.explored[r["found_as"]]
            row.update(name=r["found_as"], given_in=e.get("in"), xy_cm=round(100 * r["xy"], 1),
                       d3_cm=round(100 * r["d3"], 1) if "d3" in r else None)  # fmt: skip
        else:  # a false find near it: the name it was given instead
            near = sorted((math.dist(e["location"], t["location"]), n) for n, e in extras.items())
            if near and near[0][0] <= 0.15:
                row.update(name=near[0][1], wrong_name=True, given_in=extras[near[0][1]].get("in"),
                           xy_cm=round(100 * near[0][0], 1))  # fmt: skip
        out.append(row)
    named_wrong = {r["name"] for r in out if r.get("wrong_name")}
    false = [{"name": n, "location": e["location"], "in": e.get("in")} for n, e in extras.items() if n not in named_wrong]
    start = json.loads((run_dir / "start.json").read_text()) if (run_dir / "start.json").exists() else {}
    return {"run": run_dir.name, "detector": start.get("detector"), "layout": start.get("layout"), "rows": out,
            "false": false, "truth": run.truth, "explored": run.explored, "found": sum(r["found"] for r in out)}  # fmt: skip


def table_md(t: dict) -> str:
    lines = [f"| true object | true fixture | found | name given | fixture given | xy error (cm) | 3D error (cm) |",
             "|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in t["rows"]:
        found = "yes" if r["found"] else ("wrong name" if r.get("wrong_name") else "no")
        flag = lambda v: "" if v is None else (f"**{v}**" if v > 15 else f"{v}")
        lines.append(f"| {r['object']} | {r['true_in']} | {found} | {r.get('name', '')} | {r.get('given_in', '')} | "
                     f"{flag(r.get('xy_cm'))} | {flag(r.get('d3_cm'))} |")  # fmt: skip
    if t["false"]:
        lines.append("")
        lines.append("False finds: " + ", ".join(f"{f['name']} at ({f['location'][0]:.2f}, {f['location'][1]:.2f}) in {f['in']}" for f in t["false"]) + ".")
    return "\n".join(lines)


def table_html(t: dict) -> str:
    rows = []
    for r in t["rows"]:
        found = "yes" if r["found"] else ("wrong name" if r.get("wrong_name") else "no")
        cls = "ok" if r["found"] and (r.get("xy_cm") or 0) <= 15 else ("wrong" if r.get("wrong_name") or r["found"] else "miss")
        cell = lambda v: "" if v is None else (f'<span class="big">{v}</span>' if v > 15 else f"{v}")
        rows.append(f'<tr class="{cls}"><td>{html.escape(r["object"])}</td><td>{r["true_in"]}</td><td>{found}</td>'
                    f'<td>{html.escape(r.get("name", ""))}</td><td>{r.get("given_in", "") or ""}</td>'
                    f'<td>{cell(r.get("xy_cm"))}</td><td>{cell(r.get("d3_cm"))}</td></tr>')  # fmt: skip
    false = "".join(f"<li>{html.escape(f['name'])} at ({f['location'][0]:.2f}, {f['location'][1]:.2f}) in {f['in']}</li>" for f in t["false"])
    return ("<table><tr><th>true object</th><th>true fixture</th><th>found</th><th>name given</th><th>fixture given</th>"
            "<th>xy error (cm)</th><th>3D error (cm)</th></tr>" + "".join(rows) + "</table>"
            + (f"<p><b>False finds ({len(t['false'])}):</b></p><ul class='false'>{false}</ul>" if t["false"] else "<p>No false finds.</p>"))  # fmt: skip


# -- the map: true vs found ---------------------------------------------------------------------


def map_image(run_dir: Path, t: dict, path: Path, highlight: set | None = None, size: int = 480) -> np.ndarray:
    import cv2

    start = yaml.safe_load((run_dir / "start_map.yaml").read_text()) if (run_dir / "start_map.yaml").exists() else {"fixtures": {}}
    fixtures = {n: f for n, f in (start.get("fixtures") or {}).items() if not n.startswith(("burner",))}
    pts = [f["location"] for f in fixtures.values()] + [o["location"] for o in t["truth"].values()]
    pts = np.array(pts) if pts else np.zeros((1, 2))
    lo, hi = pts.min(axis=0) - 0.6, pts.max(axis=0) + 0.6
    scale = (size - 20) / float(max(hi - lo))
    img = np.full((size, size, 3), 250, np.uint8)

    def px(p):
        return int(10 + (p[0] - lo[0]) * scale), int(size - 10 - (p[1] - lo[1]) * scale)

    for n, f in fixtures.items():
        cv2.circle(img, px(f["location"]), 3, GREY, -1)
        if not n.startswith(("cabinet", "drawer")):
            cv2.putText(img, n, (px(f["location"])[0] + 4, px(f["location"])[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, GREY, 1, cv2.LINE_AA)
    for r in t["rows"]:
        tl = t["truth"][r["object"]]["location"]
        hot = highlight is not None and r["object"] in highlight
        color = GREEN if r["found"] and not r.get("wrong_name") and (r.get("xy_cm") or 0) <= 15 else (ORANGE if r.get("name") else RED)
        if r.get("name"):
            el = t["explored"][r["name"]]["location"]
            cv2.line(img, px(tl), px(el), color, 1, cv2.LINE_AA)
            cv2.drawMarker(img, px(el), BLUE, cv2.MARKER_TILTED_CROSS, 7, 1)
        cv2.circle(img, px(tl), 7 if hot else 4, color, 2 if hot else -1, cv2.LINE_AA)
        cv2.putText(img, r["object"], (px(tl)[0] + 5, px(tl)[1] + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)
    for f in t["false"]:
        cv2.drawMarker(img, px(f["location"]), ORANGE, cv2.MARKER_TILTED_CROSS, 7, 1)
    cv2.putText(img, "true: dot (green ok, orange wrong name or >15 cm, red missed)  found: x", (6, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (60, 60, 60), 1, cv2.LINE_AA)  # fmt: skip
    if path is not None:
        cv2.imwrite(str(path), img[..., ::-1])
    return img


# -- pictures: truth and detector boxes drawn ------------------------------------------------------


def annotate(rgb: np.ndarray, truths: list[dict], items: list[dict]) -> tuple[np.ndarray, list[str]]:
    """The picture with each true object's box (green found right, orange wrong name, red missed)
    and the detector's boxes and names; and one caption line per mistake in it."""
    import cv2

    img = rgb.copy()
    notes = []
    for it in items:
        b = [int(v) for v in it["box"]]
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), BLUE, 1)
        cv2.putText(img, it["label"], (b[0], max(10, b[1] - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, BLUE, 1, cv2.LINE_AA)
    for tr in truths:
        if tr["px"] < _SHOW_PX:
            continue
        on = [it for it in items if _overlap(it["box"], tr["box"]) >= _ON]
        right = any(_same(it["label"], tr["label"]) for it in on)
        color = GREEN if right else (ORANGE if on else RED)
        b = [int(v) for v in tr["box"]]
        cv2.rectangle(img, (b[0] - 2, b[1] - 2), (b[2] + 2, b[3] + 2), color, 2)
        cv2.putText(img, tr["label"], (b[0], min(img.shape[0] - 3, b[3] + 13)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        if not right:
            notes.append(f"{tr['label']}: " + (f"named {on[0]['label']}" if on else f"missed ({tr['px']} px)"))
    for it in items:
        if not any(_overlap(it["box"], tr["box"]) >= _ON or _overlap(tr["box"], it["box"]) >= _ON for tr in truths):
            notes.append(f"false find: {it['label']}")
    return img, notes


def _frames(run_dir: Path, sightings: list[dict]) -> list[tuple[int, Path, list[dict], list[dict]]]:
    by = {}
    for r in sightings:
        if r.get("bx0") not in (None, ""):
            by.setdefault(int(r["frame"]), []).append({"label": r["label"], "object": r["object"], "px": int(r["visible_px"]),
                                                         "box": [float(r[k]) for k in ("bx0", "by0", "bx1", "by1")],
                                                         "distance": float(r["distance_m"]), "row": r})  # fmt: skip
    out = []
    for png in sorted((run_dir / "vision").glob("*.png")):
        n = int(png.stem)
        log = json.loads(png.with_suffix(".json").read_text())
        from chefbot.explore.vision import _CONFIG, _VOCAB, normalize

        items = [dict(i, label=normalize(i["label"], dict(_CONFIG["vision"].get("synonyms") or {}), set(_VOCAB["objects"])))
                 for i in log.get("vlm", [])]  # fmt: skip
        out.append((n, png, by.get(n, []), items))
    return out


def video(run_dir: Path, t: dict, sightings: list[dict], path: Path, seconds_per_picture: float = 0.5) -> int:
    import cv2
    import imageio.v2 as imageio
    from PIL import Image

    frames = _frames(run_dir, sightings)
    if not frames:
        return 0
    fps = 8
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=6, pixelformat="yuv420p", macro_block_size=16)
    for n, png, truths, items in frames:
        rgb = np.asarray(Image.open(png).convert("RGB"))
        cam, notes = annotate(rgb, truths, items)
        mp = map_image(run_dir, t, None, highlight={tr["object"] for tr in truths if tr["px"] >= _SHOW_PX})
        panel = np.hstack([cam, mp])
        cap = np.full((64, panel.shape[1], 3), 30, np.uint8)
        cv2.putText(cap, f"{t['detector']}  layout {t['layout']}  picture {n}/{len(frames)}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)  # fmt: skip
        cv2.putText(cap, ("; ".join(notes) or "no mistakes in this picture")[:150], (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (250, 200, 120) if notes else (150, 230, 150), 1, cv2.LINE_AA)  # fmt: skip
        frame = np.vstack([panel, cap])
        frame = frame[: frame.shape[0] // 16 * 16, : frame.shape[1] // 16 * 16]
        for _ in range(max(1, round(seconds_per_picture * fps))):
            writer.append_data(frame)
    writer.close()
    return len(frames)


def gallery(run_dir: Path, t: dict, sightings: list[dict], out_dir: Path, objects_csv: list[dict]) -> list[dict]:
    """The best picture (most visible pixels) of each missed, wrongly named or >10 cm object."""
    import cv2
    from PIL import Image

    causes = {(o["run"], o["object"]): o.get("cause", "") for o in objects_csv}
    frames = {n: (png, truths, items) for n, png, truths, items in _frames(run_dir, sightings)}
    out = []
    for r in t["rows"]:
        bad = (not r["found"]) or r.get("wrong_name") or (r.get("xy_cm") or 0) > 10
        if not bad:
            continue
        views = [(tr["px"], n, tr) for n, (_, truths, _) in frames.items() for tr in truths if tr["object"] == r["object"]]
        entry = {"object": r["object"], "problem": "missed" if not r.get("name") else ("wrong name: " + r["name"] if r.get("wrong_name") else
                 (f"{r['xy_cm']} cm off" if r["found"] else "")), "cause": causes.get((t["run"], r["object"]), ""), "image": None}  # fmt: skip
        if views:
            px, n, tr = max(views, key=lambda v: v[0])
            png, truths, items = frames[n]
            rgb = np.asarray(Image.open(png).convert("RGB"))
            img, _ = annotate(rgb, [x for x in truths if x["object"] == r["object"]], items)
            b = tr["box"]
            cx, cy, half = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, max(24, max(b[2] - b[0], b[3] - b[1]) * 1.5)
            crop = img[int(max(0, cy - half)):int(min(img.shape[0], cy + half)), int(max(0, cx - half)):int(min(img.shape[1], cx + half))]
            crop = cv2.resize(crop, (240, 240), interpolation=cv2.INTER_CUBIC)
            both = np.hstack([cv2.resize(img, (320, 240)), crop])
            name = f"{t['run'].replace('_explore_robocasa_scripted_s0', '')}_{r['object'].replace(' ', '_')}.jpg"
            cv2.imwrite(str(out_dir / name), both[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
            entry.update(image=name, picture=n, px=px, distance=round(tr["distance"], 2))
        out.append(entry)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("specs", nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--objects", type=Path, action="append", default=[], help="an objects.csv from rootcause.py (failure causes); repeatable")
    parser.add_argument("--intro", type=Path, help="markdown to put before the run tables")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "media").mkdir(exist_ok=True)
    objects_csv = [r for f in args.objects for r in csv.DictReader(open(f))]
    intro = args.intro.read_text() if args.intro else ""
    md, body = [intro], []
    for spec in args.specs:
        title, rest = spec.split("=", 1)
        run_path, _, sight_path = rest.partition("@")
        run_dir = Path(run_path)
        t = object_table(run_dir)
        sightings = [r for r in csv.DictReader(open(sight_path)) if r["run"] == run_dir.name] if sight_path else []
        slug = run_dir.name.replace("_explore_robocasa_scripted_s0", "")
        map_image(run_dir, t, args.out / "media" / f"{slug}_map.png")
        n = video(run_dir, t, sightings, args.out / "media" / f"{slug}.mp4") if sightings else 0
        gal = gallery(run_dir, t, sightings, args.out / "media", objects_csv) if sightings else []
        print(f"{title}: {t['found']}/{len(t['rows'])} found, {len(t['false'])} false finds, {n} pictures in the video, {len(gal)} in the gallery", flush=True)
        md += [f"\n### {title}\n", f"Run `{run_dir.name}`: {t['found']}/{len(t['rows'])} found, {len(t['false'])} false finds.\n", table_md(t)]
        gal_html = "".join(
            f'<figure><img src="media/{g["image"]}"><figcaption><b>{html.escape(g["object"])}</b>: {html.escape(g["problem"])}'
            f'{"; " + html.escape(g["cause"]) if g["cause"] else ""}; best view: picture {g.get("picture")}, {g.get("px")} px, '
            f'{g.get("distance")} m</figcaption></figure>' if g["image"] else
            f'<figure class="noimg"><figcaption><b>{html.escape(g["object"])}</b>: {html.escape(g["problem"])}; '
            f'{html.escape(g["cause"]) or "never in a picture"}</figcaption></figure>' for g in gal)  # fmt: skip
        body.append(f'<section><h3>{html.escape(title)}</h3><p>Run <code>{run_dir.name}</code>: <b>{t["found"]}/{len(t["rows"])}</b> found, '
                    f'{len(t["false"])} false finds.</p>{table_html(t)}<div class="media"><img class="map" src="media/{slug}_map.png">'
                    + (f'<video controls preload="metadata" src="media/{slug}.mp4"></video>' if n else
                       ("<p class='note'>No camera video: this detector reads the simulator and takes no pictures.</p>"
                        if t["detector"] == "groundtruth" else "<p class='note'>Video and gallery not rendered for this run (table and map only).</p>"))
                    + f'</div>{"<h4>Where it failed</h4><div class=gallery>" + gal_html + "</div>" if gal else ""}</section>')  # fmt: skip
    (args.out / "summary.md").write_text("\n".join(md) + "\n")
    try:
        from markdown_it import MarkdownIt

        intro_html = MarkdownIt("commonmark").enable("table").render(intro)
    except ImportError:
        intro_html = f"<pre>{html.escape(intro)}</pre>"
    (args.out / "index.html").write_text(_PAGE.replace("{intro}", intro_html).replace("{runs}", "".join(body)))
    return 0


_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Exploration report</title><style>
:root{--bg:#fbfbfa;--fg:#1d1d1f;--muted:#6b6b70;--line:#e3e3e0;--ok:#e7f5ea;--wrong:#fff1dc;--miss:#fde8e8;--card:#fff}
@media (prefers-color-scheme: dark){:root{--bg:#141416;--fg:#ececee;--muted:#a0a0a8;--line:#2c2c30;--ok:#173322;--wrong:#3a2a10;--miss:#3d1717;--card:#1c1c1f}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,system-ui,sans-serif;margin:0 auto;max-width:1180px;padding:24px 16px}
h1{font-size:26px;margin:0 0 8px}h2{margin-top:36px;border-bottom:1px solid var(--line);padding-bottom:4px}h3{margin:28px 0 6px}
table{border-collapse:collapse;width:100%;margin:8px 0;font-size:13.5px;display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:4px 8px;text-align:left;white-space:nowrap}th{background:var(--card)}
tr.ok td{background:var(--ok)}tr.wrong td{background:var(--wrong)}tr.miss td{background:var(--miss)}.big{color:#d0342c;font-weight:600}
section{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin:18px 0}
.media{display:flex;flex-wrap:wrap;gap:12px;align-items:flex-start;margin-top:10px}.map{width:360px;max-width:100%;border:1px solid var(--line)}
video{width:100%;max-width:760px;border:1px solid var(--line)}.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:10px}
figure{margin:0;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:var(--bg)}figure img{width:100%;display:block}
figcaption{font-size:12.5px;padding:6px 8px;color:var(--muted)}figure.noimg figcaption{padding:12px}.note{color:var(--muted)}ul.false{columns:2;font-size:13px}
code{font-size:12.5px}
</style></head><body><h1>Exploration report</h1>
<p class="note">Colors: green found and right; orange wrong name, or found more than 15 cm off; red missed. In pictures: colored
boxes are the true objects, blue boxes are what the detector drew and named.</p>
{intro}<h2>Runs</h2>{runs}</body></html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
