"""One generator, many seeds: generate each (generate.py), check it (checks.py), render it (render.py), then judge
the batch (judge.py) and summarize it. build_one runs in a worker process; the rest in the lab's.

A batch folder: seed_<n>/ per seed (the asset: model.xml, its files, params.json, gen_log.txt; lab/: checks.json,
sheet.jpg, actuation.jpg, actuation.mp4, top_mask.npy), batch.jpg (every asset from the top), summary.json.
"""

from __future__ import annotations

import json
import math
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

from chefbot.autoresearch.objects import checks as CK
from chefbot.autoresearch.objects import generate as G
from chefbot.autoresearch.objects import judge as J


def check_ranges(gen_dir: Path, params: dict) -> dict:
    """The asset's parameters against the generator's ranges.yaml (every range needs a `why`)."""
    path = Path(gen_dir) / "ranges.yaml"
    if not path.exists():
        return {"ok": False, "why": "the generator has no ranges.yaml"}
    try:
        ranges = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        return {"ok": False, "why": f"ranges.yaml: {e}"}
    whys = []
    for k, r in ranges.items():
        if not isinstance(r, dict) or not str(r.get("why", "")).strip():
            whys.append(f"{k}: needs min, max and why")
            continue
        v = params.get(k)
        if isinstance(v, (int, float)) and isinstance(r.get("min"), (int, float)) and isinstance(r.get("max"), (int, float)):
            if not r["min"] - 1e-9 <= v <= r["max"] + 1e-9:
                whys.append(f"{k} = {v:g} is outside its range [{r['min']:g}, {r['max']:g}]")
    for k in ("length_m", "rest_tip_gap_m", "closing_force_N", "mass_kg"):
        if k not in ranges:
            whys.append(f"ranges.yaml has no {k}")
    return {"ok": not whys, **({"why": "; ".join(whys)} if whys else {})}


def build_one(gen_dir: str, seed: int, out: str, fam: dict, video: bool = True, extra_deny: tuple = ()) -> dict:
    """Generate, check and render one seed. Never raises: a failure is in the record."""
    out_p = Path(out)
    rec: dict = {"seed": seed, "dir": str(out_p)}
    try:
        g = G.run(Path(gen_dir), seed, out_p, fam["generation"]["timeout_s"], fam["generation"]["max_asset_mb"], list(extra_deny))
        rec["gen_s"] = g.get("gen_s")
        if not g["ok"]:
            rec.update(ok=False, failed=["generate"], why={"generate": g["error"]})
            return rec
        params = g["params"]
        rec["params"] = params
        c = CK.run(out_p, fam, params)
        c["checks"]["ranges"] = check_ranges(Path(gen_dir), params)
        failed = [k for k, v in c["checks"].items() if not v.get("ok")]
        c.update(ok=not failed, failed=failed)
        (out_p / "lab" / "checks.json").write_text(json.dumps(c, indent=1, default=float))
        rec.update(ok=c["ok"], failed=failed, measured=c["measured"], check_s=c["check_s"],
                   why={k: v.get("why") for k, v in c["checks"].items() if v.get("why")})  # fmt: skip
        if (out_p / "lab" / "settled.npz").exists() and (out_p / "lab" / "squeeze.npz").exists():
            title = f"seed {seed}: {params.get('style', '')}"[:110]
            try:
                from chefbot.autoresearch.objects import render

                render.render_all(out_p, title=title, video=video)
                rec["rendered"] = True
            except Exception as e:  # noqa: BLE001
                rec["render_error"] = f"{type(e).__name__}: {e}"
    except Exception as e:  # noqa: BLE001
        rec.update(ok=False, failed=rec.get("failed", []) + ["harness"], harness_error=f"{type(e).__name__}: {e}",
                   trace=traceback.format_exc()[-1500:])  # fmt: skip
    return rec


def batch_sheet(records: list[dict], path: Path) -> int:
    """Every rendered asset's top view, numbered by seed, in one image. Returns how many."""
    tiles = []
    for r in records:
        sheet = Path(r["dir"]) / "lab" / "sheet.jpg"
        if r.get("rendered") and sheet.exists():
            im = Image.open(sheet).crop((640, 70, 1280, 520)).resize((320, 225))
            ImageDraw.Draw(im).rectangle([0, 0, 90, 22], fill=(255, 255, 255))
            ImageDraw.Draw(im).text((4, 4), f"#{r['seed']}", fill=(0, 0, 0))
            tiles.append(im)
    if not tiles:
        return 0
    cols = min(4, len(tiles))
    rows = math.ceil(len(tiles) / cols)
    out = Image.new("RGB", (cols * 320, rows * 225), "white")
    for k, t in enumerate(tiles):
        out.paste(t, ((k % cols) * 320, (k // cols) * 225))
    out.save(path, quality=88)
    return len(tiles)


def silhouette_diversity(records: list[dict]) -> float | None:
    """Mean pairwise 1 - IoU of the top-view silhouettes (same camera, same scale; about 4 mm a pixel), each
    thickened by 4 pixels first so that thin arms a few millimetres apart don't count as different shapes."""
    from scipy.ndimage import binary_dilation

    masks = []
    for r in records:
        p = Path(r["dir"]) / "lab" / "top_mask.npy"
        if r.get("rendered") and p.exists():
            masks.append(binary_dilation(np.load(p), iterations=4))
    if len(masks) < 2:
        return None
    vals = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            inter, union = np.logical_and(masks[i], masks[j]).sum(), np.logical_or(masks[i], masks[j]).sum()
            vals.append(1 - inter / union if union else 0.0)
    return round(float(np.mean(vals)), 3)


def judge_records(records: list[dict], fam: dict, rubric: str, refs_image: Path | None, model: str, config_dir: Path,
                  ledger: Path | None, threads: int = 4) -> None:
    """Judge every rendered asset in place (record["judge"]), a few at a time."""
    def one(r):
        try:
            c = json.loads((Path(r["dir"]) / "lab" / "checks.json").read_text())
            r["judge"] = J.judge_asset(Path(r["dir"]) / "lab", r.get("params", {}), c, rubric, refs_image, model, config_dir, ledger)
        except Exception as e:  # noqa: BLE001
            r["judge"] = {"error": f"{type(e).__name__}: {e}"}

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(one, [r for r in records if r.get("rendered")]))


def summarize(records: list[dict], batch: dict | None = None) -> dict:
    n = len(records)
    ok = [r for r in records if r.get("ok")]
    per_check: dict = {}
    for r in records:
        for k in r.get("failed", []):
            per_check[k] = per_check.get(k, 0) + 1
    judged = [r["judge"] for r in records if isinstance(r.get("judge"), dict) and "overall" in r["judge"]]
    out = {"assets": n, "physics_pass_rate": round(len(ok) / n, 3) if n else 0.0,
           "failures_by_check": dict(sorted(per_check.items(), key=lambda kv: -kv[1])),
           "silhouette_diversity": silhouette_diversity(records)}  # fmt: skip
    if judged:
        for k in J.SCORE_KEYS:
            out[f"judge_{k}"] = round(float(np.mean([j[k] for j in judged])), 2)
        out["judged"] = len(judged)
    if batch and "variety" in batch:
        out["judge_variety"], out["judge_coverage"] = batch["variety"], batch["coverage"]
    for key in ("length_m", "rest_tip_gap_m", "closing_force_N", "mass_kg"):
        vals = [r["measured"][key] for r in records if isinstance(r.get("measured", {}).get(key), (int, float))]
        if len(vals) >= 2:
            out[f"{key}_range"] = [round(min(vals), 4), round(max(vals), 4)]
    gens = [r["gen_s"] for r in records if isinstance(r.get("gen_s"), (int, float))]
    if gens:
        out["gen_s_mean"] = round(float(np.mean(gens)), 2)
    return out


def worker_init(env: dict) -> None:
    os.environ.update(env)
