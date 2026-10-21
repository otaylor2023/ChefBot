"""Every fridge model RoboCasa ships, in a kitchen of its own split, checked with the
privileged oracle: which (layout, style, door) can be opened at all from a clear start.

    python -m experiments.fridge_lab.survey [--jobs 4]

RoboCasa's train styles/layouts (11-60) and test ones (1-10) use disjoint fridge
models. For each model, up to TRIES layouts of its split that use its fridge type
are tried (with a style that has that model) until one has a door the oracle opens
to MIN_DEG; every door of that kitchen is recorded. Writes results/survey.json.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

from experiments.astranomy.host import REPO

SCENES_DIR = Path.home() / "dev" / "robocasa" / "robocasa" / "models" / "assets" / "scenes"
OUT = REPO / "experiments" / "fridge_lab" / "results"
TRIES = 3
MIN_DEG = 60.0
TYPES = ("fridge_french_door", "fridge_side_by_side", "fridge_bottom_freezer")


def _num(p: str) -> int:
    return int(re.search(r"(\d+)\.yaml$", p).group(1))


def _layout_type(p: str) -> str | None:
    t = None
    for line in open(p):
        m = re.match(r"\s+type:\s*(fridge_(?:french_door|side_by_side|bottom_freezer))\b", line)
        if m:
            t = m.group(1)
    return t


def catalog() -> dict[str, list[dict]]:
    """split -> one entry per fridge model: {model, type, candidates: [(layout, style), ...]} (layouts spread out)."""
    out = {}
    for split in ("train", "test"):
        layouts = {_num(p): _layout_type(p) for p in sorted(glob.glob(str(SCENES_DIR / "kitchen_layouts" / split / "*.yaml")))}
        styles = {_num(p): yaml.safe_load(open(p)) for p in sorted(glob.glob(str(SCENES_DIR / "kitchen_styles" / split / "*.yaml")))}
        models: dict[tuple[str, str], list[tuple[int, int]]] = {}
        for t in TYPES:
            lts = [l for l, lt in layouts.items() if lt == t]
            for s, st in styles.items():
                if st.get(t):
                    models.setdefault((t, st[t]), [])
            for i, ((tt, model), cands) in enumerate(sorted((k, v) for k, v in models.items() if k[0] == t)):
                sts = [s for s, st in styles.items() if st.get(t) == model]
                # spread models over layouts: model i starts at layout i
                for k in range(len(lts)):
                    cands.append((lts[(i + k) % len(lts)], sts[k % len(sts)]))
        out[split] = [{"model": m, "type": t, "candidates": c} for (t, m), c in sorted(models.items())]
    return out


def check(split: str, entry: dict) -> dict:
    from experiments.fridge_lab.kitchen import LabHost
    from experiments.fridge_lab.oracle import run

    tried = []
    for layout, style in entry["candidates"][:TRIES]:
        try:
            host = LabHost(layout, style, OUT / "survey_tmp", record=False)
        except Exception as e:  # noqa: BLE001
            tried.append({"layout": layout, "style": style, "error": repr(e)[:300]})
            continue
        try:
            doors = []
            for i, D in enumerate(host.doors):
                r = run(host, i, 0)
                doors.append({"door": i, "joint": D.joint, "width": D.width, "oracle_deg": r["max_deg"], "start": r["start"]})
            rec = {"layout": layout, "style": style, "fixture": host.fixture, "doors": doors}
        except Exception:  # noqa: BLE001
            rec = {"layout": layout, "style": style, "error": traceback.format_exc()[-600:]}
        finally:
            host.close()
        tried.append(rec)
        if any(d["oracle_deg"] >= MIN_DEG for d in rec.get("doors", [])):
            break
    return {"split": split, "model": entry["model"], "type": entry["type"], "tried": tried}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    cat = catalog()
    path = OUT / "survey.json"
    done = {(r["split"], r["model"]): r for r in json.loads(path.read_text())} if path.exists() else {}
    todo = [(s, e) for s, es in cat.items() for e in es if (s, e["model"]) not in done]
    print(f"{sum(len(v) for v in cat.values())} models, {len(todo)} to check", flush=True)
    with ProcessPoolExecutor(args.jobs) as ex:
        futs = {ex.submit(check, s, e): (s, e["model"]) for s, e in todo}
        for f in as_completed(futs):
            r = f.result()
            done[(r["split"], r["model"])] = r
            best = max((d["oracle_deg"] for t in r["tried"] for d in t.get("doors", [])), default=None)
            print(f"{r['split']:5s} {r['model']} {r['type']:22s} tries {len(r['tried'])} best {best}", flush=True)
            path.write_text(json.dumps(list(done.values()), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
