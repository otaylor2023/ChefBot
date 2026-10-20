"""Results tables and failure gallery (spec §16).

    python -m experiments.ortho_grasp.report

Reads results/<config>/<scene>_<repeat>/metrics.json; writes results/summary.csv
and results/report.html (images linked relative to results/).
"""

from __future__ import annotations

import csv
import html
import json
import math
import random
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / yaml.safe_load((HERE / "configs" / "experiment.yaml").read_text())["results_dir"]

SETS = {
    "S1 representation": ["base", "p2", "b0", "r_gt", "r_heur"],
    "S2 output mode": ["base", "o2"],
    "S3 render": ["base", "v1_partial", "v4_real"],
    "S4 view handling": ["base", "seq"],
    "S5 trajectory": ["base", "t3"],
    "S6 models": ["v1_partial", "gemini_er", "gpt", "o2", "o2_gemini_er"],
}
RATES = [("e0_parse", "E0 parse"), ("resolved", "resolved"), ("e1_valid", "E1 valid"), ("e2_valid", "E2 valid"),
         ("fix_e2_valid", "E2 valid, flange corrected"), ("enclosed", "E3 enclosed"), ("lifted", "E3 lifted")]  # fmt: skip
CHECKS = ["straddle", "clear_tips", "width_ok", "height_ok", "ik_ok", "collision_ok"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for f in sorted(RESULTS.glob("*/s*_*/metrics.json")):
        m = json.loads(f.read_text())
        m["_dir"] = str(f.parent.relative_to(RESULTS))
        out.setdefault(m["config"], []).append(m)
    return out


def write_csv(rows: dict[str, list[dict]]) -> None:
    flat = [m for ms in rows.values() for m in ms]
    keys = sorted({k for m in flat for k in m if not k.startswith("_")})
    with open(RESULTS / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for m in flat:
            w.writerow({k: m.get(k) for k in keys})


def _rate(ms, key):
    vals = [m for m in ms if key in m]
    k = sum(bool(m.get(key)) for m in vals)
    return k, len(vals)


def _cell(k, n):
    if n == 0:
        return "<td class='na'>-</td>"
    lo, hi = wilson(k, n)
    return f"<td>{k}/{n} <b>{k / n:.0%}</b><br><span class='ci'>[{lo:.0%}, {hi:.0%}]</span></td>"


def _mean(ms, key, scale=1.0, fmt="{:.2f}"):
    vals = [float(m[key]) * scale for m in ms if m.get(key) is not None]
    return fmt.format(sum(vals) / len(vals)) if vals else "-"


def batch_costs() -> dict[str, float]:
    """Config -> $ from the batches' own usage (batch results carry no per-request cost)."""
    out: dict[str, float] = {}
    for f in (RESULTS / "batches").glob("*.json"):
        d = json.loads(f.read_text())
        cost = (d.get("usage") or {}).get("cost")
        if cost is not None and d.get("requests"):
            c = next(iter(d["requests"].values()))["meta"]["config"]
            out[c] = out.get(c, 0.0) + float(cost)
    return out


BATCH_COSTS: dict[str, float] = {}


def _total(ms, *keys):
    vals = [float(m[k]) for m in ms for k in keys if m.get(k) is not None]
    c = ms[0]["config"] if ms else None
    if c in BATCH_COSTS:
        return f"${BATCH_COSTS[c] + sum(vals):.2f} (batch)"
    return f"${sum(vals):.2f}" if vals else "-"


def _xdis(ms):
    vals = [float(v) for m in ms for k, v in m.items() if k.startswith("x_disagree_") and v is not None]
    return f"{1000 * sum(vals) / len(vals):.1f} mm" if vals else "-"


def table(rows, configs) -> str:
    head = "".join(f"<th>{label}</th>" for _, label in RATES)
    out = [f"<table><tr><th>config</th>{head}<th>GS angle</th><th>x disagree</th><th>flange moved</th><th>latency</th><th>cost</th></tr>"]
    for c in configs:
        ms = rows.get(c, [])
        if not ms:
            out.append(f"<tr><td>{c}</td><td colspan='12' class='na'>not run</td></tr>")
            continue
        cells = "".join(_cell(*_rate(ms, key)) for key, _ in RATES)
        out.append(f"<tr><td><a href='#{c}'>{c}</a></td>{cells}<td>{_mean(ms, 'gs_angle_deg', fmt='{:.1f}°')}</td>"
                   f"<td>{_xdis(ms)}</td><td>{_mean(ms, 'fix_flange_correction_deg', fmt='{:.0f}°')}</td><td>{_mean(ms, 'call1_latency_s', fmt='{:.1f} s')}</td>"
                   f"<td>{_total(ms, 'call1_cost', 'call2_cost')}</td></tr>")  # fmt: skip
    return "\n".join(out + ["</table>"])


def breakdown(ms) -> str:
    reasons: dict[str, int] = {}
    for m in ms:
        if not m.get("resolved"):
            r = str(m.get("reason"))
            reasons[f"unresolved: {r.split(':')[0] if r.startswith('parse') else r}"] = reasons.get(f"unresolved: {r.split(':')[0] if r.startswith('parse') else r}", 0) + 1
            continue
        for k in CHECKS:
            if m.get(k) is False:
                reasons[k] = reasons.get(k, 0) + 1
        for k, v in m.items():
            if k.startswith("flag_") and v:
                reasons[k] = reasons.get(k, 0) + 1
    items = sorted(reasons.items(), key=lambda kv: -kv[1])
    return "<ul>" + "".join(f"<li>{html.escape(k)}: {v}/{len(ms)}</li>" for k, v in items) + "</ul>" if items else "<p>no failures</p>"


def _badness(m) -> tuple:
    if not m.get("resolved"):
        return (0, 0.0)
    fails = sum(m.get(k) is False for k in CHECKS)
    return (1, -fails, -(m.get("m_to_surface") or 0.0))


def gallery(ms, n=6) -> str:
    rnd = random.Random(0)
    pick = rnd.sample(ms, min(n, len(ms)))
    worst = sorted(ms, key=_badness)[:n]
    def card(m):
        d = m["_dir"]
        status = "OK" if m.get("e2_valid") else "fail"
        fails = ", ".join(k for k in CHECKS if m.get(k) is False) or (m.get("reason") or "")
        imgs = "".join(f"<a href='{d}/overlay_{v}.png'><img src='{d}/overlay_{v}.png'></a>" for v in ("top", "side"))
        return (f"<div class='card'><div>{html.escape(d)} · {m.get('cup_model')} · <b>{status}</b> {html.escape(str(fails))}"
                f" · <a href='{d}/response.json'>response</a> · <a href='{d}/inputs/prompt_call1.txt'>prompt</a></div>{imgs}</div>")  # fmt: skip
    return "<h4>random</h4>" + "".join(card(m) for m in pick) + "<h4>worst</h4>" + "".join(card(m) for m in worst)


def grid(rows, configs) -> str:
    scenes = sorted({m["scene"] for ms in rows.values() for m in ms})
    out = ["<table class='grid'><tr><th>scene</th>" + "".join(f"<th>{c}</th>" for c in configs) + "</tr>"]
    for s in scenes:
        cells = []
        for c in configs:
            ms = [m for m in rows.get(c, []) if m["scene"] == s]
            if not ms:
                cells.append("<td class='na'></td>")
                continue
            k = sum(bool(m.get("e2_valid")) for m in ms)
            shade = int(255 * k / len(ms))
            cells.append(f"<td style='background:rgb({255 - shade},{180 + shade // 4},{120})' title='{k}/{len(ms)}'>{k}/{len(ms)}</td>")
        out.append(f"<tr><td>{s}</td>{''.join(cells)}</tr>")
    return "\n".join(out + ["</table>"])


def main() -> int:
    rows = load()
    write_csv(rows)
    BATCH_COSTS.update(batch_costs())
    configs = [c for c in dict.fromkeys(c for cs in SETS.values() for c in cs) if c in rows] + sorted(set(rows) - {c for cs in SETS.values() for c in cs})
    parts = ["<!doctype html><meta charset='utf-8'><title>Ortho grasp results</title><style>",
             "body{font:14px system-ui;margin:24px;max-width:1500px} table{border-collapse:collapse;margin:8px 0 24px}",
             "td,th{border:1px solid #ccc;padding:4px 8px;text-align:center} .ci{color:#777;font-size:11px} .na{color:#aaa}",
             ".card{display:inline-block;margin:6px;border:1px solid #ddd;padding:4px;font-size:12px;vertical-align:top}",
             ".card img{width:360px;margin:2px} .grid td{font-size:11px;padding:2px 6px}</style>",
             "<h1>Ortho-view grasp marking — results</h1>",
             "<p>Rates with 95% Wilson intervals. E1/E2 are computed on the ground-truth cup; E3 only for the configs it was run on.</p>"]  # fmt: skip
    for name, cs in SETS.items():
        parts.append(f"<h2>{name}</h2>" + table(rows, cs))
    parts.append("<h2>Scene × config (E2 valid)</h2>" + grid(rows, configs))
    for c in configs:
        parts.append(f"<h2 id='{c}'>{c}</h2><h3>failures</h3>{breakdown(rows[c])}<h3>gallery</h3>{gallery(rows[c])}")
    (RESULTS / "report.html").write_text("\n".join(parts))
    print(f"wrote {RESULTS / 'report.html'} and summary.csv ({sum(len(v) for v in rows.values())} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
