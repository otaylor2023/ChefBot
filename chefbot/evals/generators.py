"""The test of an object lab's result, after the session: its final generator on fresh seeds, the physics checks,
renders and videos, the held-out judge (Opus) against the test reference photos, and a page for people to rate the
assets.

    python -m chefbot.evals.generators RUN_DIR [--generator generators/final] [--n 24] [--jobs 6] [--no-judge]
    python -m chefbot.evals.generators --ratings ratings.json EVAL_DIR      # fold people's ratings in

RUN_DIR is a research run (runs/labs/<lab>-<time>). Output: runs/evals/objects/<lab>-<time>/ with generator/ (the
generator as tested), assets/seed_<n>/, batch.jpg, records.json, summary.json and index.html (the rating page: open
it in a browser, rate, and download the ratings).
"""

from __future__ import annotations

import argparse
import html
import json
import multiprocessing as mp
import os
import random
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from chefbot.autoresearch.generator_lab import family as F
from chefbot.autoresearch.generator_lab import judge as J
from chefbot.autoresearch.generator_lab import lab as OL
from chefbot.autoresearch.generator_lab import pipeline as P
from chefbot.autoresearch.generator_lab import refs as RF


def evaluate(run: Path, gen: str, n: int | None, jobs: int, judge: bool, out: Path | None) -> Path:
    from chefbot.autoresearch.research import RUNS

    cfg = OL.load(str(run / "private" / "lab.yaml"))
    fam = F.load(cfg.family)
    src = run / "workspace" / gen
    if not (src / "generate.py").exists():
        raise SystemExit(f"no generator at {src}")
    out = out or RUNS / "evals" / "objects" / f"{run.name}-{time.strftime('%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "generator").exists():
        shutil.copytree(src, out / "generator", ignore=shutil.ignore_patterns("__pycache__"))
    n = n or int(cfg.test.get("n", 24))
    seeds = random.SystemRandom().sample(range(1_000_000, 2_000_000_000), n)
    env = {k: v for k, v in os.environ.items() if k.startswith(OL.WORKER_ENV)}
    deny = (str(RF.root()), str(run))
    with ProcessPoolExecutor(jobs, mp_context=mp.get_context("spawn"), initializer=P.worker_init, initargs=(env,)) as ex:
        futs = [ex.submit(P.build_one, str(out / "generator"), sd, str(out / "assets" / f"seed_{sd}"), fam, True, deny) for sd in seeds]
        records = [f.result() for f in futs]
    count = P.batch_sheet(records, out / "batch.jpg")
    batch = None
    if judge:
        refs_img = RF.montage(cfg.family, "test", out / "refs_test.jpg")
        heldout = RF.root() / cfg.family / "heldout" / "rubric.md"
        rubric = (heldout if heldout.exists() else OL.family_dir(cfg) / fam["judge"]["rubric"]).read_text()
        model = cfg.judge.get("val_model", "opus")
        home, ledger = out / "claude_judge", out / "judge_ledger.jsonl"
        P.judge_records(records, fam, rubric, refs_img, model, home, ledger)
        if count >= 3:
            try:
                batch = J.judge_batch(out / "batch.jpg", count, (OL.family_dir(cfg) / "rubric_batch.md").read_text(), refs_img, model, home, ledger)
            except J.JudgeError as e:
                batch = {"error": str(e)}
        shutil.rmtree(home, ignore_errors=True)
    summary = {"run": str(run), "generator": gen, "seeds": seeds, **P.summarize(records, batch)}
    (out / "records.json").write_text(json.dumps({"records": records, "batch": batch}, indent=1, default=str))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    write_page(out, records, summary, cfg.family)
    return out


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rate generated {family}</title>
<style>
:root {{ --bg: #fbfaf8; --fg: #1d1d1b; --muted: #6b6a66; --card: #ffffff; --line: #e3e1dc; --accent: #2f6f5e; --bad: #b3412f; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg: #161615; --fg: #ecebe7; --muted: #a09e98;
  --card: #201f1d; --line: #34332f; --accent: #6fbfa6; --bad: #e07a66; }} }}
body {{ background: var(--bg); color: var(--fg); font: 15px/1.5 system-ui, sans-serif; margin: 0; padding: 24px 16px 80px; }}
main {{ max-width: 1100px; margin: 0 auto; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }} .muted {{ color: var(--muted); }}
.card {{ background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 16px; margin: 18px 0; }}
.card img, .card video {{ width: 100%; height: auto; border-radius: 6px; display: block; }}
.media {{ display: grid; grid-template-columns: 2fr 1fr; gap: 12px; }} @media (max-width: 760px) {{ .media {{ grid-template-columns: 1fr; }} }}
.nums {{ font: 13px/1.6 ui-monospace, monospace; color: var(--muted); }} .fail {{ color: var(--bad); }}
.rate {{ display: flex; flex-wrap: wrap; gap: 14px 22px; align-items: center; margin-top: 12px; }}
.rate label {{ white-space: nowrap; }} textarea {{ width: 100%; min-height: 44px; background: var(--bg); color: var(--fg);
  border: 1px solid var(--line); border-radius: 6px; padding: 6px; font: inherit; box-sizing: border-box; }}
button {{ background: var(--accent); color: #fff; border: 0; border-radius: 6px; padding: 8px 14px; font: inherit; cursor: pointer; }}
.bar {{ position: sticky; top: 0; background: var(--bg); padding: 10px 0; border-bottom: 1px solid var(--line); display: flex;
  gap: 12px; align-items: center; flex-wrap: wrap; z-index: 2; }}
details {{ margin-top: 10px; }} summary {{ cursor: pointer; color: var(--muted); }}
</style></head><body><main>
<h1>Rate the generated {family}</h1>
<p class="muted">{n} assets from <code>{gen}</code>, fresh seeds. For each: does it look like real {family}, and does it move like them
(the video squeezes the handles at the red arrows, then lets go)? Ratings save in this browser as you go; download them when done.
The judge's scores are hidden under each card so they don't sway you.</p>
<div class="bar"><span id="count" class="muted"></span><button onclick="download()">Download ratings</button></div>
{cards}
</main>
<script>
const KEY = "ratings:{eval_id}";
function load() {{ try {{ return JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{ return {{}}; }} }}
function save(r) {{ try {{ localStorage.setItem(KEY, JSON.stringify(r)); }} catch (e) {{}} }}
const ratings = load();
function update() {{
  const done = Object.values(ratings).filter(r => r.realism).length;
  document.getElementById("count").textContent = done + " of {n} rated";
}}
document.querySelectorAll("[data-seed]").forEach(card => {{
  const seed = card.dataset.seed, r = ratings[seed] || {{}};
  card.querySelectorAll("input, textarea").forEach(el => {{
    const k = el.name.split(":")[1];
    if (el.type === "radio") el.checked = r[k] === el.value; else if (r[k]) el.value = r[k];
    el.addEventListener("input", () => {{
      ratings[seed] = ratings[seed] || {{}};
      ratings[seed][k] = el.value; save(ratings); update();
    }});
  }});
}});
update();
function download() {{
  const blob = new Blob([JSON.stringify({{eval: "{eval_id}", ratings}}, null, 1)], {{type: "application/json"}});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "ratings.json"; a.click();
}}
</script></body></html>
"""


def _card(r: dict, base: Path) -> str:
    seed = r["seed"]
    rel = f"assets/seed_{seed}/lab"
    if not r.get("rendered"):
        return f'<div class="card"><b>seed {seed}</b> <span class="fail">not rendered: {html.escape(str(r.get("why") or r.get("harness_error")))}</span></div>'
    p = r.get("params", {})
    m = r.get("measured", {})
    nums = "  ".join(f"{k} {v}" for k, v in m.items())
    status = "passes every check" if r.get("ok") else '<span class="fail">fails: ' + html.escape(", ".join(r.get("failed", []))) + "</span>"
    scale = "".join(f'<label><input type="radio" name="{seed}:realism" value="{i}"> {i}</label>' for i in range(1, 11))
    moves = "".join(f'<label><input type="radio" name="{seed}:moves" value="{v}"> {v}</label>' for v in ("yes", "somewhat", "no"))
    j = r.get("judge") or {}
    judge = ""
    if "overall" in j:
        judge = (f"<details><summary>judge's scores</summary><p class='nums'>overall {j['overall']} · shape {j['shape']} · tips {j['tips']} · "
                 f"mechanism {j['mechanism']} · materials {j['materials']} · proportions {j['proportions']}</p><p>{html.escape(j.get('summary', ''))}</p>"
                 f"<ul>{''.join(f'<li>{html.escape(f)}</li>' for f in j.get('flaws', []))}</ul></details>")  # fmt: skip
    video = f'<video src="{rel}/actuation.mp4" controls loop muted playsinline></video>' if (base / rel / "actuation.mp4").exists() else ""
    return f"""<div class="card" data-seed="{seed}">
<b>seed {seed}</b> · {html.escape(str(p.get("style", "")))} · {status}
<div class="media"><img src="{rel}/sheet.jpg" alt="six views of seed {seed}" loading="lazy"><div>{video}<img src="{rel}/actuation.jpg" alt="squeeze frames" loading="lazy"></div></div>
<p class="nums">{html.escape(nums)}</p>
<div class="rate"><span>Looks real (1-10):</span> {scale}</div>
<div class="rate"><span>Moves like real tongs:</span> {moves}</div>
<textarea name="{seed}:notes" placeholder="notes (optional)"></textarea>{judge}
</div>"""


def write_page(out: Path, records: list[dict], summary: dict, family: str) -> None:
    family_words = {"tongs": "kitchen tongs"}.get(family, family)
    cards = "\n".join(_card(r, out) for r in sorted(records, key=lambda r: r["seed"]))
    (out / "index.html").write_text(PAGE.format(family=family_words, n=len(records), gen=html.escape(summary.get("generator", "")),
                                                cards=cards, eval_id=out.name))  # fmt: skip


def fold_ratings(eval_dir: Path, ratings_file: Path) -> dict:
    data = json.loads(ratings_file.read_text())["ratings"]
    real = [int(r["realism"]) for r in data.values() if r.get("realism")]
    moves = [r["moves"] for r in data.values() if r.get("moves")]
    summary = json.loads((eval_dir / "summary.json").read_text())
    summary["people"] = {"rated": len(real), "realism_mean": round(sum(real) / len(real), 2) if real else None,
                         "moves": {k: moves.count(k) for k in ("yes", "somewhat", "no")}}  # fmt: skip
    (eval_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    shutil.copy(ratings_file, eval_dir / "ratings.json")
    return summary["people"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", type=Path, help="a research run's directory (or, with --ratings, an evaluation's)")
    ap.add_argument("--generator", default=OL.FINAL)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--ratings", type=Path, default=None, help="a ratings.json downloaded from the page")
    a = ap.parse_args()
    from chefbot.runtime.models import load_env
    from chefbot.autoresearch.research import REPO

    load_env(REPO / ".env")
    if a.ratings:
        print(json.dumps(fold_ratings(a.target, a.ratings), indent=1))
        return 0
    out = evaluate(a.target, a.generator, a.n, a.jobs, not a.no_judge, a.out)
    print(json.dumps(json.loads((out / "summary.json").read_text()), indent=1))
    print(f"rating page: {out / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
