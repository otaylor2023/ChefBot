"""Render a run directory (AGENT_SPEC §11.4).

    python -m chefbot.scripts.view_run runs/<dir>          # report.html + text summary
    python -m chefbot.scripts.view_run runs/<dir> --text   # summary + the timeline as text

Mismatches between what a tool said and what ground truth shows are flagged:
a pick that succeeded with nothing becoming held, a place that succeeded with
the object still held. (The judge's input and verdict for every check are in
the expanded rows.)
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from chefbot.agent.logging.run_log import read_timeline


def load(run_dir: Path) -> tuple[dict, dict, list[dict]]:
    start = json.loads((run_dir / "start.json").read_text()) if (run_dir / "start.json").exists() else {}
    end = json.loads((run_dir / "end.json").read_text()) if (run_dir / "end.json").exists() else {}
    return start, end, read_timeline(run_dir)


def mismatches(timeline: list[dict], observed: bool = True) -> list[tuple[dict, str]]:
    """(record, why) for tool results ground truth contradicts. `observed`: the
    run had an observer (else there is no ground truth to compare with)."""
    if not observed:
        return []
    found = []
    for r in timeline:
        if r["kind"] != "action" or not r["result"]["success"]:
            continue
        if (r.get("data") or {}).get("status") == "moving":
            continue  # woken by an event before the motion ended: nothing to compare yet
        diff = r.get("world_diff") or []
        if r["tool"] == "pick" and not any(d.startswith("+ held:") and "nothing" not in d for d in diff):
            found.append((r, "pick succeeded, nothing became held"))
        if r["tool"] == "place" and not any(d.startswith("+ held: nothing") for d in diff):
            found.append((r, "place succeeded, the object is still held"))
    return found


def _clock(t: float) -> str:
    t = max(0, int(round(t)))
    return f"{t // 60}:{t % 60:02d}"


def _args(args: dict) -> str:
    return ", ".join(f"{k}={json.dumps(v)}" for k, v in args.items())


def summary(start: dict, end: dict, timeline: list[dict]) -> str:
    totals = end.get("totals", {})
    task = start.get("task", "run")
    outcome = end.get("outcome", "unfinished (no end.json)")
    live = (end.get("live_plan") or start.get("reference_plan") or {}).get("steps", [])
    ref_ids = {s["id"] for s in (start.get("reference_plan") or {}).get("steps", [])}
    done = sum(s["status"] == "done" for s in live)
    added = [s["id"] for s in live if s["id"] not in ref_ids]
    skipped = sum(s["status"] == "skipped" for s in live)
    # complete() calls that failed; a monitor's "not yet" polls are not failures.
    checks = [r for r in timeline if r["kind"] == "check" and r.get("source") == "complete"]
    failed_checks = sum(not r["passed"] for r in checks)
    replans = sum(r["kind"] == "action" and r["tool"] == "request_replan" for r in timeline)
    bad = mismatches(timeline, bool(start.get("world")))
    lines = [
        f"{task} · {outcome} · {totals.get('turns', '?')} turns · {_clock(totals.get('sim_s', 0))} sim · "
        f"{_clock(totals.get('wall_s', 0))} wall · ${totals.get('cost_usd', 0):.2f}",
        f"plan: {done}/{len(live)} done · {len(added)} step{'s' if len(added) != 1 else ''} added"
        + (f" ({', '.join(added)})" if added else "")
        + f" · {skipped} skipped · {failed_checks} check failure{'s' if failed_checks != 1 else ''}"
        + f" · {replans} replan request{'s' if replans != 1 else ''}",
        f"mismatches: {len(bad)}" + "".join(f" (turn {r['turn']}: {why})" for r, why in bad[:3]),
    ]
    errors = [r for r in timeline if r["kind"] == "action" and not r["result"]["success"]]
    if errors:
        first = errors[0]
        lines.append(f"first failed action: turn {first['turn']} {first['tool']}({_args(first['args'])}) "
                     f"-> {first['result']['error']}: {first['result']['message']}")  # fmt: skip
    if end.get("summary"):
        lines.append(f"summary: {end['summary']}")
    return "\n".join(lines)


def timeline_text(timeline: list[dict]) -> str:
    out, step = [], object()
    for r in timeline:
        if r["kind"] == "action" and r.get("step") != step:
            step = r.get("step")
            out.append(f"\n{step or '(no step)'}")
        t = _clock(r["t"])
        if r["kind"] == "action":
            ok = "✓" if r["result"]["success"] else f"✗ {r['result']['error']}"
            if r.get("reasoning"):
                out.append(f"  {t:>6}  \"{_one_line(r['reasoning'])}\"")
            line = f"  {t:>6}  {r['tool']}({_args(r['args'])}) {ok} {_one_line(r['result']['message'])}"
            out.append(line)
            for d in r.get("world_diff") or []:
                out.append(f"          world: {d}")
            for d in r.get("plan_diff") or []:
                out.append(f"          plan:  {d}")
        elif r["kind"] == "event":
            out.append(f"  {t:>6}  [{r.get('event', 'event')}] {r.get('message', '')}")
        elif r["kind"] == "check":
            out.append(f"  {t:>6}  [check {r['source']}] {r['step']}: {'passed' if r['passed'] else 'failed'} — {r['reason']}")
        elif r["kind"] in ("reprompt", "llm_error", "limit", "run_end"):
            detail = r.get("error") or r.get("limit") or r.get("outcome") or _one_line(r.get("text") or "")
            out.append(f"  {t:>6}  [{r['kind']}] {detail}")
    return "\n".join(out).lstrip("\n")


def _one_line(text: str, width: int = 140) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 1] + "…"


# -- HTML -----------------------------------------------------------------------

_CSS = """
:root { --bg:#fbfaf7; --fg:#1d1d1b; --muted:#6b6a65; --line:#e4e1d8; --bad:#b3261e; --ok:#2e7d32; --hl:#fff4cc; --card:#ffffff; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#171716; --fg:#ecebe6; --muted:#a3a19a; --line:#34332f; --bad:#ff8a80; --ok:#81c784; --hl:#3d3517; --card:#1f1f1d; } }
body { background:var(--bg); color:var(--fg); font:14px/1.45 -apple-system, system-ui, sans-serif; margin:0 auto; max-width:1100px; padding:16px; }
h1 { font-size:20px; margin:8px 0 } h2 { font-size:16px; margin:28px 0 8px; border-bottom:1px solid var(--line); padding-bottom:4px }
pre, code, .mono { font-family: ui-monospace, Menlo, monospace; font-size:12.5px }
pre { background:var(--card); border:1px solid var(--line); padding:10px; overflow-x:auto; white-space:pre-wrap }
.filters label { margin-right:14px; color:var(--muted) }
.step { margin-top:14px; font-weight:600 }
.row { border-top:1px solid var(--line); padding:4px 0 }
.row summary { cursor:pointer; list-style:none; display:grid; grid-template-columns:56px 1fr; gap:8px }
.row summary::-webkit-details-marker { display:none }
.t { color:var(--muted) } .why { color:var(--muted); font-style:italic }
.bad { color:var(--bad) } .ok { color:var(--ok) } .mismatch { background:var(--hl) }
.diff { color:var(--muted); font-size:12.5px } .kind { color:var(--muted) }
table { border-collapse:collapse } td, th { border:1px solid var(--line); padding:3px 8px; text-align:left; vertical-align:top }
"""

_JS = """
function apply() {
  const errs = document.getElementById('f-errors').checked, mism = document.getElementById('f-mism').checked,
        hide = document.getElementById('f-percep').checked, step = document.getElementById('f-step').value;
  document.querySelectorAll('.row').forEach(r => {
    let show = true;
    if (errs && !r.classList.contains('is-error')) show = false;
    if (mism && !r.classList.contains('mismatch')) show = false;
    if (hide && r.dataset.percep === '1') show = false;
    if (step && r.dataset.step !== step) show = false;
    r.style.display = show ? '' : 'none';
  });
}
document.querySelectorAll('.filters input, .filters select').forEach(e => e.addEventListener('change', apply));
"""


def report_html(start: dict, end: dict, timeline: list[dict]) -> str:
    e = html.escape
    bad = {id(r): why for r, why in mismatches(timeline, bool(start.get("world")))}
    steps = sorted({str(r.get("step")) for r in timeline if r["kind"] == "action" and r.get("step")})
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>ChefBot run</title><style>{_CSS}</style></head><body>",
        f"<h1>{e(start.get('task', 'ChefBot run'))}</h1>",
        f"<pre>{e(summary(start, end, timeline))}</pre>",
        f"<p class='kind'>models: {e(str(start.get('models', {})))} · commit {e(str(start.get('git_commit')))}</p>",
        "<h2>Start state</h2>",
        f"<pre>{e(_plan_text(start.get('reference_plan')))}</pre>",
        f"<pre>{e(chr(10).join(start.get('world', [])) or '(no observer)')}</pre>",
        "<h2>Timeline</h2>",
        "<div class='filters'><label><input type='checkbox' id='f-errors'> errors only</label>"
        "<label><input type='checkbox' id='f-mism'> mismatches only</label>"
        "<label><input type='checkbox' id='f-percep'> hide perception calls</label>"
        "<label>step <select id='f-step'><option value=''>all</option>"
        + "".join(f"<option>{e(s)}</option>" for s in steps)
        + "</select></label></div>",
    ]
    current = object()
    for r in timeline:
        step = str(r.get("step")) if r.get("step") else ""
        if r["kind"] == "action" and r.get("step") != current:
            current = r.get("step")
            parts.append(f"<div class='step'>{e(step or '(no step)')}</div>")
        classes = ["row"]
        if r["kind"] == "action" and not r["result"]["success"]:
            classes.append("is-error")
        if r["kind"] in ("llm_error", "limit"):
            classes.append("is-error")
        if id(r) in bad:
            classes.append("mismatch")
        percep = "1" if r["kind"] == "action" and r["tool"] in ("detect", "nearby", "get_pose", "get_status", "holding") else "0"
        head, body = _row(r, bad.get(id(r)))
        parts.append(
            f"<details class='{' '.join(classes)}' data-step='{e(step)}' data-percep='{percep}'>"
            f"<summary><span class='t mono'>{_clock(r['t'])}</span><span>{head}</span></summary>{body}</details>"
        )
    parts += [
        "<h2>Plan over time</h2>", _plan_chart(timeline, end),
        "<h2>End state</h2>",
        f"<pre>{e(_plan_text(end.get('live_plan')))}</pre>",
        f"<pre>{e(chr(10).join(end.get('world', [])) or '(no observer)')}</pre>",
        "<h2>Edit log</h2><table><tr><th>time</th><th>step</th><th>change</th><th>reason</th></tr>"
        + "".join(f"<tr><td class='mono'>{_clock(x['t'])}</td><td>{e(str(x['step']))}</td><td>{e(x['change'])}</td>"
                  f"<td>{e(str(x.get('reason') or ''))}</td></tr>" for x in end.get("edits", []))  # fmt: skip
        + "</table>",
        f"<script>{_JS}</script></body></html>",
    ]
    return "\n".join(parts)


def _row(r: dict, mismatch: str | None) -> tuple[str, str]:
    e = html.escape
    if r["kind"] == "action":
        ok = r["result"]["success"]
        mark = "<span class='ok'>✓</span>" if ok else f"<span class='bad'>✗ {e(str(r['result']['error']))}</span>"
        why = f"<div class='why'>“{e(_one_line(r['reasoning']))}”</div>" if r.get("reasoning") else ""
        diffs = "".join(f"<div class='diff'>world: {e(d)}</div>" for d in r.get("world_diff") or [])
        diffs += "".join(f"<div class='diff'>plan: {e(d)}</div>" for d in r.get("plan_diff") or [])
        flag = f"<div class='bad'>mismatch: {e(mismatch)}</div>" if mismatch else ""
        head = f"{why}<span class='mono'>{e(r['tool'])}({e(_args(r['args']))})</span> {mark} {e(_one_line(r['result']['message']))}{diffs}{flag}"
        body = f"<pre>{e(json.dumps({k: r[k] for k in ('result', 'data', 'events', 'duration_s', 'wall_s', 'request') if k in r}, indent=2, default=str))}</pre>"
        return head, body
    if r["kind"] == "check":
        head = f"<span class='kind'>[check · {e(r['source'])}]</span> {e(r['step'])}: " + (
            "<span class='ok'>passed</span>" if r["passed"] else "<span class='bad'>failed</span>") + f" — {e(r['reason'])}"
        return head, f"<pre>{e(json.dumps(r, indent=2, default=str))}</pre>"
    label = r.get("event", r["kind"]) if r["kind"] == "event" else r["kind"]
    detail = r.get("message") or r.get("error") or r.get("limit") or r.get("outcome") or r.get("text") or ""
    return f"<span class='kind'>[{e(label)}]</span> {e(_one_line(str(detail)))}", f"<pre>{e(json.dumps(r, indent=2, default=str))}</pre>"


def _plan_text(plan: dict | None) -> str:
    if not plan:
        return "(no plan)"
    lines = [f"GOAL: {plan.get('goal')}"]
    for s in plan.get("steps", []):
        extra = f"  [added: {s['added_because']}]" if s.get("added_because") else ""
        lines.append(f"  {s['id']:<4}{s.get('status', 'pending'):<10}{s['goal']}{extra}")
        if s.get("log"):
            lines.append(f"{'':16}log: {s['log']}")
    return "\n".join(lines)


_COLORS = {"pending": "#c9c6bd", "active": "#3b82f6", "pausing": "#f59e0b", "paused": "#f97316",
           "waiting": "#a855f7", "done": "#22c55e", "failed": "#ef4444", "skipped": "#9ca3af"}  # fmt: skip


def _plan_chart(timeline: list[dict], end: dict) -> str:
    """One bar per step: its status over time, from the plan edits."""
    edits = [r for r in timeline if r["kind"] == "plan_edit" and str(r.get("change", "")).startswith(("status:", "added"))]
    steps = [s["id"] for s in (end.get("live_plan") or {}).get("steps", [])]
    t_end = max([r["t"] for r in timeline] + [1.0])
    width, row_h, left = 900, 18, 50
    rows = []
    for i, step in enumerate(steps):
        status, since = "pending", 0.0
        segments = []
        for r in edits:
            if r.get("step") != step:
                continue
            new = r["change"].split("->")[-1].strip() if "->" in r["change"] else "pending"
            segments.append((since, r["t"], status))
            status, since = new, r["t"]
        segments.append((since, t_end, status))
        y = i * (row_h + 4)
        rows.append(f"<text x='0' y='{y + 13}' font-size='12' fill='currentColor'>{html.escape(step)}</text>")
        for a, b, st in segments:
            x = left + (width - left) * a / t_end
            w = max(1.0, (width - left) * (b - a) / t_end)
            rows.append(f"<rect x='{x:.1f}' y='{y}' width='{w:.1f}' height='{row_h}' fill='{_COLORS.get(st, '#999')}'>"
                        f"<title>{html.escape(step)}: {st} {_clock(a)}–{_clock(b)}</title></rect>")  # fmt: skip
    legend = " ".join(f"<span style='color:{c}'>■</span> {s}" for s, c in _COLORS.items())
    height = max(1, len(steps)) * (row_h + 4)
    return (f"<svg viewBox='0 0 {width} {height}' width='100%' style='max-width:{width}px'>{''.join(rows)}</svg>"
            f"<div class='kind'>{legend} · 0:00–{_clock(t_end)}</div>")  # fmt: skip


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a ChefBot run directory.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--text", action="store_true", help="print the timeline as text too")
    args = parser.parse_args()
    start, end, timeline = load(args.run_dir)
    print(summary(start, end, timeline))
    if args.text:
        print()
        print(timeline_text(timeline))
    out = args.run_dir / "report.html"
    out.write_text(report_html(start, end, timeline))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
