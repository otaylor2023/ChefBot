"""One self-contained HTML page per run: source with evidence highlighted, findings, step graph, state trace."""
from __future__ import annotations
import html
import json
from .recipe_pipeline import PipelineResult, Validation

CSS = """
:root{--bg:#fbfaf7;--fg:#1d1c1a;--mut:#6b6760;--card:#fff;--line:#e4e0d8;--stated:#d7ecd9;--inferred:#fbe7c6;
--gap:#c0392b;--err:#b3261e;--errbg:#fbe9e7;--warn:#8a5a00;--warnbg:#fdf3dc;--ok:#1e6b3a;--okbg:#e3f3e8;--code:#f2efe9}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#171614;--fg:#ece9e3;--mut:#a29d94;--card:#201f1c;
--line:#34322e;--stated:#23412a;--inferred:#4a3a17;--gap:#ff7b6b;--err:#ff8a80;--errbg:#3a1f1c;--warn:#f0c060;--warnbg:#3a3018;
--ok:#7fd49a;--okbg:#1c3324;--code:#2a2925}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}h1{font-size:24px;margin:0 0 4px}h2{font-size:17px;margin:32px 0 10px}
.mut{color:var(--mut)}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;overflow-x:auto}
.badge{display:inline-block;padding:2px 10px;border-radius:99px;font-weight:600;font-size:13px}
.valid{background:var(--okbg);color:var(--ok)}.recipe_findings{background:var(--warnbg);color:var(--warn)}
.extraction_errors{background:var(--errbg);color:var(--err)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-top:14px}
.kpi b{display:block;font-size:20px}pre.src{white-space:pre-wrap;margin:0;font:14px/1.7 ui-monospace,Menlo,monospace}
mark{border-radius:3px;padding:1px 0;color:inherit}mark.stated{background:var(--stated)}mark.inferred{background:var(--inferred)}
span.gap{text-decoration:underline wavy var(--gap);text-underline-offset:3px}
table{border-collapse:collapse;width:100%;font-size:13.5px}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mut);font-weight:600}code{background:var(--code);padding:1px 5px;border-radius:4px;font-size:12.5px}
td.err{color:var(--err)}td.warn{color:var(--warn)}.legend span{margin-right:14px}
details summary{cursor:pointer;color:var(--mut)}pre.json{font-size:12px;max-height:420px;overflow:auto}
.mermaid{text-align:center}
"""


def _source_html(v: Validation, text: str) -> str:
    marks = [(*ev.span, ev.prov, oid) for oid, ev in v.recipe.evidenced() if ev.span] if v.recipe else []
    cls = [None] * len(text)
    tip = [""] * len(text)
    for a, b, prov, oid in sorted(marks, key=lambda m: m[1] - m[0], reverse=True):  # short spans paint last
        for i in range(a, min(b, len(text))):
            cls[i] = prov
            tip[i] = oid
    gap = [False] * len(text)
    for a, b in v.gaps:
        for i in range(a, b):
            gap[i] = True
    out, i = [], 0
    while i < len(text):
        j = i
        while j < len(text) and (cls[j], tip[j], gap[j]) == (cls[i], tip[i], gap[i]):
            j += 1
        chunk = html.escape(text[i:j])
        if gap[i]:
            chunk = f'<span class="gap" title="not cited by any element">{chunk}</span>'
        if cls[i]:
            chunk = f'<mark class="{cls[i]}" title="{html.escape(tip[i])}">{chunk}</mark>'
        out.append(chunk)
        i = j
    return "".join(out)


def _mermaid(v: Validation) -> str:
    if not v.recipe:
        return ""
    lines = ["flowchart TD"]
    safe = lambda s: "".join(ch if ch.isalnum() else "_" for ch in s)
    producers = {s.output for s in v.recipe.steps if s.output}
    containers = {c.id for c in v.recipe.containers}
    for s in v.recipe.steps:
        label = s.verb + (f" ({s.style})" if s.style else "") + (f"<br/>→ {s.output}" if s.output else "")
        shape = f'S_{safe(s.id)}["{s.id}: {label}"]' if not s.optional else f'S_{safe(s.id)}(["{s.id}: {label} (optional)"])'
        lines.append("  " + shape)
        for i in s.inputs:
            if i.ref not in producers and i.ref not in containers:
                lines.append(f'  I_{safe(i.ref)}(("{i.ref}")) --> S_{safe(s.id)}')
    data = {(p, s.id) for s in v.recipe.steps for i in s.inputs for p in
            [next((x.id for x in v.recipe.steps if x.output == i.ref), None)] if p}
    for a, b in v.edges:
        arrow = "-->" if (a, b) in data else "-. state .->"
        lines.append(f"  S_{safe(a)} {arrow} S_{safe(b)}")
    return "\n".join(lines)


def _findings_rows(v: Validation) -> str:
    rows = []
    for x in v.extraction_errors:
        rows.append(("err", "extraction error", x.code, x.step, x.message))
    for x in v.recipe_findings:
        rows.append(("warn", "recipe finding", x.code, x.step, x.message))
    if v.engine:
        rows += [("", "unresolved", x.code, x.step, x.message) for x in v.engine.unresolved]
    rows += [("", "warning", x.code, x.step, x.message) for x in v.warnings]
    if not rows:
        return '<tr><td colspan="4" class="mut">None</td></tr>'
    return "".join(f'<tr><td class="{c}">{k}</td><td><code>{html.escape(code)}</code></td><td>{html.escape(step or "")}</td>'
                   f'<td>{html.escape(str(msg))}</td></tr>' for c, k, code, step, msg in rows)


def _steps_rows(v: Validation) -> str:
    if not v.recipe:
        return ""
    out = []
    for s in v.recipe.steps:
        ins = ", ".join(f"{i.ref}" + (f"×{i.fraction}" if i.fraction is not None and i.fraction != 1.0 else "")
                        + (" (req)" if i.mode == "require" else "")
                        + "".join(f" {r.key}={r.value if r.unresolved is None else '?'}" for r in i.required_state)
                        for i in s.inputs)
        until = f"{s.until.kind}: {s.until.target if s.until.target is not None else s.until.raw}" if s.until else ""
        ev = s.evidence
        evs = (f'<mark class="{ev.prov}">{html.escape(ev.quote)}</mark>' if ev.quote else
               f'<span class="mut">{ev.prov}: {html.escape(ev.reason or ev.implied_by or "")}</span>')
        params = {k: v for k, v in s.params.model_dump().items() if v is not None}
        out.append(f"<tr><td>{s.id}{' <span class=mut>(opt)</span>' if s.optional else ''}</td>"
                   f"<td><b>{s.verb}</b>{' · ' + html.escape(s.style) if s.style else ''}"
                   f"{' · ' + html.escape(s.verb_raw) if s.verb_raw else ''}</td>"
                   f"<td>{html.escape(ins)}</td><td>{html.escape(s.output or '')}</td><td>{html.escape(s.container or '')}</td>"
                   f"<td>{html.escape(json.dumps(params)) if params else ''}</td>"
                   f"<td>{html.escape(until)}</td><td>{evs}</td></tr>")
    return "".join(out)


def _rules_rows(v: Validation) -> str:
    if not v.recipe or not v.recipe.rules:
        return '<tr><td colspan="4" class="mut">None extracted</td></tr>'
    return "".join(f"<tr><td>{r.id}</td><td>{r.kind}</td><td>{html.escape(', '.join(r.applies_to) or 'whole recipe')}</td>"
                   f"<td>{html.escape(r.text)}<br><mark class='{r.evidence.prov}'>"
                   f"{html.escape(r.evidence.quote or '')}</mark></td></tr>" for r in v.recipe.rules)


def _uncited_rows(v: Validation, text: str) -> str:
    u = v.uncited(text) if v.recipe else {"action": [], "rule": []}
    rows = [("action", s) for s in u["action"]] + [("rule", s) for s in u["rule"]]
    if not rows:
        return '<tr><td colspan="2" class="mut">None: every uncited sentence looks like commentary</td></tr>'
    return "".join(f"<tr><td class='warn'>{k}</td><td>{html.escape(s)}</td></tr>" for k, s in rows)


def _trace_rows(v: Validation) -> str:
    if not v.engine:
        return ""
    out = []
    for t in v.engine.trace:
        if t["status"] != "ok":
            detail = t.get("violations", "")
        elif "equipment" in t:
            detail = "equipment → " + json.dumps(t["equipment"], default=str)
        else:
            detail = (f"consumed {json.dumps(t.get('consumed'))} → <b>{html.escape(str(t.get('output')))}</b>"
                      f" @ {t.get('location') or '—'}; props {json.dumps(t.get('properties'), default=str)};"
                      f" cooked {t.get('cooked_components')}; contents {json.dumps(t.get('container_contents'))}")
        out.append(f"<tr><td>{t['step']}</td><td>{t['verb']}</td><td>{t['status']}</td><td>{detail}</td></tr>")
    skipped = [s.id for s in v.recipe.steps if s.id not in {t['step'] for t in v.engine.trace}] if v.recipe else []
    if skipped:
        out.append(f'<tr><td colspan="4" class="mut">Not run (optional / unselected choice): {", ".join(skipped)}</td></tr>')
    return "".join(out)


def render(result: PipelineResult, title: str = "Recipe extraction") -> str:
    v = result.final.validation
    src = result.source.text
    s = v.summary()
    u = result.usage_totals()
    attempts = "".join(
        f"<tr><td>{a.kind}</td><td>{html.escape(a.call.usage.model)}</td><td><span class='badge {a.validation.status}'>"
        f"{a.validation.status}</span></td><td>{len(a.validation.extraction_errors)}</td>"
        f"<td>{a.call.usage.input_tokens}/{a.call.usage.output_tokens}</td><td>{a.call.usage.latency_s:.1f}s</td>"
        f"<td>{'✓' if i == result.final_index else ''}</td></tr>" for i, a in enumerate(result.attempts))
    raw = html.escape(result.final.call.raw if len(result.final.call.raw) < 200_000 else result.final.call.raw[:200_000])
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(v.recipe.title if v.recipe else title)}</title><style>{CSS}</style></head><body><main>
<h1>{html.escape(v.recipe.title if v.recipe else title)}</h1>
<div class="mut">Repair: {html.escape(result.repair_decision)}</div>
<div class="kpis">
 <div class="card kpi"><span class="mut">Status</span><b><span class="badge {s['status']}">{s['status'].replace('_', ' ')}</span></b></div>
 <div class="card kpi"><span class="mut">Quotes verified</span><b>{s['grounding_rate']:.0%}</b></div>
 <div class="card kpi"><span class="mut">Instruction coverage</span><b>{s['instruction_coverage']:.0%}</b></div>
 <div class="card kpi"><span class="mut">Steps / edges</span><b>{s['n_steps']} / {s['n_edges']}</b></div>
 <div class="card kpi"><span class="mut">Tokens in / out</span><b>{u['input_tokens']:,} / {u['output_tokens']:,}</b></div>
 <div class="card kpi"><span class="mut">LLM time</span><b>{u['latency_s']}s</b></div>
</div>
<h2>Source evidence</h2>
<div class="legend mut"><span><mark class="stated">stated</mark></span><span><mark class="inferred">inferred</mark></span>
<span><span class="gap">instruction text no element cites</span></span> Hover a highlight to see which element cites it.</div>
<div class="card"><pre class="src">{_source_html(v, src)}</pre></div>
<h2>Findings</h2><div class="card"><table><tr><th>Kind</th><th>Code</th><th>Step</th><th>Detail</th></tr>{_findings_rows(v)}</table></div>
<h2>Possibly omitted</h2><p class="mut">Uncited instruction sentences that look like actions or rules
(heuristic; commentary such as tips and storage notes is left out).</p>
<div class="card"><table><tr><th>Looks like</th><th>Sentence</th></tr>{_uncited_rows(v, src)}</table></div>
<h2>Steps</h2><div class="card"><table><tr><th>Id</th><th>Verb</th><th>Inputs</th><th>Output</th><th>Container</th>
<th>Params</th><th>Until</th><th>Evidence</th></tr>{_steps_rows(v)}</table></div>
<h2>Rules</h2><div class="card"><table><tr><th>Id</th><th>Kind</th><th>Applies to</th><th>Rule / evidence</th></tr>{_rules_rows(v)}</table></div>
<h2>Dependency graph</h2><p class="mut">Solid: dataflow. Dotted: equipment state (e.g. preheat → bake). Unconnected steps can run in parallel.</p>
<div class="card"><pre class="mermaid">{html.escape(_mermaid(v))}</pre></div>
<h2>State trace</h2><div class="card"><table><tr><th>Step</th><th>Verb</th><th>Status</th><th>Effect</th></tr>{_trace_rows(v)}</table></div>
<h2>LLM calls</h2><div class="card"><table><tr><th>Call</th><th>Model</th><th>Result</th><th>Extraction errors</th>
<th>Tokens in/out</th><th>Time</th><th>Used</th></tr>{attempts}</table></div>
<h2>Raw output</h2><details class="card"><summary>Model output used for this report</summary><pre class="json">{raw}</pre></details>
</main>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"></script>
<script>try{{mermaid.initialize({{startOnLoad:true,theme:matchMedia('(prefers-color-scheme: dark)').matches?'dark':'neutral'}})}}catch(e){{}}</script>
</body></html>"""
