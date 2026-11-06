"""A run as a conversation: who said what to whom, in order (conversation.html in the run folder).

    python -m chefbot.scripts.view_conversation runs/<dir>

The user's request, the planner's questions, the answers and the plan it submitted (planner.json); then every loop
iteration: what the harness sent the executor (the TASK STATE; the system prompt and tools once, at the top), what the
executor replied (its words, its thinking if the API returned any, its tool call), the tool's result, and on complete
the judge's input and verdict (requests/, run.jsonl, checks/). Everything comes from the run's own files.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from chefbot.agent.logging.run_log import read_timeline

_CSS = """
:root { --bg:#f6f7f5; --fg:#1c2220; --muted:#5f6a65; --line:#dce1de; --card:#fff; --code:#eef1ef;
  --user:#7a4fb0; --planner:#b9481a; --exec:#1f6fb2; --robot:#2b7a4b; --judge:#a3317a; --harness:#5f6a65; --bad:#b3261e; }
@media (prefers-color-scheme: dark) { :root { --bg:#121614; --fg:#e4eae6; --muted:#9aa5a0; --line:#2c3431; --card:#1a1f1d;
  --code:#222a27; --user:#b996e6; --planner:#ee7d45; --exec:#6cb0ee; --robot:#5cc489; --judge:#e07bbd; --harness:#a3ada8; --bad:#f2867c; } }
body { background:var(--bg); color:var(--fg); font:15px/1.5 -apple-system, system-ui, sans-serif; margin:0 auto; max-width:980px; padding:20px 16px 60px }
h1 { font-size:22px; margin:0 0 4px } h2 { font-size:17px; margin:30px 0 10px }
.muted { color:var(--muted) }
pre { font:12.5px/1.45 ui-monospace, Menlo, monospace; background:var(--code); padding:10px; border-radius:6px; white-space:pre-wrap; word-break:break-word; margin:6px 0 0; max-height:420px; overflow:auto }
.legend span { display:inline-block; margin:0 10px 6px 0; font-weight:600; font-size:13px }
.msg { background:var(--card); border:1px solid var(--line); border-left:5px solid var(--c); border-radius:8px; padding:10px 14px; margin:8px 0 }
.msg.llm { margin-right:10% } .msg.code { margin-left:10% }
.who { font-weight:700; color:var(--c); font-size:13px } .to { color:var(--muted); font-size:13px }
.say { margin-top:4px } .big { font-size:20px; font-weight:600 }
.call { font:600 13.5px ui-monospace, Menlo, monospace; background:var(--code); border-radius:6px; padding:6px 10px; margin-top:6px; word-break:break-word }
.think { font-style:italic; color:var(--muted); margin-top:4px }
.ok { color:var(--robot); font-weight:700 } .bad { color:var(--bad); font-weight:700 }
.turn { margin:22px 0 4px; font:700 12px ui-monospace, Menlo, monospace; color:var(--muted); border-bottom:1px solid var(--line); padding-bottom:3px }
details summary { cursor:pointer; color:var(--muted); font-size:13px; margin-top:4px }
.user { --c:var(--user) } .planner { --c:var(--planner) } .exec { --c:var(--exec) } .robot { --c:var(--robot) }
.judge { --c:var(--judge) } .harness { --c:var(--harness) }
"""

NAMES = {"user": "User", "planner": "Planner (LLM)", "exec": "Executor (LLM)", "judge": "Judge (LLM)",
         "harness": "Harness (code)", "robot": "Robot and tools (code + simulator)"}  # fmt: skip
LLMS = {"planner", "exec", "judge"}
HARNESS_TOOLS = {"complete", "set_status", "add_step", "skip_step", "request_replan", "finish", "set_timer", "cancel_timer", "watch"}


def _e(x) -> str:
    return html.escape(str(x if x is not None else ""))


def _msg(frm: str, to: str | None, body: str) -> str:
    side = "llm" if frm in LLMS else "code"
    arrow = f" <span class='to'>→ {NAMES[to]}</span>" if to else ""
    return f"<div class='msg {side} {frm}'><span class='who'>{NAMES[frm]}</span>{arrow}{body}</div>"


def _fold(label: str, text: str, open_: bool = False) -> str:
    return f"<details{' open' if open_ else ''}><summary>{_e(label)}</summary><pre>{_e(text)}</pre></details>"


def _text(content) -> str:
    return content if isinstance(content, str) else "".join(x.get("text", "") for x in content or [])


def _args(args: dict) -> str:
    return ", ".join(f"{k}={json.dumps(v)}" for k, v in (args or {}).items())


def _plan(plan: dict) -> str:
    lines = [f"GOAL: {plan.get('goal')}"]
    for s in plan.get("steps", []):
        deps = f"   (after {', '.join(s['depends_on'])})" if s.get("depends_on") else ""
        lines.append(f"{s['id']}  {s['goal']}{deps}\n    done_when: {s['done_when']}" + (f"\n    hint: {s['hint']}" if s.get("hint") else ""))
    return "\n".join(lines)


def conversation_html(run_dir: Path) -> str:
    run_dir = Path(run_dir)
    start = json.loads((run_dir / "start.json").read_text()) if (run_dir / "start.json").exists() else {}
    end = json.loads((run_dir / "end.json").read_text()) if (run_dir / "end.json").exists() else {}
    planner = json.loads((run_dir / "planner.json").read_text()) if (run_dir / "planner.json").exists() else None
    timeline = read_timeline(run_dir)
    checks = {r["turn"]: r for r in timeline if r["kind"] == "check"}
    out = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>{_e(start.get('task', 'run'))}: conversation</title><style>{_CSS}</style></head><body>",
           f"<h1>{_e(start.get('task', 'run'))}</h1><p class='muted'>{_e(run_dir.name)} · env {_e((start.get('session') or {}).get('env'))}"
           f" · outcome {_e(end.get('outcome'))} · ground truth {_e('SUCCESS' if (end.get('ground_truth') or {}).get('success') else 'FAILURE' if end.get('ground_truth') else 'n/a')}</p>",
           "<p class='legend'>" + "".join(f"<span class='{k}' style='color:var(--c)'>■ {v}</span>" for k, v in NAMES.items()) + "</p>"]  # fmt: skip

    # 1. the user and the planner
    out.append("<h2>1. The user's request and the planner</h2>")
    if planner:
        out.append(_msg("user", "planner", f"<div class='say big'>“{_e(planner.get('request'))}”</div>"))
        for call in planner.get("calls", []):
            if call.get("kind") == "replan":
                out.append(_msg("exec", "planner", f"<div class='call'>request_replan(reason={_e(json.dumps(call.get('reason')))})</div>"))
            answers = dict((q, a) for q, a in call.get("questions") or [])
            for t in call.get("transcript", []):
                a = t.get("args", {})
                if t["tool"] == "ask_user":
                    out.append(_msg("planner", "user", f"<div class='call'>ask_user(question={_e(json.dumps(a.get('question')))})</div>"))
                    out.append(_msg("user", "planner", f"<div class='say big'>“{_e(answers.get(a.get('question')) or '(nobody to ask)')}”</div>"))
                elif t["tool"] == "submit_plan":
                    out.append(_msg("planner", "harness", f"<div class='call'>submit_plan(goal={_e(json.dumps(a.get('goal')))}, steps=[{len(a.get('steps', []))} steps])</div>"
                                    + _fold("the plan it submitted", _plan(a), open_=True)))  # fmt: skip
                else:
                    out.append(_msg("planner", "harness", f"<div class='call'>{_e(t['tool'])}({_e(_args(a))})</div>"))
            n_back = len(call.get("rejected") or [])
            verdict = (f"cannot_do: {call['refused']}" if call.get("refused") else
                       f"plan accepted ({n_back} sent back first by the check)" if call.get("plan") else "no plan")  # fmt: skip
            out.append(_msg("harness", None, f"<div class='say'>{_e(verdict)}</div>"))
    else:
        plan = start.get("reference_plan") or {}
        out.append(_msg("harness", None, "<div class='say'>No user request: this run used the job file's hand-written plan.</div>"
                        + _fold("the plan", _plan(plan), open_=True)))  # fmt: skip

    # 2. the executor's prompt, once
    reqs = sorted((run_dir / "requests").glob("*.json"))
    if reqs:
        first = json.loads(reqs[0].read_text())
        sys_text = _text(first["messages"][0]["content"])
        tools = [t.get("function", t) for t in first.get("tools") or []]
        out.append("<h2>2. What the executor is told once (the same every turn)</h2>")
        out.append(_msg("harness", "exec", _fold(f"system prompt ({len(sys_text.splitlines())} lines)", sys_text)
                        + _fold(f"tools ({len(tools)})", "\n\n".join(f"{t['name']}({', '.join(t.get('parameters', {}).get('properties', {}))})\n  {t.get('description')}" for t in tools))))  # fmt: skip

    # 3. the loop
    out.append("<h2>3. The loop: one iteration per turn</h2>")
    for r in timeline:
        if r["kind"] != "action":
            continue
        req = json.loads((run_dir / r["request"]).read_text()) if r.get("request") and (run_dir / r["request"]).exists() else None
        reply = ((req or {}).get("response") or {}).get("choices", [{}])[0].get("message", {}) if req else {}
        state = _text(req["messages"][-1]["content"]) if req else ""
        window = sum(m["role"] == "tool" for m in req["messages"]) if req else 0
        out.append(f"<div class='turn'>turn {r['turn']}{' · step ' + _e(r['step']) if r.get('step') else ''} · sim {_e(round(r['t']))} s</div>")
        if req:
            out.append(_msg("harness", "exec", f"<div class='say'>system prompt + tools + the last {window} turns + this TASK STATE:</div>" + _fold("TASK STATE", state)))
        said = (reply.get("content") or r.get("reasoning") or "").strip()
        thought = (reply.get("reasoning") or "").strip()
        body = (f"<div class='say'>“{_e(said)}”</div>" if said else "") + (f"<div class='think'>thinking: {_e(thought)}</div>" if thought else "")
        out.append(_msg("exec", "harness" if r["tool"] in HARNESS_TOOLS else "robot", body + f"<div class='call'>{_e(r['tool'])}({_e(_args(r['args']))})</div>"))
        ck = checks.get(r["turn"])
        if ck:
            cfile = run_dir / ck["file"] if ck.get("file") else None
            cdata = json.loads(cfile.read_text()) if cfile and cfile.exists() else {}
            out.append(_msg("harness", "judge", _fold(f"CONDITION: {ck['condition']}  (and the simulator's STATE)",
                                                      f"CONDITION\n{ck['condition']}\n\nSTATE\n{cdata.get('observation', '')}")))  # fmt: skip
            out.append(_msg("judge", "harness", f"<div class='call'>{_e(cdata.get('judge_reply') or json.dumps({'passed': ck['passed'], 'reason': ck['reason']}))}</div>"))
        res = r["result"]
        mark = "<span class='ok'>ok</span>" if res["success"] else f"<span class='bad'>{_e(res['error'])}</span>"
        out.append(_msg("harness" if r["tool"] in HARNESS_TOOLS else "robot", "exec", f"<div class='say'>{mark} {_e(res['message'])}</div>"))

    out.append("<h2>4. The end</h2>")
    gt = end.get("ground_truth") or {}
    out.append(_msg("harness", None, f"<div class='say'><b>{_e(end.get('outcome'))}</b>: {_e(end.get('summary'))}</div>"
                    f"<div class='say'>ground truth: {'SUCCESS' if gt.get('success') else 'FAILURE'}"
                    + (f" · missing {_e(gt.get('missing'))}" if gt.get("missing") else "")
                    + (f" · forbidden {_e(gt.get('forbidden_present'))}" if gt.get("forbidden_present") else "") + "</div>"))  # fmt: skip
    out.append("</body></html>")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a ChefBot run as a conversation (conversation.html).")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    out = args.run_dir / "conversation.html"
    out.write_text(conversation_html(args.run_dir))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
