"""Talk to the kitchen agent in the terminal and watch it work, live (after HW1's chat.py).

    python -m chefbot.scripts.chat [--task chefbot/jobs/fried_sausage.yaml] [--env symbolic|robocasa] [--video]
                                   [--state first|all|none] [--quiet-prompts]

You type the request (you>). Every message then prints as it happens, by who sends it:
the system prompts and the user message the planner and the executor get, the planner's reply in words and the plan
it submits, its questions (you answer them), and each turn of the loop: the TASK STATE the harness sends, the
executor's reply in words and its tool call, the judge's verdict on a step, the robot's result. Both LLMs run with
--narrate: they say what they are doing and why before each call. The run is saved as usual (runs/<run>/, with
conversation.html).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from chefbot import agents
from chefbot.agent.check.checker import StateChecker
from chefbot.agent.loop.llm import OpenRouterClient
from chefbot.scripts import run_agent

_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
_CODES = {"bold": "1", "dim": "2", "red": "31", "green": "32", "yellow": "33", "blue": "34", "magenta": "35", "cyan": "36"}
WHO = {"you": ("magenta", "bold"), "planner": ("yellow", "bold"), "executor": ("blue", "bold"), "judge": ("magenta",),
       "robot": ("green",), "harness": ("dim",)}  # fmt: skip


def paint(text, *styles: str) -> str:
    if not _COLOR or not styles:
        return str(text)
    return "".join(f"\033[{_CODES[s]}m" for s in styles) + str(text) + "\033[0m"


def say(who: str, text: str) -> None:
    lines = str(text).rstrip().splitlines() or [""]
    print(f"{paint(who + '>', *WHO[who])} {lines[0]}")
    for line in lines[1:]:
        print(" " * (len(who) + 2) + line)
    sys.stdout.flush()


def block(title: str, text: str) -> None:
    print(paint(f"\n──── {title} " + "─" * max(4, 70 - len(title)), "dim"))
    print(paint(text.rstrip(), "dim"))
    print(paint("─" * 76, "dim"), flush=True)


def _text(content) -> str:
    return content if isinstance(content, str) else "".join(x.get("text", "") for x in content or [])


def _call(name: str, arguments: str) -> str:
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return f"{name}({arguments})"
    return f"{name}({', '.join(f'{k}={json.dumps(v)}' for k, v in args.items())})"


def _plan(arguments: str) -> str:
    try:
        a = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return arguments
    lines = [f"goal: {a.get('goal')}"]
    for s in a.get("steps") or []:
        deps = f"  (after {', '.join(s['depends_on'])})" if s.get("depends_on") else ""
        lines.append(f"  {s.get('id')}  {s.get('goal')}{deps}\n      done_when: {s.get('done_when')}"
                     + (f"\n      hint: {s['hint']}" if s.get("hint") else ""))  # fmt: skip
    return "\n".join(lines)


class PlannerView:
    """Wraps the planner's LLM: prints what it is sent and what comes out."""

    def __init__(self, llm, show_prompts: bool):
        self.llm, self.show_prompts, self.calls = llm, show_prompts, 0

    def chat(self, messages, tools=None, **kw):
        if self.calls == 0 and self.show_prompts:
            block("PLANNER · system prompt", _text(messages[0]["content"]))
            block("PLANNER · user message (the request, what the robot knows, what it can do)", _text(messages[1]["content"]))
        elif messages[-1].get("role") == "tool" and messages[-1].get("content"):
            say("harness", messages[-1]["content"])
        self.calls += 1
        r = self.llm.chat(messages, tools=tools, **kw)
        if r.text:
            say("planner", r.text.strip())
        for c in r.tool_calls:
            if c.name == "submit_plan":
                say("planner", "submit_plan(...)\n" + _plan(c.arguments))
            elif c.name != "ask_user":  # the question prints when it is put to you
                say("planner", _call(c.name, c.arguments))
        return r


class ExecutorView:
    """Wraps the executor's LLM: each turn prints the last tool result, the TASK STATE, and what comes out."""

    def __init__(self, llm, show_prompts: bool, state: str):
        self.llm, self.show_prompts, self.state, self.turn = llm, show_prompts, state, 0

    def chat(self, messages, tools=None, **kw):
        if self.turn == 0 and self.show_prompts:
            block("EXECUTOR · system prompt (sent every turn, with the tools)", _text(messages[0]["content"]))
            block(f"EXECUTOR · tools ({len(tools or [])})", "\n".join(f"{t['function']['name']}: {t['function']['description'][:110]}" for t in tools or []))
        prev = messages[-2] if len(messages) >= 2 else {}
        if prev.get("role") == "tool":
            try:
                res = json.loads(prev["content"])
                say("robot", ("ok  " if res.get("success") else f"{res.get('error')}  ") + str(res.get("message")))
            except (json.JSONDecodeError, TypeError):
                say("robot", prev["content"])
        self.turn += 1
        state = _text(messages[-1]["content"])
        print(paint(f"\n── turn {self.turn} " + "─" * 66, "dim"))
        if self.state == "all" or (self.state == "first" and self.turn == 1):
            block("EXECUTOR · user message: TASK STATE", state)
        else:
            active = next((l.strip() for l in state.splitlines() if " active " in l), None)
            say("harness", "TASK STATE sent" + (f" · {active}" if active else ""))
        r = self.llm.chat(messages, tools=tools, **kw)
        if r.text and r.text.strip():
            say("executor", r.text.strip())
        for c in r.tool_calls:
            say("executor", "→ " + _call(c.name, c.arguments))
        return r


class JudgeView:
    def __init__(self, llm):
        self.llm = llm

    def chat(self, messages, **kw):
        cond = _text(messages[-1]["content"]).split("\n\nSTATE", 1)[0].replace("CONDITION\n", "")
        say("harness", f"asks the judge: does \"{cond}\" hold? (with the simulator's state)")
        r = self.llm.chat(messages, **kw)
        say("judge", (r.text or "").strip())
        return r


def ask(question: str) -> str:
    say("planner", f"ask_user: {question}")
    try:
        return input(paint("\nyou> ", *WHO["you"])).strip()
    except EOFError:
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Talk to the kitchen agent and watch it work.")
    parser.add_argument("--task", type=Path, default=Path("chefbot/jobs/fried_sausage.yaml"),
                        help="the kitchen and its success check (default: fried_sausage)")
    parser.add_argument("--env", choices=["symbolic", "robocasa"], default="symbolic")
    parser.add_argument("--video", action="store_true", help="robocasa: record runs/<run>/video.mp4")
    parser.add_argument("--state", choices=["first", "all", "none"], default="first", help="which TASK STATEs to print in full")
    parser.add_argument("--quiet-prompts", action="store_true", help="don't print the system prompts")
    parser.add_argument("--prompt", help="the request (otherwise you type it)")
    args = parser.parse_args()

    agent = agents.load(agents.DEFAULT)
    config = agent.loop_config()
    task = __import__("yaml").safe_load(args.task.read_text())
    print(paint(f"ChefBot · {task.get('name', args.task.stem)} kitchen · {args.env}", "bold"))
    print(paint(f"(the job's own request was: \"{task.get('prompt')}\"; its success check: {task.get('goal')})", "dim"))
    request = args.prompt or input(paint("\nyou> ", *WHO["you"])).strip()
    if not request:
        return 0
    show = not args.quiet_prompts
    planner = PlannerView(OpenRouterClient(config.model, temperature=config.temperature, provider=config.provider), show)
    executor = ExecutorView(OpenRouterClient(config.model, temperature=config.temperature, provider=config.provider), show, args.state)
    judge = StateChecker(JudgeView(OpenRouterClient(config.judge_model, provider=config.provider)), config.judge_model, config.judge_temperature)
    r = run_agent.run(args.task, 0, Path("runs"), llm=executor, planner_llm=planner, judge=judge, user=ask, quiet=True,
                      env=args.env, video=args.video, plan_from="prompt", prompt=request, narrate=True,
                      operator=lambda reason: "abort")  # fmt: skip
    print()
    say("harness", f"run over: {r['outcome']} · {r['summary_text'].splitlines()[-1] if r.get('summary_text') else ''}")
    gt = "SUCCESS" if r.get("success") else "FAILURE" if r.get("success") is False else "not judged"
    say("harness", f"ground truth: {gt}" + (f" · missing {r['missing']}" if r.get("missing") else ""))
    run_dir = Path(r["run_dir"])
    print(paint(f"\nsaved: {run_dir}/conversation.html  (and report.html{', video.mp4' if args.video else ''})", "dim"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
