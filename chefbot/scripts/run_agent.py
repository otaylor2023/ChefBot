"""Run the agent on a task (AGENT_SPEC §12).

    python -m chefbot.scripts.run_agent --task chefbot/tasks/fried_egg.yaml
    python -m chefbot.scripts.run_agent --task chefbot/tasks/breakfast.yaml --seed 3 --model anthropic/claude-opus-5.5

`--env symbolic` (default) runs in the symbolic kitchen: no simulator, same
agent, tools and run log. Writes runs/<date>_<task>_s<seed>/ with report.html,
and says whether the dish is done by the task's ground-truth `success` rules.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from chefbot.agent.check.checker import StateChecker
from chefbot.agent.config import AgentConfig, LoopConfig
from chefbot.agent.logging.run_log import git_commit
from chefbot.agent.loop.llm import OpenRouterClient
from chefbot.agent.loop.runner import AgentRunner, AgentSession
from chefbot.agent.plan.model import Plan
from chefbot.agent.tools.plan_tools import terminal_operator
from chefbot.agent.world.map import WorldMap
from chefbot.scripts.view_run import load, report_html, summary

REPO = Path(__file__).resolve().parents[2]


def _short(text: str, width: int = 110) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 1] + "…"


def run(
    task_path: Path,
    seed: int = 0,
    out: Path = Path("runs"),
    config: LoopConfig | None = None,
    llm=None,
    judge=None,
    operator=terminal_operator,
    quiet: bool = False,
    noise: str = "task",
) -> dict:
    from chefbot.envs.symbolic.scenario import Scenario

    config = config or LoopConfig.load()
    scenario = Scenario.load(task_path, seed, noise)
    # A long task can raise its own run limits: `limits: {max_turns: 250}` in the task file.
    limits = {k: v for k, v in (scenario.task.get("limits") or {}).items() if hasattr(config, k)}
    if limits:
        import dataclasses

        config = dataclasses.replace(config, **limits)
    plan = Plan.from_dict(scenario.task)
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = out / f"{stamp}_{scenario.name}_s{seed}"
    llm = llm or OpenRouterClient(config.model, temperature=config.temperature, provider=config.provider)
    judge = judge or StateChecker(
        OpenRouterClient(config.judge_model, provider=config.provider), config.judge_model, config.judge_temperature
    )
    world_map = WorldMap.from_dict(scenario.map)
    tools = scenario.task.get("tools") or {}  # a task can withhold tools: {exclude: [...], only: [...]}
    session = AgentSession.build(
        plan, scenario.robot, scenario.detector, world_map, scenario.clock, run_dir, judge, scenario.observer,
        config, AgentConfig.load(), operator=operator,
        tools_exclude=tuple(tools.get("exclude") or ()), tools_only=tuple(tools["only"]) if tools.get("only") else None,
    )  # fmt: skip
    session.bus.add_safety(scenario.safety)
    session.log.start(
        task=scenario.name, task_file=str(task_path), env="symbolic", seed=seed,
        reference_plan=plan.to_dict(), models={"agent": config.model, "judge": config.judge_model},
        tools=[s["function"]["name"] for s in session.registry.schemas],
        config=config.__dict__, git_commit=git_commit(REPO),
        disturbances=[d.spec for d in scenario.disturbances], failures=scenario.robot.rates.__dict__, noise=noise,
    )  # fmt: skip
    for t, what in scenario.applied:
        session.log.record("disturbance", 0, at=t, what=what)
    logged = len(scenario.applied)

    def before_turn(s: AgentSession) -> None:
        nonlocal logged
        scenario.on_plan({step.id: step.status for step in s.store.live.steps})
        for t, what in scenario.applied[logged:]:
            s.log.record("disturbance", s.turn, at=t, what=what)
            if not quiet:
                print(f"         !! disturbance at {t:.0f}s: {what}", flush=True)
        logged = len(scenario.applied)

    def after_action(s: AgentSession, tool: str, args: dict, result, reasoning) -> None:
        scenario.on_tool(tool)
        if quiet:
            return
        mark = "ok " if result.success else f"ERR {result.error}"
        if reasoning:
            print(f"         {_short(reasoning)}", flush=True)
        call = f"{tool}({', '.join(f'{k}={json.dumps(v)}' for k, v in args.items())})"
        print(f"{s.turn:>4} {s.clock.now():>6.0f}s  {call:<48} {mark}  {_short(result.message, 70)}", flush=True)

    def on_limit() -> None:
        for burner in scenario.world.burners.values():
            scenario.world.set_power(burner.name, False)

    runner = AgentRunner(session, llm, on_limit=on_limit, before_turn=before_turn, after_action=after_action)
    outcome = runner.run()
    from chefbot.agent.logging.run_log import read_timeline

    called = [r["tool"] for r in read_timeline(run_dir) if r["kind"] == "action" and r["result"]["success"]]
    verdict = scenario.success(called)
    end_path = run_dir / "end.json"
    end = json.loads(end_path.read_text())
    end["ground_truth"] = verdict
    end["disturbances_applied"] = scenario.applied
    end_path.write_text(json.dumps(end, default=str))
    start, end, timeline = load(run_dir)
    (run_dir / "report.html").write_text(report_html(start, end, timeline))
    text = summary(start, end, timeline)
    if not quiet:
        print("\n" + text)
        print(f"ground truth: {'SUCCESS' if verdict['success'] else 'FAILURE'}"
              + (f" · missing {verdict['missing']}" if verdict["missing"] else "")
              + (f" · forbidden {verdict['forbidden_present']}" if verdict["forbidden_present"] else ""))  # fmt: skip
        print(f"report: {run_dir / 'report.html'}")
    return {"run_dir": str(run_dir), "outcome": outcome[0], "summary_text": text, **verdict, "totals": end.get("totals", {})}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the ChefBot agent on a task.")
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--env", choices=["symbolic"], default="symbolic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=REPO / "runs")
    parser.add_argument("--model")
    parser.add_argument("--judge-model")
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--max-cost", type=float, dest="max_llm_cost_usd")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--noise", choices=["off", "task", "on"], default="task",
                        help="random failures: off (clean run), task (the task file's rates), on (noise.yaml under them)")  # fmt: skip
    parser.add_argument("--operator", choices=["terminal", "continue", "abort"], default="terminal",
                        help="answer to request_replan (non-interactive runs: continue or abort)")  # fmt: skip
    args = parser.parse_args()
    config = LoopConfig.load(model=args.model, judge_model=args.judge_model, max_turns=args.max_turns,
                             max_llm_cost_usd=args.max_llm_cost_usd)  # fmt: skip
    operator = terminal_operator if args.operator == "terminal" else (lambda reason, a=args.operator: a)
    result = run(args.task, args.seed, args.out, config, operator=operator, quiet=args.quiet, noise=args.noise)
    if args.quiet:
        print(json.dumps({k: v for k, v in result.items() if k != "summary_text"}, default=str))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
