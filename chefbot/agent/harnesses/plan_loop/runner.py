"""The agent loop (AGENT_SPEC §8): one tool call per turn until finish or a limit.

`AgentSession.build` wires the pieces: plan store, event bus, checker, the tool
registry with the loop's tools, context, run log. The robot, detector, map,
clock and observer come from outside (in-process stack or remote), as do the
LLM clients.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from chefbot.agent.check.checker import CheckResult, StateChecker
from chefbot.agent.config import AgentConfig, LoopConfig
from chefbot.agent.events.bus import SAFETY, EventBus
from chefbot.agent.logging.run_log import RunLog, world_diff
from chefbot.agent.harnesses.plan_loop.context import REPROMPT, ContextBuilder, TaskState, tool_message
from chefbot.agent.harnesses.plan_loop.context import system_prompt as build_system_prompt
from chefbot.agent.loop.llm import LLMError, parse_arguments
from chefbot.agent.harnesses.plan_loop.perception import FreshnessConfig, Perception
from chefbot.agent.plan.model import Plan
from chefbot.agent.plan.store import PlanStore
from chefbot.agent.tools.plan_tools import Operator, PlanTools, terminal_operator
from chefbot.agent.tools.registry import ToolRegistry
from chefbot.agent.tools.time_tools import TimeTools
from chefbot.agent.world.map import WorldMap
from chefbot.interface.clock import Clock
from chefbot.interface.detector import Detector
from chefbot.interface.observer import Observer
from chefbot.interface.robot_api import RobotAPI
from chefbot.interface.types import ToolResult


@dataclass
class Totals:
    turns: int = 0
    tool_calls: int = 0
    errors: int = 0
    reprompts: int = 0
    llm_errors: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    checks: int = 0
    check_failures: int = 0
    wall_s: float = 0.0
    sim_s: float = 0.0


@dataclass
class AgentSession:
    """Everything one run needs, wired together."""

    registry: ToolRegistry
    store: PlanStore
    bus: EventBus
    time_tools: TimeTools
    plan_tools: PlanTools
    perception: Perception
    task_state: TaskState
    context: ContextBuilder
    robot: RobotAPI
    clock: Clock
    log: RunLog
    config: LoopConfig
    totals: Totals = field(default_factory=Totals)
    turn: int = 0

    @classmethod
    def build(
        cls,
        plan: Plan,
        robot: RobotAPI,
        detector: Detector,
        world_map: WorldMap,
        clock: Clock,
        run_dir: str | Path,
        checker: StateChecker | None,
        observer: Observer | None,
        config: LoopConfig | None = None,
        agent_config: AgentConfig | None = None,
        operator: Operator = terminal_operator,
        system_prompt: str | None = None,
        tools_exclude: tuple[str, ...] = (),
        tools_only: tuple[str, ...] | None = None,
        prompts_dir: str | Path | None = None,
        agent_tools: tuple[str, ...] | None = None,
        robot_skills: list | None = None,
        skill_tools: list = (),
        reload: Callable[[], None] | None = None,
        narrate: bool = False,
    ) -> "AgentSession":
        config = config or LoopConfig.load()
        log = RunLog(run_dir, clock, observer, config.log_full_requests)
        holder: dict = {}

        def check(step_id: str, condition: str) -> CheckResult:
            session: AgentSession = holder["session"]
            observation = observer.observe() if observer is not None else None
            if checker is None or observation is None:
                result = CheckResult(False, "no state checker in this run")
            else:
                result = checker.check(condition, observation)
            session.totals.checks += 1
            session.totals.check_failures += not result.passed and not session._in_wait  # complete() only
            log.check(session.turn, step_id, condition, getattr(observation, "text", None), result,
                      source="monitor" if session._in_wait else "complete")  # fmt: skip
            return result

        store = PlanStore(plan, clock.now, on_edit=lambda edit: log.plan_edit(holder["session"].turn, edit))
        bus = EventBus(clock, check)
        registry = ToolRegistry(
            robot, detector, world_map, agent_config or AgentConfig.load(),
            exclude=tuple(config.tools_exclude) + tuple(tools_exclude),
            only=tuple(tools_only) if tools_only is not None else (tuple(config.tools_only) if config.tools_only else None),
            agent_tools=agent_tools, robot_skills=robot_skills,
        )  # fmt: skip
        time_tools = TimeTools(robot, bus, clock, store)
        def open_doors() -> list[str]:
            states = registry.tools.kitchen.states
            return sorted(n for n, f in states.items() if f.value == "open" and f.source == "you")

        plan_tools = PlanTools(store, check, bus, operator, reload, open_doors)
        registry.add_loop_tools(plan_tools, time_tools)
        for schema, handler in skill_tools:  # chefbot-v1 skills: (schema, handler running an episode)
            registry.add_tool(schema, handler)
        registry.tools.now = clock.now  # the kitchen belief's "N ago" is in run time
        perception = Perception(FreshnessConfig(
            config.stale_after_calls, config.stale_move_m, config.stale_turn_deg,
            config.clear_move_m, config.clear_turn_deg,
        ))  # fmt: skip
        task_state = TaskState(store, bus, perception, registry.tools, robot, world_map, clock, time_tools)
        prompt = system_prompt if system_prompt is not None else build_system_prompt({s["function"]["name"] for s in registry.schemas}, prompts_dir, narrate)
        context = ContextBuilder(prompt, config.window_n)
        session = cls(registry, store, bus, time_tools, plan_tools, perception, task_state, context, robot, clock, log, config)
        holder["session"] = session
        return session

    _in_wait: bool = False


class AgentRunner:
    def __init__(
        self,
        session: AgentSession,
        llm,
        on_limit: Callable[[], None] | None = None,
        wall_clock: Callable[[], float] = time.monotonic,
        before_turn: Callable[["AgentSession"], None] | None = None,
        after_action: Callable[["AgentSession", str, dict, ToolResult, str | None], None] | None = None,
        before_action: Callable[["AgentSession", str, dict, str | None], None] | None = None,
    ):
        self.s = session
        self._llm = llm
        self._on_limit = on_limit  # e.g. turn the burners off
        self._before_turn = before_turn  # test harnesses: disturbances
        self._after_action = after_action  # (session, tool, args, result, reasoning): printing, disturbances
        self._before_action = before_action  # (session, tool, args, reasoning): video captions
        self._wall = wall_clock
        self._wall_start = wall_clock()
        self._last_error: str | None = None
        self._same_error = 0
        self._reprompts = 0

    def run(self) -> tuple[str, str]:
        """Turns until finish, abort or a limit. Returns (outcome, summary)."""
        s = self.s
        outcome: tuple[str, str] | None = s.plan_tools.outcome  # already set: the run ends before a turn (refused)
        try:
            while outcome is None:
                limit = self._limit_reached()
                if limit is not None:
                    outcome = self._stop_for_limit(limit)
                    break
                s.turn += 1
                s.totals.turns = s.turn
                self._turn()
                outcome = s.plan_tools.outcome
        except KeyboardInterrupt:
            s.robot.stop()
            outcome = ("interrupted", "stopped by the operator")
        except Exception as exc:  # leave a log behind, then re-raise
            s.robot.stop()
            s.log.record("run_end", s.turn, outcome="crashed", summary=repr(exc))
            self._write_end(("crashed", repr(exc)))
            raise
        s.log.record("run_end", s.turn, outcome=outcome[0], summary=outcome[1])
        self._write_end(outcome)
        return outcome

    # -- one turn -----------------------------------------------------------

    def _turn(self) -> None:
        s = self.s
        if self._before_turn is not None:
            self._before_turn(s)
        s.time_tools.poll_motion()
        s._in_wait = True
        s.bus.poll()  # monitors due while the agent was thinking
        s._in_wait = False
        events = s.bus.drain()
        for event in events:
            s.log.event(s.turn, event)
        messages = s.context.build(s.task_state.render(events))
        tools = s.registry.schemas
        try:
            reply = self._llm.chat(messages, tools=tools, parallel_tool_calls=False)
        except LLMError as exc:
            s.totals.llm_errors += 1
            s.log.record("llm_error", s.turn, error=str(exc))
            self._count_error(f"llm_error: {exc}")
            return
        s.totals.prompt_tokens += reply.prompt_tokens
        s.totals.completion_tokens += reply.completion_tokens
        s.totals.cost_usd += reply.cost_usd
        request = s.log.request(s.turn, messages, tools, reply)

        if not reply.tool_calls:
            self._reprompts += 1
            s.totals.reprompts += 1
            s.log.record("reprompt", s.turn, text=reply.text, request=request)
            s.context.add_turn([reply.assistant_message(), {"role": "user", "content": REPROMPT}])
            if self._reprompts > s.config.max_reprompts:
                self._count_error("no_tool_call")
            return
        self._reprompts = 0

        call = reply.tool_calls[0]
        step = s.store.active()
        world_before = s.log.world()
        edits_before = len(s.store.edits)
        started, wall_started = s.clock.now(), self._wall()
        try:
            args = parse_arguments(call)
        except (ValueError, json.JSONDecodeError) as exc:
            args, result = {}, ToolResult(False, "invalid_arguments", f"{call.name}: arguments are not valid JSON ({exc})")
        else:
            s._in_wait = call.name == "wait" or call.name in _MOTION_TOOLS
            if self._before_action is not None:
                self._before_action(s, call.name, args, reply.text)
            try:
                result = s.registry.dispatch(call.name, args)
            finally:
                s._in_wait = False
        s.totals.tool_calls += 1
        took = s.clock.now() - started
        if took >= 1.0 and isinstance(result.data, dict):
            result.data["took"] = f"{int(took) // 60}:{int(took) % 60:02d}"  # so the agent learns how long things take
        s.perception.update(call.name, args, result, s.robot.get_base_pose())
        s.task_state.record(call.name, args, result)
        delivered = result.data.get("events", []) if isinstance(result.data, dict) else []

        tool_messages = [tool_message(call.id, result)]
        for extra in reply.tool_calls[1:]:
            tool_messages.append(tool_message(extra.id, ToolResult(
                False, "one_call_per_turn", f"Only the first tool call ({call.name}) ran. Call {extra.name} next turn if you still need it.",
            )))  # fmt: skip
        s.context.add_turn([reply.assistant_message(), *tool_messages])

        plan_diff = [f"{e.step}: {e.change}" for e in s.store.edits[edits_before:]]
        s.log.action(
            s.turn,
            step=args.get("step_id") or (step.id if step else None),
            reasoning=reply.text,
            tool=call.name,
            args=args,
            result={"success": result.success, "error": result.error, "message": result.message},
            data=result.data,
            world_diff=world_diff(world_before, s.log.world()),
            plan_diff=plan_diff,
            events=delivered,
            duration_s=round(s.clock.now() - started, 3),
            wall_s=round(self._wall() - wall_started, 3),
            extra_calls=[c.name for c in reply.tool_calls[1:]],
            request=request,
        )  # fmt: skip
        active = s.store.active()
        if result.success and active is not None and call.name in _NOTED and result.data.get("status") != "moving":
            # The step keeps a record of what was done for it (the window forgets).
            target = f'("{args["id"]}")' if "id" in args else f"({args['seconds']:g} s)" if "seconds" in args else "()"
            s.store.add_note(active.id, f"did: {call.name}{target} -> {result.message.split(': ', 1)[-1][:90]}")
        elif not result.success and active is not None and call.name in _NOTED | {"align"}:
            # Failed attempts too (BUMBLE's execution record): what was tried for the step, and why it failed.
            target = f'("{args["id"]}")' if "id" in args else "()"
            s.store.add_note(active.id, f"tried: {call.name}{target} -> {result.error}: {result.message.split(': ', 1)[-1][:80]}")
        s.plan_tools.note_action(call.name, result.success)
        if self._after_action is not None:
            self._after_action(s, call.name, args, result, reply.text)
        if result.success:
            self._last_error, self._same_error = None, 0
        else:
            s.totals.errors += 1
            self._count_error(f"{call.name}: {result.error}")

    # -- limits -------------------------------------------------------------

    def _count_error(self, key: str) -> None:
        if key == self._last_error:
            self._same_error += 1
        else:
            self._last_error, self._same_error = key, 1

    def _limit_reached(self) -> str | None:
        s, cfg = self.s, self.s.config
        if s.turn >= cfg.max_turns:
            return f"max_turns ({cfg.max_turns})"
        if s.totals.cost_usd >= cfg.max_llm_cost_usd:
            return f"max_llm_cost_usd ({cfg.max_llm_cost_usd})"
        if s.clock.now() >= cfg.max_sim_time_s:
            return f"max_sim_time_s ({cfg.max_sim_time_s})"
        if self._wall() - self._wall_start >= cfg.max_wall_time_s:
            return f"max_wall_time_s ({cfg.max_wall_time_s})"
        if self._same_error >= cfg.max_consecutive_errors:
            return f"max_consecutive_errors ({self._last_error} x{self._same_error})"
        return None

    def _stop_for_limit(self, limit: str) -> tuple[str, str]:
        s = self.s
        event = s.bus.post(SAFETY, f"run limit reached: {limit}", data={"limit": limit})
        s.log.event(s.turn, event)
        s.log.record("limit", s.turn, limit=limit)
        s.robot.stop()
        if self._on_limit is not None:
            self._on_limit()
        return ("limit_reached", limit)

    def _write_end(self, outcome: tuple[str, str]) -> None:
        s = self.s
        s.totals.wall_s = round(self._wall() - self._wall_start, 2)
        s.totals.sim_s = round(s.clock.now(), 2)
        s.log.end(
            outcome=outcome[0],
            summary=outcome[1],
            live_plan=s.store.live.to_dict(),
            reference_plan=s.store.reference.to_dict(),
            edits=[e.to_dict() for e in s.store.edits],
            totals=s.totals.__dict__,
        )
        s.log.close()


# Successful calls that change the kitchen: noted on the active step.
_NOTED = {"pick_up", "place", "open", "close", "turn_on", "turn_off", "lower_into", "squeeze", "stir", "pour"}
# Tools during which a monitor check counts as "monitor" in the log (time passes inside them).
_MOTION_TOOLS = {"go_to", "align", "pick_up", "place", "open", "close", "turn_on", "turn_off"}
