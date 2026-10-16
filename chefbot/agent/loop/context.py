"""Each turn's messages (AGENT_SPEC §9).

    [1] system prompt     static for the run
    [2] tool schemas      static, passed alongside (not a message)
    [3] rolling window    the last N tool calls with their results and reasoning
    [4] task state        always the last message; only the newest copy is kept

[1] and [2] never change within a run, so providers can cache them. The task
state is built from the agent's own records only: the plan, timers and monitors,
the latest perception, what the robot reports. Never from the Observer.
"""

from __future__ import annotations

import json
from pathlib import Path

from chefbot.agent.loop.perception import Perception
from chefbot.agent.plan.model import ACTIVE, DONE, FAILED, PAUSED, PAUSING, PENDING, SKIPPED, WAITING
from chefbot.agent.plan.store import PlanStore
from chefbot.agent.world.kitchen import SEEN, YOU
from chefbot.agent.world.map import label_of
from chefbot.interface.types import MotionStatus, ToolResult

SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "system.md"
# Extra sections, included when the registry offers their tools.
PROMPT_SECTIONS = {"squeeze": "handheld.md", "stir": "handheld.md", "pour": "handheld.md"}


def system_prompt(tool_names: set[str]) -> str:
    """The system prompt for a run: the base plus a section for each kind of tool offered."""
    text = SYSTEM_PROMPT_PATH.read_text()
    for name in dict.fromkeys(f for tool, f in PROMPT_SECTIONS.items() if tool in tool_names):
        text += (SYSTEM_PROMPT_PATH.parent / name).read_text()
    return text
REPROMPT = "Respond with exactly one tool call."


class ContextBuilder:
    def __init__(self, system_prompt: str, window_n: int = 8, cache: bool = True):
        # Marked for prompt caching (Anthropic via OpenRouter): tools + system prompt are the static prefix.
        content = [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}] if cache else system_prompt
        self._system = {"role": "system", "content": content}
        self._window_n = window_n
        self._turns: list[list[dict]] = []  # each: assistant message (+ tool result or re-prompt)

    def add_turn(self, messages: list[dict]) -> None:
        self._turns.append(messages)

    def build(self, task_state: str) -> list[dict]:
        window = [m for turn in self._turns[-self._window_n :] for m in turn]
        return [self._system, *window, {"role": "user", "content": task_state}]


def tool_message(call_id: str, result: ToolResult) -> dict:
    """A tool result as the model sees it."""
    body = {"success": result.success, "error": result.error, "message": result.message}
    if result.data:
        body["data"] = result.data
    return {"role": "tool", "tool_call_id": call_id, "content": json.dumps(body, default=str)}


class TaskState:
    """Renders [4]. `bus` is the EventBus (for timers and monitors), `nav` the
    NavTools (what is held), `robot` the RobotAPI (motion status, pose)."""

    def __init__(self, store: PlanStore, bus, perception: Perception, nav, robot, world_map, clock, time_tools=None):
        self._store, self._bus, self._perception = store, bus, perception
        self._nav, self._robot, self._map, self._clock = nav, robot, world_map, clock
        self._time = time_tools
        # Unproductive calls over the whole run (the window only keeps the last few):
        # key -> [count, last time, what happened].
        self._fruitless: dict[str, list] = {}
        # go_to visits since the last progress (a pick, place, open/close, turn on/off or completed step).
        self._visits: dict[str, int] = {}

    def record(self, tool: str, args: dict, result: ToolResult) -> None:
        """After every tool call: count calls that got nowhere, so loops show up in the task state."""
        if result.success and tool in _PROGRESS:
            self._visits.clear()
        elif result.success and tool == "go_to":
            self._visits[args.get("place", "?")] = self._visits.get(args.get("place", "?"), 0) + 1
        call = f"{tool}({', '.join(f'{k}={json.dumps(v)}' for k, v in args.items() if k != 'wait')})"
        if not result.success:
            outcome = f"failed: {result.error}"
        elif tool == "detect" and not result.data.get("detections"):
            outcome = "found nothing"
        else:
            # Progress resets the count of the same call (e.g. a detect that now finds something).
            self._fruitless.pop(call, None)
            return
        entry = self._fruitless.setdefault(call, [0, 0.0, outcome])
        entry[0] += 1
        entry[1], entry[2] = self._clock.now(), outcome

    def render(self, events: list) -> str:
        plan = self._store.live
        lines = ["TASK STATE", "", f"GOAL: {plan.goal}", "", "PLAN"]
        next_pending = self._next_pending()
        for step in plan.steps:
            lines += self._step_lines(step, step.status in (ACTIVE, WAITING, PAUSING) or step.id == next_pending)
        lines += ["", "NOW", *("  " + line for line in self._now_lines())]
        lines += ["", self._perception.block(self._robot.get_base_pose(), held=self._held_id())]
        missing = getattr(self._nav, "not_found", {})
        if missing:
            lines.append("  not where the map says: " + ", ".join(
                f"{name} (map: {self._map.get(name).inside if self._map.get(name) else '?'}; looked {n}x)"
                for name, n in missing.items()))  # fmt: skip
        searched = getattr(self._nav, "searched", {})
        if searched:
            places = [e.name for e in self._map.list() if e.kind == "fixture" and e.label != "burner"]
            lines += ["", "SEARCHING (asked detect for it, not found)"]
            for label, looked in searched.items():
                left = [p for p in places if p not in looked]
                lines.append(f"  {label}: looked near {', '.join(sorted(looked)) or 'nowhere yet'} · "
                             + (f"not yet: {', '.join(left)}" if left else "looked everywhere on the map"))  # fmt: skip
        repeated = [(k, v) for k, v in self._fruitless.items() if v[0] >= 3]
        circling = [(p, n) for p, n in self._visits.items() if n >= 3]
        if repeated or circling:
            lines += ["", "REPEATED WITHOUT PROGRESS (whole run)"]
            for call, (count, last, outcome) in sorted(repeated, key=lambda kv: -kv[1][0])[:6]:
                lines.append(f"  {call} {outcome} x{count}, last {_duration(self._clock.now() - last)} ago")
            if circling:
                lines.append("  went to " + ", ".join(f"{p} {n}x" for p, n in sorted(circling, key=lambda pn: -pn[1]))
                             + " since your last progress (pick, place, open/close, turn on/off, complete)")  # fmt: skip
        lines += ["", *self._kitchen_lines()]
        lines += ["", "EVENTS SINCE LAST TURN"]
        lines += [f"  [{e.kind}] {e.message}" for e in events] or ["  none"]
        return "\n".join(lines)

    def _step_lines(self, step, show_done_when: bool) -> list[str]:
        head = f"  {step.id:<4}{step.status:<10}{step.goal}"
        if step.added_because:
            head += f"   [added: {step.added_because}]"
        if step.status == PENDING and step.depends_on:
            head += "   depends: " + ", ".join(step.depends_on)
        detail = []
        if step.status == DONE and step.log:
            detail.append(f"log: {step.log}")
        if step.status in (FAILED, SKIPPED) and step.reason:
            detail.append(f"reason: {step.reason}")
        if step.status == PAUSED:
            resume = next((n for n in reversed(step.notes) if n.startswith("resume:")), None)
            if resume:
                detail.append(resume)
        if show_done_when:
            detail.append(f"done_when: {step.done_when}")
        if step.status in (ACTIVE, WAITING, PAUSING, PAUSED):
            done = [n.removeprefix("did: ") for n in step.notes if n.startswith("did:")]
            if done:
                detail.append("done so far: " + "; ".join(done[-4:]) + (f" (+{len(done) - 4} earlier)" if len(done) > 4 else ""))
        if step.hint and step.status in (PENDING, ACTIVE):
            detail.append(f"hint: {step.hint}")
        extras = []
        if step.status in (WAITING, ACTIVE, PAUSING):
            since = self._since(step.id, step.status)
            if since is not None:
                extras.append(f"{step.status} {_duration(since)}")
        extras += [self._timer(t) for t in self._bus.timers.values() if t.step_id == step.id]
        if step.status not in (DONE, SKIPPED, FAILED):
            fired = self._went_off(step.id)
            extras += fired
            unwatched = step.id not in self._bus.monitors and not any(t.step_id == step.id for t in self._bus.timers.values())
            if step.status == WAITING and unwatched and not fired:
                extras.append("NO TIMER OR MONITOR: nothing will tell you when it is ready")
        monitor = self._bus.monitors.get(step.id)
        if monitor is not None:
            extras.append(f"monitor every {_duration(monitor.every_s)}")
        if step.status in (ACTIVE, WAITING, PAUSING):
            failed = next((n for n in reversed(step.notes) if n.startswith("check failed:")), None)
            if failed:
                extras.append(failed)
        if extras:
            detail.append(" · ".join(extras))
        return [head] + [" " * 16 + d for d in detail]

    # -- KITCHEN: the map as a list of places and what is on / in them ---------------

    def _kitchen_lines(self) -> list[str]:
        nav, belief, now = self._nav, self._nav.kitchen, self._clock.now()
        gone = nav.missing()
        entries = [e for e in self._map.list() if e.name not in gone]
        fixtures = [e for e in entries if e.kind == "fixture"]
        holder_of = {e.name: (belief.where[e.name].value if e.name in belief.where else e.inside)
                     for e in entries if e.kind == "object"}  # fmt: skip
        held_id = self._held_id()
        if held_id:  # what is in the held tool is off the map but still known
            for obj in nav.held_contents(held_id):
                holder_of.setdefault(obj, belief.where[obj].value)
        children: dict[str, list[str]] = {}
        for obj, holder in holder_of.items():
            children.setdefault(holder or "", []).append(obj)

        def obj_text(obj: str, depth: int = 0) -> str:
            text = obj
            fact = belief.where.get(obj)
            if fact is not None and fact.source == YOU:
                text += f" (you put it here {_duration(now - fact.t)} ago)"
            elif fact is not None and fact.source == SEEN:
                text += f" (seen here {_duration(now - fact.t)} ago)"
            inner = children.get(obj, [])
            if inner and depth < 3:
                text += " · " + ("inside" if label_of(obj) in _CONTAINERS else "on") + ": "
                text += ", ".join(obj_text(o, depth + 1) for o in inner)
            return text

        def fixture_text(name: str) -> str:
            text = name
            fact = belief.states.get(name)
            if fact is not None:
                ago = _duration(now - fact.t)
                text += (f" ({_DID[fact.value]} by you {ago} ago)" if fact.source == YOU
                         else f" (seen {fact.value} {ago} ago)")  # fmt: skip
            elif label_of(name) in _OPENABLE:
                text += " (not opened yet)"
            return text

        held = self._held_id()
        use = getattr(nav, "tool_use", {})
        how = [f"lowered into {use['lowered_into']}"] if use.get("lowered_into") else []
        how += ["squeezed"] if use.get("squeezing") else []
        holding = (obj_text(held) + (f"   [{', '.join(how)}]" if how else "")) if held else "nothing"
        lines = ["KITCHEN  (what you know: from the map, and from what you have seen and done)",
                 f"  holding: {holding}"]  # fmt: skip
        burners = [f for f in fixtures if f.label == "burner"]
        quiet = []
        for fixture in fixtures:
            if fixture.label == "burner":
                continue
            contents = ", ".join(obj_text(o) for o in children.get(fixture.name, []))
            parts = burners if fixture.label == "stove" else []
            if not contents and not parts and fixture.name not in belief.states and label_of(fixture.name) not in _OPENABLE:
                quiet.append(fixture.name)
                continue
            lines.append(_row("  ", fixture_text(fixture.name), contents))
            idle = []
            for burner in parts:
                stuff = ", ".join(obj_text(o) for o in children.get(burner.name, []))
                fact = belief.states.get(burner.name)
                if stuff or (fact is not None and fact.value == "on"):
                    lines.append(_row("    ", fixture_text(burner.name), stuff))
                else:
                    idle.append(burner.name)
            if idle:
                lines.append("    " + ", ".join(idle) + "   (off, nothing on them)")
        if burners and not any(f.label == "stove" for f in fixtures):
            lines.append("  " + ", ".join(fixture_text(b.name) for b in burners))
        if quiet:
            lines.append("  " + ", ".join(quiet) + "   (nothing known on them)")
        known = {f.name for f in fixtures} | set(holder_of) | {"", held_id}
        stray = [o for o, h in holder_of.items() if h not in known]
        if stray:
            lines.append("  elsewhere: " + ", ".join(obj_text(o) for o in stray))
        if gone:
            lines.append("  missing: " + ", ".join(
                f"{n} (map said {self._map.get(n).inside if self._map.get(n) else '?'}; not found there)" for n in sorted(gone)))  # fmt: skip
        return lines

    def _held_id(self) -> str | None:
        held = self._nav.held
        return held["id"] if held else None

    def _now_lines(self) -> list[str]:
        lines = [f"holding: {self._held_id() or 'nothing'}"]
        if self._robot.get_status().status == MotionStatus.MOVING:
            name = getattr(self._time, "motion_name", None) or "a motion"
            lines.append(f"motion: {name} in progress")
        else:
            lines.append("motion: none")
        loose = [t for t in self._bus.timers.values() if t.step_id is None]
        if loose:
            lines.append("timers: " + ", ".join(self._timer(t) for t in loose))
        fired = [e for e in self._bus.delivered if e.kind == "timer" and e.step_id is None
                 and self._clock.now() - e.time < 600]  # fmt: skip
        if fired:
            lines.append("timers gone off: " + ", ".join(
                f"'{e.data.get('timer')}' {_duration(self._clock.now() - e.time)} ago" for e in fired))  # fmt: skip
        alerts = [e for e in self._bus.delivered if e.kind == "safety" and self._clock.now() - e.time < 600]
        if alerts:
            lines.append("recent alerts: " + "; ".join(
                f"{e.message} ({_duration(self._clock.now() - e.time)} ago)" for e in alerts[-3:]))  # fmt: skip
        return lines

    def _went_off(self, step_id: str | None) -> list[str]:
        """Timers and monitor results for this step that already fired: they stay
        visible until the step is finished (the event itself is shown only once)."""
        out = []
        for event in self._bus.delivered:
            if event.step_id != step_id or event.kind not in ("timer", "monitor"):
                continue
            ago = _duration(self._clock.now() - event.time)
            if event.kind == "timer":
                out.append(f"TIMER '{event.data.get('timer')}' WENT OFF {ago} ago")
            elif event.data.get("passed"):
                out.append(f"MONITOR PASSED {ago} ago")
            else:
                out.append(f"monitor timed out {ago} ago")
        return out

    def _timer(self, timer) -> str:
        return f"timer '{timer.name}' {_duration(max(0.0, timer.due - self._clock.now()))} left"

    def _since(self, step_id: str, status: str) -> float | None:
        for edit in reversed(self._store.edits):
            if edit.step == step_id and edit.change.endswith(f"-> {status}"):
                return self._clock.now() - edit.t
        return None

    def _next_pending(self) -> str | None:
        done = {s.id for s in self._store.live.steps if s.status in (DONE, SKIPPED)}
        for step in self._store.live.steps:
            if step.status == PENDING and all(d in done for d in step.depends_on):
                return step.id
        return None


def _row(indent: str, place: str, contents: str, width: int = 30) -> str:
    """A KITCHEN line: the place, then what is on / in it in a column (or after a gap)."""
    head = indent + place
    if not contents:
        return head
    return f"{head:<{width}}  {contents}" if len(head) < width else f"{head}   {contents}"


_PROGRESS = {"pick", "place", "open", "close", "turn_on", "turn_off", "complete"}
_OPENABLE = {"fridge", "cabinet", "microwave", "oven", "drawer", "dishwasher"}
_CONTAINERS = {"fridge", "cabinet", "microwave", "drawer", "oven", "toaster", "pan", "pot", "bowl", "mug", "sink", "egg cracker"}
_DID = {"open": "opened", "closed": "closed", "on": "turned on", "off": "turned off"}


def _duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"
