"""ScriptedRearranger: after exploring, move things around, and keep the map right.

It answers the agent loop's chat() like a model would (as ScriptedExplorer
does), one tool call per turn, and sees only what a model sees: the map it
starts from (the explored map) and the tool results. Each move: go_to an
object, detect, align, pick (one retry), drive to some other fixture on the
way, go_to a random spot along a surface, detect, align, place (one retry;
with no room there, another surface, then back where it came from), and
detect again to look at where it landed (or fell).

Nothing here reads the true map. The map the agent ends with is scored
against the true map made after the run (chefbot.scripts.rearrange).
"""

from __future__ import annotations

import copy
import itertools
import json
import random
from dataclasses import dataclass, field

from chefbot.agent.check.checker import CheckResult
from chefbot.agent.loop.llm import LLMResponse, ToolCall
from chefbot.agent.world.map import WorldMap, label_of
from chefbot.interface.observer import Observation, Observer


def initial_map(start: dict, explored: dict) -> dict:
    """What the agent knows after exploring: the fixtures it was given (with their
    footprints) and the objects the explored map holds."""
    out = copy.deepcopy(start)
    for name, spec in (explored.get("fixtures") or {}).items():
        out["fixtures"].setdefault(name, copy.deepcopy(spec))
    out["objects"] = copy.deepcopy(explored.get("objects") or {})
    return out


@dataclass
class Move:
    object: str
    source: str | None  # where the agent's map had it
    target: str  # where it was put down (or last tried)
    picked: bool = False
    pick_attempts: int = 0
    placed: bool = False
    place_attempts: int = 0
    dropped: bool = False  # the gripper was empty after a failed place
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class ScriptedRearranger:
    def __init__(self, start_map: WorldMap, moves: int, seed: int = 0, surfaces: tuple[str, ...] = ("counter", "table", "island"),
                 skip_in: tuple[str, ...] = (), detour: bool = True, look_after_place: bool = True,
                 step_id: str = "s1"):  # fmt: skip
        """`surfaces`: fixture labels things are put down on. `skip_in`: fixture labels whose
        objects are left alone (behind a door: the fridge, cabinets)."""
        self._map = start_map
        self._n = moves
        self._rng = random.Random(seed)
        self._surfaces = [e for e in start_map.list() if e.kind == "fixture" and e.label in surfaces]
        self._skip_in = set(skip_in)
        self._detour = detour
        self._look = look_after_place
        self._step = step_id
        self.moves: list[Move] = []
        self._ids = itertools.count(1)
        self._policy = self._run()
        self._started = False

    def chat(self, messages, tools=None, **kwargs) -> LLMResponse:
        try:
            if self._started:
                tool, args = self._policy.send(_last_tool_result(messages))
            else:
                self._started = True
                tool, args = next(self._policy)
        except StopIteration:
            tool, args = "finish", {"summary": "rearranged", "abandon": True}
        return LLMResponse(None, [ToolCall(f"scripted_{next(self._ids)}", tool, json.dumps(args))])

    def _movable(self, where: dict[str, str | None]) -> list[str]:
        fixtures = {e.name: e for e in self._map.list() if e.kind == "fixture"}
        return sorted(name for name, holder in where.items()
                      if holder in fixtures and fixtures[holder].label not in self._skip_in)  # fmt: skip

    def _run(self):
        yield ("set_status", {"step_id": self._step, "status": "active"})
        # Where the agent believes each object is: the start map, then where it put things.
        where = {e.name: e.inside for e in self._map.list() if e.kind == "object"}
        failed: set[str] = set()  # objects that would not come up: not tried again
        for _ in range(self._n):
            candidates = [o for o in self._movable(where) if o not in failed]
            if not candidates or not self._surfaces:
                break
            obj = self._rng.choice(candidates)
            targets = [s for s in self._surfaces if s.name != where[obj]] or self._surfaces
            target = self._rng.choice(targets)
            move = Move(obj, where[obj], target.name)
            self.moves.append(move)
            yield from self._pick(move)
            if not move.picked:
                failed.add(obj)
                continue
            where[obj] = None
            if self._detour:
                others = [e for e in self._map.list() if e.kind == "fixture" and e.label != "burner"
                          and e.name not in (target.name, move.source)]  # fmt: skip
                if others:
                    r = yield ("go_to", {"place": self._rng.choice(others).name})
                    _note(move, "detour", r)
            # No room there: another surface, at most two more, then back where it came from.
            rest = [s for s in self._surfaces if s.name not in (target.name, move.source)]
            self._rng.shuffle(rest)
            source = [e for e in self._map.list() if e.name == move.source]
            for surface in [target, *rest[:2], *source]:
                yield from self._place(move, surface)
                if move.placed or move.dropped:
                    break
            if move.placed:
                where[obj] = move.target
            elif not move.dropped:
                where[obj] = None  # still holding it: nothing more can be picked
                break
        done = yield ("complete", {"step_id": self._step})
        yield ("finish", {"summary": f"{len(self.moves)} moves", "abandon": not done.get("success")})

    def _pick(self, move: Move):
        moved = yield ("go_to", {"place": move.object})
        if not _note(move, "go_to", moved):
            return
        for _ in range(2):
            found = yield ("detect", {"classes": [label_of(move.object)]})
            ids = _ids(found)
            if not ids:
                move.errors.append("pick: not seen")
                return
            id = move.object if move.object in ids else ids[0]
            if not _note(move, "align", (yield ("align", {"id": id}))):
                continue
            move.pick_attempts += 1
            picked = yield ("pick", {"id": id})
            if _note(move, "pick", picked):
                move.picked = True
                return

    def _place(self, move: Move, target):
        move.target = target.name
        along = None
        if target.size is not None and target.size[0] > 1.0:
            along = round(self._rng.uniform(-0.8, 0.8), 2)
        for attempt in range(2):
            where = {"place": target.name, **({"along": along} if along is not None and attempt == 0 else {})}
            if not _note(move, "go_to", (yield ("go_to", where))):
                continue
            if target.name not in _ids((yield ("detect", {"classes": [target.label]}))):
                move.errors.append(f"place: {target.name} not seen")
                continue
            if not _note(move, "align", (yield ("align", {"id": target.name}))):
                continue
            move.place_attempts += 1
            placed = yield ("place", {"id": target.name})
            if _note(move, "place", placed):
                move.placed = True
                if self._look:
                    yield ("detect", {"classes": [label_of(move.object)]})
                return
            held = ((yield ("holding", {})).get("data") or {}).get("holding")
            if held is None:
                move.dropped = True  # it fell somewhere near: look for it, and the map follows
                yield ("detect", {"classes": [label_of(move.object)]})
                return


class RearrangeObserver(Observer):
    """What complete() sees: how many moves were made (nothing about the world)."""

    def __init__(self, policy: ScriptedRearranger):
        self._policy = policy

    def observe(self) -> Observation:
        done = sum(m.placed for m in self._policy.moves)
        return Observation(text=f"{done} of {len(self._policy.moves)} moves done", extra={})


class RearrangeChecker:
    def check(self, condition: str, observation: Observation) -> CheckResult:
        return CheckResult(True, observation.text)


def _note(move: Move, what: str, result: dict) -> bool:
    if not result.get("success"):
        move.errors.append(f"{what}: {result.get('error')}")
        return False
    return True


def _ids(result: dict) -> list[str]:
    return [d["id"] for d in (result.get("data") or {}).get("detections", [])]


def _last_tool_result(messages) -> dict:
    for message in reversed(messages):
        if message.get("role") == "tool":
            return json.loads(message["content"])
    return {}
