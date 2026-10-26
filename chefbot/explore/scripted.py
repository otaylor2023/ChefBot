"""ScriptedExplorer: a fixed exploration policy in place of the LLM.

It answers the agent loop's chat() like a model would, one tool call per turn,
and sees only what a model sees: the map it was given and the tool results.
Visit every fixture, detect, and for a fixture with a door: find its handle,
align, open, detect, close. A baseline for the tools, the coverage check and
the scorer, and for what an LLM explorer should reach.
"""

from __future__ import annotations

import itertools
import json
import math
import re

from chefbot.agent.loop.llm import LLMResponse, ToolCall
from chefbot.agent.world.map import WorldMap


class ScriptedExplorer:
    def __init__(self, start_map: WorldMap, open_labels: set[str], step_id: str = "s1",
                 storage_max_bottom: float = math.inf, approach: dict | None = None):  # fmt: skip
        """`approach` (distance, per_stop): look closer at objects detect says it saw only small."""
        self._map = start_map
        self._approach = approach
        self._small: dict[str, str] = {}  # place -> label: seen only small, not yet looked at closer
        self._approached: set[str] = set()
        self.approaches = {"stops": 0, "unreachable": 0}
        self._max_bottom = storage_max_bottom  # cabinets higher than this cannot be looked into
        self._open = open_labels
        self._step = step_id
        self._ids = itertools.count(1)
        self._policy = self._run()
        self._started = False

    def chat(self, messages, tools=None, **kwargs) -> LLMResponse:
        try:
            if self._started:  # the previous call's result is the last tool message
                result = _last_tool_result(messages)
                self._note_small(result)
                tool, args = self._policy.send(result)
            else:
                self._started = True
                tool, args = next(self._policy)
        except StopIteration:
            tool, args = "finish", {"summary": "explored", "abandon": True}
        return LLMResponse(None, [ToolCall(f"scripted_{next(self._ids)}", tool, json.dumps(args))])

    def _run(self):
        fixtures = [e for e in self._map.list() if e.kind == "fixture" and e.label != "burner"]
        storage = [e for e in fixtures if e.z is not None and e.z[0] <= self._max_bottom]  # cabinets, drawers
        yield ("set_status", {"step_id": self._step, "status": "active"})
        for entry in (e for e in fixtures if e.z is None):
            yield from self._visit(entry)
            yield from self._look_closer()
        yield from self._storage(storage)
        done = yield ("complete", {"step_id": self._step})

        # Long fixtures (an island) may have ends not yet in view: complete names them.
        for _ in range(2):
            if done.get("success"):
                break
            for place, part, along in _PART_HINT.findall(done.get("message") or ""):
                where = {"part": part} if part else {"along": float(along)}
                moved = yield ("go_to", {"place": place, **where})
                if moved.get("success"):
                    yield ("detect", {})
            done = yield ("complete", {"step_id": self._step})
        summary = "explored every fixture" if done.get("success") else f"explored; left: {done.get('message')}"
        yield ("finish", {"summary": summary, "abandon": not done.get("success")})

    def _visit(self, entry):
        """A fixture: go to it and look; behind its door, if it has one."""
        moved = yield ("go_to", {"place": entry.name})
        if not moved.get("success"):
            return
        yield ("detect", {})
        if entry.label not in self._open:
            return
        handle = f"{entry.label} handle"
        ids = _ids((yield ("detect", {"classes": [handle]})))
        if not ids:
            return
        yield ("align", {"id": ids[0]})
        opened = yield ("open", {"id": ids[0]})
        if not opened.get("success"):
            return
        yield ("detect", {})
        # An open door can swing its handle out of view: close by the fixture itself then.
        ids = _ids((yield ("detect", {"classes": [entry.label, handle]})))
        handles = [i for i in ids if i.startswith(handle)]
        if handles:
            yield ("close", {"id": handles[0]})
        elif entry.name in ids:
            yield ("align", {"id": entry.name})
            yield ("close", {"id": entry.name})

    def _storage(self, storage):
        """Cabinets and drawers, one stop per group of neighbours: at each, one door at a time
        (open, detect to look inside, close), so no open door is in the way of the next."""
        left = [e for e in storage if e.label in self._open]
        while left:
            stop = left[0]
            group = [e for e in left if math.dist(e.location, stop.location) <= _GROUP_RADIUS and e.facing == stop.facing]
            left = [e for e in left if e not in group]
            moved = yield ("go_to", {"place": stop.name})
            if not moved.get("success"):
                continue
            for entry in group:
                classes = [f"{entry.label} handle"]
                handle = _handle_of((yield ("detect", {"classes": classes})), entry.name)
                if handle is None:
                    continue
                yield ("align", {"id": handle})
                if not (yield ("open", {"id": handle})).get("success"):
                    yield ("close", {"id": handle})  # an open that stopped part-way leaves it ajar, in the way
                    continue
                yield ("detect", {})
                # Close it again: by its name (the handle it was opened with), else by whichever
                # handle here shows open (a swung door's).
                if (yield ("close", {"id": entry.name})).get("success"):
                    continue
                yield ("align", {"id": entry.name})  # the open door moved its handle out of reach
                if (yield ("close", {"id": entry.name})).get("success"):
                    continue
                for _ in range(3):
                    found = (yield ("detect", {"classes": classes})).get("data", {}).get("detections", [])
                    mine = [d["id"] for d in found if d.get("of") == entry.name] + [d["id"] for d in found if d.get("state") in ("open", "ajar")]
                    if not mine:
                        break
                    if (yield ("close", {"id": mine[0]})).get("success"):
                        break
                    yield ("align", {"id": mine[0]})
            yield from self._look_closer()

    def _note_small(self, result: dict) -> None:
        """From a detect's result: objects it saw only small join the list; seen well, they leave it."""
        if self._approach is None:
            return
        for d in (result.get("data") or {}).get("detections", []):
            place = d.get("place") or d.get("id")
            if d.get("seen_small") and place not in self._approached:
                self._small[place] = d["label"]
            elif not d.get("seen_small"):
                self._small.pop(place, None)

    def _look_closer(self):
        """approach: stand nearer to up to per_stop of the objects seen only small, and look again
        (the detector's tracking and name vote merge what it sees)."""
        if self._approach is None:
            return
        for _ in range(int(self._approach.get("per_stop", 2))):
            if not self._small:
                return
            place = next(iter(self._small))
            self._small.pop(place)
            self._approached.add(place)
            moved = yield ("go_to", {"place": place, "distance": float(self._approach.get("distance", 0.7))})
            if not moved.get("success"):
                self.approaches["unreachable"] += 1
                continue
            self.approaches["stops"] += 1
            yield ("detect", {})


_GROUP_RADIUS = 0.6  # m: cabinets and drawers this close to a stop's are opened and looked into together


def _handle_of(result: dict, fixture: str) -> str | None:
    """The detected handle that belongs to `fixture` (detect says which, when there are several like it)."""
    handles = [d for d in (result.get("data") or {}).get("detections", []) if d.get("of") == fixture]
    return handles[0]["id"] if handles else None


_PART_HINT = re.compile(r'go_to\("([^"]+)", (?:part="(left|right)"|along=(-?[0-9.]+))\)')


def _ids(result: dict) -> list[str]:
    return [d["id"] for d in (result.get("data") or {}).get("detections", [])]


def _last_tool_result(messages) -> dict:
    for message in reversed(messages):
        if message.get("role") == "tool":
            return json.loads(message["content"])
    return {}
