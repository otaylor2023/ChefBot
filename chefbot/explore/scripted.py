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

from chefbot.agent.loop.llm import LLMResponse, ToolCall
from chefbot.agent.world.map import WorldMap


class ScriptedExplorer:
    def __init__(self, start_map: WorldMap, open_labels: set[str], step_id: str = "s1"):
        self._map = start_map
        self._open = open_labels
        self._step = step_id
        self._ids = itertools.count(1)
        self._policy = self._run()
        self._started = False

    def chat(self, messages, tools=None, **kwargs) -> LLMResponse:
        try:
            if self._started:  # the previous call's result is the last tool message
                tool, args = self._policy.send(_last_tool_result(messages))
            else:
                self._started = True
                tool, args = next(self._policy)
        except StopIteration:
            tool, args = "finish", {"summary": "explored", "abandon": True}
        return LLMResponse(None, [ToolCall(f"scripted_{next(self._ids)}", tool, json.dumps(args))])

    def _run(self):
        fixtures = [e for e in self._map.list() if e.kind == "fixture" and e.label != "burner"]
        yield ("set_status", {"step_id": self._step, "status": "active"})
        for entry in fixtures:
            moved = yield ("go_to", {"place": entry.name})
            if not moved.get("success"):
                continue
            yield ("detect", {})
            if entry.label not in self._open:
                continue
            handle = f"{entry.label} handle"
            ids = _ids((yield ("detect", {"classes": [handle]})))
            if not ids:
                continue
            yield ("align", {"id": ids[0]})
            opened = yield ("open", {"id": ids[0]})
            if not opened.get("success"):
                continue
            yield ("detect", {})
            # An open door can swing its handle out of view: close by the fixture itself then.
            ids = _ids((yield ("detect", {"classes": [entry.label, handle]})))
            handles = [i for i in ids if i.startswith(handle)]
            if handles:
                yield ("close", {"id": handles[0]})
            elif entry.name in ids:
                yield ("align", {"id": entry.name})
                yield ("close", {"id": entry.name})
        done = yield ("complete", {"step_id": self._step})
        summary = "explored every fixture" if done.get("success") else f"explored; left: {done.get('message')}"
        yield ("finish", {"summary": summary, "abandon": not done.get("success")})


def _ids(result: dict) -> list[str]:
    return [d["id"] for d in (result.get("data") or {}).get("detections", [])]


def _last_tool_result(messages) -> dict:
    for message in reversed(messages):
        if message.get("role") == "tool":
            return json.loads(message["content"])
    return {}
