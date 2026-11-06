"""The executor's memory beyond the window: what acting on things taught (LEARNED BY DOING, after MessyMem's
interaction analyzer), and the kitchen memory kept across runs (run_agent --memory)."""

import json
from pathlib import Path

import yaml

from chefbot.agent.loop.llm import LLMResponse, ToolCall
from chefbot.scripts import run_agent

REPO = Path(run_agent.__file__).resolve().parents[2]


class Calls:
    def __init__(self, calls):
        self.calls, self.i, self.sent = calls, 0, []

    def chat(self, messages, tools=None, **kw):
        last = messages[-1]["content"]
        self.sent.append(last if isinstance(last, str) else "".join(x.get("text", "") for x in last))
        name, args = self.calls[min(self.i, len(self.calls) - 1)]
        self.i += 1
        return LLMResponse(None, [ToolCall(f"c{self.i}", name, json.dumps(args))])


def _run(tmp_path, job, calls, **kw):
    llm = Calls(calls)
    r = run_agent.run(REPO / f"chefbot/jobs/{job}.yaml", 0, tmp_path, llm=llm, operator=lambda x: "abort", quiet=True,
                      env="symbolic", noise="off", **kw)  # fmt: skip
    return llm, r


def test_a_failure_and_what_fixed_it_are_learned_for_the_whole_run(tmp_path):
    calls = [("pick_up", {"id": "egg"}), ("go_to", {"place": "fridge"}), ("detect", {}), ("open", {"id": "fridge handle_1"}),
             ("detect", {"classes": ["egg"]}), ("align", {"id": "egg"}), ("pick_up", {"id": "egg"}),
             ("finish", {"summary": "x", "abandon": True})]  # fmt: skip
    llm, _ = _run(tmp_path, "egg_to_pan", calls)
    assert "LEARNED BY DOING" not in llm.sent[0]
    after_fail = llm.sent[1]
    assert "pick_up egg: unknown_id 1x; not solved yet" in after_fail or "pick_up egg: not_visible 1x; not solved yet" in after_fail
    final = llm.sent[-1]
    learned = next(line for line in final.splitlines() if line.strip().startswith("pick_up egg:"))
    assert "then it worked after go_to, detect, open, align" in learned


def test_the_kitchen_memory_is_saved_and_the_next_run_starts_from_it(tmp_path):
    memory = tmp_path / "kitchen.yaml"
    _, r1 = _run(tmp_path / "a", "seq_mug_B", [("finish", {"summary": "x", "abandon": True})], memory=memory)
    assert memory.exists() and json.loads((Path(r1["run_dir"]) / "start.json").read_text())["memory"]["loaded"] is False
    saved = yaml.safe_load(memory.read_text())
    assert saved["objects"]["mug"]["in"] == "counter_1"  # what this run knew: the env's map
    saved["objects"]["mug"]["in"] = "cabinet_2"  # as if the last run had put it away there
    memory.write_text(yaml.safe_dump(saved))
    llm, r2 = _run(tmp_path / "b", "seq_mug_B", [("finish", {"summary": "x", "abandon": True})], memory=memory)
    assert json.loads((Path(r2["run_dir"]) / "start.json").read_text())["memory"]["loaded"] is True
    kitchen = llm.sent[0].split("KITCHEN", 1)[1]
    cab = next(line for line in kitchen.splitlines() if line.strip().startswith("cabinet_2"))
    assert "mug" in cab


def test_without_memory_the_run_starts_from_the_envs_map(tmp_path):
    llm, _ = _run(tmp_path, "seq_mug_B", [("finish", {"summary": "x", "abandon": True})])
    kitchen = llm.sent[0].split("KITCHEN", 1)[1]
    assert "mug" in next(line for line in kitchen.splitlines() if line.strip().startswith("counter_1"))
