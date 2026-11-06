"""Offline tests: no API keys, no network. Fake Ollama / OpenRouter servers serve the gold extractions.

  cd recipe_planning && python -m pytest tests -q
"""
from __future__ import annotations
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from recipe_ir.actions import feasibility, instructions, load_skills, mentioned_tools  # noqa: E402
from recipe_ir.adapters import load_source  # noqa: E402
from recipe_ir.corpus import features, select  # noqa: E402
from recipe_ir.recipe_pipeline import run_pipeline, validate  # noqa: E402
from recipe_ir.schema import Recipe  # noqa: E402

GOLD = {p.name.removesuffix(".gold.json"): p for p in (HERE / "recipes" / "gold").glob("*.gold.json")}


def gold_validation(name):
    src = load_source(str(HERE / "recipes" / f"{name}.txt"))
    return validate(Recipe.model_validate_json(GOLD[name].read_text()), src), src


# ------------------------------------------------------------------ pipeline + ordering
@pytest.mark.parametrize("name,status", [("scrambled_eggs", "valid"), ("onion_egg", "recipe_findings"),
                                         ("blueberry_muffins", "valid")])
def test_gold_status(name, status):
    v, _ = gold_validation(name)
    assert v.status == status


def test_turn_off_waits_for_cooking():
    v, _ = gold_validation("scrambled_eggs")
    assert ("s6", "s7") in v.edges     # remove from heat only after the eggs are cooked (write after read)


# ------------------------------------------------------------------ action-ready instructions
def test_instructions_scrambled_eggs():
    v, _ = gold_validation("scrambled_eggs")
    ins = instructions(v)
    by = {s["id"]: s for s in ins["steps"]}
    assert ins["stages"][-1] == ["s7", "s9"] or "s7" in ins["stages"][-1]
    assert by["s6"]["done_when"]["sensed_by"] == "camera"
    assert {"tool": "spatula", "how": "mentioned"} in by["s6"]["tools"]
    assert by["s6"]["equipment"][0]["state"] == {"heat": "low"}
    assert by["s7"]["equipment"][0]["state"] == {}               # heat removed
    assert set(by["s6"]["after"]) == {"s3", "s5"}
    assert all(s["instruction"] for s in ins["steps"])
    assert ins["counts"]["main_path"] == 7                      # optional chives steps are off the main path


def test_open_questions_and_findings_surface():
    v, _ = gold_validation("onion_egg")
    ins = instructions(v)
    assert any("raw" in q for q in ins["open_questions"])       # recipe never cooks the egg
    assert any("heat" in q for s in ins["steps"] for q in s["open_questions"])


def test_feasibility_is_swappable(tmp_path):
    v, _ = gold_validation("scrambled_eggs")
    ins = instructions(v)
    before = json.dumps(ins, sort_keys=True, default=str)
    sim = feasibility(ins, load_skills("chefbot_sim"))
    everything = {"name": "all", "version": 1, "verbs": {k: {"status": "supported"} for k in
                                                         ["crack", "mix", "heat", "melt", "transfer", "cook", "turn_off"]}}
    full = feasibility(ins, everything)
    assert json.dumps(ins, sort_keys=True, default=str) == before   # profiles never touch the instructions
    assert full["fully_supported"] and not sim["fully_supported"]
    assert sim["supported"] + sim["partial"] + sim["unsupported"] == sim["steps"] == 7


def test_tool_matching():
    assert mentioned_tools("Add 1 tablespoon butter and a teaspoon of salt") == []
    assert mentioned_tools("Stir with a wooden spoon") == ["wooden spoon"]
    assert mentioned_tools("Line with parchment; use a whisk") == ["parchment", "whisk"]


# ------------------------------------------------------------------ fake model servers
class _Fake(BaseHTTPRequestHandler):
    mode = "ok"          # ok | truncate_once
    calls = 0

    def log_message(self, *a):
        pass

    def _gold_for(self, text):
        for name, p in GOLD.items():
            g = json.loads(p.read_text())
            if g["title"] in text:
                return p.read_text()
        raise AssertionError("no gold for prompt")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).calls += 1
        user = body["messages"][-1]["content"]
        content = self._gold_for(user)
        truncated = type(self).mode == "truncate_once" and type(self).calls == 1
        if self.path == "/api/chat":
            assert "format" in body and "$ref" not in json.dumps(body["format"])
            out = {"message": {"role": "assistant", "content": content[:50] if truncated else content},
                   "done_reason": "length" if truncated else "stop", "prompt_eval_count": 1000, "eval_count": 500}
        else:   # OpenAI-compatible chat completions (OpenRouter)
            assert body["response_format"]["type"] == "json_schema"
            assert body["provider"] == {"require_parameters": True}
            out = {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                   "choices": [{"index": 0, "finish_reason": "length" if truncated else "stop",
                                "message": {"role": "assistant", "content": content[:50] if truncated else content}}],
                   "usage": {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500, "cost": 0.0123}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def fake_server():
    _Fake.mode, _Fake.calls = "ok", 0
    srv = HTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_ollama_extractor(fake_server):
    from recipe_ir.providers import OllamaExtractor
    ex = OllamaExtractor("fake:1b", host=fake_server)
    res = run_pipeline(load_source(str(HERE / "recipes" / "scrambled_eggs.txt")), ex)
    assert res.final.validation.status == "valid"
    assert res.final.call.usage.cost_usd == 0.0 and res.final.call.usage.model == "ollama:fake:1b"


def test_ollama_truncation_retries_with_bigger_cap(fake_server):
    from recipe_ir.providers import OllamaExtractor
    _Fake.mode = "truncate_once"
    ex = OllamaExtractor("fake:1b", host=fake_server)
    res = run_pipeline(load_source(str(HERE / "recipes" / "scrambled_eggs.txt")), ex)
    assert [a.kind for a in res.attempts] == ["extract", "retry"]
    assert res.attempts[0].call.error.startswith("truncated")
    assert res.final.validation.status == "valid"


def test_openrouter_extractor_records_cost(fake_server):
    from recipe_ir.providers import OpenRouterExtractor
    from recipe_ir.runio import run_cost
    ex = OpenRouterExtractor("vendor/model", base_url=fake_server + "/v1", api_key="test")
    res = run_pipeline(load_source(str(HERE / "recipes" / "onion_egg.txt")), ex)
    assert res.final.validation.status == "recipe_findings"
    assert run_cost(res) == 0.0123


def test_make_extractor_dispatch(monkeypatch):
    from recipe_ir.llm_extractor import make_extractor
    from recipe_ir.providers import OllamaExtractor, OpenRouterExtractor
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    assert isinstance(make_extractor("ollama:qwen3.5:9b"), OllamaExtractor)
    assert isinstance(make_extractor("openrouter:openai/gpt-6.1-sol"), OpenRouterExtractor)
    with pytest.raises(ValueError):
        make_extractor("claude-sonnet-5-5")


# ------------------------------------------------------------------ corpus + discovery
def test_features_and_robot_plausible():
    egg = features((HERE / "recipes" / "scrambled_eggs.txt").read_text())
    muffin = features((HERE / "recipes" / "blueberry_muffins.txt").read_text())
    assert egg["robot_plausible"] and egg["stovetop"]
    assert muffin["oven"] and not muffin["robot_plausible"]


def test_select_stratified():
    rows = [{"name": f"r{i}", "site": f"s{i % 3}", "size": ["small", "medium", "large"][i % 3],
             "robot_plausible": i % 2 == 0} for i in range(30)]
    pick = select(rows, n=6, seed=1)
    sizes = {next(r["size"] for r in rows if r["name"] == n) for n in pick}
    assert len(pick) == 6 and sizes == {"small", "medium", "large"}
    assert all(next(r for r in rows if r["name"] == n)["robot_plausible"] for n in select(rows, subset="robot_plausible"))
    assert select(rows, n=6, seed=1) == pick                    # seeded


def test_sitemap_parsing():
    from discover_recipes import SKIP, is_index, locs
    idx = b'<?xml version="1.0"?><sitemapindex><sitemap><loc>https://x.com/post-sitemap.xml</loc></sitemap></sitemapindex>'
    assert is_index(idx) and locs(idx) == ["https://x.com/post-sitemap.xml"]
    assert locs(b"<urlset><url><loc><![CDATA[https://x.com/eggs/]]></loc></url></urlset>") == ["https://x.com/eggs/"]
    assert SKIP.search("https://x.com/category/dinner/") and not SKIP.search("https://x.com/shakshuka-recipe/")


# ------------------------------------------------------------------ end to end: two "models" through eval_models
def test_eval_models_end_to_end(fake_server, tmp_path):
    import os
    env = {**os.environ, "OLLAMA_HOST": fake_server}
    rc = subprocess.call([sys.executable, str(HERE / "eval_models.py"), "--recipes", str(HERE / "recipes"),
                          "--only", "scrambled_eggs", "onion_egg", "--models", "ollama:a:1b", "ollama:b:1b",
                          "--reference", "ollama:a:1b", "--out", str(tmp_path)], cwd=HERE, env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert rc == 0
    batch = next(tmp_path.glob("models__*"))
    m = json.loads((batch / "matrix.json").read_text())
    b = m["models"]["ollama:b:1b"]
    assert b["valid"] == 1.0 and b["silver_strict_step_f1"] == 1.0 and b["gold_strict_step_f1"] == 1.0
    assert b["robot_supported_share"] is not None and b["cost_usd"] == 0.0
    assert (batch / "matrix.md").exists() and (batch / "cells.csv").exists()
