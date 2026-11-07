"""Extractors beyond OpenAI's Responses API. Same prompts, same schema, same pipeline; only the transport differs.

OpenRouterExtractor  any hosted model through OpenRouter's OpenAI-compatible chat completions (OPENROUTER_API_KEY).
                     Structured output is requested strictly and `require_parameters` keeps OpenRouter from routing
                     to a provider that would ignore the schema. Cost is OpenRouter's own figure for the call.
OllamaExtractor      a local model through Ollama's native /api/chat with `format` = the JSON schema (constrained
                     decoding), so even small models return parseable JSON. Cost is 0.
"""
from __future__ import annotations
import copy
import json
import os
import time
import urllib.error
import urllib.request
from .llm_extractor import CappedExtractor, LLMCall, Usage
from .schema import Recipe


def recipe_schema() -> dict:
    from openai.lib._pydantic import to_strict_json_schema
    return to_strict_json_schema(Recipe)


def inline_refs(schema: dict) -> dict:
    """Replace local {"$ref": "#/$defs/X"} with the definition. Some grammar converters handle $ref poorly."""
    defs = schema.get("$defs", {})

    def walk(node, depth=0):
        if depth > 40:
            raise ValueError("schema nesting too deep (recursive $ref?)")
        if isinstance(node, dict):
            if "$ref" in node and node["$ref"].startswith("#/$defs/"):
                target = copy.deepcopy(defs[node["$ref"].removeprefix("#/$defs/")])
                extra = {k: v for k, v in node.items() if k != "$ref"}
                return walk({**target, **extra}, depth + 1)
            return {k: walk(v, depth + 1) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [walk(x, depth + 1) for x in node]
        return node

    return walk(schema)


def _loop_note(raw: str) -> str:
    tail = raw[-200:]
    return " (degenerate whitespace loop)" if len(tail) == 200 and not tail.strip() else ""


class OpenRouterExtractor(CappedExtractor):
    base_url = "https://openrouter.ai/api/v1"

    def __init__(self, model: str, temperature: float | None = 0.0, reasoning_effort: str | None = None,
                 max_output_tokens: int = 8000, retry_temperature: float = 0.4, base_url: str | None = None,
                 api_key: str | None = None):
        from openai import OpenAI
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self.client = OpenAI(base_url=base_url or self.base_url, api_key=key, timeout=1800)
        self.model = "openrouter:" + model
        self.remote_model = model
        self.temperature, self.reasoning_effort = temperature, reasoning_effort
        self.max_output_tokens, self.retry_temperature, self.cap = max_output_tokens, retry_temperature, max_output_tokens
        self.response_format = {"type": "json_schema",
                                "json_schema": {"name": "Recipe", "strict": True, "schema": recipe_schema()}}
        self.adaptations: set[str] = set()
        # max_tokens counts hidden reasoning AND the answer. Thinking models (Claude Haiku: ~3-5k reasoning tokens)
        # otherwise truncate a 4-6k-token plan at an 8k cap. Headroom is billed only if used.
        self.reasoning_headroom = int(os.environ.get("OPENROUTER_REASONING_HEADROOM", "8000"))

    def _create(self, system, user, extra, kwargs):
        """Vendor limits, handled once per extractor and recorded in self.adaptations:
        - no endpoint accepts `temperature` (some Claude models): drop it
        - the schema is too big for the provider's grammar compiler (Anthropic: >16 union-typed fields, strict or
          not): JSON mode with the schema in the system prompt; our parser + retry reject anything invalid"""
        import openai
        while True:
            sys_msg = system
            if self.response_format["type"] == "json_object":
                sys_msg += ("\n\nOUTPUT FORMAT\nReply with ONE JSON object that matches this JSON schema exactly "
                            "(every property present, null where allowed, no extra keys, no prose):\n"
                            + json.dumps(recipe_schema(), separators=(",", ":")))
            try:
                return self.client.chat.completions.create(
                    model=self.remote_model, messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user}],
                    response_format=self.response_format, max_tokens=self.cap + self.reasoning_headroom,
                    extra_body=extra, **kwargs)
            except openai.NotFoundError as e:
                if "temperature" in kwargs and "No endpoints found" in str(e):
                    kwargs.pop("temperature")
                    self.temperature = None
                    self.adaptations.add("no temperature")
                    continue
                raise
            except openai.BadRequestError as e:
                msg = str(e)
                if self.response_format["type"] == "json_schema" and ("grammar is too large" in msg or "union types" in msg):
                    # the provider compiles any json_schema into a grammar, strict or not: JSON mode + schema in prompt
                    self.response_format = {"type": "json_object"}
                    self.adaptations.add("JSON mode, schema in prompt (provider grammar limit)")
                    continue
                raise

    def _call(self, system, user):
        t = time.perf_counter()
        extra = {"provider": {"require_parameters": True}, "usage": {"include": True}}
        if self.reasoning_effort:
            extra["reasoning"] = {"effort": self.reasoning_effort}
        kwargs = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.retry_temperature if self.retrying else self.temperature
        resp = self._create(system, user, extra, kwargs)
        u = resp.usage
        details = getattr(u, "completion_tokens_details", None) if u else None
        cost = getattr(u, "cost", None) if u else None
        if cost is None and u is not None and getattr(u, "model_extra", None):
            cost = u.model_extra.get("cost")
        usage = Usage(self.model, getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0,
                      time.perf_counter() - t, getattr(details, "reasoning_tokens", 0) or 0,
                      float(cost) if cost is not None else None)
        if not resp.choices:
            return LLMCall(None, "", usage, "empty response (no choices)")
        choice = resp.choices[0]
        raw = choice.message.content or ""
        if getattr(choice.message, "refusal", None):
            return LLMCall(None, choice.message.refusal, usage, "refusal")
        if choice.finish_reason == "length":
            return LLMCall(None, raw, usage, f"truncated: length{_loop_note(raw)} after {usage.output_tokens} output tokens"
                           f" ({usage.reasoning_tokens} reasoning)")
        return self._parse(raw, usage)


class OllamaExtractor(CappedExtractor):
    """reasoning_effort: None = model default; "off" = think false; "low"/"medium"/"high" = think level
    (gpt-oss takes the level; other thinking models treat any level as think=true)."""

    def __init__(self, model: str, temperature: float | None = 0.0, reasoning_effort: str | None = None,
                 max_output_tokens: int = 8000, retry_temperature: float = 0.4, host: str | None = None,
                 timeout: float = 3600):
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.model = "ollama:" + model
        self.local_model = model
        self.temperature, self.reasoning_effort, self.timeout = temperature, reasoning_effort, timeout
        self.max_output_tokens, self.retry_temperature, self.cap = max_output_tokens, retry_temperature, max_output_tokens
        self.schema = inline_refs(recipe_schema())
        self.num_ctx_override = int(os.environ["OLLAMA_NUM_CTX"]) if os.environ.get("OLLAMA_NUM_CTX") else None

    def _think(self):
        e = (self.reasoning_effort or "").strip().lower()
        if not e:
            return None
        if e in ("off", "none", "false"):
            return False
        return e if self.local_model.startswith("gpt-oss") else True

    def _call(self, system, user):
        t = time.perf_counter()
        prompt_tokens_est = (len(system) + len(user)) // 3
        num_ctx = self.num_ctx_override or min(131072, 1024 * -(-(prompt_tokens_est + self.cap + 512) // 1024))
        options = {"num_ctx": num_ctx, "num_predict": self.cap}
        # OLLAMA_OPTIONS (JSON) adds runtime options that change speed and heat, not the output: e.g.
        # {"num_gpu": 20, "num_batch": 128} keeps some layers on the CPU and prefills in small chunks (cooler GPU).
        if os.environ.get("OLLAMA_OPTIONS"):
            options.update(json.loads(os.environ["OLLAMA_OPTIONS"]))
        if self.temperature is not None:
            options["temperature"] = self.retry_temperature if self.retrying else self.temperature
        body = {"model": self.local_model, "stream": False, "format": self.schema, "options": options,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if (think := self._think()) is not None:
            body["think"] = think
        req = urllib.request.Request(self.host + "/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            raise RuntimeError(f"ollama {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"ollama not reachable at {self.host} ({e.reason}); start it with `ollama serve`") from e
        msg = data.get("message") or {}
        raw = msg.get("content") or ""
        thinking = msg.get("thinking") or ""
        usage = Usage(self.model, data.get("prompt_eval_count", 0) or 0, data.get("eval_count", 0) or 0,
                      time.perf_counter() - t, len(thinking) // 4, 0.0)
        if data.get("done_reason") == "length":
            return LLMCall(None, raw, usage, f"truncated: length{_loop_note(raw)} after {usage.output_tokens} output tokens")
        return self._parse(raw, usage)
