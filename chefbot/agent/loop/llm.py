"""OpenRouter chat client (OpenAI-compatible tool calling), AGENT_SPEC §8.2.

Only the agent loop and the StateChecker talk to an LLM.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_RETRY_STATUS = {408, 429, 500, 502, 503, 504}


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON text, as the model wrote it


@dataclass
class LLMResponse:
    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    model: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def assistant_message(self) -> dict:
        """This reply as an assistant message for the next request."""
        message: dict[str, Any] = {"role": "assistant", "content": self.text or ""}
        if self.tool_calls:
            message["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                for c in self.tool_calls
            ]
        return message


class LLMError(Exception):
    pass


def api_key(env_file: str | Path | None = None) -> str | None:
    """OPENROUTER_API_KEY from the environment, else from a .env file (repo root by default)."""
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    path = Path(env_file) if env_file else Path(__file__).resolve().parents[3] / ".env"
    if path.is_file():
        for line in path.read_text().splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key.strip() == "OPENROUTER_API_KEY":
                return value.strip().strip('"').strip("'")
    return None


class OpenRouterClient:
    def __init__(
        self,
        model: str,
        key: str | None = None,
        temperature: float | None = None,
        provider: dict | None = None,
        timeout_s: float = 120.0,
        retries: int = 6,
        url: str = OPENROUTER_URL,
    ):
        self.model = model
        self._key = key or api_key()
        if not self._key:
            raise LLMError("no OpenRouter key: set OPENROUTER_API_KEY or put it in .env")
        self._temperature = temperature
        self._provider = provider
        self._retries = retries
        self._url = url
        self._http = httpx.Client(timeout=timeout_s)

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        tool_choice: str = "auto",
        parallel_tool_calls: bool = False,
    ) -> LLMResponse:
        body: dict[str, Any] = {"model": model or self.model, "messages": messages, "usage": {"include": True}}
        if tools:
            body.update(tools=tools, tool_choice=tool_choice, parallel_tool_calls=parallel_tool_calls)
        temp = self._temperature if temperature is None else temperature
        if temp is not None:
            body["temperature"] = temp
        if self._provider:
            body["provider"] = self._provider
        headers = {"Authorization": f"Bearer {self._key}", "X-Title": "ChefBot"}
        last: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                response = self._http.post(self._url, json=body, headers=headers)
            except httpx.HTTPError as exc:
                last = exc
            else:
                if response.status_code == 200:
                    data = response.json()
                    if "error" in data:  # OpenRouter can report provider errors in a 200
                        last = LLMError(str(data["error"]))
                    else:
                        return parse_response(data)
                elif response.status_code in _RETRY_STATUS:
                    last = LLMError(f"HTTP {response.status_code}: {response.text[:300]}")
                else:
                    raise LLMError(f"HTTP {response.status_code}: {response.text[:500]}")
            if attempt < self._retries:
                time.sleep(min(2.0 ** (attempt + 1), 30.0))  # 2, 4, 8, 16, 30, 30 s: rides out a short outage
        raise LLMError(f"OpenRouter request failed after {self._retries + 1} tries: {last}")


def parse_response(data: dict) -> LLMResponse:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    calls = [
        ToolCall(
            c.get("id") or f"call_{i}",
            (c.get("function") or {}).get("name", ""),
            (c.get("function") or {}).get("arguments") or "{}",
        )
        for i, c in enumerate(message.get("tool_calls") or [])
    ]
    usage = data.get("usage") or {}
    return LLMResponse(
        text=message.get("content") or None,
        tool_calls=calls,
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        cost_usd=float(usage.get("cost") or 0.0),
        model=data.get("model"),
        raw=data,
    )


def parse_arguments(call: ToolCall) -> dict:
    """A tool call's arguments as a dict. Raises ValueError for malformed JSON or a non-object."""
    text = call.arguments.strip() or "{}"
    args = json.loads(text)
    if not isinstance(args, dict):
        raise ValueError("arguments must be a JSON object")
    return args
