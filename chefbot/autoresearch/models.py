"""The models a skill may call, metered: Gemini (paid, from GEMINI_API_KEY, which the
skill never sees) and the open-vocabulary detectors / SAM 2 on the local GPU (free in
dollars, paid in time).

(agentic-skills-exploration's runtime/models.py.)

Every Gemini call is charged to a ledger shared by all workers (a JSON-lines file
under a lock) with a total cap, and to the episode with a per-episode cap. A call
that would not fit is refused before it is sent (charged at its worst case: prompt
tokens estimated, max_output_tokens + thinking budget).
"""

from __future__ import annotations

import base64
import fcntl
import io
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image

import urllib.error
import urllib.request

GEMINI = "https://generativelanguage.googleapis.com/v1beta"


def load_env(path: Path) -> None:
    """KEY=value lines into os.environ (values are never printed)."""
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _http(url: str, body: dict | None, headers: dict, method: str = "POST", timeout: float = 600) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json", **headers})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:2000]
            if e.code in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(5 * 2**attempt)
                continue
            raise RuntimeError(f"HTTP {e.code} from {url.split('?')[0]}: {msg}") from None
        except (urllib.error.URLError, TimeoutError):
            if attempt < 4:
                time.sleep(5 * 2**attempt)
                continue
            raise
    raise RuntimeError("unreachable")


def _gm_headers() -> dict:
    return {"x-goog-api-key": os.environ["GEMINI_API_KEY"]}

# USD per 1M tokens (input, output incl. thinking), standard tier, Oct 2026 (the 2.5 models are closed to new keys).
GEMINI_PRICES = {
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.6-flash": (0.75, 3.75),
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-robotics-er-2-preview": (1.00, 5.00),
}
IMAGE_TOKENS = 1300  # worst-case estimate per image before the call (Gemini tiles large images at 258 tokens each)
VISION_MODELS = ("owlv2", "grounding-dino", "sam2")


class BudgetExceeded(Exception):
    pass


class Ledger:
    """Total model spend across every worker and episode: one JSON line per call."""

    def __init__(self, path: Path, cap_usd: float):
        self.path, self.cap = Path(path), cap_usd
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch()

    @contextmanager
    def _locked(self):
        with open(self.path.with_suffix(".lock"), "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def spent(self) -> float:
        total = 0.0
        for line in self.path.read_text().splitlines():
            try:
                total += float(json.loads(line).get("cost_usd") or 0.0)
            except json.JSONDecodeError:
                pass
        return total

    def reserve(self, usd: float, tag: str) -> None:
        """Refuse when the worst case would pass the cap; otherwise hold it (a 'reserve' line)."""
        with self._locked():
            if self.spent() + usd > self.cap:
                raise BudgetExceeded(f"model budget: ${self.spent():.3f} of ${self.cap:.2f} spent; this call could cost ${usd:.4f}")
            with open(self.path, "a") as f:
                f.write(json.dumps({"t": time.time(), "tag": tag, "kind": "reserve", "cost_usd": usd}) + "\n")

    def settle(self, reserved: float, actual: float, record: dict) -> None:
        with self._locked(), open(self.path, "a") as f:
            f.write(json.dumps({"t": time.time(), "kind": "settle", "cost_usd": actual - reserved, **record}) + "\n")


def _png_b64(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(img, dtype=np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def gemini(model: str, prompt: str, images: list[np.ndarray], max_output_tokens: int, thinking_budget: int | None,
           response_schema: dict | None, temperature: float | None, ledger: Ledger, episode_left_usd: float, tag: str) -> dict:  # fmt: skip
    if model not in GEMINI_PRICES:
        raise ValueError(f"model {model!r} is not allowed; allowed: {sorted(GEMINI_PRICES)}")
    pin, pout = GEMINI_PRICES[model]
    max_output_tokens = int(min(max(max_output_tokens, 1), 8192))
    think = None if thinking_budget is None else int(min(max(thinking_budget, 0), 8192))
    worst_out = max_output_tokens + (think if think is not None else 8192)
    worst = (len(prompt) / 3.0 + IMAGE_TOKENS * len(images)) * pin / 1e6 + worst_out * pout / 1e6
    if worst > episode_left_usd:
        raise BudgetExceeded(f"episode model budget: ${episode_left_usd:.4f} left; this call could cost ${worst:.4f}")
    ledger.reserve(worst, tag)
    parts = [{"text": prompt}] + [{"inline_data": {"mime_type": "image/png", "data": _png_b64(im)}} for im in images]
    gen = {"maxOutputTokens": max_output_tokens}
    if think is not None and "robotics" not in model:
        gen["thinkingConfig"] = {"thinkingBudget": think}
    if temperature is not None:
        gen["temperature"] = float(temperature)
    if response_schema:
        gen["responseMimeType"] = "application/json"
        gen["responseSchema"] = response_schema
    body = {"contents": [{"role": "user", "parts": parts}], "generationConfig": gen}
    t0 = time.time()
    actual, rec = 0.0, {}
    try:
        try:
            resp = _http(f"{GEMINI}/models/{model}:generateContent", body, _gm_headers(), timeout=120)
        except RuntimeError as e:
            if "INVALID_ARGUMENT" not in str(e) or "thinkingConfig" not in gen:
                raise
            # some models take a thinking level, not a token budget
            level = "minimal" if think == 0 else "low" if think <= 1024 else "medium" if think <= 4096 else "high"
            gen["thinkingConfig"] = {"thinkingLevel": level}
            resp = _http(f"{GEMINI}/models/{model}:generateContent", body, _gm_headers(), timeout=120)
        um = resp.get("usageMetadata") or {}
        n_in = int(um.get("promptTokenCount") or 0)
        n_out = int(um.get("candidatesTokenCount") or 0) + int(um.get("thoughtsTokenCount") or 0)
        actual = n_in * pin / 1e6 + n_out * pout / 1e6
        cand = (resp.get("candidates") or [{}])[0]
        text = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts", []) if not p.get("thought"))
        rec = {"model": model, "input_tokens": n_in, "output_tokens": n_out, "latency_s": time.time() - t0, "error": resp.get("error")}
        return {"text": text, "cost_usd": actual, "latency_s": rec["latency_s"], "input_tokens": n_in, "output_tokens": n_out,
                "finish_reason": cand.get("finishReason")}  # fmt: skip
    finally:
        ledger.settle(worst, actual, {"tag": tag, **rec})


class Vision:
    """Client for the vision server (autoresearch/vision_server.py, in its own venv), over localhost HTTP."""

    def __init__(self, url: str | None):
        self.url = url

    def __call__(self, op: str, **args) -> dict:
        if not self.url:
            raise RuntimeError("no vision server is running (chefbot/autoresearch/vision_server.py; VISION_URL)")
        return _http(f"{self.url}/{op}", args, {}, timeout=60)


def model_menu() -> dict:
    return {"llm": {m: {"input_per_mtok": p[0], "output_per_mtok": p[1]} for m, p in GEMINI_PRICES.items()},
            "vision": {"detect": ["owlv2", "grounding-dino"], "segment": ["sam2"]}}  # fmt: skip


def gemini_key_present() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))
