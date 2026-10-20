"""Model clients (spec §12): one Request shape, cached responses, two ways to run.

sync:  one HTTP call per request (OpenRouter chat completions with inline
       images; Gemini generateContent). For prompt tests and small runs.
batch: half price, results within 24 h. OpenRouter's Batch API takes images
       only by public URL, so they are uploaded to a public-read GCS bucket
       first; Gemini's batch API takes them inline (split under 20 MB).

Every response is cached under results/cache/<key>.json, the key a hash of
everything that defines the call (provider, model, prompt, image bytes,
temperature, max tokens, repeat index), so re-running a config re-uses it.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

OPENROUTER = "https://openrouter.ai/api/v1"
GEMINI = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_INLINE_LIMIT = 18_000_000  # bytes per batch (the API's limit is 20 MB)


@dataclass
class Request:
    provider: str  # openrouter | gemini
    model: str
    text: str
    images: list[str]  # paths to PNGs, in order
    temperature: float | None
    max_tokens: int
    repeat: int
    key: str = ""
    meta: dict = field(default_factory=dict)  # config, scene, call: for bookkeeping only

    def __post_init__(self):
        if not self.key:
            h = hashlib.sha256()
            for part in (self.provider, self.model, self.text, repr(self.temperature), str(self.max_tokens), str(self.repeat)):
                h.update(part.encode())
            for img in self.images:
                h.update(_sha(img).encode())
            self.key = h.hexdigest()[:24]


def _sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_env(path: Path) -> None:
    """KEY=value lines into os.environ (values never printed)."""
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class Cache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> dict | None:
        p = self.root / f"{key}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def put(self, key: str, record: dict) -> None:
        (self.root / f"{key}.json").write_text(json.dumps(record, indent=1))


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


def _or_headers() -> dict:
    return {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}


def _gm_headers() -> dict:
    return {"x-goog-api-key": os.environ["GEMINI_API_KEY"]}


def _or_body(req: Request, image_urls: list[str]) -> dict:
    content = [{"type": "text", "text": req.text}] + [{"type": "image_url", "image_url": {"url": u}} for u in image_urls]
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": req.max_tokens, "usage": {"include": True}}
    if req.temperature is not None:
        body["temperature"] = req.temperature
    return body


def _gm_body(req: Request) -> dict:
    parts = [{"text": req.text}] + [{"inline_data": {"mime_type": "image/png", "data": base64.b64encode(Path(p).read_bytes()).decode()}}
                                    for p in req.images]  # fmt: skip
    cfg = {"maxOutputTokens": req.max_tokens}
    if req.temperature is not None:
        cfg["temperature"] = req.temperature
    return {"contents": [{"role": "user", "parts": parts}], "generationConfig": cfg}


def _data_url(path: str) -> str:
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def _or_record(req: Request, resp: dict, mode: str, latency: float | None) -> dict:
    choice = (resp.get("choices") or [{}])[0]
    usage = resp.get("usage") or {}
    return {"key": req.key, "provider": req.provider, "model": req.model, "mode": mode, "meta": req.meta,
            "text": (choice.get("message") or {}).get("content") or "", "finish_reason": choice.get("finish_reason"),
            "input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens"),
            "cost": usage.get("cost"), "latency_s": latency, "served_model": resp.get("model"), "error": resp.get("error")}  # fmt: skip


def _gm_record(req: Request, resp: dict, mode: str, latency: float | None) -> dict:
    cand = (resp.get("candidates") or [{}])[0]
    text = "".join(p.get("text", "") for p in ((cand.get("content") or {}).get("parts") or []) if not p.get("thought"))
    usage = resp.get("usageMetadata") or {}
    return {"key": req.key, "provider": req.provider, "model": req.model, "mode": mode, "meta": req.meta, "text": text,
            "finish_reason": cand.get("finishReason"), "input_tokens": usage.get("promptTokenCount"),
            "output_tokens": (usage.get("candidatesTokenCount") or 0) + (usage.get("thoughtsTokenCount") or 0),
            "cost": None, "latency_s": latency, "served_model": resp.get("modelVersion"), "error": resp.get("error")}  # fmt: skip


# -- sync ---------------------------------------------------------------------------------------


def run_sync(reqs: list[Request], cache: Cache) -> list[dict]:
    out = []
    for req in reqs:
        hit = cache.get(req.key)
        if hit is not None:
            out.append(hit)
            continue
        t0 = time.time()
        try:
            if req.provider == "openrouter":
                resp = _http(f"{OPENROUTER}/chat/completions", {"model": req.model, **_or_body(req, [_data_url(p) for p in req.images])},
                             _or_headers())  # fmt: skip
                rec = _or_record(req, resp, "sync", time.time() - t0)
            else:
                resp = _http(f"{GEMINI}/models/{req.model}:generateContent", _gm_body(req), _gm_headers())
                rec = _gm_record(req, resp, "sync", time.time() - t0)
        except Exception as e:  # recorded, not cached: a retry may work
            out.append({"key": req.key, "provider": req.provider, "model": req.model, "mode": "sync", "meta": req.meta,
                        "text": "", "error": str(e)[:1000]})  # fmt: skip
            continue
        cache.put(req.key, rec)
        out.append(rec)
    return out


# -- batch --------------------------------------------------------------------------------------


def upload_public(paths: list[str], bucket: str) -> dict[str, str]:
    """path -> public https URL (object named by content hash; already-uploaded skipped)."""
    by_sha = {p: _sha(p) for p in paths}
    listed = subprocess.run(["gcloud", "storage", "ls", f"gs://{bucket}/img/"], capture_output=True, text=True)
    have = {line.rsplit("/", 1)[-1][:-4] for line in listed.stdout.split()}
    todo = sorted({sha: p for p, sha in by_sha.items() if sha not in have}.items())
    if todo:
        staging = Path("/tmp/ortho_grasp_upload")
        staging.mkdir(exist_ok=True)
        for f in staging.glob("*.png"):
            f.unlink()
        for sha, p in todo:
            (staging / f"{sha}.png").write_bytes(Path(p).read_bytes())
        subprocess.run(["gcloud", "storage", "cp", *[str(f) for f in staging.glob("*.png")], f"gs://{bucket}/img/"], check=True,
                       capture_output=True)  # fmt: skip
    return {p: f"https://storage.googleapis.com/{bucket}/img/{sha}.png" for p, sha in by_sha.items()}


def submit_batches(reqs: list[Request], cache: Cache, batch_dir: Path, bucket: str | None) -> list[str]:
    """Submit every uncached request (grouped per provider and model). Returns batch ids."""
    batch_dir.mkdir(parents=True, exist_ok=True)
    in_flight = set()
    for f in batch_dir.glob("*.json"):
        d = json.loads(f.read_text())
        if d.get("status") not in ("done", "failed"):
            in_flight |= set(d["requests"])
    todo = [r for r in reqs if cache.get(r.key) is None and r.key not in in_flight]
    groups: dict[tuple[str, str], list[Request]] = {}
    for r in {r.key: r for r in todo}.values():
        groups.setdefault((r.provider, r.model), []).append(r)
    ids = []
    for (provider, model), group in groups.items():
        if provider == "openrouter":
            if not bucket:
                raise RuntimeError("OpenRouter batches need images at public URLs: set gcs_bucket in experiment.yaml")
            urls = upload_public(sorted({p for r in group for p in r.images}), bucket)
            body = {"endpoint": "/v1/chat/completions", "model": model, "completion_window": "24h",
                    "requests": [{"custom_id": r.key, "body": _or_body(r, [urls[p] for p in r.images])} for r in group]}  # fmt: skip
            resp = _http(f"{OPENROUTER}/batches", body, _or_headers())
            ids.append(_record_batch(batch_dir, resp["id"], provider, model, group))
        else:
            chunk, size = [], 0
            for r in group + [None]:
                body = None if r is None else _gm_body(r)
                n = 0 if r is None else len(json.dumps(body))
                if chunk and (r is None or size + n > GEMINI_INLINE_LIMIT):
                    payload = {"batch": {"display_name": f"ortho-{int(time.time())}", "input_config": {"requests": {"requests": [
                        {"request": b, "metadata": {"key": rr.key}} for rr, b in chunk]}}}}  # fmt: skip
                    resp = _http(f"{GEMINI}/models/{model}:batchGenerateContent", payload, _gm_headers())
                    ids.append(_record_batch(batch_dir, resp["name"], provider, model, [rr for rr, _ in chunk]))
                    chunk, size = [], 0
                if r is not None:
                    chunk.append((r, body))
                    size += n
    return ids


def _record_batch(batch_dir: Path, batch_id: str, provider: str, model: str, group: list[Request]) -> str:
    safe = batch_id.replace("/", "_")
    (batch_dir / f"{safe}.json").write_text(json.dumps({"id": batch_id, "provider": provider, "model": model,
        "status": "submitted", "submitted_at": time.time(), "requests": {r.key: asdict(r) for r in group}}, indent=1))  # fmt: skip
    return batch_id


def collect_batches(cache: Cache, batch_dir: Path) -> dict[str, int]:
    """Poll every unfinished batch; cache the results of finished ones. Returns status counts."""
    counts: dict[str, int] = {}
    for f in sorted(batch_dir.glob("*.json")):
        d = json.loads(f.read_text())
        if d["status"] in ("done", "failed"):
            counts[d["status"]] = counts.get(d["status"], 0) + 1
            continue
        reqs = {k: Request(**v) for k, v in d["requests"].items()}
        if d["provider"] == "openrouter":
            resp = _http(f"{OPENROUTER}/batches/{d['id']}", None, _or_headers(), method="GET")
            status = resp.get("status")
            if status == "completed":
                for item in resp.get("results") or []:
                    req = reqs.get(item.get("custom_id"))
                    if req is None:
                        continue
                    if item.get("response") and item["response"].get("status_code") == 200:
                        cache.put(req.key, _or_record(req, item["response"]["body"], "batch", None))
                    else:
                        cache.put(req.key, {"key": req.key, "provider": req.provider, "model": req.model, "mode": "batch",
                                            "meta": req.meta, "text": "", "error": item.get("error") or item.get("response")})  # fmt: skip
                d["status"], d["usage"] = "done", resp.get("usage")
            elif status in ("failed", "expired", "cancelled"):
                d["status"], d["error"] = "failed", resp.get("error")
        else:
            resp = _http(f"{GEMINI}/{d['id']}", None, _gm_headers(), method="GET")
            state = (resp.get("metadata") or {}).get("state") or resp.get("state")
            if state in ("JOB_STATE_SUCCEEDED", "BATCH_STATE_SUCCEEDED"):
                holder = resp.get("response") or (resp.get("metadata") or {}).get("output") or resp.get("dest") or {}
                inlined = holder.get("inlinedResponses") or {}
                items = inlined.get("inlinedResponses", inlined) if isinstance(inlined, dict) else inlined
                for item in items or []:
                    req = reqs.get((item.get("metadata") or {}).get("key"))
                    if req is None:
                        continue
                    if "response" in item:
                        cache.put(req.key, _gm_record(req, item["response"], "batch", None))
                    else:
                        cache.put(req.key, {"key": req.key, "provider": req.provider, "model": req.model, "mode": "batch",
                                            "meta": req.meta, "text": "", "error": item.get("error")})  # fmt: skip
                d["status"] = "done"
            elif state in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED", "BATCH_STATE_FAILED"):
                d["status"], d["error"] = "failed", resp.get("error")
            else:
                d["state"] = state
        f.write_text(json.dumps(d, indent=1))
        counts[d["status"]] = counts.get(d["status"], 0) + 1
    return counts
