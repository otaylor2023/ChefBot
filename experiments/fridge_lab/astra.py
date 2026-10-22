"""GPT-6 Astra as the robot's policy model, called through Codex (the same login as the researcher).

One call = `codex exec` with every tool switched off (the model can only look at the images and
answer), the policy's own instructions in place of Codex's, and optionally a JSON schema for the
answer. `n` > 1 sends that many identical requests at once and returns the first good answer
(the others are stopped). A `session` keeps a conversation: its first call starts a thread, later
calls continue it (one request at a time).

The episode runner calls `start` in a background thread and makes the answer available only once
the simulated clock has lived through its real latency, so the robot can keep moving meanwhile.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np

MODEL = "gpt-6-astra"
EFFORTS = ("low", "medium", "high", "xhigh")
TIERS = ("default", "fast", "priority")
MAX_N = 4
MAX_IMAGES = 6
MAX_TEXT = 20000
TIMEOUT_S = 120.0
# Everything but plain answering is off: no shell, no files, no web, no image tools.
DISABLED = ("shell_tool", "unified_exec", "shell_snapshot", "image_generation", "tool_suggest", "skill_search", "sleep_tool",
            "apps", "view_image", "browser_use", "browser_use_external", "computer_use")  # fmt: skip
DEFAULT_INSTRUCTIONS = "You decide for a mobile robot. Look at the images and answer exactly what is asked. You cannot run tools."


class AstraError(ValueError):
    pass


def check(req: dict) -> dict:
    """Validate a request from the policy; returns the normalized arguments."""
    a = {"prompt": str(req.get("prompt") or ""), "images": list(req.get("images") or []), "schema": req.get("schema"),
         "effort": req.get("effort") or "low", "tier": req.get("tier") or "default", "n": int(req.get("n") or 1),
         "instructions": req.get("instructions"), "session": req.get("session")}  # fmt: skip
    if not a["prompt"] or len(a["prompt"]) > MAX_TEXT:
        raise AstraError(f"prompt: 1..{MAX_TEXT} characters")
    if a["effort"] not in EFFORTS:
        raise AstraError(f"effort: one of {EFFORTS}")
    if a["tier"] not in TIERS:
        raise AstraError(f"tier: one of {TIERS}")
    if not 1 <= a["n"] <= MAX_N:
        raise AstraError(f"n: 1..{MAX_N}")
    if len(a["images"]) > MAX_IMAGES:
        raise AstraError(f"at most {MAX_IMAGES} images")
    if a["instructions"] is not None and (not isinstance(a["instructions"], str) or len(a["instructions"]) > MAX_TEXT):
        raise AstraError(f"instructions: a string up to {MAX_TEXT} characters")
    if a["schema"] is not None and not isinstance(a["schema"], dict):
        raise AstraError("schema: a JSON schema object")
    if a["session"] is not None:
        a["session"] = "".join(c for c in str(a["session"]) if c.isalnum() or c in "-_")[:40] or None
        if a["n"] > 1:
            raise AstraError("a session call is one request (n=1)")
    return a


def _codex_cmd(a: dict, work: Path, images: list[Path], thread: str | None) -> list[str]:
    cmd = ["codex"] + [x for f in DISABLED for x in ("--disable", f)] + ["exec"]
    if thread:
        cmd += ["resume", thread]
    cmd += ["--json", "--skip-git-repo-check", "-m", MODEL, "-c", f"model_reasoning_effort={a['effort']}",
            "-c", f"model_instructions_file={work / 'instructions.md'}", "-c", "model_reasoning_summary=none"]  # fmt: skip
    if a["tier"] != "default":
        cmd += ["-c", f"service_tier={a['tier']}"]
    if not thread:
        cmd += ["-s", "read-only"] + ([] if a["session"] else ["--ephemeral"])
        if a["schema"] is not None:
            cmd += ["--output-schema", str(work / "schema.json")]
    for im in images:
        cmd += ["-i", str(im)]
    return cmd + ["--", a["prompt"]]


def _parse(stdout: str) -> dict:
    ev = [json.loads(l) for l in stdout.splitlines() if l.startswith("{")]
    tid = next((e.get("thread_id") for e in ev if e.get("type") == "thread.started"), None)
    msgs = [e["item"].get("text") for e in ev if e.get("type") == "item.completed" and e["item"].get("type") == "agent_message"]
    usage = next((e.get("usage") for e in ev if e.get("type") == "turn.completed"), None)
    errs = [str(e.get("message") or e.get("error"))[:300] for e in ev if e.get("type") in ("error", "turn.failed")]
    return {"thread": tid, "text": msgs[-1] if msgs else None, "usage": usage, "errors": errs}


class Call:
    """One policy request (n parallel Codex processes), run in a background thread."""

    def __init__(self, a: dict, image_arrays: list[np.ndarray], sessions: dict, env: dict):
        self.a, self.sessions, self.env = a, sessions, env
        self.work = Path(tempfile.mkdtemp(prefix="astra_"))
        (self.work / "instructions.md").write_text(a["instructions"] or DEFAULT_INSTRUCTIONS)
        if a["schema"] is not None:
            (self.work / "schema.json").write_text(json.dumps(a["schema"]))
        self.images = []
        for i, im in enumerate(image_arrays):
            p = self.work / f"im{i}.png"
            iio.imwrite(p, np.asarray(im, dtype=np.uint8))
            self.images.append(p)
        self.result: dict | None = None
        self.t0 = time.time()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        a, procs = self.a, []
        thread = self.sessions.get(a["session"]) if a["session"] else None
        try:
            cmd = _codex_cmd(a, self.work, self.images, thread)
            for _ in range(a["n"]):
                procs.append(subprocess.Popen(cmd, cwd=self.work, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                              stderr=subprocess.DEVNULL, text=True, env=self.env))  # fmt: skip
            done: dict | None = None
            outs = {}
            readers = [threading.Thread(target=lambda p=p: outs.__setitem__(p.pid, p.communicate()[0]), daemon=True) for p in procs]
            for r in readers:
                r.start()
            failures = []
            while time.time() - self.t0 < TIMEOUT_S and done is None:
                for p in procs:
                    if p.pid in outs and p.pid not in [f["pid"] for f in failures]:
                        r = _parse(outs[p.pid])
                        if r["text"] is not None:
                            done = {**r, "pid": p.pid}
                            break
                        failures.append({**r, "pid": p.pid})
                if done is None and len(failures) == len(procs):
                    break
                time.sleep(0.02)
            latency = time.time() - self.t0
            if done is None:
                errs = [e for f in failures for e in f["errors"]] or ["timed out" if latency >= TIMEOUT_S else "no answer"]
                self.result = {"ok": False, "error": "; ".join(errs)[:500], "latency_s": round(latency, 3)}
                return
            if a["session"] and done["thread"] and not thread:
                self.sessions[a["session"]] = done["thread"]
            text, parsed = done["text"], None
            if a["schema"] is not None:
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    pass
            self.result = {"ok": True, "text": text, "json": parsed, "latency_s": round(latency, 3), "usage": done["usage"],
                           "n_sent": a["n"]}  # fmt: skip
        except Exception as e:  # noqa: BLE001
            self.result = {"ok": False, "error": f"{type(e).__name__}: {e}"[:500], "latency_s": round(time.time() - self.t0, 3)}
        finally:
            for p in procs:
                if p.poll() is None:
                    p.kill()
            shutil.rmtree(self.work, ignore_errors=True)

    def done(self) -> bool:
        return self.result is not None

    def wait(self) -> dict:
        self.thread.join()
        return self.result


class Replay:
    """A recorded answer from another episode, given back with its recorded latency (the answer-swap check)."""

    def __init__(self, result: dict):
        self.result = dict(result)
        self.thread = threading.Thread(target=lambda: None)

    def done(self) -> bool:
        return True

    def wait(self) -> dict:
        return self.result


def codex_env() -> dict:
    """The policy's Codex runs with the researcher's login but none of the lab's environment."""
    keep = ("HOME", "CODEX_HOME", "PATH", "LANG", "SSL_CERT_FILE", "SSL_CERT_DIR", "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY")
    return {k: v for k, v in os.environ.items() if k in keep}
