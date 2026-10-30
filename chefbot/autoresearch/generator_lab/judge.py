"""The judge: Claude (the `claude` CLI, on its own account: CLAUDE_JUDGE_TOKEN) looks at an asset's renders next to
photos of real tongs and scores how real it looks, against a rubric (families/<family>/rubric*.md); and looks at a
batch's assets together for variety.

The CLI runs with only its Read tool, in a temporary folder holding just the images it is to look at, with no
settings, MCP servers or session saved. Each call's usage is appended to a ledger (JSON lines).
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

ASSET_SCHEMA = {
    "type": "object",
    "properties": {
        "shape": {"type": "integer", "minimum": 1, "maximum": 10},
        "tips": {"type": "integer", "minimum": 1, "maximum": 10},
        "mechanism": {"type": "integer", "minimum": 1, "maximum": 10},
        "materials": {"type": "integer", "minimum": 1, "maximum": 10},
        "proportions": {"type": "integer", "minimum": 1, "maximum": 10},
        "overall": {"type": "integer", "minimum": 1, "maximum": 10},
        "closest_real": {"type": "string"},
        "flaws": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["shape", "tips", "mechanism", "materials", "proportions", "overall", "closest_real", "flaws", "summary"],
}
BATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "variety": {"type": "integer", "minimum": 1, "maximum": 10},
        "coverage": {"type": "integer", "minimum": 1, "maximum": 10},
        "missing": {"type": "array", "items": {"type": "string"}},
        "near_duplicates": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["variety", "coverage", "missing", "near_duplicates", "summary"],
}
SCORE_KEYS = ("shape", "tips", "mechanism", "materials", "proportions", "overall")


class JudgeError(RuntimeError):
    pass


def _ledger(path: Path | None, entry: dict) -> None:
    if path is None:
        return
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps({"t": time.time(), **entry}) + "\n")


def calls_made(path: Path | None, kind: str | None = None) -> int:
    if path is None or not Path(path).exists():
        return 0
    return sum(1 for line in Path(path).read_text().splitlines() if line.strip() and (kind is None or json.loads(line).get("kind") == kind))


def ask(prompt: str, images: dict[str, Path], schema: dict, model: str, config_dir: Path, timeout_s: float = 300,
        ledger: Path | None = None, kind: str = "") -> dict:
    """One judge call: the images copied into a fresh folder under the given names, the prompt, a JSON answer."""
    token = os.environ.get("CLAUDE_JUDGE_TOKEN")
    if not token:
        raise JudgeError("CLAUDE_JUDGE_TOKEN is not set (the judge's account)")
    work = Path(tempfile.mkdtemp(prefix="chefbot_judge_"))
    try:
        for name, src in images.items():
            shutil.copy(src, work / name)
        config_dir.mkdir(parents=True, exist_ok=True)
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(config_dir), "CLAUDE_CONFIG_DIR": str(config_dir),
               "CLAUDE_CODE_OAUTH_TOKEN": token, "LANG": "C.UTF-8", "DISABLE_AUTOUPDATER": "1"}  # fmt: skip
        cmd = [os.environ.get("CLAUDE_BIN", "claude"), "-p", prompt, "--model", model, "--tools", "Read", "--output-format", "json",
               "--json-schema", json.dumps(schema), "--no-session-persistence", "--strict-mcp-config", "--setting-sources", ""]  # fmt: skip
        last = None
        for attempt in range(3):
            t0 = time.time()
            try:
                p = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True, timeout=timeout_s)
            except subprocess.TimeoutExpired:
                last = f"timed out after {timeout_s:g} s"
                continue
            try:
                r = json.loads(p.stdout)
            except json.JSONDecodeError:
                last = f"not JSON (exit {p.returncode}): {(p.stdout + p.stderr)[-400:]}"
                time.sleep(5 * (attempt + 1))
                continue
            _ledger(ledger, {"kind": kind, "model": model, "s": round(time.time() - t0, 1), "usd_equiv": r.get("total_cost_usd"),
                             "error": r.get("is_error")})  # fmt: skip
            out = r.get("structured_output")
            if r.get("is_error") or not isinstance(out, dict):
                last = f"judge error: {str(r.get('result'))[:400]}"
                time.sleep(5 * (attempt + 1))
                continue
            return out
        raise JudgeError(last or "the judge failed")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _card(params: dict, checks: dict) -> str:
    keep = {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool)) and len(str(v)) < 200}
    m = checks.get("measured", {})
    failed = checks.get("failed", [])
    return (f"Declared by its generator: {json.dumps(keep)}\n"
            f"Measured in simulation: {json.dumps(m)}\n"
            f"Physics checks failed: {failed or 'none'}")


def judge_asset(lab_dir: Path, params: dict, checks: dict, rubric: str, refs_image: Path | None, model: str, config_dir: Path,
                ledger: Path | None = None) -> dict:
    lab_dir = Path(lab_dir)
    images = {"asset_views.jpg": lab_dir / "sheet.jpg", "asset_squeeze.jpg": lab_dir / "actuation.jpg"}
    if refs_image is not None:
        images["real_tongs.jpg"] = Path(refs_image)
    prompt = rubric.replace("{CARD}", _card(params, checks)).replace("{REFS}", "real_tongs.jpg" if refs_image else "(none)")
    out = ask(prompt, images, ASSET_SCHEMA, model, config_dir, ledger=ledger, kind="asset")
    out["mean"] = round(sum(out[k] for k in SCORE_KEYS) / len(SCORE_KEYS), 2)
    return out


def judge_batch(contact_sheet: Path, n: int, rubric: str, refs_image: Path | None, model: str, config_dir: Path,
                ledger: Path | None = None) -> dict:
    images = {"batch.jpg": Path(contact_sheet)}
    if refs_image is not None:
        images["real_tongs.jpg"] = Path(refs_image)
    prompt = rubric.replace("{N}", str(n)).replace("{REFS}", "real_tongs.jpg" if refs_image else "(none)")
    return ask(prompt, images, BATCH_SCHEMA, model, config_dir, ledger=ledger, kind="batch")
