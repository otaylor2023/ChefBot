"""A scene's dynamics (its own logic: an elevator's controller) in a separate process, one JSON line each way per
control tick. The caller gives the sandbox to run it in (a command prefix: the environment imports no runtime).

    child = DynamicsProcess(folder, annotations, params, seed, python=..., prefix=[...])
    commands = child.step(t, sensed)       # {"ctrl": {...}, "displays": {...}, "events": [...]}
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RUNNER = r'''
import importlib.util, json, sys, traceback
folder, name = sys.argv[1], sys.argv[2]
sys.path.insert(0, folder)
channel, sys.stdout = sys.stdout, sys.stderr  # the dynamics' own prints go to the log, not the channel
def reply(obj):
    channel.write(json.dumps(obj) + "\n"); channel.flush()
try:
    spec = importlib.util.spec_from_file_location("scene_dynamics", folder + "/" + name)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    init = json.loads(sys.stdin.readline())
    dyn = mod.Dynamics(init["annotations"], init["params"], init["seed"])
    reply({"ok": True})
except BaseException:
    reply({"error": traceback.format_exc()[-2000:]}); sys.exit(3)
for line in sys.stdin:
    req = json.loads(line)
    try:
        out = dyn.step(req["t"], req["sensed"]) or {}
        reply({"ctrl": out.get("ctrl") or {}, "displays": out.get("displays") or {}, "events": out.get("events") or []})
    except BaseException:
        reply({"error": traceback.format_exc()[-2000:]})
'''


class DynamicsError(RuntimeError):
    pass


class DynamicsProcess:
    def __init__(self, folder: Path, annotations: dict, params: dict, seed: int, python: str | None = None,
                 prefix: list[str] | tuple = (), writable: Path | None = None, timeout_s: float = 30.0):
        self.folder = Path(folder).resolve()
        name = annotations.get("dynamics", "dynamics.py")
        if not (self.folder / name).exists():
            raise DynamicsError(f"no {name}")
        self.tmp = Path(tempfile.mkdtemp(prefix="chefbot_dyn_"))
        runner = self.tmp / "runner.py"
        runner.write_text(RUNNER)
        os.chmod(self.tmp, 0o755)
        os.chmod(runner, 0o644)
        python = python or sys.executable
        env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(writable or self.tmp), "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1"}
        cmd = list(prefix) + [python, "-I", str(runner), str(self.folder), name]
        self.log = open(self.tmp / "stderr.txt", "w")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, env=env, bufsize=1)
        self.timeout_s = timeout_s
        r = self._ask({"annotations": annotations, "params": params, "seed": int(seed)})
        if "error" in r:
            self.close()
            raise DynamicsError(f"the dynamics did not start: {r['error']}")

    def _ask(self, msg: dict) -> dict:
        try:
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            raise DynamicsError(f"the dynamics process is gone: {self.stderr()}") from None
        line = self.proc.stdout.readline()
        if not line:
            raise DynamicsError(f"the dynamics process exited: {self.stderr()}")
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            raise DynamicsError(f"not a reply from the dynamics: {line[:200]!r}") from None

    def stderr(self) -> str:
        self.log.flush()
        try:
            return (self.tmp / "stderr.txt").read_text(errors="replace")[-1500:]
        except OSError:
            return ""

    def step(self, t: float, sensed: dict) -> dict:
        r = self._ask({"t": round(t, 6), "sensed": sensed})
        if "error" in r:
            raise DynamicsError(f"the dynamics raised at t={t:.2f}: {r['error']}")
        return r

    def close(self) -> None:
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()
        shutil.rmtree(self.tmp, ignore_errors=True)
