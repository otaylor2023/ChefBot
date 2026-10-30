"""Relay between a researcher workspace on this machine and a lab running elsewhere (the pod).

Codex runs here (its sandbox works here; the pod's container cannot run it), and the
lab, with the simulator, the GPU models, the keys and the val/test data, runs on the
pod. Codex's `./lab` writes requests into <ws>/.lab/ipc as usual; this relay, outside
Codex's sandbox:
  - before forwarding a request, pushes <ws>/skills/ to the pod's workspace;
  - forwards the request over SSH and waits for the lab's answer;
  - pulls the run folders the answer refers to (and, every SYNC_S, all of runs/) back
    into <ws>/runs/ before handing the answer to Codex.
Nothing from the lab's private directory comes back.
(agentic-skills-exploration's autoresearch/relay.py, unchanged.)
"""

from __future__ import annotations

import json
import shlex
import subprocess
import threading
import time
from pathlib import Path

SYNC_S = 60.0


class Relay:
    def __init__(self, ws: Path, host: str, remote_ws: str):
        self.ws, self.host, self.rws = Path(ws), host, remote_ws.rstrip("/")
        self.ipc = self.ws / ".lab" / "ipc"
        self.ipc.mkdir(parents=True, exist_ok=True)
        (self.ws / "runs").mkdir(exist_ok=True)
        self._push_lock = threading.Lock()
        self.log = open(self.ws.parent / "relay.log", "a")

    def _ssh(self, cmd: str, inp: str | None = None, timeout: float = 120) -> subprocess.CompletedProcess:
        return subprocess.run(["ssh", "-o", "ServerAliveInterval=30", "-o", "ConnectTimeout=20", self.host, cmd], input=inp,
                              capture_output=True, text=True, timeout=timeout)  # fmt: skip

    def _rsync(self, args: list[str], timeout: float = 600) -> None:
        subprocess.run(["rsync", "-az", "--no-owner", "--no-group", "-e", "ssh -o ServerAliveInterval=30"] + args,
                       capture_output=True, timeout=timeout)  # fmt: skip

    def push_skills(self) -> None:
        with self._push_lock:
            self._rsync(["--chmod=Da+rx,Fa+r", "--exclude", "__pycache__", f"{self.ws}/skills/", f"{self.host}:{self.rws}/skills/"])

    def pull_runs(self, sub: str = "") -> None:
        sub = sub.strip("/")
        src = f"{self.host}:{self.rws}/runs/{sub + '/' if sub else ''}"
        dst = self.ws / "runs" / sub if sub else self.ws / "runs"
        dst.mkdir(parents=True, exist_ok=True)
        self._rsync([src, f"{dst}/"])

    def _note(self, msg: str) -> None:
        self.log.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        self.log.flush()

    def handle(self, rid: str, req: dict) -> None:
        try:
            if req.get("cmd") in ("run", "suite", "val"):
                self.push_skills()
            wait = float(req.get("wait_s") or 0) + 120
            r = self._ssh(f"python3 -c {shlex.quote(_REMOTE_CALL)} {shlex.quote(self.rws)} {wait}", inp=json.dumps(req), timeout=wait + 60)
            resp = json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else {"error": f"relay: lab unreachable ({r.stderr.strip()[-300:]})"}
            d = resp.get("dir")
            if isinstance(d, str) and d.startswith("runs/"):
                self.pull_runs(d[len("runs/") :])
        except Exception as e:  # noqa: BLE001
            resp = {"error": f"relay: {type(e).__name__}: {e}"}
        tmp = self.ipc / f".resp_{rid}.json"
        tmp.write_text(json.dumps(resp, indent=1))
        tmp.rename(self.ipc / f"resp_{rid}.json")
        self._note(f"{req.get('cmd')} -> {sorted(resp)[:6]}")

    def serve_forever(self, stop: threading.Event) -> None:
        busy: dict[str, threading.Thread] = {}
        last_sync = 0.0
        while not stop.is_set():
            for p in sorted(self.ipc.glob("req_*.json")):
                rid = p.stem[4:]
                if rid in busy:
                    continue
                try:
                    req = json.loads(p.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                p.unlink(missing_ok=True)
                t = threading.Thread(target=self.handle, args=(rid, req), daemon=True)
                busy[rid] = t
                t.start()
            for rid in [r for r, t in busy.items() if not t.is_alive()]:
                busy.pop(rid)
            if time.time() - last_sync > SYNC_S:
                last_sync = time.time()
                threading.Thread(target=self.pull_runs, daemon=True).start()
            time.sleep(0.2)


# Runs on the pod (system python3): hand a request to the lab's IPC folder and print its answer.
_REMOTE_CALL = r"""
import json, sys, time, uuid
from pathlib import Path
ipc = Path(sys.argv[1]) / ".lab" / "ipc"
timeout = float(sys.argv[2])
req = sys.stdin.read()
rid = "relay" + uuid.uuid4().hex[:10]
tmp = ipc / f".req_{rid}.json"
tmp.write_text(req)
tmp.rename(ipc / f"req_{rid}.json")
resp = ipc / f"resp_{rid}.json"
t_end = time.time() + timeout
while time.time() < t_end:
    if resp.exists():
        sys.stdout.write(resp.read_text())
        resp.unlink()
        sys.exit(0)
    time.sleep(0.3)
print(json.dumps({"error": f"no answer from the lab in {timeout:.0f} s"}))
"""
