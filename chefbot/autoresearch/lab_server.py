"""The lab server: answers the researcher's ./lab requests (file IPC in the workspace) by running episodes in a pool
of simulator worker processes (each keeps its last kitchen loaded), with the lab's visibility rules:

    dev   the researcher's items: full results (video, trace with the task's ground-truth measurements, logs)
    val   held-out items: only aggregate scores go back; at most `submissions` per lab
    test  never reachable from here: run by us after the lab (python -m chefbot.evals.evaluate)

Layout of a lab run (runs/labs/<id>/): workspace/ (the researcher's), private/ (ledger, val results, logs: not the
researcher's), snapshots/ (every skill version as it ran, readable by the skill's sandbox user).
(agentic-skills-exploration's autoresearch/lab_server.py, without Astra.)
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from chefbot import setups, suites

from chefbot.runtime import worker
from chefbot.autoresearch.config import LabConfig
from chefbot.autoresearch.summary import summarize

MAX_DEV_SEEDS = 5
WRAP_UP_MIN = 40
WORKER_ENV = ("MUJOCO", "PYOPENGL", "GEMINI", "EGL", "CUDA", "SKILL_", "CODEX")  # what the workers inherit


def _world_readable(path: Path) -> None:
    for p in [path, *path.rglob("*")]:
        os.chmod(p, 0o755 if p.is_dir() else 0o644)


class IpcServer:
    """Answers ./lab requests: request files in <workspace>/.lab/ipc, one thread per request, each answer with the
    lab's notice (root/notice.txt) and, near the deadline, a note to wrap up. Subclasses: handle(req) -> dict."""

    WRAP_UP = "skills/final, commit, REPORT.md"
    root: Path
    ipc: Path
    log: object
    deadline: float | None = None

    def handle(self, req: dict) -> dict:
        raise NotImplementedError

    def serve_forever(self, stop: threading.Event | None = None) -> None:
        stop = stop or threading.Event()
        busy: dict[str, threading.Thread] = {}
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
                t = threading.Thread(target=self._answer, args=(rid, req), daemon=True)
                busy[rid] = t
                t.start()
            for rid in [r for r, t in busy.items() if not t.is_alive()]:
                busy.pop(rid)
            time.sleep(0.2)

    def _answer(self, rid: str, req: dict) -> None:
        left = None if self.deadline is None else (self.deadline - time.time()) / 60
        try:
            if left is not None and left <= 0 and req.get("cmd") in ("run", "suite", "val"):
                resp = {"error": "the research session is over: no new episodes"}
            else:
                resp = self.handle(req)
        except Exception as e:  # noqa: BLE001
            resp = {"error": f"{type(e).__name__}: {e}"}
        notice = self.root / "notice.txt"  # a message for the researcher from us, attached to every answer
        if notice.exists() and notice.read_text().strip():
            resp["lab_notice"] = notice.read_text().strip()
        if left is not None and left <= WRAP_UP_MIN:
            resp["lab_note"] = f"{max(left, 0):.0f} min left in the research session: wrap up ({self.WRAP_UP})."
        self.log.write(json.dumps({"t": time.time(), "req": req, "resp_keys": list(resp)}) + "\n")
        self.log.flush()
        tmp = self.ipc / f".resp_{rid}.json"
        tmp.write_text(json.dumps(resp, indent=1))
        tmp.rename(self.ipc / f"resp_{rid}.json")


class Lab(IpcServer):
    def __init__(self, cfg: LabConfig, root: Path, jobs: int, vision_url: str | None, limits: dict | None = None):
        self.cfg, self.root = cfg, Path(root)
        self.ws, self.private, self.snapshots = self.root / "workspace", self.root / "private", self.root / "snapshots"
        self.ipc = self.ws / ".lab" / "ipc"
        self.ipc.mkdir(parents=True, exist_ok=True)
        (self.ws / "runs").mkdir(exist_ok=True)
        self.private.mkdir(parents=True, exist_ok=True)
        self.snapshots.mkdir(parents=True, exist_ok=True)
        os.chmod(self.snapshots, 0o755)
        self.dev = {it.id: it for it in suites.load(cfg.suite("dev")).items}
        self.val = suites.load(cfg.suite("val")).items if "val" in cfg.suites else []
        # (setup, task, seed) that val and test hold: refused as dev episodes (a lab whose parts share a scene and
        # differ by seed only would otherwise let dev reach them)
        held = [it for part in ("val", "test") if part in cfg.suites for it in suites.load(cfg.suite(part)).items]
        self.reserved = {(it.setup, it.task, sd) for it in held for sd in it.seeds}
        self.ledger = (str(self.private / "model_ledger.jsonl"), cfg.model_usd)
        self.vision_url, self.limits = vision_url, limits
        env = {k: v for k, v in os.environ.items() if k.startswith(WORKER_ENV)}
        self.pool = ProcessPoolExecutor(jobs, mp_context=mp.get_context("spawn"), initializer=worker.init, initargs=(env,))
        self.jobs: dict[str, dict] = {}
        self.state_path = self.private / "lab_state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"val_used": 0, "val_history": []}
        self.log = open(self.private / "lab_server.log", "a")
        self.deadline: float | None = None

    # -- jobs ----------------------------------------------------------------------------

    def snapshot(self, skill: str) -> tuple[Path, str]:
        src = (self.ws / skill).resolve()
        if not str(src).startswith(str((self.ws / "skills").resolve()) + os.sep) or not (src / "main.py").exists():
            raise ValueError(f"a skill is a directory under skills/ with a main.py (got {skill!r})")
        h = hashlib.sha256()
        for p in sorted(src.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                h.update(str(p.relative_to(src)).encode())
                h.update(p.read_bytes())
        digest = h.hexdigest()[:12]
        dst = self.snapshots / f"{src.name}-{digest}"
        if not dst.exists():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            _world_readable(dst)
        return dst, digest

    def submit(self, kind: str, skill: str, episodes: list[tuple], record: bool) -> dict:
        snap, digest = self.snapshot(skill)
        spec = {"dir": str(snap), "interface": self.cfg.interface, "access": self.cfg.access()}
        job = f"{kind}-{time.strftime('%H%M%S')}-{uuid.uuid4().hex[:4]}"
        base = (self.ws / "runs" / job) if kind == "dev" else (self.private / "val_runs" / job)
        futs = []
        for it, seed in episodes:
            out = base / f"{it.id}_s{seed}"
            futs.append(self.pool.submit(worker.run, {"id": it.id, "setup": it.setup, "task": it.task}, seed, spec, self.cfg.robot, None, str(out),
                                         self.ledger, self.vision_url, record, f"{job}/{it.id}_s{seed}", self.limits))  # fmt: skip
        self.jobs[job] = {"kind": kind, "skill": skill, "digest": digest, "futures": futs, "dir": base, "t0": time.time()}
        self.log.write(json.dumps({"t": time.time(), "submit": job, "skill": skill, "digest": digest, "n": len(episodes)}) + "\n")
        self.log.flush()
        return {"job": job, "skill_digest": digest, "episodes": len(episodes)}

    def status(self, job: str, wait_s: float = 0.0) -> dict:
        j = self.jobs.get(job)
        if j is None:
            return {"error": f"unknown job {job!r}"}
        t_end = time.time() + max(0.0, wait_s)
        while time.time() < t_end and not all(f.done() for f in j["futures"]):
            time.sleep(0.5)
        done = [f.result() for f in j["futures"] if f.done()]
        out = {"job": job, "kind": j["kind"], "skill": j["skill"], "skill_digest": j["digest"], "done": len(done),
               "total": len(j["futures"]), "elapsed_s": round(time.time() - j["t0"], 1)}  # fmt: skip
        if j["kind"] == "dev":
            out["dir"] = str(j["dir"].relative_to(self.ws))
            keep = ("item", "seed", "success", "lift_cm", "max_lift_cm", "held", "robot_time_s", "sim_s", "think_s", "model_s",
                    "model_usd", "n_astra", "astra_s", "end", "sandbox_blocked", "error")  # fmt: skip
            out["episodes"] = [{k: r.get(k) for k in keep if k in r} for r in done]
            out["summary"] = summarize(done)
        elif len(done) == len(j["futures"]):
            out["summary"] = summarize(done)  # val: aggregate only
            if not j.get("recorded"):
                j["recorded"] = True
                self.state["val_history"].append({"job": job, "skill": j["skill"], "digest": j["digest"], **out["summary"]})
                self._save_state()
                (self.private / "val_results").mkdir(exist_ok=True)
                (self.private / "val_results" / f"{job}.json").write_text(json.dumps({"summary": out["summary"], "episodes": done}, indent=1))
        if j["kind"] == "dev" and len(done) == len(j["futures"]) and not j.get("written"):
            j["written"] = True
            j["dir"].mkdir(parents=True, exist_ok=True)
            (j["dir"] / "summary.json").write_text(json.dumps(out, indent=1))
        return out

    def _save_state(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=1))

    # -- requests from ./lab -------------------------------------------------------------------

    def items(self) -> list[dict]:
        out = []
        for it in self.dev.values():
            s = setups.load(it.setup)
            out.append({"id": it.id, "task": it.task, **{k: v for k, v in s.attributes.items() if k != "split"}})
        return out

    def handle(self, req: dict) -> dict:
        cmd = req.get("cmd")
        if cmd == "scenes":
            return {"scenes": self.items()}
        if cmd in ("run", "suite"):
            ids = req.get("scenes") or ["all"]
            picked = list(self.dev.values()) if ids == ["all"] else [self.dev[i] for i in ids if i in self.dev]
            bad = [i for i in ids if i != "all" and i not in self.dev]
            if bad:
                return {"error": f"not dev scenes: {bad} (see ./lab scenes)"}
            seeds = [int(s) for s in req.get("seeds") or [0]]
            if len(seeds) > MAX_DEV_SEEDS:
                return {"error": f"at most {MAX_DEV_SEEDS} seeds per scene per job"}
            taken = sorted({sd for it in picked for sd in seeds if (it.setup, it.task, sd) in self.reserved})
            if taken:
                return {"error": f"seeds {taken} are held out for val or test on these scenes; pick other seeds"}
            r = self.submit("dev", req["skill"], [(it, sd) for it in picked for sd in seeds], record=not req.get("no_video"))
            if cmd == "run":
                return self.status(r["job"], wait_s=float(req.get("wait_s") or 600))
            return r
        if cmd == "val":
            n = self.cfg.val_submissions
            if not self.val:
                return {"error": "this lab has no val suite"}
            if self.state["val_used"] >= n:
                return {"error": f"all {n} val submissions are used", "val_history": self.state["val_history"]}
            self.state["val_used"] += 1
            self._save_state()
            r = self.submit("val", req["skill"], [(it, sd) for it in self.val for sd in it.seeds], record=False)
            r["val_submissions_left"] = n - self.state["val_used"]
            return r
        if cmd == "status":
            return self.status(req["job"], wait_s=float(req.get("wait_s") or 0))
        if cmd == "jobs":
            return {"jobs": [{"job": k, "kind": j["kind"], "skill": j["skill"], "done": sum(f.done() for f in j["futures"]),
                              "total": len(j["futures"])} for k, j in self.jobs.items()]}  # fmt: skip
        if cmd == "budget":
            from chefbot.runtime.models import Ledger

            spent = Ledger(Path(self.ledger[0]), self.ledger[1]).spent()
            out = {"model_usd_spent": round(spent, 4), "model_usd_cap": self.ledger[1],
                   "val_submissions_left": self.cfg.val_submissions - self.state["val_used"], "val_history": self.state["val_history"],
                   "episode_limits": self.limits or {}}  # fmt: skip
            if self.deadline:
                out["research_time_left_min"] = round((self.deadline - time.time()) / 60, 1)
            usage = codex_usage()
            if usage:
                out["codex_weekly_used_percent"] = usage
            return out
        return {"error": f"unknown command {cmd!r}"}

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)


RESEARCHER_HOME = Path(os.environ.get("RESEARCHER_HOME", Path.home()))  # whose ~/.codex holds the researcher's sessions
CODEX_DIR = Path(os.environ["CODEX_HOME"]) if os.environ.get("CODEX_HOME") else None  # the researcher's Codex home, if not ~/.codex


def codex_usage(max_age_s: float | None = None) -> float | None:
    """The weekly Codex usage percent from the newest Codex session log, if any (and, with max_age_s, only if that log
    was written that recently: an old log's percent can be far behind)."""
    root = (CODEX_DIR or RESEARCHER_HOME / ".codex") / "sessions"
    files = sorted(root.glob("*/*/*/*.jsonl"), key=lambda p: p.stat().st_mtime)
    if max_age_s is not None:
        files = [f for f in files if time.time() - f.stat().st_mtime <= max_age_s]
    for f in reversed(files[-5:]):
        best = None
        for line in f.read_text(errors="replace").splitlines():
            if '"rate_limits"' in line:
                try:
                    rl = _find(json.loads(line), "rate_limits")
                    windows = [w for w in (rl.get("primary"), rl.get("secondary")) if w] if rl else []
                    weekly = [w for w in windows if (w.get("window_minutes") or 0) >= 7 * 24 * 60]  # Plus: primary is a 5 h window
                    if windows:
                        w = (weekly or windows)[0]
                        best = 0.0 if (w.get("resets_at") or float("inf")) <= time.time() else float(w["used_percent"])  # reset since
                except (json.JSONDecodeError, TypeError, KeyError):
                    pass
        if best is not None:
            return best
    return None


def _find(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = _find(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find(v, key)
            if r is not None:
                return r
    return None


def main() -> int:
    """Serve a lab without a researcher (for trying ./lab by hand)."""
    import argparse

    from chefbot.autoresearch import config as C
    from chefbot.runtime.episode import sandbox_kind
    from chefbot.runtime.models import load_env
    from chefbot.autoresearch.research import REPO, brief, setup_workspace

    ap = argparse.ArgumentParser()
    ap.add_argument("lab")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--vision-url", help="the vision server (default: VISION_URL, from the environment or .env)")
    ap.add_argument("--hours", type=float, default=None, help="research session length: wrap-up notes near the end, no new jobs after")
    args = ap.parse_args()
    load_env(REPO / ".env")
    args.vision_url = args.vision_url or os.environ.get("VISION_URL")
    cfg = C.load(args.lab)
    setup_workspace(cfg, args.root / "workspace", brief(cfg, args.hours or 3, 40, sandbox_kind(), False))
    lab = Lab(cfg, args.root, args.jobs, args.vision_url)
    os.chmod(args.root / "private", 0o700)
    if args.hours:
        lab.deadline = time.time() + args.hours * 3600
    print(f"lab {cfg.name} serving {args.root / 'workspace'}", flush=True)
    try:
        lab.serve_forever()
    finally:
        lab.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
