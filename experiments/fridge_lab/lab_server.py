"""The lab: runs skill episodes for the researcher (Codex) and for the final eval.

Outside Codex's sandbox, it watches <workspace>/.lab/ipc for requests written by the
`./lab` command (lab_cli.py, copied into the workspace) and runs episodes on a pool
of simulator worker processes (each keeps its last kitchen loaded).

    dev   the researcher's scenes: full results (video, trace with the door angle and handle
          pose, skill log) under <workspace>/runs/<job>/
    val   held-out scenes: only aggregate scores go back; at most VAL_SUBMISSIONS per run
    test  never served to the researcher; evaluate.py runs it

Skills are snapshotted when a job is submitted (later edits do not touch running jobs).
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from experiments.fridge_lab import splits

LAB_ROOT = Path(os.environ.get("FRIDGE_LAB_ROOT", Path.home() / "fridge_lab"))
VAL_SUBMISSIONS = 6
MAX_DEV_SEEDS = 5
WRAP_UP_MIN = 40

_host = None  # per worker: (layout, style) -> LabHost


def _worker_init(env: dict) -> None:
    os.environ.update(env)


def _worker_run(scene: dict, seed: int, skill_dir: str, out: str, ledger: tuple, vision_url: str | None, record: bool, tag: str,
                limits: dict | None) -> dict:  # fmt: skip
    global _host
    from experiments.fridge_lab import models as M
    from experiments.fridge_lab.episode import run_skill_episode
    from experiments.fridge_lab.kitchen import LabHost

    try:
        key = (scene["layout"], scene["style"])
        if _host is None or _host[0] != key:
            if _host is not None:
                _host[1].close()
            _host = (key, LabHost(scene["layout"], scene["style"], Path(out), record=record))
        r = run_skill_episode(_host[1], scene, seed, Path(skill_dir), Path(out), M.Ledger(Path(ledger[0]), ledger[1]), M.Vision(vision_url),
                              limits=limits, record=record, tag=tag)  # fmt: skip
        return r
    except Exception:  # noqa: BLE001
        err = traceback.format_exc()[-1500:]
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "harness_error.txt").write_text(err)
        if _host is not None:  # a broken kitchen is rebuilt next time
            try:
                _host[1].close()
            except Exception:  # noqa: BLE001
                pass
            _host = None
        return {"scene": scene["id"], "seed": seed, "error": err.splitlines()[-1], "success60": False, "success85": False, "final_deg": 0.0}


def snapshot_dir() -> Path:
    """Frozen copies of skills as they ran: readable by the skill's user (outside the private directory)."""
    d = LAB_ROOT / "snapshots"
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(d, 0o755)
    return d


def _world_readable(path: Path) -> None:
    for p in [path, *path.rglob("*")]:
        os.chmod(p, 0o755 if p.is_dir() else 0o644)


def summarize(results: list[dict]) -> dict:
    ok = [r for r in results if "error" not in r]
    n = len(results)

    def mean(k):
        v = [r[k] for r in ok if k in r]
        return round(sum(v) / len(v), 3) if v else None

    return {"episodes": n, "harness_errors": n - len(ok), "success60": round(sum(r["success60"] for r in results) / max(n, 1), 3),
            "success85": round(sum(r["success85"] for r in results) / max(n, 1), 3), "mean_final_deg": mean("final_deg"),
            "mean_robot_time_s": mean("robot_time_s"), "mean_sim_s": mean("sim_s"), "mean_think_s": mean("think_s"),
            "mean_model_s": mean("model_s"), "mean_model_usd": mean("model_usd"), "mean_wall_s": mean("wall_s"),
            "sandbox_blocked_total": sum(r.get("sandbox_blocked", 0) for r in ok)}  # fmt: skip


class Lab:
    def __init__(self, workspace: Path, private: Path, jobs: int, vision_url: str | None, model_cap_usd: float,
                 limits: dict | None = None):  # fmt: skip
        self.ws, self.private = Path(workspace), Path(private)
        self.ipc = self.ws / ".lab" / "ipc"
        self.ipc.mkdir(parents=True, exist_ok=True)
        (self.ws / "runs").mkdir(exist_ok=True)
        self.private.mkdir(parents=True, exist_ok=True)
        self.scenes = {s["id"]: s for s in splits.load()["scenes"]}
        self.val_seeds = splits.load()["val_seeds"]
        self.ledger = (str(self.private / "model_ledger.jsonl"), model_cap_usd)
        self.vision_url, self.limits = vision_url, limits
        env = {k: v for k, v in os.environ.items() if k.startswith(("MUJOCO", "PYOPENGL", "GEMINI", "FRIDGE", "EGL", "CUDA"))}
        self.pool = ProcessPoolExecutor(jobs, mp_context=mp.get_context("spawn"), initializer=_worker_init, initargs=(env,))
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
        dst = snapshot_dir() / f"{src.name}-{digest}"
        if not dst.exists():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            _world_readable(dst)
        return dst, digest

    def submit(self, kind: str, skill: str, episodes: list[tuple[dict, int]], record: bool) -> dict:
        snap, digest = self.snapshot(skill)
        job = f"{kind}-{time.strftime('%H%M%S')}-{uuid.uuid4().hex[:4]}"
        base = (self.ws / "runs" / job) if kind == "dev" else (self.private / "val_runs" / job)
        futs = []
        for scene, seed in episodes:
            out = base / f"{scene['id']}_s{seed}"
            futs.append(self.pool.submit(_worker_run, scene, seed, str(snap), str(out), self.ledger, self.vision_url, record,
                                         f"{job}/{scene['id']}_s{seed}", self.limits))  # fmt: skip
        self.jobs[job] = {"kind": kind, "skill": skill, "digest": digest, "futures": futs, "episodes": episodes, "dir": base,
                          "t0": time.time()}  # fmt: skip
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
            out["episodes"] = [{k: r.get(k) for k in ("scene", "seed", "final_deg", "max_deg", "success60", "success85", "robot_time_s", "sim_s",
                                                      "think_s", "model_s", "model_usd", "end", "sandbox_blocked", "error")} for r in done]  # fmt: skip
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

    def handle(self, req: dict) -> dict:
        cmd = req.get("cmd")
        if cmd == "scenes":
            return {"scenes": [{k: s[k] for k in ("id", "type", "model", "door")} for s in self.scenes.values() if s["split"] == "dev"]}
        if cmd in ("run", "suite"):
            ids = req.get("scenes") or ["all"]
            dev = [s for s in self.scenes.values() if s["split"] == "dev"]
            picked = dev if ids == ["all"] else [self.scenes[i] for i in ids if i in self.scenes and self.scenes[i]["split"] == "dev"]
            bad = [i for i in ids if i != "all" and (i not in self.scenes or self.scenes[i]["split"] != "dev")]
            if bad:
                return {"error": f"not dev scenes: {bad} (see ./lab scenes)"}
            seeds = [int(s) for s in req.get("seeds") or [0]]
            if len(seeds) > MAX_DEV_SEEDS:
                return {"error": f"at most {MAX_DEV_SEEDS} seeds per scene per job"}
            r = self.submit("dev", req["skill"], [(s, sd) for s in picked for sd in seeds], record=not req.get("no_video"))
            if cmd == "run":
                return self.status(r["job"], wait_s=float(req.get("wait_s") or 600))
            return r
        if cmd == "val":
            if self.state["val_used"] >= VAL_SUBMISSIONS:
                return {"error": f"all {VAL_SUBMISSIONS} val submissions are used", "val_history": self.state["val_history"]}
            self.state["val_used"] += 1
            self._save_state()
            val = [s for s in self.scenes.values() if s["split"] == "val"]
            r = self.submit("val", req["skill"], [(s, sd) for s in val for sd in self.val_seeds], record=False)
            r["val_submissions_left"] = VAL_SUBMISSIONS - self.state["val_used"]
            return r
        if cmd == "status":
            return self.status(req["job"], wait_s=float(req.get("wait_s") or 0))
        if cmd == "jobs":
            return {"jobs": [{"job": k, "kind": j["kind"], "skill": j["skill"], "done": sum(f.done() for f in j["futures"]),
                              "total": len(j["futures"])} for k, j in self.jobs.items()]}  # fmt: skip
        if cmd == "budget":
            from experiments.fridge_lab.models import Ledger

            spent = Ledger(Path(self.ledger[0]), self.ledger[1]).spent()
            out = {"model_usd_spent": round(spent, 4), "model_usd_cap": self.ledger[1], "val_submissions_left": VAL_SUBMISSIONS - self.state["val_used"],
                   "val_history": self.state["val_history"], "episode_limits": self.limits or {}}  # fmt: skip
            if self.deadline:
                out["research_time_left_min"] = round((self.deadline - time.time()) / 60, 1)
            usage = codex_usage()
            if usage:
                out["codex_weekly_used_percent"] = usage
            return out
        return {"error": f"unknown command {cmd!r}"}

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
        notice = LAB_ROOT / "notice.txt"  # a message for the researcher from us, attached to every answer
        if notice.exists() and notice.read_text().strip():
            resp["lab_notice"] = notice.read_text().strip()
        if left is not None and left <= WRAP_UP_MIN:
            resp["lab_note"] = (f"{max(left, 0):.0f} min left in the research session: wrap up (skills/final, commit, REPORT.md).")
        self.log.write(json.dumps({"t": time.time(), "req": req, "resp_keys": list(resp)}) + "\n")
        self.log.flush()
        tmp = self.ipc / f".resp_{rid}.json"
        tmp.write_text(json.dumps(resp, indent=1))
        tmp.rename(self.ipc / f"resp_{rid}.json")

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)


CODEX_HOME = Path(os.environ.get("FRIDGE_CODEX_HOME", Path.home()))


def codex_usage() -> float | None:
    """The weekly Codex usage percent from the newest Codex session log, if any."""
    root = CODEX_HOME / ".codex" / "sessions"
    files = sorted(root.glob("*/*/*/*.jsonl"), key=lambda p: p.stat().st_mtime)
    for f in reversed(files[-5:]):
        best = None
        for line in f.read_text(errors="replace").splitlines():
            if '"rate_limits"' in line:
                try:
                    rl = _find(json.loads(line), "rate_limits")
                    if rl and rl.get("primary"):
                        best = float(rl["primary"]["used_percent"])
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
    """Serve the lab without a researcher (for trying ./lab by hand)."""
    import argparse

    from experiments.astranomy.host import REPO
    from experiments.fridge_lab.research import setup_workspace
    from experiments.ortho_grasp.query.models import load_env

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=LAB_ROOT)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--vision-url", default=os.environ.get("FRIDGE_VISION_URL"))
    ap.add_argument("--model-usd", type=float, default=1.0)
    ap.add_argument("--hours", type=float, default=None, help="research session length: wrap-up notes near the end, no new jobs after")
    args = ap.parse_args()
    load_env(REPO / ".env")
    setup_workspace(args.root / "workspace", args.hours or 8, 40, args.model_usd)
    lab = Lab(args.root / "workspace", args.root / "private", args.jobs, args.vision_url, args.model_usd)
    os.chmod(args.root / "private", 0o700)
    if args.hours:
        lab.deadline = time.time() + args.hours * 3600
    print(f"lab serving {args.root / 'workspace'}", flush=True)
    try:
        lab.serve_forever()
    finally:
        lab.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
