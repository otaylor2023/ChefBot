"""Object-creation labs: the researcher writes a generator (families/<family>/contract.md) and the lab builds, checks,
renders and judges its assets. research.py runs these labs as it runs skill labs (labs/<name>.yaml with
`kind: object`).

    dev   ./lab build GEN --seeds ...: full results (assets, checks, renders); ./lab judge JOB: the dev judge's
          scores and flaws (a budget of judge calls)
    val   ./lab val GEN: fresh seeds, the held-out rubric, a stronger judge, the val reference photos: aggregate only,
          a few submissions
    test  after the lab: python -m chefbot.autoresearch.objects.evaluate (fresh seeds, the test references, you)
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import random
import shutil
import subprocess
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from chefbot.autoresearch import lab_server
from chefbot.autoresearch.objects import checks as CK
from chefbot.autoresearch.objects import judge as J
from chefbot.autoresearch.objects import pipeline as P
from chefbot.autoresearch.objects import refs as RF

HERE = Path(__file__).resolve().parent
AUTORESEARCH = HERE.parent
LABS = AUTORESEARCH / "labs"
MAX_SEEDS = 16
WORKER_ENV = ("MUJOCO", "PYOPENGL", "EGL", "CUDA", "GEN_", "CHEFBOT_")
FINAL = "generators/final"


@dataclass
class ObjectLabConfig:
    name: str
    path: Path
    family: str
    brief: str
    example: str
    judge: dict = field(default_factory=dict)  # dev_model, val_model, dev_calls
    val: dict = field(default_factory=dict)  # n, submissions
    test: dict = field(default_factory=dict)  # n
    budgets: dict = field(default_factory=dict)
    start_from: dict | None = None
    model_usd: float = 0.0  # no paid models for generators

    @property
    def val_submissions(self) -> int:
        return int(self.val.get("submissions", 4))

    @property
    def dev_judge_calls(self) -> int:
        return int(self.judge.get("dev_calls", 300))


def kind(name_or_path: str) -> str:
    p = Path(name_or_path)
    p = p if p.suffix else LABS / f"{name_or_path}.yaml"
    return yaml.safe_load(p.read_text()).get("kind", "skill")


def load(name_or_path: str) -> ObjectLabConfig:
    p = Path(name_or_path)
    p = p if p.suffix else LABS / f"{name_or_path}.yaml"
    raw = yaml.safe_load(p.read_text())
    return ObjectLabConfig(p.stem, p, raw["family"], raw["brief"], raw["example"], raw.get("judge") or {}, raw.get("val") or {},
                           raw.get("test") or {}, raw.get("budgets") or {}, raw.get("start_from"))  # fmt: skip


def family_dir(cfg: ObjectLabConfig) -> Path:
    return HERE / "families" / cfg.family


def _gen_packages() -> str:
    return "numpy, scipy, trimesh, manifold3d, coacd, shapely, mapbox-earcut, build123d, mujoco, pillow, pyyaml, networkx, rtree"


def contract_text(cfg: ObjectLabConfig, fam: dict) -> str:
    import mujoco

    g = fam["generation"]
    text = (family_dir(cfg) / fam["contract"]).read_text()
    for k, v in {"{GEN_TIMEOUT_S}": g["timeout_s"], "{MAX_ASSET_MB}": g["max_asset_mb"], "{MAX_COLLISION_GEOMS}": g["max_collision_geoms"],
                 "{MAX_COLLISION_VERTS}": g["max_collision_verts"], "{GEN_PACKAGES}": _gen_packages(), "{MUJOCO_VERSION}": mujoco.__version__}.items():  # fmt: skip
        text = text.replace(k, str(v))
    return text


WEB_TEXT = """   - You have the internet (web search, and commands may download). Use it: product pages, catalogs, reviews,
     patents, papers, mesh and CAD tools. Downloaded things may inform your generator, but the tongs it makes
     must come from its parameters, not be downloaded meshes or photos.
"""


def brief(cfg: ObjectLabConfig, hours: float, usage_cap: float, sandbox: str, web: bool) -> str:
    text = (LABS / cfg.brief).read_text()
    for k, v in {"{HOURS}": f"{hours:g}", "{USAGE_CAP}": f"{usage_cap:g}", "{VAL_SUBMISSIONS}": str(cfg.val_submissions),
                 "{VAL_N}": str(cfg.val.get("n", 12)), "{DEV_JUDGE_CALLS}": str(cfg.dev_judge_calls), "{SANDBOX}": sandbox,
                 "{WEB}": WEB_TEXT if web else "", "{GEN_PYTHON}": os.environ.get("GEN_PYTHON", "python"), "{MAX_SEEDS}": str(MAX_SEEDS)}.items():  # fmt: skip
        text = text.replace(k, v)
    return text


def setup_workspace(cfg: ObjectLabConfig, ws: Path, text: str) -> None:
    fam = CK.load_family(cfg.family)
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "docs").mkdir(exist_ok=True)
    (ws / "generators").mkdir(exist_ok=True)
    shutil.copy(HERE / "lab_cli.py", ws / "lab")
    os.chmod(ws / "lab", 0o755)
    (ws / "docs" / "generator_contract.md").write_text(contract_text(cfg, fam))
    checks_doc = (family_dir(cfg) / "checks.md").read_text()
    thresholds = {k: v for k, v in fam["checks"].items()}
    (ws / "docs" / "checks.md").write_text(checks_doc + "\n## Thresholds (family.yaml)\n\n```yaml\n" + yaml.safe_dump(thresholds, sort_keys=False) + "```\n")
    shutil.copy(family_dir(cfg) / fam["judge"]["rubric"], ws / "docs" / "judge_rubric.md")
    shutil.copy(family_dir(cfg) / "rubric_batch.md", ws / "docs" / "judge_rubric_batch.md")
    refs_ws = ws / "refs"
    if not refs_ws.exists():
        refs_ws.mkdir()
        for it in RF.items(cfg.family, "dev"):
            shutil.copytree(it["dir"], refs_ws / it["id"])
        RF.montage(cfg.family, "dev", refs_ws / "real_tongs.jpg")
        (refs_ws / "README.md").write_text("# Real tongs (the dev reference set)\n\nThe dev judge sees `real_tongs.jpg` next to your "
                                           "renders. Val and test use other products.\n\n" + RF.spec_table(cfg.family, "dev") + "\n")  # fmt: skip
    (ws / "AGENTS.md").write_text(text)
    if not (ws / "generators" / "example").exists():
        shutil.copytree(AUTORESEARCH / "templates" / cfg.example, ws / "generators" / "example")
    if not (ws / "research_log.md").exists():
        (ws / "research_log.md").write_text("# Research log\n\n")
    (ws / ".gitignore").write_text("runs/\n.lab/\n__pycache__/\n")
    if not (ws / ".git").exists():
        subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
        subprocess.run(["git", "add", "-A"], cwd=ws, check=True)
        subprocess.run(["git", "-c", "user.name=lab", "-c", "user.email=lab@localhost", "commit", "-q", "-m", "workspace"], cwd=ws, check=True)


def default_hide(cfg: ObjectLabConfig) -> list[str]:
    """What the researcher must not see: the held-out references and rubric, other runs, the keys."""
    from chefbot.autoresearch.research import REPO, RUNS

    fam_refs = RF.root() / cfg.family
    return [str(fam_refs / "val"), str(fam_refs / "test"), str(fam_refs / "heldout"), str(RUNS), str(REPO / ".env")]


class ObjectLab(lab_server.IpcServer):
    WRAP_UP = f"{FINAL}, commit, REPORT.md"

    def __init__(self, cfg: ObjectLabConfig, root: Path, jobs: int):
        self.cfg, self.root = cfg, Path(root)
        self.fam = CK.load_family(cfg.family)
        self.ws, self.private, self.snapshots = self.root / "workspace", self.root / "private", self.root / "snapshots"
        self.ipc = self.ws / ".lab" / "ipc"
        self.ipc.mkdir(parents=True, exist_ok=True)
        (self.ws / "runs").mkdir(exist_ok=True)
        self.private.mkdir(parents=True, exist_ok=True)
        self.snapshots.mkdir(parents=True, exist_ok=True)
        os.chmod(self.snapshots, 0o755)
        self.ledger = self.private / "judge_ledger.jsonl"
        self.judge_home = self.private / "claude_judge"
        self.dev_refs = self.ws / "refs" / "real_tongs.jpg"
        self.val_refs = RF.montage(cfg.family, "val", self.private / "refs_val.jpg")
        heldout = RF.root() / cfg.family / "heldout" / "rubric.md"
        self.val_rubric = (heldout if heldout.exists() else family_dir(cfg) / self.fam["judge"]["rubric"]).read_text()
        self.dev_rubric = (family_dir(cfg) / self.fam["judge"]["rubric"]).read_text()
        self.batch_rubric = (family_dir(cfg) / "rubric_batch.md").read_text()
        env = {k: v for k, v in os.environ.items() if k.startswith(WORKER_ENV)}
        self.pool = ProcessPoolExecutor(jobs, mp_context=mp.get_context("spawn"), initializer=P.worker_init, initargs=(env,))
        self.jobs: dict[str, dict] = {}
        self.state_path = self.private / "lab_state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"val_used": 0, "val_history": []}
        self.log = open(self.private / "lab_server.log", "a")
        self.lock = threading.Lock()
        self.extra_deny = (str(RF.root()), str(self.private), str(self.ws))

    # -- generators and jobs ---------------------------------------------------------------

    def snapshot(self, gen: str) -> tuple[Path, str]:
        src = (self.ws / gen).resolve()
        if not str(src).startswith(str((self.ws / "generators").resolve()) + os.sep) or not (src / "generate.py").exists():
            raise ValueError(f"a generator is a directory under generators/ with a generate.py (got {gen!r})")
        h = hashlib.sha256()
        for p in sorted(src.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                h.update(str(p.relative_to(src)).encode())
                h.update(p.read_bytes())
        digest = h.hexdigest()[:12]
        dst = self.snapshots / f"{src.name}-{digest}"
        if not dst.exists():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            for p in [dst, *dst.rglob("*")]:
                os.chmod(p, 0o755 if p.is_dir() else 0o644)
        return dst, digest

    def submit(self, kind: str, gen: str, seeds: list[int], video: bool) -> dict:
        snap, digest = self.snapshot(gen)
        job = f"{kind}-{time.strftime('%H%M%S')}-{uuid.uuid4().hex[:4]}"
        base = (self.ws / "runs" / job) if kind == "dev" else (self.private / "val_runs" / job)
        base.mkdir(parents=True, exist_ok=True)
        futs = [self.pool.submit(P.build_one, str(snap), sd, str(base / f"seed_{sd}"), self.fam, video, self.extra_deny) for sd in seeds]
        self.jobs[job] = {"kind": kind, "gen": gen, "digest": digest, "futures": futs, "dir": base, "t0": time.time(), "seeds": seeds}
        self._log({"submit": job, "gen": gen, "digest": digest, "seeds": seeds})
        return {"job": job, "generator_digest": digest, "seeds": seeds if kind == "dev" else len(seeds)}

    def _log(self, entry: dict) -> None:
        self.log.write(json.dumps({"t": time.time(), **entry}) + "\n")
        self.log.flush()

    def _records(self, j: dict) -> list[dict]:
        return [f.result() for f in j["futures"] if f.done()]

    def _finish(self, job: str) -> None:
        """Once every seed is built: the batch image and the summary (and, for val, the judging)."""
        j = self.jobs[job]
        with self.lock:
            if j.get("finishing"):
                return
            j["finishing"] = True
        recs = self._records(j)
        n = P.batch_sheet(recs, j["dir"] / "batch.jpg")
        if j["kind"] == "val":
            P.judge_records(recs, self.fam, self.val_rubric, self.val_refs, self.cfg.judge.get("val_model", "opus"), self.judge_home, self.ledger)
            batch = None
            if n >= 3:
                try:
                    batch = J.judge_batch(j["dir"] / "batch.jpg", n, self.batch_rubric, self.val_refs, self.cfg.judge.get("val_model", "opus"),
                                          self.judge_home, self.ledger)  # fmt: skip
                except J.JudgeError as e:
                    batch = {"error": str(e)}
            summary = P.summarize(recs, batch)
            (j["dir"] / "records.json").write_text(json.dumps({"records": recs, "batch": batch}, indent=1, default=str))
            self.state["val_history"].append({"job": job, "gen": j["gen"], "digest": j["digest"], **summary})
            self._save_state()
            j["summary"] = summary
        else:
            j["summary"] = P.summarize(recs)
            (j["dir"] / "summary.json").write_text(json.dumps({"summary": j["summary"], "records": recs}, indent=1, default=str))
        j["finished"] = True

    def status(self, job: str, wait_s: float = 0.0) -> dict:
        j = self.jobs.get(job)
        if j is None:
            return {"error": f"unknown job {job!r}"}
        t_end = time.time() + max(0.0, wait_s)
        while time.time() < t_end and not j.get("finished"):
            if all(f.done() for f in j["futures"]) and not j.get("finishing"):
                threading.Thread(target=self._finish, args=(job,), daemon=True).start()
            time.sleep(0.5)
        if all(f.done() for f in j["futures"]) and not j.get("finishing"):
            threading.Thread(target=self._finish, args=(job,), daemon=True).start()
        recs = self._records(j)
        out = {"job": job, "kind": j["kind"], "generator": j["gen"], "generator_digest": j["digest"], "done": len(recs),
               "total": len(j["futures"]), "elapsed_s": round(time.time() - j["t0"], 1), "finished": bool(j.get("finished"))}  # fmt: skip
        if j["kind"] == "dev":
            out["dir"] = str(j["dir"].relative_to(self.ws))
            out["assets"] = [{k: r.get(k) for k in ("seed", "ok", "failed", "why", "measured", "gen_s", "harness_error", "render_error") if r.get(k) is not None}
                             for r in sorted(recs, key=lambda r: r["seed"])]  # fmt: skip
        elif j.get("finished"):
            out["note"] = "val: aggregate only"
        if j.get("finished"):
            out["summary"] = j["summary"]
        return out

    def judge(self, job: str, seeds: list[int] | None) -> dict:
        j = self.jobs.get(job)
        if j is None or j["kind"] != "dev":
            return {"error": f"not a dev build job: {job!r}"}
        if not j.get("finished"):
            return {"error": "the build is not finished (./lab wait it first)"}
        recs = [r for r in self._records(j) if r.get("rendered") and (not seeds or r["seed"] in seeds)]
        used = J.calls_made(self.ledger, "asset") + J.calls_made(self.ledger, "batch") - self.state.get("val_calls", 0)
        need = len(recs) + (1 if len(recs) >= 3 else 0)
        left = self.cfg.dev_judge_calls - used
        if need > left:
            return {"error": f"this needs {need} judge calls; {left} are left (judge fewer seeds: --seeds)"}
        model = self.cfg.judge.get("dev_model", "sonnet")
        P.judge_records(recs, self.fam, self.dev_rubric, self.dev_refs if self.dev_refs.exists() else None, model, self.judge_home, self.ledger)
        batch = None
        if len(recs) >= 3:
            sub = j["dir"] / f"batch_{'-'.join(map(str, sorted(r['seed'] for r in recs)))}.jpg" if seeds else j["dir"] / "batch.jpg"
            n = P.batch_sheet(recs, sub)
            try:
                batch = J.judge_batch(sub, n, self.batch_rubric, self.dev_refs if self.dev_refs.exists() else None, model, self.judge_home, self.ledger)
            except J.JudgeError as e:
                batch = {"error": str(e)}
        out = {"job": job, "model": model, "assets": [{"seed": r["seed"], **r.get("judge", {})} for r in recs], "batch": batch,
               "summary": P.summarize(recs, batch), "judge_calls_left": self.cfg.dev_judge_calls - used - need}  # fmt: skip
        (j["dir"] / f"judge-{time.strftime('%H%M%S')}.json").write_text(json.dumps(out, indent=1))
        return out

    def _save_state(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=1))

    # -- ./lab requests --------------------------------------------------------------------------

    def handle(self, req: dict) -> dict:
        cmd = req.get("cmd")
        if cmd == "build":
            seeds = [int(s) for s in req.get("seeds") or [0]]
            if len(seeds) > MAX_SEEDS:
                return {"error": f"at most {MAX_SEEDS} seeds per build"}
            r = self.submit("dev", req["generator"], seeds, video=not req.get("no_video"))
            if req.get("wait_s"):
                return self.status(r["job"], wait_s=float(req["wait_s"]))
            return r
        if cmd == "status":
            return self.status(req["job"], wait_s=float(req.get("wait_s") or 0))
        if cmd == "judge":
            return self.judge(req["job"], [int(s) for s in req.get("seeds") or []] or None)
        if cmd == "val":
            n = self.cfg.val_submissions
            if self.state["val_used"] >= n:
                return {"error": f"all {n} val submissions are used", "val_history": self.state["val_history"]}
            self.state["val_used"] += 1
            k = int(self.cfg.val.get("n", 12))
            self.state["val_calls"] = self.state.get("val_calls", 0) + k + 1  # judge calls that are not the researcher's
            self._save_state()
            seeds = random.SystemRandom().sample(range(1_000_000, 2_000_000_000), k)
            r = self.submit("val", req["generator"], seeds, video=False)
            r["val_submissions_left"] = n - self.state["val_used"]
            return r
        if cmd == "jobs":
            return {"jobs": [{"job": k, "kind": j["kind"], "generator": j["gen"], "done": sum(f.done() for f in j["futures"]),
                              "total": len(j["futures"]), "finished": bool(j.get("finished"))} for k, j in self.jobs.items()]}  # fmt: skip
        if cmd == "budget":
            used = J.calls_made(self.ledger) - self.state.get("val_calls", 0)
            out = {"dev_judge_calls_left": self.cfg.dev_judge_calls - used, "val_submissions_left": self.cfg.val_submissions - self.state["val_used"],
                   "val_history": self.state["val_history"]}  # fmt: skip
            if self.deadline:
                out["research_time_left_min"] = round((self.deadline - time.time()) / 60, 1)
            usage = lab_server.codex_usage()
            if usage:
                out["codex_weekly_used_percent"] = usage
            return out
        return {"error": f"unknown command {cmd!r}"}

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)


def main() -> int:
    """Serve an object lab without a researcher (to try ./lab by hand)."""
    import argparse

    from chefbot.autoresearch.episode import sandbox_kind
    from chefbot.autoresearch.models import load_env
    from chefbot.autoresearch.research import REPO

    ap = argparse.ArgumentParser()
    ap.add_argument("lab")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--hours", type=float, default=None)
    args = ap.parse_args()
    load_env(REPO / ".env")
    cfg = load(args.lab)
    setup_workspace(cfg, args.root / "workspace", brief(cfg, args.hours or 3, 30, sandbox_kind(), True))
    lab = ObjectLab(cfg, args.root, args.jobs)
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
