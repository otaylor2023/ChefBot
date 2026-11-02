"""A research run: Codex (the researcher) iterates on a skill for a lab's goal against its dev suite through ./lab,
scoring on val a few times, until the deadline or its usage cap.

    python -m chefbot.autoresearch.research robocasa_pick [--hours 4] [--usage-cap 60] [--jobs 4] [--effort high]
        [--web-search] [--codex-sandbox workspace-write|none] [--hide PATHS] [--remote HOST:WORKSPACE] [--codex-user USER]

The run's directory (runs/labs/<lab>-<time>/, or --root): workspace/ (Codex's working directory, a git repo:
AGENTS.md, docs/, lab, skills/, runs/, research_log.md), private/ (the lab's ledger, val results, Codex's event log:
not the researcher's), snapshots/. The run ends at the deadline, when Codex's weekly usage passes the cap, or after
MAX_SESSIONS turns; Codex is resumed whenever it ends its turn early. Afterwards: python -m chefbot.autoresearch.register.
(agentic-skills-exploration's autoresearch/research.py.)
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from chefbot.autoresearch import config as C
from chefbot.autoresearch import lab_server
from chefbot import embodiments, skills
from chefbot.runtime.episode import INTERFACES, LIMITS, sandbox_kind
from chefbot.runtime.models import load_env

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
REPO = PACKAGE.parent
RUNS = Path(os.environ.get("CHEFBOT_RUNS", REPO / "runs"))  # lab runs and evals; gitignored
MAX_SESSIONS = 400
WRAP_UP_MIN = 40
QUICK_S = 30  # a session shorter than this did nothing much: after a few in a row, back off (or, wrapped up, stop)

WEB_TEXT = """   - You have web search (every query is logged). Use it for general knowledge: papers,
     library and controller docs, robotics methods. Don't use it to look up RoboCasa's
     kitchens, layouts, fixtures or object models, or anything about this lab's scenes.
"""


def brief(cfg: C.LabConfig, hours: float, usage_cap: float, sandbox: str, web_search: bool) -> str:
    """The researcher's AGENTS.md: the lab's brief with the robot, limits and budgets filled in from the config."""
    text = (C.LABS / cfg.brief).read_text()
    prior = ""
    if cfg.start_from:
        prior = "## Prior work\n\n" + cfg.start_from.get("note", "A previous research session's skill is in `prior/`.").strip() + "\n\n"
    for k, v in {"{HOURS}": f"{hours:g}", "{USAGE_CAP}": f"{usage_cap:g}", "{MODEL_USD}": f"{cfg.model_usd:g}", "{LIMITS}": json.dumps(cfg.limits()),
                 "{VAL_SUBMISSIONS}": str(cfg.val_submissions), "{SANDBOX}": sandbox, "{WEB}": WEB_TEXT if web_search else "",
                 "{ROBOT}": embodiments.load(cfg.robot).brief, "{PRIOR}": prior}.items():  # fmt: skip
        text = text.replace(k, v)
    return text


def setup_workspace(cfg: C.LabConfig, ws: Path, text: str) -> None:
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "docs").mkdir(exist_ok=True)
    (ws / "skills").mkdir(exist_ok=True)
    shutil.copy(HERE / "lab_cli.py", ws / "lab")
    os.chmod(ws / "lab", 0o755)
    shutil.copy(INTERFACES[cfg.interface], ws / "docs" / "robot_api.py")
    if cfg.start_from and not (ws / "prior").exists():
        prior = skills.load(cfg.start_from["skill"])
        (ws / "prior").mkdir()
        shutil.copytree(prior.dir, ws / "prior" / "skill", ignore=shutil.ignore_patterns("__pycache__", "skill.yaml", "REPORT.md"))
        if (prior.dir / "REPORT.md").exists():
            shutil.copy(prior.dir / "REPORT.md", ws / "prior" / "REPORT.md")
        for f in cfg.start_from.get("files", []):  # reference files for the prior skill (e.g. the API it was written against)
            shutil.copy(PACKAGE / "labs" / "templates" / f, ws / "prior" / Path(f).name)
    (ws / "AGENTS.md").write_text(text)
    if not (ws / "skills" / "example").exists():
        shutil.copytree(PACKAGE / "labs" / "templates" / (cfg.example or "example_pick"), ws / "skills" / "example")
    if not (ws / "research_log.md").exists():
        (ws / "research_log.md").write_text("# Research log\n\n")
    (ws / ".gitignore").write_text("runs/\n.lab/\n__pycache__/\n")
    if not (ws / ".git").exists():
        subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
        subprocess.run(["git", "add", "-A"], cwd=ws, check=True)
        subprocess.run(["git", "-c", "user.name=lab", "-c", "user.email=lab@localhost", "commit", "-q", "-m", "workspace"], cwd=ws, check=True)


def hide_prefix(hide: list[str], workspace: Path) -> list[str]:
    """bubblewrap around Codex (Linux, no root needed): the whole filesystem read-only, `hide` paths empty, the
    workspace and Codex's own home writable. For a lab where the researcher runs as the same user as the lab."""
    args = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--die-with-parent"]
    for h in hide:
        p = Path(h)
        if p.is_dir():
            args += ["--tmpfs", str(p)]
        elif p.exists():
            args += ["--ro-bind", "/dev/null", str(p)]
    codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    args += ["--tmpfs", "/tmp", "--dir", f"/tmp/user/{os.getuid()}"]  # its own /tmp (Codex's own sandbox makes mount points there)
    for w in (workspace, codex_home):
        args += ["--bind", str(w.resolve()), str(w.resolve())]
    return args + ["--"]


def default_hide() -> list[str]:
    """What the researcher must not see: the lab's setups and suites (val and test), the task types (how episodes are
    scored, the fragile foods' breaking forces), the exported kitchens and generated assets, other runs, the keys."""
    from chefbot.environments.mujoco_kitchen.export import kitchens_root

    return [str(PACKAGE / "setups"), str(PACKAGE / "suites"), str(PACKAGE / "tasks"), str(kitchens_root()), str(RUNS), str(REPO / ".env")]


def codex_cmd(model: str, effort: str, sandbox: str, resume: str | None, prompt: str, search: bool = False, network: bool = False) -> list[str]:
    cmd = ["codex"] + (["--search"] if search else []) + ["exec"] + (["resume", resume] if resume else []) + ["--json", "--skip-git-repo-check", "-m", model,
                                                                         "-c", f"model_reasoning_effort={effort}"]  # fmt: skip
    if network and sandbox == "workspace-write":
        cmd += ["-c", "sandbox_workspace_write.network_access=true"]  # its commands may reach the internet
    if sandbox == "none":
        cmd.append("--dangerously-bypass-approvals-and-sandbox")
    elif resume:
        cmd += ["-c", f'sandbox_mode="{sandbox}"']  # `exec resume` takes no -s
    else:
        cmd += ["-s", sandbox]
    return cmd + ["--", prompt]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lab", help="a lab config (labs/<name>.yaml, or its name)")
    ap.add_argument("--root", type=Path, help="the run's directory (default runs/labs/<lab>-<time>)")
    ap.add_argument("--hours", type=float, help="default: the lab's budgets.hours")
    ap.add_argument("--usage-cap", type=float, help="stop when Codex's weekly usage passes this percent (default: the lab's)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--vision-url", help="the vision server (default: VISION_URL, from the environment or .env)")
    ap.add_argument("--model", default="gpt-6-astra", help="the researcher's model")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--codex-sandbox", default="workspace-write", help="workspace-write, or none inside a container")
    ap.add_argument("--codex-user", default=None, help="run Codex as this user (Linux, as root); its workspace is chowned to it")
    ap.add_argument("--remote", default=None, help="HOST:WORKSPACE of a lab running elsewhere (lab_server there); Codex runs here")
    ap.add_argument("--remote-sandbox", default="user skilluser + guard", help="the remote lab's skill sandbox, for the brief")
    ap.add_argument("--hide", default=None, help="colon list of paths Codex must not see (Linux: run in bubblewrap); 'default' hides "
                    "the lab's setups/ and suites/, runs/ and .env")  # fmt: skip
    ap.add_argument("--web-search", action="store_true", help="give Codex its web search tool (each query is logged); commands stay offline")
    ap.add_argument("--codex-network", action="store_true", help="let Codex's commands reach the network too (downloads)")
    args = ap.parse_args()
    load_env(REPO / ".env")
    args.vision_url = args.vision_url or os.environ.get("VISION_URL")
    from chefbot.autoresearch.generator_lab import lab as OL

    objects = OL.kind(args.lab) == "generator"  # a generator lab (assets or scenes), not a skill lab
    cfg = OL.load(args.lab) if objects else C.load(args.lab)
    final = OL.FINAL if objects else "skills/final"
    hours = args.hours or float(cfg.budgets.get("hours", 3))
    usage_cap = args.usage_cap or float(cfg.budgets.get("codex_usage_pct", 40))
    if not args.usage_cap and "codex_usage_add_pct" in cfg.budgets:  # a cap relative to where usage is now
        start = lab_server.codex_usage(max_age_s=1800)
        if start is None:
            raise SystemExit("codex_usage_add_pct: no Codex session in the last 30 min to read usage from; pass --usage-cap")
        usage_cap = start + float(cfg.budgets["codex_usage_add_pct"])
    root = args.root or RUNS / "labs" / f"{cfg.name}-{time.strftime('%m%d-%H%M%S')}"
    ws, private = root / "workspace", root / "private"
    sandbox = args.remote_sandbox if args.remote else sandbox_kind()
    if objects:
        OL.setup_workspace(cfg, ws, OL.brief(cfg, hours, usage_cap, sandbox, args.web_search or args.codex_network))
    else:
        setup_workspace(cfg, ws, brief(cfg, hours, usage_cap, sandbox, args.web_search))
    private.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg.path, private / "lab.yaml")  # what this run was
    (private / "lab_name").write_text(cfg.name)
    codex_home = Path.home()
    if args.codex_user:
        import pwd

        pw = pwd.getpwnam(args.codex_user)
        codex_home = Path(pw.pw_dir)
        lab_server.RESEARCHER_HOME, lab_server.CODEX_DIR = codex_home, None
        (ws / ".lab" / "ipc").mkdir(parents=True, exist_ok=True)
        (ws / "runs").mkdir(exist_ok=True)
        subprocess.run(["chown", "-R", f"{pw.pw_uid}:{pw.pw_gid}", str(ws)], check=True)
    hide = (OL.default_hide(cfg) if objects else default_hide()) if args.hide == "default" else [h for h in (args.hide or "").split(":") if h]

    t0 = time.time()
    deadline = t0 + hours * 3600
    stop = threading.Event()
    if args.remote:  # the lab runs elsewhere: relay ./lab to it
        from chefbot.autoresearch.relay import Relay

        host, rws = args.remote.split(":", 1)
        lab = Relay(ws, host, rws)
    elif objects:
        lab = OL.GeneratorLab(cfg, root, args.jobs)
        os.chmod(private, 0o700)
        lab.deadline = deadline
    else:
        lab = lab_server.Lab(cfg, root, args.jobs, args.vision_url)
        os.chmod(private, 0o700)  # the lab's ledger, val results: not the researcher's
        lab.deadline = deadline
    threading.Thread(target=lab.serve_forever, args=(stop,), daemon=True).start()
    events = open(private / "codex_events.jsonl", "a")
    run_log = open(private / "research_run.log", "a")

    def note(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        run_log.write(line + "\n")
        run_log.flush()

    note(f"lab {cfg.name}; workspace {ws}; deadline in {hours} h; usage cap {usage_cap}%; model cap ${cfg.model_usd}; "
         f"vision {'on' if args.vision_url else 'off'}; sandbox: {sandbox}; hide: {hide}")  # fmt: skip
    thread_id, sessions, capacity, over_cap = None, 0, False, False
    quick = 0  # sessions in a row that ended within QUICK_S
    proc = None

    def _stop(signum, _frame):  # SIGTERM / SIGINT (a background job ignores SIGINT): end the run cleanly
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    try:
        while sessions < MAX_SESSIONS:
            left_min = (deadline - time.time()) / 60
            usage = lab_server.codex_usage()
            if left_min <= 0:
                note("deadline reached")
                break
            if usage is not None and usage >= usage_cap:
                note(f"Codex weekly usage {usage}% >= cap {usage_cap}%")
                break
            if capacity:
                time.sleep(60)
            if thread_id is None and (ws / "research_log.md").read_text().strip() != "# Research log":  # a restarted run
                prompt = (f"Read AGENTS.md. The lab was restarted, and this is a new conversation: your earlier work this session is in "
                          f"the workspace (research_log.md, {final.split('/')[0]}/, runs/). Continue from your research log. You have {left_min / 60:.1f} "
                          "hours. ./lab budget shows what is left at any time.")  # fmt: skip
            elif thread_id is None:
                prompt = f"Read AGENTS.md and begin. You have {left_min / 60:.1f} hours. ./lab budget shows what is left at any time."
            elif left_min <= WRAP_UP_MIN:
                prompt = (f"{left_min:.0f} minutes are left. Wrap up now: put your best {'generator' if objects else 'skill'} in {final} "
                          "(run it once with ./lab to check it), commit, and write REPORT.md as AGENTS.md describes.")  # fmt: skip
            else:
                prompt = (f"Your turn ended, but the research session continues: {left_min / 60:.1f} hours left. Keep going with the "
                          "next experiment from your research log. (./lab budget for budgets.)")  # fmt: skip
            cmd = codex_cmd(args.model, args.effort, args.codex_sandbox, thread_id, prompt, args.web_search, args.codex_network)
            if hide:
                cmd = hide_prefix(hide, ws) + cmd
            env = None
            if args.codex_user:  # Codex runs as an unprivileged user (no access to the lab's private files)
                cmd = ["setpriv", f"--reuid={args.codex_user}", f"--regid={args.codex_user}", "--init-groups", "--"] + cmd
                env = {"HOME": str(codex_home), "USER": args.codex_user, "PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8"}
            proc = subprocess.Popen(cmd, cwd=ws, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
            q: queue.Queue = queue.Queue()
            threading.Thread(target=lambda: [q.put(l) for l in proc.stdout] + [q.put(None)], daemon=True).start()
            sessions += 1
            session_t0 = time.time()
            capacity, last_check = False, time.time()
            note(f"session {sessions} ({'resume' if thread_id else 'new'})")
            while True:
                try:
                    line = q.get(timeout=5)
                except queue.Empty:
                    line = ""
                if line is None:
                    break
                if line:
                    events.write(line)
                    events.flush()
                    if '"thread.started"' in line and thread_id is None:
                        try:
                            thread_id = json.loads(line).get("thread_id")
                        except json.JSONDecodeError:
                            pass
                    if '"turn.failed"' in line and "capacity" in line:
                        capacity = True
                    if '"agent_message"' in line:
                        try:
                            note("agent: " + json.loads(line)["item"]["text"][:300].replace("\n", " "))
                        except (json.JSONDecodeError, KeyError):
                            pass
                if time.time() > deadline + 15 * 60:  # past the deadline and a grace period: stop it
                    note("stopping Codex (deadline + grace)")
                    proc.terminate()
                    break
                if time.time() - last_check > 300:
                    last_check = time.time()
                    usage = lab_server.codex_usage()
                    if usage is not None and usage >= usage_cap:
                        note(f"stopping Codex: weekly usage {usage}% >= cap {usage_cap}%")
                        proc.terminate()
                        over_cap = True
                        break
            proc.wait()
            if time.time() > deadline or over_cap:
                break
            quick = quick + 1 if time.time() - session_t0 < QUICK_S else 0
            done = (ws / "REPORT.md").exists() and (ws / final).is_dir()
            if done and (deadline - time.time()) / 60 <= WRAP_UP_MIN and quick >= 3:
                note(f"wrapped up (REPORT.md and {final} exist; Codex keeps ending its turn at once): ending the run")
                break
            if quick >= 3:
                time.sleep(min(60, 5 * quick))  # an idle researcher: don't spin
    except KeyboardInterrupt as e:
        note(f"stopped: {e}")
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
        stop.set()
        if hasattr(lab, "close"):
            lab.close()
        if args.remote:
            lab.pull_runs()
        note(f"research run over after {(time.time() - t0) / 3600:.2f} h, {sessions} sessions; usage {lab_server.codex_usage()}%")
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # not waiting at exit on the episode pool's workers: run 2's lab process outlived its run by 10 h
