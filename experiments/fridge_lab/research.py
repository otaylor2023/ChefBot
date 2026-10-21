"""The overnight research run: Codex (GPT-6 Astra) iterates on a fridge-opening skill
against the dev scenes through ./lab, scoring on val a few times, until the deadline.

    python -m experiments.fridge_lab.research --hours 8 --usage-cap 40 --model-usd 10 \
        [--jobs 4] [--vision-url http://127.0.0.1:8777] [--effort high] [--codex-sandbox workspace-write|none]

Layout under $FRIDGE_LAB_ROOT (default ~/fridge_lab):
  workspace/   Codex's working directory (a git repo): AGENTS.md, docs/, lab, skills/, runs/, research_log.md
  private/     the lab's ledger, val results, skill snapshots, Codex's event log (not for the researcher)
The run ends at the deadline, when Codex's weekly usage passes --usage-cap percent, or
after MAX_SESSIONS turns. Resumes Codex whenever it ends its turn early.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path

from experiments.astranomy.host import REPO
from experiments.fridge_lab import lab_server
from experiments.fridge_lab.episode import LIMITS, sandbox_kind
from experiments.ortho_grasp.query.models import load_env

HERE = Path(__file__).resolve().parent
MAX_SESSIONS = 400
WRAP_UP_MIN = 40


def setup_workspace(ws: Path, hours: float, usage_cap: float, model_usd: float, sandbox: str | None = None) -> None:
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "docs").mkdir(exist_ok=True)
    (ws / "skills").mkdir(exist_ok=True)
    shutil.copy(HERE / "lab_cli.py", ws / "lab")
    os.chmod(ws / "lab", 0o755)
    shutil.copy(HERE / "skill_api" / "fridge_robot.py", ws / "docs" / "fridge_robot.py")
    text = (HERE / "AGENTS.md").read_text()
    text = text.replace("{HOURS}", f"{hours:g}").replace("{USAGE_CAP}", f"{usage_cap:g}").replace("{MODEL_USD}", f"{model_usd:g}")
    text = text.replace("{LIMITS}", json.dumps(LIMITS)).replace("{VAL_SUBMISSIONS}", str(lab_server.VAL_SUBMISSIONS))
    text = text.replace("{SANDBOX}", sandbox or sandbox_kind())
    (ws / "AGENTS.md").write_text(text)
    if not (ws / "skills" / "example").exists():
        shutil.copytree(HERE / "example_skill", ws / "skills" / "example")
    if not (ws / "research_log.md").exists():
        (ws / "research_log.md").write_text("# Research log\n\n")
    (ws / ".gitignore").write_text("runs/\n.lab/\n__pycache__/\n")
    if not (ws / ".git").exists():
        subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
        subprocess.run(["git", "add", "-A"], cwd=ws, check=True)
        subprocess.run(["git", "-c", "user.name=lab", "-c", "user.email=lab@localhost", "commit", "-q", "-m", "workspace"], cwd=ws, check=True)


def codex_cmd(model: str, effort: str, sandbox: str, resume: str | None, prompt: str) -> list[str]:
    cmd = ["codex", "exec"] + (["resume", resume] if resume else []) + ["--json", "--skip-git-repo-check", "-m", model,
                                                                         "-c", f"model_reasoning_effort={effort}"]  # fmt: skip
    if sandbox == "none":
        cmd.append("--dangerously-bypass-approvals-and-sandbox")
    elif resume:
        cmd += ["-c", f'sandbox_mode="{sandbox}"']  # `exec resume` takes no -s
    else:
        cmd += ["-s", sandbox]
    return cmd + ["--", prompt]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--usage-cap", type=float, default=40.0, help="stop when Codex's weekly usage passes this percent")
    ap.add_argument("--model-usd", type=float, default=10.0, help="total Gemini spend for all dev and val episodes")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--vision-url", default=os.environ.get("FRIDGE_VISION_URL"))
    ap.add_argument("--model", default="gpt-6-astra")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--codex-sandbox", default="workspace-write", help="workspace-write, or none inside a container")
    ap.add_argument("--root", type=Path, default=lab_server.LAB_ROOT)
    ap.add_argument("--codex-user", default=None, help="run Codex as this user (Linux, as root); its workspace is chowned to it")
    ap.add_argument("--remote", default=None, help="HOST:WORKSPACE of a lab running elsewhere (lab_server.py there); Codex runs here")
    ap.add_argument("--remote-sandbox", default="user fridgeskill + guard", help="the remote lab's skill sandbox, for the brief")
    args = ap.parse_args()
    load_env(REPO / ".env")
    ws, private = args.root / "workspace", args.root / "private"
    setup_workspace(ws, args.hours, args.usage_cap, args.model_usd, args.remote_sandbox if args.remote else None)
    codex_home = Path.home()
    if args.codex_user:
        import pwd

        pw = pwd.getpwnam(args.codex_user)
        codex_home = Path(pw.pw_dir)
        lab_server.CODEX_HOME = codex_home
        (ws / ".lab" / "ipc").mkdir(parents=True, exist_ok=True)
        (ws / "runs").mkdir(exist_ok=True)
        subprocess.run(["chown", "-R", f"{pw.pw_uid}:{pw.pw_gid}", str(ws)], check=True)

    private.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    deadline = t0 + args.hours * 3600
    stop = threading.Event()
    if args.remote:  # the lab runs elsewhere: relay ./lab to it
        from experiments.fridge_lab.relay import Relay

        host, rws = args.remote.split(":", 1)
        lab = Relay(ws, host, rws)
    else:
        lab = lab_server.Lab(ws, private, args.jobs, args.vision_url, args.model_usd)
        os.chmod(private, 0o700)  # the lab's ledger, val results, snapshots: not the researcher's
        lab.deadline = deadline
    threading.Thread(target=lab.serve_forever, args=(stop,), daemon=True).start()
    events = open(private / "codex_events.jsonl", "a")
    run_log = open(private / "research_run.log", "a")

    def note(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        run_log.write(line + "\n")
        run_log.flush()

    note(f"workspace {ws}; deadline in {args.hours} h; usage cap {args.usage_cap}%; model cap ${args.model_usd}; "
         f"vision {'on' if args.vision_url else 'off'}; skill sandbox: {sandbox_kind()}")  # fmt: skip
    thread_id, sessions, capacity, over_cap = None, 0, False, False
    try:
        while sessions < MAX_SESSIONS:
            left_min = (deadline - time.time()) / 60
            usage = lab_server.codex_usage()
            if left_min <= 0:
                note("deadline reached")
                break
            if usage is not None and usage >= args.usage_cap:
                note(f"Codex weekly usage {usage}% >= cap {args.usage_cap}%")
                break
            if capacity:
                time.sleep(60)
            if thread_id is None:
                prompt = (f"Read AGENTS.md and begin. You have {left_min / 60:.1f} hours. ./lab budget shows what is left at any time.")
            elif left_min <= WRAP_UP_MIN:
                prompt = (f"{left_min:.0f} minutes are left. Wrap up now: put your best skill in skills/final (run it once with ./lab to "
                          "check it), commit, and write REPORT.md as AGENTS.md describes.")  # fmt: skip
            else:
                prompt = (f"Your turn ended, but the research session continues: {left_min / 60:.1f} hours left. Keep going with the "
                          "next experiment from your research log. (./lab budget for budgets.)")  # fmt: skip
            cmd = codex_cmd(args.model, args.effort, args.codex_sandbox, thread_id, prompt)
            env = None
            if args.codex_user:  # the pod: Codex runs as an unprivileged user (no access to the lab's private files)
                cmd = ["setpriv", f"--reuid={args.codex_user}", f"--regid={args.codex_user}", "--init-groups", "--"] + cmd
                env = {"HOME": str(codex_home), "USER": args.codex_user, "PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8"}
            proc = subprocess.Popen(cmd, cwd=ws, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
            q: queue.Queue = queue.Queue()
            threading.Thread(target=lambda: [q.put(l) for l in proc.stdout] + [q.put(None)], daemon=True).start()
            sessions += 1
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
                    if usage is not None and usage >= args.usage_cap:
                        note(f"stopping Codex: weekly usage {usage}% >= cap {args.usage_cap}%")
                        proc.terminate()
                        over_cap = True
                        break
            proc.wait()
            if time.time() > deadline or over_cap:
                break
    finally:
        stop.set()
        if hasattr(lab, "close"):
            lab.close()
        if args.remote:
            lab.pull_runs()
        note(f"research run over after {(time.time() - t0) / 3600:.2f} h, {sessions} sessions; usage {lab_server.codex_usage()}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
