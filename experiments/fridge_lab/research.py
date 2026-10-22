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
import signal
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


ROBOT_TEXT = {
    "panda": """- **Robot.** A mobile Franka arm with a Robotiq-style parallel gripper, on a holonomic
  base. It sits in a RoboCasa kitchen, about 0.6–1.0 m from a closed fridge door,
  roughly facing it.
- **Sensors.** Three RGB-D cameras: two shoulder cameras on the base and a wrist camera.
  Each comes with intrinsics and its pose. You also get the arm's joint state, the
  flange pose, the gripper and the base odometry. That is what a real robot has, and
  it is all the skill gets.""",
    "yam": """- **Robot.** TidyBot++: a small holonomic base (0.48 m square, 0.34 m tall) with an
  I2RT YAM arm on top and the YAM's linear parallel gripper. It sits in a RoboCasa
  kitchen, about 0.5–0.7 m (base center) from a closed fridge door, roughly facing it.
  - The YAM is a small 6-DoF arm: its base is on a riser 0.12 m ahead of the base center
    and 0.40 m above the floor, and it reaches about 0.6 m. With the gripper level it
    reaches handle heights up to about 0.95 m; tilting the approach upward gets to about
    1.05–1.15 m. Fridge handles mostly come down to 0.7–0.9 m, a few only to 1.0–1.1 m.
  - The wrist (joint 6) turns only ±120°: of the two finger orientations about an approach
    direction, usually only one is reachable.
  - Joint torques are small (28 N·m on joints 1–3, 10 N·m on 4–6). The base is much
    stronger: let it do the pulling.
  - The gripper: fingers slide straight across, 0–94 mm apart, with 50 N of grip force.
    Each fingertip pad is 27 mm long (along the fingers) and 28 mm wide.
- **Sensors.** Three cameras, as on the real robot this one copies (two with depth):
  - `pole`: a ZED 2 on a pole beside the arm, about 0.72 m up and 0.18 m left of center,
    turned about 25° to the right (across the arm) and tilted about 12° down;
  - `base`: a ZED 2 low on the front of the base (about 0.39 m up), looking straight ahead;
  - `wrist`: a fisheye USB camera (Arducam IMX291, 160° diagonal, about 140° x 78° here) on the wrist,
    above the fingers: color only, no depth (`depth` is None); calibrated with OpenCV's fisheye model (`K`, `D`).
  The ZEDs are wide (about 100° x 68°). Like a real robot's, the cameras are not exactly
  where the drawings say: every episode each one sits up to a few cm off and a few degrees
  turned, with a slightly different field of view. Each observation comes with that
  camera's true intrinsics and pose (its calibration), so use them rather than assuming
  a fixed view. You also get the arm's joint state (6 joints), the flange pose, the
  gripper and the base odometry. That is what a real robot has, and it is all the skill gets.""",
}

WEB_TEXT = """   - You have web search (every query is logged). Use it for general knowledge: papers,
     library and controller docs, robotics methods. Don't use it to look up RoboCasa's
     kitchens, layouts, fixtures or fridge models, or anything about this lab's scenes.
"""

PRIOR_TEXT = {"panda": """## Prior work

A previous research session wrote a fridge-opening skill for a different robot (a
mobile Franka with a Robotiq gripper, bigger reach, a 7-DoF arm) under the same rules
and lab. It is in `prior/`: the skill (`prior/skill/`), its `REPORT.md` and its
`research_log.md`. On that robot it opened the held-out val fridges 28/28 past 60° and
26/28 past 85° in about 21 s for about $0.0024 per episode. Its perception (finding
doors, handles and hinges from RGB-D) does not depend on the robot; its grasping and
pulling do. Start from it, but check every assumption against this robot.

""", "yam": """## Prior work

A previous research session wrote a fridge-opening skill for this robot under the same
rules and lab, but with a different camera set (two level cameras on a 1 m mast at the
back of the base) and the arm mounted at the base center, 6 cm lower. It is in `prior/`:
the skill (`prior/skill/`), its `REPORT.md` and its `research_log.md`. Its grasping and
pulling transfer, after moving the stance by the arm's new offset; its perception was
built for the old cameras' views and names. Start from it, and make its perception work
from the new cameras and hold up when they shift between episodes.

"""}


def setup_workspace(ws: Path, hours: float, usage_cap: float, model_usd: float, sandbox: str | None = None,
                    robot: str = "panda", prior: Path | None = None, prior_robot: str = "panda", web_search: bool = False,
                    astra: bool = False) -> None:
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "docs").mkdir(exist_ok=True)
    (ws / "skills").mkdir(exist_ok=True)
    shutil.copy(HERE / "lab_cli.py", ws / "lab")
    os.chmod(ws / "lab", 0o755)
    shutil.copy(HERE / "skill_api" / "fridge_robot.py", ws / "docs" / "fridge_robot.py")
    text = (HERE / ("AGENTS_astra.md" if astra else "AGENTS.md")).read_text()
    text = text.replace("{HOURS}", f"{hours:g}").replace("{USAGE_CAP}", f"{usage_cap:g}").replace("{MODEL_USD}", f"{model_usd:g}")
    limits = {k: v for k, v in LIMITS.items() if k in ("control_steps", "think_s", "wall_s", "astra_calls", "astra_in_flight")} if astra else \
        {k: v for k, v in LIMITS.items() if not k.startswith("astra")}
    text = text.replace("{LIMITS}", json.dumps(limits)).replace("{VAL_SUBMISSIONS}", str(lab_server.VAL_SUBMISSIONS))
    text = text.replace("{SANDBOX}", sandbox or sandbox_kind())
    text = text.replace("{WEB}", WEB_TEXT if web_search else "")
    text = text.replace("{ROBOT}", ROBOT_TEXT[robot]).replace("{PRIOR}", PRIOR_TEXT[prior_robot] if prior else "")
    if prior and not (ws / "prior").exists():
        # prior/: a finished workspace (skills/final, REPORT.md, research_log.md).
        (ws / "prior").mkdir()
        shutil.copytree(prior / "skills" / "final", ws / "prior" / "skill", ignore=shutil.ignore_patterns("__pycache__"))
        for f in ("REPORT.md", "research_log.md"):
            shutil.copy(prior / f, ws / "prior" / f)
    (ws / "AGENTS.md").write_text(text)
    if not (ws / "skills" / "example").exists():
        shutil.copytree(HERE / ("example_astra" if astra else "example_skill"), ws / "skills" / "example")
    if not (ws / "research_log.md").exists():
        (ws / "research_log.md").write_text("# Research log\n\n")
    (ws / ".gitignore").write_text("runs/\n.lab/\n__pycache__/\n")
    if not (ws / ".git").exists():
        subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
        subprocess.run(["git", "add", "-A"], cwd=ws, check=True)
        subprocess.run(["git", "-c", "user.name=lab", "-c", "user.email=lab@localhost", "commit", "-q", "-m", "workspace"], cwd=ws, check=True)


def hide_prefix(hide: list[str], workspace: Path) -> list[str]:
    """bubblewrap around Codex (no root needed): the whole filesystem read-only, `hide` paths empty, the workspace and
    Codex's own home writable. For a lab where the researcher runs as the same user as the lab (no separate account)."""
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


def codex_cmd(model: str, effort: str, sandbox: str, resume: str | None, prompt: str, search: bool = False) -> list[str]:
    cmd = ["codex"] + (["--search"] if search else []) + ["exec"] + (["resume", resume] if resume else []) + ["--json", "--skip-git-repo-check", "-m", model,
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
    ap.add_argument("--robot", default=os.environ.get("FRIDGE_ROBOT", "panda"), choices=sorted(ROBOT_TEXT),
                    help="the lab's robot (set FRIDGE_ROBOT to the same on the lab's side), for the brief")
    ap.add_argument("--prior", type=Path, default=None, help="a finished workspace to give the researcher as prior work")
    ap.add_argument("--hide", default=os.environ.get("FRIDGE_HIDE", ""), help="colon list of paths Codex must not see "
                    "(run in bubblewrap; for a lab without a separate researcher account)")
    ap.add_argument("--web-search", action="store_true", help="give Codex its web search tool (each query is logged); commands stay offline")
    ap.add_argument("--prior-robot", default="panda", choices=sorted(ROBOT_TEXT), help="the robot the prior work was done on")
    args = ap.parse_args()
    load_env(REPO / ".env")
    ws, private = args.root / "workspace", args.root / "private"
    setup_workspace(ws, args.hours, args.usage_cap, args.model_usd, args.remote_sandbox if args.remote else None, args.robot, args.prior, args.prior_robot, args.web_search,
                    os.environ.get("FRIDGE_MODELS") == "astra")
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
            cmd = codex_cmd(args.model, args.effort, args.codex_sandbox, thread_id, prompt, args.web_search)
            if args.hide:
                cmd = hide_prefix([h for h in args.hide.split(":") if h], ws) + cmd
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
    raise SystemExit(main())
