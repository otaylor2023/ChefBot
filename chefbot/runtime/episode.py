"""One skill episode: the skill's process (sandboxed, speaking the robot API of its interface, skill_api.py) drives
an episode host (host.py) through JSON lines; this side serves its requests and keeps time, and the task scores
the end.

The world does not wait for the skill. Whatever time the skill spends computing (from each reply sent to its next
request) or waiting on its Gemini and vision calls, the simulator is advanced by that much before the next request
is served, the robot holding its last command (the base still driving to its last target). So simulated time is
the episode's wall time: robot_time = sim time, which includes think and model time (reported separately too).
Time the harness itself spends (stepping, rendering) is not charged.

Limits per episode (LIMITS): control steps, think seconds, model dollars, Astra calls, and a hard wall-clock cap.
Which models the skill may call is the episode's `access`: {"llm": bool, "vision": bool or [names], "astra": bool,
"cameras", "arm_only"}. An Astra answer (runtime/astra.py) arrives its real latency after the request, in world time.
Whatever the skill does, the result has the task's score (from privileged measurements).

(agentic-skills-exploration's runtime/episode.py.)
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import platform
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from chefbot.runtime import astra as A
from chefbot.runtime import models as M

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
INTERFACES = {"chefbot-v1": HERE.parent / "interface" / "skill_api.py"}  # the skill-side robot API, by version (frozen once skills use it)
LIMITS = {"control_steps": 1350, "think_s": 90.0, "model_usd": 0.10, "wall_s": 900.0, "astra_calls": 16, "astra_in_flight": 2}  # 1350 steps = 90 s of world time
MAX_ROWS = 600  # 40 s per call, so one move is one call
ACT_OPS = {"act": ("whole", 11), "act_arm": ("arm", 8), "act_joints": ("joints", 8), "act_base": ("base", 3)}  # op -> (host step mode, row width)
STARTUP_GRACE_S = 15.0
CONTROL_HZ = 15.0


def skill_python() -> str:
    """The skill's interpreter: numpy, scipy, opencv (headless), pillow. Default: this one."""
    return os.environ.get("SKILL_PYTHON", sys.executable)


def protected_paths() -> list[str]:
    """What the skill may not read: the repo (code, simulator, results, keys), the researcher's tools, other dirs given."""
    home = Path.home()
    paths = [str(REPO), str(home / "dev"), str(home / ".codex"), str(home / ".ssh"), str(home / ".config"), "/workspace", "/root"]
    for pkg in ("robocasa", "robosuite"):  # where they are, without importing them (and their warnings)
        spec = importlib.util.find_spec(pkg)
        if spec is not None and spec.origin:
            paths.append(str(Path(spec.origin).resolve().parents[1]))
    return paths + [p for p in os.environ.get("SKILL_EXTRA_DENY", "").split(os.pathsep) if p]


def _sandbox_prefix(skill_dir: Path, xchg: Path, deny: list[str], api: Path, python: str | None = None) -> list[str]:
    """OS-level sandbox on top of the in-process guard: sandbox-exec on macOS, bubblewrap on Linux when usable.
    Readable: skill_dir, the interpreter's environment, the API's folder; writable: xchg only. No network."""
    python = python or skill_python()
    if platform.system() == "Darwin" and shutil.which("sandbox-exec"):
        rules = ["(version 1)", "(allow default)", "(deny network*)"]
        rules += [f'(deny file-read* (subpath "{d}"))' for d in deny]
        for a in (skill_dir, xchg, Path(python).parents[1], api.parent):
            rules.append(f'(allow file-read* (subpath "{Path(a).resolve()}"))')
        rules.append(f'(deny file-write* (require-not (require-any (subpath "{xchg.resolve()}") (subpath "/private/var/folders") (literal "/dev/null"))))')
        return ["sandbox-exec", "-p", "\n".join(rules)]
    pre = []
    if platform.system() == "Linux" and os.geteuid() == 0 and _user_exists(SKILL_USER):
        # as root (a pod): the skill runs as its own unprivileged user, which can read only world-readable files
        pre = ["setpriv", f"--reuid={SKILL_USER}", f"--regid={SKILL_USER}", "--clear-groups", "--"]
    if platform.system() == "Linux" and _bwrap_ok():
        args = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--unshare-net", "--die-with-parent"]
        for d in deny:
            if Path(d).exists() and Path(d).is_dir():
                args += ["--tmpfs", d]
        for a in (skill_dir, Path(python).parents[1], api.parent):
            args += ["--ro-bind", str(Path(a).resolve()), str(Path(a).resolve())]
        args += ["--bind", str(xchg.resolve()), str(xchg.resolve())]
        return args + pre
    return pre


SKILL_USER = "skilluser"


def _user_exists(name: str) -> bool:
    import pwd

    try:
        pwd.getpwnam(name)
        return True
    except KeyError:
        return False


_BWRAP: bool | None = None


def _bwrap_ok() -> bool:
    global _BWRAP
    if _BWRAP is None:
        _BWRAP = bool(shutil.which("bwrap")) and subprocess.run(
            ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--unshare-net", "true"], capture_output=True).returncode == 0  # fmt: skip
    return _BWRAP


def sandbox_kind() -> str:
    if platform.system() == "Darwin" and shutil.which("sandbox-exec"):
        return "sandbox-exec + guard"
    kinds = ["bwrap"] if platform.system() == "Linux" and _bwrap_ok() else []
    if platform.system() == "Linux" and os.geteuid() == 0 and _user_exists(SKILL_USER):
        kinds.append(f"user {SKILL_USER}")
    return " + ".join(kinds + ["guard"])


def run_episode(host, task, seed: int, skill_dir: Path, out: Path, ledger: M.Ledger, vision: M.Vision, access: dict,
                interface: str = "chefbot-v1", limits: dict | None = None, record: bool = True, tag: str = "", item: str = "",
                restart: bool = True, args: dict | None = None) -> dict:  # fmt: skip
    """Start `task` on the host at `seed`, run the skill to the end, score it.
    access: which models the skill may call ({"llm", "vision"}), its cameras, whether it is arm only.
    restart: False continues the world as the last skill left it (an agent's session): odom is fixed where the robot
    is now, nothing is reset. args: the skill's arguments (run(robot, **args)); what run returns is `returned`."""
    limits = {**LIMITS, **(limits or {})}
    out.mkdir(parents=True, exist_ok=True)
    host.out, host.record = out, record
    pose = host.start(task, seed) if task is not None and restart else host.start_here()
    if args is None and task is not None and restart:
        args = task.type.skill_args(task) or None  # what the task tells the skill (pick_up: the object's name)
    shm = Path("/dev/shm") if Path("/dev/shm").is_dir() else None
    # The skill may read its API's folder: a copy of the one file, not the package (with its setups and suites).
    api_dir = Path(tempfile.mkdtemp(prefix="chefbot_api_", dir=shm))
    api = Path(shutil.copy(INTERFACES[interface], api_dir / "skill_api.py"))
    os.chmod(api_dir, 0o755)
    os.chmod(api, 0o644)
    xchg = Path(tempfile.mkdtemp(prefix="chefbot_xchg_", dir=shm))
    os.chmod(xchg, 0o777)  # the skill may run as another user
    if args:
        (xchg / "args.json").write_text(json.dumps(args))
        os.chmod(xchg / "args.json", 0o644)
    skill_dir = Path(skill_dir).resolve()
    deny = protected_paths()
    env = {"PATH": "/usr/bin:/bin", "HOME": str(xchg), "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "4",
           "SKILL_DENY": os.pathsep.join(deny), "SKILL_ALLOW": ""}  # fmt: skip
    cmd = _sandbox_prefix(skill_dir, xchg, deny, api) + [skill_python(), "-I", str(api), str(skill_dir), str(xchg)]
    log = open(out / "skill_log.txt", "w")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, env=env, cwd=str(xchg), bufsize=1)
    lines: queue.Queue = queue.Queue()
    threading.Thread(target=lambda: [lines.put(l) for l in proc.stdout] + [lines.put(None)], daemon=True).start()

    trace = open(out / "trace.jsonl", "w")
    calls = open(out / "calls.jsonl", "w")
    clock = {"think_s": 0.0, "model_s": 0.0, "model_usd": 0.0, "n_llm": 0, "n_vision": 0, "n_observe": 0, "n_astra": 0, "astra_s": 0.0,
             "astra_in_tokens": 0, "astra_out_tokens": 0}  # fmt: skip
    tickets: dict[int, dict] = {}  # Astra calls: ticket -> {call, sim_t0, args}
    sessions: dict[str, str] = {}  # Astra conversations: session name -> Codex thread
    transcript = open(out / "astra.jsonl", "w") if access.get("astra") else None
    t_start = time.time()
    reason, images_saved, returned = "skill returned", 0, None
    history: list[dict] = []  # the task's measurements, every control step

    def over(why: str) -> dict:
        return {"episode_over": True, "reason": why}

    def budget() -> dict:
        return {"control_steps": limits["control_steps"] - host.step_count, "think_s": round(limits["think_s"] - clock["think_s"], 2),
                "model_usd": round(limits["model_usd"] - clock["model_usd"], 4),
                **({"astra_calls": limits["astra_calls"] - clock["n_astra"]} if access.get("astra") else {})}  # fmt: skip

    hold = {"row": None, "mode": "whole", "carry": 0.0}

    def hold_row() -> list[float]:
        return hold["row"] if hold["row"] is not None else host.hold_row()

    def trace_step(row, rec, held: bool) -> None:
        meas = host.measure()  # the task's privileged measurements (none on a real robot)
        history.append(meas)
        trace.write(json.dumps({"step": host.step_count, "hold": held, "target": [round(v, 4) for v in row],
                                "pos_err": round(rec["pos_error_m"], 4), "rot_err": round(rec["rot_error_rad"], 4),
                                "base_err": round(rec["base_error_m"], 4), **_rounded(meas), **host.debug()}) + "\n")  # fmt: skip

    def advance(seconds: float) -> None:
        """The world goes on while the skill thinks: step the sim `seconds`, holding the last command.
        A real robot (host.realtime) lived through that time already, holding the command on its own."""
        if getattr(host, "realtime", False):
            return
        hold["carry"] += seconds * CONTROL_HZ
        n = int(hold["carry"])
        hold["carry"] -= n
        row, mode = hold_row(), hold["mode"] if hold["row"] is not None else "whole"
        for _ in range(n):
            if host.step_count >= limits["control_steps"]:
                break
            trace_step(row, host.step(row, mode), True)

    try:
        t_reply, first = time.time(), True
        debt = 0.0  # model-call seconds not yet lived through
        while True:
            left = limits["think_s"] - clock["think_s"]
            try:
                line = lines.get(timeout=max(0.05, min(left, limits["wall_s"] - (time.time() - t_start))))
            except queue.Empty:
                clock["think_s"] += time.time() - t_reply
                reason = "think budget exhausted" if clock["think_s"] >= limits["think_s"] else "wall-clock cap"
                break
            waited = time.time() - t_reply
            think = max(0.0, waited - STARTUP_GRACE_S) if first else waited  # process start + imports are free (up to a point)
            clock["think_s"] += think
            first = False
            advance(think + debt)
            debt = 0.0
            if line is None:
                reason = "skill exited" if proc.poll() not in (0, None) else "skill returned"
                break
            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                log.write(f"[harness] not a request: {line[:200]}\n")
                t_reply = time.time()
                continue
            op = req.get("op")
            t0 = time.time()
            resp: dict
            try:
                if op == "done":
                    returned = req.get("returned")
                    if req.get("error"):
                        reason = f"skill raised {req['error']}"[:200]
                    resp = {}
                elif clock["think_s"] >= limits["think_s"]:
                    resp = over("think budget exhausted")
                elif host.step_count >= limits["control_steps"] and op not in ("log", "budget"):
                    resp = over("time budget exhausted (90 s of world time)")
                elif op == "proprio":
                    resp = {"proprio": host.proprio()}
                elif op == "observe" and not access.get("cameras", True):
                    resp = {"error": "this skill declares cameras: false, so it may not call robot.observe"}
                elif op == "observe" and isinstance(access.get("cameras"), list) and not set(req.get("cameras") or access["cameras"]) <= set(access["cameras"]):
                    resp = {"error": f"this skill declares cameras: {access['cameras']}"}
                elif op == "observe":
                    clock["n_observe"] += 1
                    cams = {}
                    declared = access.get("cameras") if isinstance(access.get("cameras"), list) else None
                    wanted = req.get("cameras") or declared
                    # a real robot grabs only the cameras asked for (each one's frame takes time); a sim renders all
                    rgbd = host.rgbd(wanted) if getattr(host, "realtime", False) and wanted else host.rgbd()
                    unknown = [n for n in req.get("cameras") or [] if n not in rgbd]
                    if unknown:
                        raise KeyError(f"no camera {unknown}; cameras: {sorted(rgbd)}")
                    for name in req.get("cameras") or declared or list(rgbd):
                        c = rgbd[name]
                        rn, dn = f"o{clock['n_observe']}_{name}_rgb.npy", f"o{clock['n_observe']}_{name}_depth.npy"
                        np.save(xchg / rn, c["rgb"])
                        if c["depth"] is None:
                            dn = None
                        else:
                            np.save(xchg / dn, c["depth"])
                        cams[name] = {"rgb": rn, "depth": dn, "K": c["K"].tolist(), "D": c.get("D"), "model": c.get("model", "pinhole"),
                                      "cam_to_odom": c["cam_to_odom"].tolist()}  # fmt: skip
                    resp = {"proprio": host.proprio(), "cameras": cams}
                elif op in ACT_OPS and op not in ("act_arm", "act_joints") and access.get("arm_only"):
                    resp = {"error": f"this skill is arm only: {op} is not available (use act_arm or act_joints; the base holds)"}
                elif op in ACT_OPS:
                    mode, width = ACT_OPS[op]
                    rows = np.asarray(req.get("rows") or [], dtype=float)
                    if rows.ndim != 2 or rows.shape[1] != width or not len(rows) or len(rows) > MAX_ROWS or not np.isfinite(rows).all():
                        resp = {"error": f"rows must be 1..{MAX_ROWS} finite {width}-D rows"}
                    elif host.step_count >= limits["control_steps"]:
                        resp = over("control-step budget exhausted")
                    else:
                        if access.get("confirm_moves") and hasattr(host, "confirm_move"):
                            host.confirm_move(rows, mode)  # a real robot: Enter at the laptop first (skill.yaml confirm_moves)
                        steps = []
                        for r in rows:
                            if host.step_count >= limits["control_steps"]:
                                break
                            rec = host.step(r.tolist(), mode)
                            hold["row"], hold["mode"] = r.tolist(), mode
                            trace_step(hold["row"], rec, False)
                            steps.append({k: rec[k] for k in ("pos_error_m", "rot_error_rad", "base_error_m", "ik_ok")})
                        resp = {"proprio": host.proprio(), "steps": steps}
                elif op in ("gripper_move", "gripper_grasp"):
                    vals = [req.get(k) for k in (("width", "speed") if op == "gripper_move" else ("speed", "force"))]
                    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
                        resp = {"error": f"{op} needs finite numbers"}
                    else:
                        getattr(host, op)(*(float(v) for v in vals))
                        resp = {"proprio": host.proprio()}
                elif op == "llm" and not access.get("llm"):
                    resp = {"error": "this skill may not call Gemini" + (": the only model is robot.astra(...)" if access.get("astra") else "")}
                elif op in ("detect", "segment") and not access.get("vision"):
                    resp = {"error": "this skill may not call the vision models" + (": the only model is robot.astra(...)" if access.get("astra") else "")}
                elif op == "astra_start":
                    if not access.get("astra"):
                        raise A.AstraError("robot.astra is not available to this skill")
                    a = A.check(req)
                    if clock["n_astra"] >= limits["astra_calls"]:
                        raise A.AstraError(f"at most {limits['astra_calls']} Astra calls per episode")
                    if sum(1 for t in tickets.values() if not t.get("collected")) >= limits["astra_in_flight"]:
                        raise A.AstraError(f"at most {limits['astra_in_flight']} Astra calls in flight; collect one first")
                    clock["n_astra"] += 1
                    imgs = [np.load(xchg / Path(n).name) for n in a["images"]]
                    tickets[clock["n_astra"]] = {"call": A.Call(a, imgs, sessions, A.codex_env()), "sim_t0": host.step_count / CONTROL_HZ, "args": a}
                    resp = {"ticket": clock["n_astra"]}
                elif op in ("astra_poll", "astra_wait"):
                    t = tickets.get(int(req.get("ticket") or 0))
                    if t is None:
                        raise A.AstraError("unknown ticket")
                    call, sim_now = t["call"], host.step_count / CONTROL_HZ
                    if op == "astra_poll" and not (call.done() and sim_now >= t["sim_t0"] + call.result["latency_s"]):
                        resp = {"ready": False}
                    else:
                        r = call.wait()
                        debt += max(0.0, t["sim_t0"] + r["latency_s"] - sim_now)  # the answer arrives `latency` after the request, in world time
                        if not t.get("collected"):
                            t["collected"] = True
                            clock["astra_s"] += r["latency_s"]
                            u = r.get("usage") or {}
                            clock["astra_in_tokens"] += int(u.get("input_tokens") or 0)
                            clock["astra_out_tokens"] += int(u.get("output_tokens") or 0)
                            transcript.write(json.dumps({"ticket": req.get("ticket"), "sim_t0": round(t["sim_t0"], 2),
                                                         "args": {k: v for k, v in t["args"].items() if k != "images"},
                                                         "n_images": len(t["args"]["images"]), "result": r}) + "\n")  # fmt: skip
                            transcript.flush()
                        resp = {"ready": True, "ok": r.get("ok"), "text": r.get("text"), "json": r.get("json"), "failure": r.get("error"),
                                "latency_s": r.get("latency_s"), "usage": r.get("usage")}  # fmt: skip
                elif op in ("detect", "segment") and isinstance(access["vision"], list) and \
                        req.get("model", "owlv2" if op == "detect" else "sam2") not in access["vision"]:  # fmt: skip
                    resp = {"error": f"vision model {req.get('model', 'owlv2' if op == 'detect' else 'sam2')!r} is not available; available: {access['vision']}"}
                elif op == "llm":
                    imgs = [np.load(xchg / Path(n).name) for n in req.get("images") or []]
                    clock["n_llm"] += 1
                    r = M.gemini(req["model"], str(req.get("prompt", "")), imgs, int(req.get("max_output_tokens") or 1024), req.get("thinking_budget"),
                                 req.get("response_schema"), req.get("temperature"), ledger, limits["model_usd"] - clock["model_usd"], tag)  # fmt: skip
                    clock["model_usd"] += r["cost_usd"]
                    resp = r
                elif op in ("detect", "segment"):
                    clock["n_vision"] += 1
                    vargs = {k: v for k, v in req.items() if k != "op"}
                    vargs["image"] = str(xchg / Path(req["image"]).name)
                    r = vision(op, **vargs)
                    if op == "segment":
                        mn = f"m{clock['n_vision']}.npy"
                        np.save(xchg / mn, np.load(r["masks"]))
                        Path(r["masks"]).unlink(missing_ok=True)
                        resp = {"masks": mn}
                    else:
                        resp = {"detections": r["detections"]}
                elif op == "models":
                    resp = {"models": M.model_menu()}
                elif op == "budget":
                    resp = {"budget": budget()}
                elif op == "log":
                    log.write(f"[step {host.step_count}] {str(req.get('text'))[:2000]}\n")
                    log.flush()
                    resp = {}
                elif op in ("save_image", "show"):
                    images_saved += 1
                    if images_saved <= 60:
                        (out / "images").mkdir(exist_ok=True)
                        a = np.load(xchg / Path(req["image"]).name)
                        if a.dtype == bool:
                            a = a.astype(np.uint8) * 255
                        name = "".join(ch for ch in str(req.get("name")) if ch.isalnum() or ch in "-_")[:40] or "img"
                        path = out / "images" / f"{images_saved:02d}_{name}.png"
                        iio.imwrite(path, np.asarray(a, dtype=np.uint8))
                        if op == "show" and hasattr(host, "show"):
                            host.show(path)  # a real robot: open it on the laptop
                    resp = {}
                else:
                    resp = {"error": f"unknown op {op!r}"}
            except (M.BudgetExceeded, A.AstraError) as e:
                resp = {"error": str(e)}
            except Exception as e:  # noqa: BLE001
                resp = {"error": f"{type(e).__name__}: {e}"[:500]}
            dt = time.time() - t0
            if op in ("llm", "detect", "segment"):
                clock["model_s"] += dt
                debt += dt  # the world goes on during the call: lived through before the next request
            calls.write(json.dumps({"op": op, "step": host.step_count, "dt_s": round(dt, 3), "error": resp.get("error"),
                                    **({"cost_usd": resp.get("cost_usd"), "model": req.get("model")} if op == "llm" else {})}) + "\n")  # fmt: skip
            if op == "done":
                break
            try:
                proc.stdin.write(json.dumps(resp) + "\n")
                proc.stdin.flush()
            except (BrokenPipeError, OSError):
                reason = "skill exited"
                break
            if resp.get("episode_over"):
                reason = resp["reason"]
            t_reply = time.time()
            if time.time() - t_start > limits["wall_s"]:
                reason = "wall-clock cap"
                break
    finally:
        try:
            proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        trace.close()
        calls.close()
        if transcript:
            transcript.close()
        log.close()
        shutil.rmtree(xchg, ignore_errors=True)
        shutil.rmtree(api_dir, ignore_errors=True)
    history.append(host.measure())  # the world as the skill left it
    score = task.type.score(history, task) if task is not None else {}
    sim_s = host.step_count / CONTROL_HZ
    blocked = (out / "skill_log.txt").read_text(errors="replace").count("[skill sandbox] blocked")
    result = {"item": item, "setup": getattr(host, "setup", None) and host.setup.name, "task": task.name if task else None, "seed": seed,
              **score, "end": reason, "steps": host.step_count, "sim_s": round(sim_s, 2),
              "think_s": round(clock["think_s"], 2), "model_s": round(clock["model_s"], 2),
              "robot_time_s": round(sim_s, 2), "model_usd": round(clock["model_usd"], 5),
              "n_llm": clock["n_llm"], "n_vision": clock["n_vision"], "n_observe": clock["n_observe"],
              "n_astra": clock["n_astra"], "astra_s": round(clock["astra_s"], 2), "astra_in_tokens": clock["astra_in_tokens"],
              "astra_out_tokens": clock["astra_out_tokens"],
              "wall_s": round(time.time() - t_start, 2), "sandbox": sandbox_kind(), "sandbox_blocked": blocked,
              "start_pose": None if pose is None else [pose.x, pose.y, pose.yaw], "returned": returned}  # fmt: skip
    if record:
        host.save_video("video.mp4")
        _keyframes(host, out)
    (out / "result.json").write_text(json.dumps(result, indent=1))
    return result


def _rounded(meas: dict) -> dict:
    """Measurements for the trace, floats rounded."""
    def r(v):
        return round(v, 3) if isinstance(v, float) else {k: r(x) for k, x in v.items()} if isinstance(v, dict) else v
    return {k: r(v) for k, v in meas.items()}


def _keyframes(host, out: Path, every_s: float = 3.0) -> None:
    """One contact sheet of the episode (every few seconds of sim time), for a quick look."""
    if not host.frames:
        return
    k = max(1, int(every_s * 10))
    picks = host.frames[::k][:24]
    small = [f[::2, ::2] for f in picks]
    while len(small) % 4:
        small.append(np.zeros_like(small[0]))
    rows = [np.concatenate(small[i : i + 4], 1) for i in range(0, len(small), 4)]
    iio.imwrite(out / "keyframes.jpg", np.concatenate(rows, 0), quality=80)
