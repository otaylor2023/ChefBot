"""One skill episode: the skill's process (sandboxed, see skill_api/fridge_robot.py)
drives a LabHost through JSON lines; this side serves its requests and keeps time.

The world does not wait for the skill. Whatever time the skill spends computing (from
each reply sent to its next request) or waiting on its Gemini and vision calls, the
simulator is advanced by that much before the next request is served, the robot
holding its last command (the base still driving to its last target). So simulated
time is the episode's wall time: robot_time = sim time, which includes think and model
time (reported separately too). Time the harness itself spends (stepping, rendering)
is not charged.

Limits per episode (LIMITS): control steps, think seconds, model dollars, and a hard
wall-clock cap. Whatever the skill does, the result has the final and max door angles
(privileged, from the simulator) and success at 60 and 85 degrees on the final angle.
"""

from __future__ import annotations

import json
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

from experiments.fridge_lab import astra as A
from experiments.fridge_lab import models as M

SKILL_API = Path(__file__).resolve().parent / "skill_api" / "fridge_robot.py"
LIMITS = {"control_steps": 1350, "think_s": 90.0, "model_usd": 0.10, "wall_s": 900.0, "astra_calls": 16, "astra_in_flight": 2}  # 1350 steps = 90 s of world time
# Which models a skill may call: "gemini" (Gemini + OWLv2/DINO/SAM, the skill labs) or "astra" (GPT-6 Astra through Codex only).
MODELS = os.environ.get("FRIDGE_MODELS", "gemini")
MAX_ROWS = 150
STARTUP_GRACE_S = 15.0


def skill_python() -> str:
    return os.environ.get("FRIDGE_SKILL_PYTHON", str(Path.home() / "fridge_lab" / "venvs" / "skill" / "bin" / "python"))


def protected_paths() -> list[str]:
    """What the skill may not read: the code and simulator, the lab's private data, keys, the researcher's tools."""
    from experiments.astranomy.host import REPO

    home = Path.home()
    paths = [str(REPO), str(home / "dev"), str(home / ".codex"), str(home / ".ssh"), str(home / ".config"),
             str(Path(os.environ.get("FRIDGE_LAB_ROOT", home / "fridge_lab"))), "/workspace", "/root"]  # fmt: skip
    try:
        import robocasa

        paths.append(str(Path(robocasa.__file__).resolve().parents[1]))
    except ImportError:
        pass
    return paths + [p for p in os.environ.get("FRIDGE_EXTRA_DENY", "").split(os.pathsep) if p]


def _sandbox_prefix(skill_dir: Path, xchg: Path, deny: list[str]) -> list[str]:
    """OS-level sandbox on top of the in-process guard: sandbox-exec on macOS, bubblewrap on Linux when usable."""
    if platform.system() == "Darwin" and shutil.which("sandbox-exec"):
        rules = ["(version 1)", "(allow default)", "(deny network*)"]
        rules += [f'(deny file-read* (subpath "{d}"))' for d in deny]
        for a in (skill_dir, xchg, Path(skill_python()).parents[1], SKILL_API.parent):
            rules.append(f'(allow file-read* (subpath "{Path(a).resolve()}"))')
        rules.append(f'(deny file-write* (require-not (require-any (subpath "{xchg.resolve()}") (subpath "/private/var/folders") (literal "/dev/null"))))')
        return ["sandbox-exec", "-p", "\n".join(rules)]
    pre = []
    if platform.system() == "Linux" and os.geteuid() == 0 and _user_exists(SKILL_USER):
        # as root (the pod): the skill runs as its own unprivileged user, which can read only world-readable files
        pre = ["setpriv", f"--reuid={SKILL_USER}", f"--regid={SKILL_USER}", "--clear-groups", "--"]
    if platform.system() == "Linux" and _bwrap_ok():
        args = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--unshare-net", "--die-with-parent"]
        for d in deny:
            if Path(d).exists() and Path(d).is_dir():
                args += ["--tmpfs", d]
        for a in (skill_dir, Path(skill_python()).parents[1], SKILL_API.parent):
            args += ["--ro-bind", str(Path(a).resolve()), str(Path(a).resolve())]
        args += ["--bind", str(xchg.resolve()), str(xchg.resolve())]
        return args + pre
    return pre


SKILL_USER = "fridgeskill"


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


def run_skill_episode(host, scene: dict, seed: int, skill_dir: Path, out: Path, ledger: M.Ledger, vision: M.Vision,
                      limits: dict | None = None, record: bool = True, tag: str = "", astra_replay: list | None = None) -> dict:  # fmt: skip
    """Reset the host at the scene's door with this seed, run the skill to the end, score it.
    astra_replay: Astra answers recorded in another episode, given back in order instead of calling Astra (the
    answer-swap check: does the skill still succeed when the model's decisions are about a different fridge?)."""
    limits = {**LIMITS, **(limits or {})}
    out.mkdir(parents=True, exist_ok=True)
    host.out, host.record = out, record
    host.success_deg = 60.0
    pose = host.find_start(scene["door"], seed)
    if pose is None:
        raise RuntimeError(f"no clear start pose for {scene['id']} seed {seed}")
    shm = Path("/dev/shm") if Path("/dev/shm").is_dir() else None
    xchg = Path(tempfile.mkdtemp(prefix="fridge_xchg_", dir=shm))
    os.chmod(xchg, 0o777)  # the skill may run as another user
    skill_dir = Path(skill_dir).resolve()
    deny = protected_paths()
    env = {"PATH": "/usr/bin:/bin", "HOME": str(xchg), "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "4",
           "FRIDGE_DENY": os.pathsep.join(deny), "FRIDGE_ALLOW": ""}  # fmt: skip
    cmd = _sandbox_prefix(skill_dir, xchg, deny) + [skill_python(), "-I", str(SKILL_API), str(skill_dir), str(xchg)]
    log = open(out / "skill_log.txt", "w")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, env=env, cwd=str(xchg), bufsize=1)
    lines: queue.Queue = queue.Queue()
    threading.Thread(target=lambda: [lines.put(l) for l in proc.stdout] + [lines.put(None)], daemon=True).start()

    trace = open(out / "trace.jsonl", "w")
    calls = open(out / "calls.jsonl", "w")
    clock = {"think_s": 0.0, "model_s": 0.0, "model_usd": 0.0, "n_llm": 0, "n_vision": 0, "n_observe": 0, "n_astra": 0, "astra_s": 0.0,
             "astra_in_tokens": 0, "astra_out_tokens": 0}  # fmt: skip
    tickets: dict[int, dict] = {}  # Astra calls: ticket -> {call, sim_t0}
    sessions: dict[str, str] = {}
    transcript = open(out / "astra.jsonl", "w") if MODELS == "astra" else None
    replay = list(astra_replay or [])
    t_start = time.time()
    reason, max_deg, images_saved = "skill returned", 0.0, 0

    def over(why: str) -> dict:
        return {"episode_over": True, "reason": why}

    def budget() -> dict:
        return {"control_steps": limits["control_steps"] - host.step_count, "think_s": round(limits["think_s"] - clock["think_s"], 2),
                "model_usd": round(limits["model_usd"] - clock["model_usd"], 4)}  # fmt: skip

    from experiments.astranomy.host import quat_wxyz

    hold = {"row": None, "carry": 0.0}

    def hold_row() -> list[float]:
        if hold["row"] is not None:
            return hold["row"]
        p = host.proprio()
        F = np.array(p["flange_pose_odom"])
        return [*F[:3, 3].tolist(), *quat_wxyz(F[:3, :3]), 1.0 if p["gripper_closed_cmd"] else 0.0, *p["base_pose_odom"]]

    def record(row, rec, held: bool) -> None:
        nonlocal max_deg
        dbg = host.debug()
        ang = [d["angle_deg"] for d in dbg["doors"]]  # none on a real robot
        max_deg = max(max_deg, *ang) if ang else max_deg
        trace.write(json.dumps({"step": host.step_count, "hold": held, "target": [round(v, 4) for v in row],
                                "pos_err": round(rec["pos_error_m"], 4), "rot_err": round(rec["rot_error_rad"], 4),
                                "base_err": round(rec["base_error_m"], 4), "door_deg": [round(a, 2) for a in ang],
                                "handle_odom": [[round(v, 3) for v in d["handle_odom"]] for d in dbg["doors"]],
                                "touching": dbg["robot_touching"]}) + "\n")  # fmt: skip

    def advance(seconds: float) -> None:
        """The world goes on while the skill thinks: step the sim `seconds`, holding the last command.
        A real robot (host.realtime) lived through that time already, holding the command on its own."""
        if getattr(host, "realtime", False):
            return
        hold["carry"] += seconds * 15.0
        n = int(hold["carry"])
        hold["carry"] -= n
        row = hold_row()
        for _ in range(n):
            if host.step_count >= limits["control_steps"]:
                break
            record(row, host.step(row), True)

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
                    resp = {}
                elif clock["think_s"] >= limits["think_s"]:
                    resp = over("think budget exhausted")
                elif host.step_count >= limits["control_steps"] and op not in ("log", "budget"):
                    resp = over("time budget exhausted (90 s of world time)")
                elif op == "proprio":
                    resp = {"proprio": host.proprio()}
                elif op == "observe":
                    clock["n_observe"] += 1
                    cams = {}
                    rgbd = host.rgbd()
                    for name in req.get("cameras") or list(rgbd):
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
                elif op == "act":
                    rows = np.asarray(req.get("rows") or [], dtype=float)
                    if rows.ndim != 2 or rows.shape[1] != 11 or not len(rows) or len(rows) > MAX_ROWS or not np.isfinite(rows).all():
                        resp = {"error": f"rows must be 1..{MAX_ROWS} finite 11-D rows"}
                    elif host.step_count >= limits["control_steps"]:
                        resp = over("control-step budget exhausted")
                    else:
                        steps = []
                        for r in rows:
                            if host.step_count >= limits["control_steps"]:
                                break
                            rec = host.step(r.tolist())
                            hold["row"] = r.tolist()
                            record(hold["row"], rec, False)
                            steps.append({k: rec[k] for k in ("pos_error_m", "rot_error_rad", "base_error_m", "ik_ok")})
                        resp = {"proprio": host.proprio(), "steps": steps}
                elif op in ("llm", "detect", "segment") and MODELS == "astra":
                    resp = {"error": "this lab has no Gemini or vision models: the only model is robot.astra(...)"}
                elif op == "astra_start":
                    if MODELS != "astra":
                        raise A.AstraError("robot.astra is not available in this lab")
                    a = A.check(req)
                    if clock["n_astra"] >= limits["astra_calls"]:
                        raise A.AstraError(f"at most {limits['astra_calls']} Astra calls per episode")
                    if sum(1 for t in tickets.values() if not t.get("collected")) >= limits["astra_in_flight"]:
                        raise A.AstraError(f"at most {limits['astra_in_flight']} Astra calls in flight; collect one first")
                    clock["n_astra"] += 1
                    imgs = [np.load(xchg / Path(n).name) for n in a["images"]]
                    call = A.Replay(replay.pop(0) if replay else {"ok": False, "error": "no recorded answer", "latency_s": 0.0}) \
                        if astra_replay is not None else A.Call(a, imgs, sessions, A.codex_env())
                    tickets[clock["n_astra"]] = {"call": call, "sim_t0": host.step_count / 15.0, "args": a}
                    resp = {"ticket": clock["n_astra"]}
                elif op in ("astra_poll", "astra_wait"):
                    t = tickets.get(int(req.get("ticket") or 0))
                    if t is None:
                        raise A.AstraError("unknown ticket")
                    call, sim_now = t["call"], host.step_count / 15.0
                    if op == "astra_poll" and not (call.done() and sim_now >= t["sim_t0"] + call.result["latency_s"]):
                        resp = {"ready": False}
                    else:
                        r = call.wait()
                        # the answer arrives `latency` after the request, in world time: wait for it if it hasn't yet
                        debt += max(0.0, t["sim_t0"] + r["latency_s"] - sim_now)
                        if not t.get("collected"):
                            t["collected"] = True
                            clock["astra_s"] += r["latency_s"]
                            u = r.get("usage") or {}
                            clock["astra_in_tokens"] += int(u.get("input_tokens") or 0)
                            clock["astra_out_tokens"] += int(u.get("output_tokens") or 0)
                            transcript.write(json.dumps({"ticket": req.get("ticket"), "sim_t0": round(t["sim_t0"], 2), "args": {k: v for k, v in t["args"].items() if k != "images"},
                                                         "n_images": len(t["args"]["images"]), "result": r}) + "\n")  # fmt: skip
                            transcript.flush()
                        resp = {"ready": True, "ok": r.get("ok"), "text": r.get("text"), "json": r.get("json"), "failure": r.get("error"),
                                "latency_s": r.get("latency_s"), "usage": r.get("usage")}  # fmt: skip
                elif op == "llm":
                    imgs = [np.load(xchg / Path(n).name) for n in req.get("images") or []]
                    clock["n_llm"] += 1
                    r = M.gemini(req["model"], str(req.get("prompt", "")), imgs, int(req.get("max_output_tokens") or 1024), req.get("thinking_budget"),
                                 req.get("response_schema"), req.get("temperature"), ledger, limits["model_usd"] - clock["model_usd"], tag)  # fmt: skip
                    clock["model_usd"] += r["cost_usd"]
                    resp = r
                elif op in ("detect", "segment"):
                    clock["n_vision"] += 1
                    args = {k: v for k, v in req.items() if k != "op"}
                    args["image"] = str(xchg / Path(req["image"]).name)
                    r = vision(op, **args)
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
                elif op == "save_image":
                    images_saved += 1
                    if images_saved <= 60:
                        (out / "images").mkdir(exist_ok=True)
                        a = np.load(xchg / Path(req["image"]).name)
                        if a.dtype == bool:
                            a = a.astype(np.uint8) * 255
                        name = "".join(ch for ch in str(req.get("name")) if ch.isalnum() or ch in "-_")[:40] or "img"
                        iio.imwrite(out / "images" / f"{images_saved:02d}_{name}.png", np.asarray(a, dtype=np.uint8))
                    resp = {}
                else:
                    resp = {"error": f"unknown op {op!r}"}
            except M.BudgetExceeded as e:
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
            if proc.poll() is not None and op != "done":
                pass
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
        for t in tickets.values():
            if not t["call"].done():
                t["call"].thread.join(timeout=0)
        log.close()
        shutil.rmtree(xchg, ignore_errors=True)
    final = host.door_deg()  # None on a real robot: nothing measures the door
    max_deg = max(max_deg, final) if final is not None else None
    sim_s = host.step_count / 15.0
    blocked = (out / "skill_log.txt").read_text(errors="replace").count("[skill sandbox] blocked")
    rnd = lambda v: None if v is None else round(v, 2)  # noqa: E731
    result = {"scene": scene["id"], "seed": seed, "door": scene["door"], "final_deg": rnd(final), "max_deg": rnd(max_deg),
              "success60": None if final is None else final >= 60.0, "success85": None if final is None else final >= 85.0,
              "end": reason, "steps": host.step_count, "sim_s": round(sim_s, 2),
              "think_s": round(clock["think_s"], 2), "model_s": round(clock["model_s"], 2),
              "robot_time_s": round(sim_s, 2), "model_usd": round(clock["model_usd"], 5),
              "n_llm": clock["n_llm"], "n_vision": clock["n_vision"], "n_observe": clock["n_observe"],
              "n_astra": clock["n_astra"], "astra_s": round(clock["astra_s"], 2), "astra_in_tokens": clock["astra_in_tokens"],
              "astra_out_tokens": clock["astra_out_tokens"],
              "wall_s": round(time.time() - t_start, 2), "sandbox": sandbox_kind(), "sandbox_blocked": blocked,
              "start_pose": [pose.x, pose.y, pose.yaw]}  # fmt: skip
    if record:
        host.save_video("video.mp4")
        _keyframes(host, out)
    (out / "result.json").write_text(json.dumps(result, indent=1))
    return result


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
