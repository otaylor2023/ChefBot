"""Astranomy's Agentic Policy on the cup scenes: a persistent Codex session
(GPT-6 Astra) controlling the robot one target at a time.

    python -m experiments.astranomy.agentic_codex --scenes 0 [--model gpt-6-astra] [--effort medium]

As in the paper: the expanded-interface developer prompt; each turn the agent
calls execute_eef_target with one absolute flange target held for 1-5 control
steps at 15 Hz (within 0.05 m and 0.35 rad of the latest measured pose) and
gets back fresh RGB views, measured state, motion feedback (tracking error),
remaining budget and subtask progress; memory_* tools return past
observations, the action history and notes; the session keeps its full history;
only the host ends the episode.

One change, forced by this Codex install (its enterprise policy disables MCP
servers, which is how the paper attaches the tools): the tools are a `./robot`
command in the session's workspace. The command hands each call to this
process through files and prints the result, including the paths of the new
camera images, which the agent then views with Codex's image viewer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from experiments.astranomy.host import REPO, Host, R_of, write_json
from experiments.astranomy.prompts import agentic_prompt, agentic_prompt_mobile
from experiments.ortho_grasp import scenes as scene_set

INSTRUCTION = "Pick up the cup."
MAX_STEPS = 600  # 40 s of simulation at 15 Hz
MAX_SESSIONS = 6  # the agent ending its turn early is resumed this many times
TARGET_POS_LIMIT, TARGET_ROT_LIMIT = 0.05, 0.35
BASE_POS_LIMIT, BASE_ROT_LIMIT = 0.10, 0.25  # fridge (mobile base): base target vs the measured base pose
FRIDGE_INSTRUCTION = "Open the fridge door."
FRIDGE_MAX_STEPS = 900

ROBOT_CLI = r'''#!/opt/miniconda3/envs/chefbot/bin/python
"""The robot tools for this episode. Each call is handed to the host process and its result printed."""
import json, os, sys, time, uuid
ipc = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ipc")
rid = uuid.uuid4().hex[:12]
tmp = os.path.join(ipc, f".req_{rid}.json")
with open(tmp, "w") as f:
    json.dump({"id": rid, "argv": sys.argv[1:]}, f)
os.rename(tmp, os.path.join(ipc, f"req_{rid}.json"))
resp = os.path.join(ipc, f"resp_{rid}.json")
for _ in range(36000):
    if os.path.exists(resp):
        print(json.load(open(resp))["text"])
        sys.exit(0)
    time.sleep(0.1)
print("error: the robot host did not answer")
sys.exit(1)
'''


class Server:
    """The tools, on the host side."""

    def __init__(self, host: Host, ws: Path, task: str = "cup", max_steps: int = MAX_STEPS):
        self.host, self.ws, self.task, self.max_steps = host, ws, task, max_steps
        self.ipc = ws / "ipc"
        self.ipc.mkdir(parents=True, exist_ok=True)
        self.notes: list[str] = []
        self.ended, self.success = False, False
        self.log = open(ws / "tool_calls.jsonl", "a")

    def poll(self) -> int:
        n = 0
        for req in sorted(self.ipc.glob("req_*.json"), key=os.path.getmtime):
            d = json.loads(req.read_text())
            try:
                text = self.handle(d["argv"])
            except Exception as e:  # a malformed call: tell the agent, keep the episode
                text = f"error: {e}"
            (self.ipc / f".resp_{d['id']}.json").write_text(json.dumps({"text": text}))
            os.rename(self.ipc / f".resp_{d['id']}.json", self.ipc / f"resp_{d['id']}.json")
            req.unlink()
            self.log.write(json.dumps({"t": time.time(), "argv": d["argv"], "step": self.host.step_count, "result": text[:4000]}) + "\n")
            self.log.flush()
            n += 1
        return n

    def handle(self, argv: list[str]) -> str:
        if not argv:
            return "error: no tool given"
        tool, args = argv[0], argv[1:]
        if self.ended and tool == "execute_eef_target":
            return self._ended_text()
        if tool == "execute_eef_target":
            return self.execute(args)
        if tool == "memory_get_observation":
            k = int(_flag(args, "--step", 1)[0])
            obs = self.host.at(k)
            return json.dumps({"step": obs.step, "sim_time_s": round(obs.time - self.host.history[0].time, 3), "measured_state": obs.state,
                               "images": obs.paths}, indent=1)  # fmt: skip
        if tool == "memory_list_actions":
            return json.dumps(self.host.actions[-60:], indent=0) if self.host.actions else "no actions yet"
        if tool == "memory_add_note":
            self.notes.append(" ".join(_flag(args, "--text", None)))
            return f"note {len(self.notes)} saved"
        if tool == "memory_get_notes":
            return "\n".join(f"{i + 1}. {n}" for i, n in enumerate(self.notes)) or "no notes"
        return f"error: unknown tool {tool}"

    def execute(self, args: list[str]) -> str:
        target = [float(v) for v in _flag(args, "--target", 8)]
        steps = int(_flag(args, "--steps", 1)[0])
        if not 1 <= steps <= 5:
            return "rejected: --steps must be 1 to 5"
        fridge = self.task == "fridge"
        if fridge:
            mb = self.host.base_in_odom()
            base = [float(v) for v in _flag(args, "--base", 3)] if "--base" in args else list(mb)
            db = math.hypot(base[0] - mb[0], base[1] - mb[1])
            dyaw = abs((base[2] - mb[2] + math.pi) % (2 * math.pi) - math.pi)
            if db > BASE_POS_LIMIT or dyaw > BASE_ROT_LIMIT:
                return (f"rejected: the base target is {db:.3f} m and {dyaw:.3f} rad from the latest measured base pose "
                        f"(limits {BASE_POS_LIMIT} m, {BASE_ROT_LIMIT} rad). Nothing moved.")  # fmt: skip
            target = target + base
        F = self.host.flange_in_odom() if fridge else self.host.flange_pose()
        dp = float(np.linalg.norm(np.asarray(target[:3]) - F[:3, 3]))
        dr = float(np.linalg.norm(Rotation.from_matrix(R_of(target[3:7]) @ F[:3, :3].T).as_rotvec()))
        if dp > TARGET_POS_LIMIT or dr > TARGET_ROT_LIMIT:
            return (f"rejected: the target is {dp:.3f} m and {dr:.3f} rad from the latest measured EEF pose "
                    f"(limits {TARGET_POS_LIMIT} m, {TARGET_ROT_LIMIT} rad). Nothing moved.")  # fmt: skip
        rec, done = None, 0
        for _ in range(steps):
            rec = self.host.step(target)
            done += 1
            obs = self.host.observe()
            if self.host.success():
                self.ended, self.success = True, True
                break
            if self.host.step_count >= self.max_steps:
                self.ended = True
                break
        obs = self.host.history[-1]
        if fridge:
            held = bool(self.host.env._grasped_roots() & {int(self.host.env.model.body_rootid[self.host.env.model.jnt_bodyid[self.host.door_joint]])})
            progress = {"grasp the fridge door handle": "holding" if held else "not holding",
                        "open the door": f"{max(0.0, self.host.door_deg()):.1f} of 60.0 deg" + (" (done)" if self.success else "")}  # fmt: skip
        else:
            lift = self.host.cup_bottom() - self.host.cfg["counter_z"]
            held = self.host.cup in self.host.env._grasped_roots()
            progress = {"grasp the cup": "done" if held else "not yet", "lift the cup": f"{max(0.0, lift):.3f} of 0.050 m" + (" (done)" if self.success else "")}
        out = {"executed_steps": done,
               "motion_feedback": {"position_error_m": round(rec["pos_error_m"], 4), "rotation_error_rad": round(rec["rot_error_rad"], 4),
                                   "ik_ok": rec["ik_ok"]},
               "measured_state": obs.state, "remaining_budget_steps": self.max_steps - self.host.step_count,
               "subtask_progress": progress, "episode_ended": self.ended,
               "images": {"step": obs.step, **obs.paths}}  # fmt: skip
        text = json.dumps(out, indent=1)
        if self.ended:
            text += "\n" + self._ended_text()
        else:
            text += "\nView the three new images (left shoulder, right shoulder, wrist) before choosing the next action."
        return text

    def _ended_text(self) -> str:
        return "The episode has ended (" + ("task complete" if self.success else "budget used up") + "). Stop now."


def _flag(args: list[str], name: str, n: int | None) -> list[str]:
    if name not in args:
        raise ValueError(f"missing {name}")
    i = args.index(name) + 1
    return args[i:] if n is None else args[i : i + n]


def run_episode(host: Host, scene, model: str, effort: str, ws: Path, task: str = "cup") -> dict:
    ws.mkdir(parents=True, exist_ok=True)
    host.out = ws
    fridge = task == "fridge"
    obs = host.reset_fridge(scene) if fridge else host.reset(scene)
    robot = ws / "robot"
    robot.write_text(ROBOT_CLI)
    robot.chmod(0o755)
    max_steps = FRIDGE_MAX_STEPS if fridge else MAX_STEPS
    server = Server(host, ws, task, max_steps)
    dev = agentic_prompt_mobile("./robot") if fridge else agentic_prompt("./robot")
    (ws / "developer_prompt.txt").write_text(dev)
    first = (f"Task: {FRIDGE_INSTRUCTION if fridge else INSTRUCTION}\n\nInitial observation (control step 0). measured_state = {json.dumps(obs.state)}\n"
             f"The three attached images are, in order, the left shoulder, right shoulder and wrist camera views at step 0.")  # fmt: skip
    base = ["codex", "exec", "--json", "--skip-git-repo-check", "-s", "workspace-write", "-m", model,
            "-c", f"model_reasoning_effort={effort}", "-c", f"developer_instructions={json.dumps(dev)}"]  # fmt: skip
    events = open(ws / "codex_events.jsonl", "a")
    thread_id, sessions, t0 = None, 0, time.time()
    while not server.ended and sessions < MAX_SESSIONS:
        if thread_id is None:
            cmd = base + [a for p in obs.paths.values() for a in ("-i", p)] + ["--", first]  # -i takes several files: end it
        else:
            cmd = base[:2] + ["resume", thread_id] + base[2:] + [
                f"The episode has not ended (control step {host.step_count}, {max_steps - host.step_count} steps left). "
                "Continue controlling the robot with ./robot until it reports that the episode has ended."]  # fmt: skip
        proc = subprocess.Popen(cmd, cwd=ws, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        lines_q: queue.Queue[str] = queue.Queue()
        reader = threading.Thread(target=lambda: [lines_q.put(l.rstrip("\n")) for l in proc.stdout], daemon=True)
        reader.start()
        sessions += 1
        ended_at = None
        while True:
            server.poll()
            lines = []
            while not lines_q.empty():
                lines.append(lines_q.get())
            if lines:
                for line in lines:
                    events.write(line + "\n")
                    if '"thread.started"' in line and thread_id is None:
                        try:
                            thread_id = json.loads(line).get("thread_id")
                        except json.JSONDecodeError:
                            pass
                    if '"agent_message"' in line:
                        try:
                            print("   agent:", json.loads(line)["item"]["text"][:160].replace("\n", " "), flush=True)
                        except (json.JSONDecodeError, KeyError):
                            pass
                events.flush()
            if server.ended:
                if ended_at is None:
                    ended_at = time.time()
                if time.time() - ended_at > 60.0:  # it was told to stop; give it a minute to end its turn (and report usage)
                    proc.terminate()
                    break
            if proc.poll() is not None:
                server.poll()
                reader.join(timeout=5.0)
                while not lines_q.empty():
                    events.write(lines_q.get() + "\n")
                break
            time.sleep(0.05)
        status = f"door {host.door_deg():.1f} deg" if fridge else f"cup lift {host.cup_bottom() - host.cfg['counter_z']:+.3f} m"
        print(f"  session {sessions} ended: step {host.step_count}, {status}", flush=True)
    events.close()
    video = host.save_video()
    usage = _usage(ws / "codex_events.jsonl")
    result = {"scene": f"seed{scene}" if fridge else scene.name, "task": task, "success": server.success, "control_steps": host.step_count,
              "sim_time_s": host.step_count / 15.0, "sessions": sessions, "tool_calls": sum(1 for _ in open(ws / "tool_calls.jsonl")),
              "wall_time_s": time.time() - t0, "model": model, "effort": effort, "codex_usage": usage,
              "video": str(video) if video else None}  # fmt: skip
    result.update({"door_deg": host.door_deg()} if fridge else {"cup_model": scene.cup_model, "cup_lift_m": host.cup_bottom() - host.cfg["counter_z"]})
    write_json(ws / "result.json", result)
    return result


def _usage(path: Path) -> dict:
    total = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
    for line in open(path):
        if '"turn.completed"' in line:
            try:
                u = json.loads(line).get("usage") or {}
            except json.JSONDecodeError:
                continue
            for k in total:
                total[k] += int(u.get(k) or 0)
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description="Astranomy Agentic Policy (Codex) on the cup scenes or the fridge.")
    ap.add_argument("--task", choices=["cup", "fridge"], default="cup")
    ap.add_argument("--scenes", default="0", help="cup: scene indices; fridge: start-pose seeds")
    ap.add_argument("--model", default="gpt-6-astra")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--out", type=Path, default=REPO / "experiments" / "astranomy" / "results")
    args = ap.parse_args()
    from experiments.ortho_grasp.run import parse_range

    if args.task == "fridge":
        picked = parse_range(args.scenes, 1000)
    else:
        scenes = scene_set.load(REPO / "experiments" / "ortho_grasp" / "results" / "scenes.json")
        picked = [scenes[i] for i in parse_range(args.scenes, len(scenes))]
    host, results = None, []
    try:
        for s in picked:
            name = f"seed{s}" if args.task == "fridge" else s.name
            ws = args.out / (f"fridge_agentic_codex_{args.model}" if args.task == "fridge" else f"agentic_codex_{args.model}") / name
            if host is None:
                if args.task == "fridge":
                    from experiments.astranomy.fridge import FridgeHost

                    host = FridgeHost(ws)
                else:
                    host = Host(ws)
            print(f"{name}:", flush=True)
            r = run_episode(host, s, args.model, args.effort, ws, args.task)
            results.append(r)
            print(f"{name}: {'SUCCESS' if r['success'] else 'failure'} after {r['tool_calls']} tool calls, {r['control_steps']} steps, "
                  f"{r['wall_time_s'] / 60:.1f} min", flush=True)  # fmt: skip
    finally:
        if host is not None:
            host.close()
    print(f"agentic codex {args.task}: {sum(r['success'] for r in results)}/{len(results)} succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
