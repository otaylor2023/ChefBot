"""The robot as a skill sees it (interface chefbot-v1). A skill is a Python package with `main.py` defining
`run(robot, **args)`; it runs in its own sandboxed process (no simulator, no network, no API keys) and talks to the
robot through this object only. Standard library and numpy only: this file is all the skill gets of ChefBot.

Frames: odom = the robot root (the arm base frame) where the episode started, fixed for the episode: +x forward,
+y left, +z up, meters. The flange is the arm's mounting flange (Franka's): +z points out of the flange toward the
fingers (the approach direction); the robot's brief says where the fingertip pads are and which way they close.

Control: `act(rows)` runs one 15 Hz control step per row. A row is an absolute whole-body target
[x, y, z, qw, qx, qy, qz, gripper, base_x, base_y, base_yaw]: the flange pose in odom, the gripper (1 closed, 0 open;
closed at >= 0.5) and where the robot root should be (x, y, yaw in odom). The gripper column is the simple way to use
the hand: closed is a grasp at the robot's default speed and force, open is a move to its full width, sent only when
the column changes between open and closed (so a gripper_move or gripper_grasp call stands while your rows keep the
matching value: 1 after a grasp, 0 after a move). Each step the base drives toward its target and the arm
takes a clipped step toward the flange target (a few mm and a few degrees per 20 Hz tick, the speed profile's
rates), re-solved by IK against where the base really is. At most 600 rows (40 s) per call.

`act_arm(rows)`: arm only. A row is [x, y, z, qw, qx, qy, qz, gripper]: the flange pose in the robot root frame (the
arm base frame, which rides with the base), solved by the arm's IK alone; the base holds its last target, whatever
happens to the arm. `act_base(rows)`: base only. A row is [base_x, base_y, base_yaw], the root in odom; the arm gets
no commands and holds its joints (so the flange rides with the base), the gripper as it was.

Time: the world does not wait for you. While your process computes, or waits on an llm/detect/segment call, the
simulation keeps running for that long, the robot holding its last command (the base still driving to its last
target, the gripper as it was), before your next request is served. An episode is 90 s of world time (1350 steps
of 15 Hz), thinking included, and is scored on that time (`robot_time_s`). You may compute in a thread while
`act()` executes motion: that overlap is free, as on a real robot. Calls to the robot are served one at a time.
Budgets (see `robot.budget()`): control steps (world time) left, think seconds, model dollars (and Astra calls, in an Astra lab).

(agentic-skills-exploration's interface/wholebody_v2.py, on ChefBot's frames, without Astra.)
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

import numpy as np


class EpisodeOver(Exception):
    """The episode ended (a budget ran out); return from run()."""


class Robot:
    """One call at a time (calls from several threads are serialized)."""

    def __init__(self, xchg: Path, req_out, resp_in):
        self._x = Path(xchg)
        self._out, self._in = req_out, resp_in
        self._lock = threading.Lock()

    # -- transport -------------------------------------------------------------------

    def _call(self, op: str, **args) -> dict:
        with self._lock:
            self._out.write(json.dumps({"op": op, **args}) + "\n")
            self._out.flush()
            line = self._in.readline()
        if not line:
            raise EpisodeOver("robot connection closed")
        resp = json.loads(line)
        if resp.get("episode_over"):
            raise EpisodeOver(resp.get("reason", "episode over"))
        if "error" in resp:
            raise RuntimeError(resp["error"])
        return resp

    def _put(self, arr: np.ndarray) -> str:
        p = self._x / f"in_{uuid.uuid4().hex[:10]}.npy"
        np.save(p, np.asarray(arr))
        return p.name

    def _get(self, name: str) -> np.ndarray:
        p = self._x / name
        a = np.load(p)
        p.unlink(missing_ok=True)
        return a

    # -- sensing ---------------------------------------------------------------------

    def proprio(self) -> dict:
        """Cheap state: step, sim_time_s, flange_pose_odom (4x4), flange_pose_root (4x4, in the root
        frame: what act_arm rows give), base_pose_odom [x, y, yaw], gripper_command (the last gripper command sent:
        {"kind": "move", "width_m", "speed_mps"} or {"kind": "grasp", "speed_mps", "force_n"}; not on every robot),
        gripper_closed_cmd (that was a grasp), gripper_closure (0 open .. 1 closed, measured), gripper_width_m
        (measured, between the fingers), arm_q, arm_q_cmd, base_in_contact."""
        return _with_root(self._call("proprio")["proprio"])

    def observe(self, cameras: tuple[str, ...] | None = None) -> dict:
        """proprio plus, per camera (all of them by default): rgb (480x640x3 uint8), depth (480x640 float32,
        meters along the optical axis, 0 where nothing; None for a camera without depth), model ("pinhole"),
        K (3x3 intrinsics), cam_to_odom (4x4, OpenCV axes: x right, y down, z forward). The wrist camera rides
        on the hand, the pole camera on the base. The robot's camera names are in AGENTS.md."""
        r = self._call("observe", cameras=list(cameras) if cameras else None)
        cams = {}
        for name, c in r["cameras"].items():
            cams[name] = {"rgb": self._get(c["rgb"]), "depth": self._get(c["depth"]) if c.get("depth") else None, "K": np.array(c["K"]),
                          "model": c.get("model", "pinhole"), "D": None if c.get("D") is None else np.array(c["D"]),
                          "cam_to_odom": np.array(c["cam_to_odom"])}  # fmt: skip
        return {"proprio": _with_root(r["proprio"]), "cameras": cams}

    # -- acting ----------------------------------------------------------------------

    def act(self, rows) -> dict:
        """Run each 11-D row for one 15 Hz step (at most 600 rows per call). Returns
        {"proprio": ..., "steps": [{"pos_error_m", "rot_error_rad", "base_error_m", "ik_ok"}, ...]}."""
        rows = np.asarray(rows, dtype=float).reshape(-1, 11)
        return _with_root_steps(self._call("act", rows=rows.tolist()))

    def act_arm(self, rows) -> dict:
        """Arm only: each 8-D row [x, y, z, qw, qx, qy, qz, gripper] (the flange in the root frame) for one
        15 Hz step, the base holding its last target. Returns as act(); pos/rot errors are in the root frame."""
        rows = np.asarray(rows, dtype=float).reshape(-1, 8)
        return _with_root_steps(self._call("act_arm", rows=rows.tolist()))

    def act_joints(self, rows) -> dict:
        """Arm joints only: each row [q1 .. q7, gripper] (rad) for one 15 Hz step, no IK, the base holding its last
        target; each step's change clipped to the joint limits and the speed profile's joint rate. Returns as act();
        a step's pos_error_m is the largest joint's error (rad) after it."""
        rows = np.asarray(rows, dtype=float).reshape(-1, 8)
        return _with_root_steps(self._call("act_joints", rows=rows.tolist()))

    def act_base(self, rows) -> dict:
        """Base only: each 3-D row [base_x, base_y, base_yaw] (the root in odom) for one 15 Hz step, the arm
        holding its joints. Returns as act() (no arm errors)."""
        rows = np.asarray(rows, dtype=float).reshape(-1, 3)
        return _with_root_steps(self._call("act_base", rows=rows.tolist()))

    def gripper_move(self, width_m: float, speed_mps: float = 0.05) -> dict:
        """The Franka Hand's move (libfranka's, as the real robot runs it): the fingers to `width_m` apart (0 ..
        0.08) at `speed_mps` (0.01 .. 0.1), position-controlled: it pushes as hard as it must, up to the hand's 70 N.
        If something stops the fingers short, the move ends there and the hand keeps pushing toward the width, holding
        what is between them as hard as it can (use gripper_grasp to squeeze with a chosen force). Returns at once with
        {"proprio"}; the hand runs the command over the next steps. The hand finishes each command before it takes
        the next: a command sent while one runs waits, and a later one replaces the waiting one. A gripper that is
        only open or closed closes for a width under 0.04."""
        return _with_root_steps(self._call("gripper_move", width=float(width_m), speed=float(speed_mps)))

    def gripper_grasp(self, force_n: float = 20.0, speed_mps: float = 0.05) -> dict:
        """The Franka Hand's grasp (libfranka's, as the real robot runs it): close at `speed_mps` (0.01 .. 0.1) until
        the fingers stop on something, then squeeze it with `force_n` (0.1 .. 70 N) for as long as the grasp holds.
        The hand sets that force itself; nothing measures it, and only the fingers' width comes back. Returns at
        once, as gripper_move, and waits its turn the same way. A gripper that is only open or closed just closes."""
        return _with_root_steps(self._call("gripper_grasp", speed=float(speed_mps), force=float(force_n)))

    # -- models (metered; see budget()) ------------------------------------------------

    def llm(self, model: str, prompt: str, images: list[np.ndarray] | None = None, max_output_tokens: int = 1024,
            thinking_budget: int | None = 0, response_schema: dict | None = None, temperature: float | None = None) -> dict:
        """A Gemini call. images: RGB uint8 arrays (sent as PNG, in order, after the prompt).
        Returns {"text", "cost_usd", "latency_s", "input_tokens", "output_tokens"}. Models: robot.models()."""
        names = [self._put(np.asarray(im, dtype=np.uint8)) for im in images or []]
        return self._call("llm", model=model, prompt=prompt, images=names, max_output_tokens=max_output_tokens,
                          thinking_budget=thinking_budget, response_schema=response_schema, temperature=temperature)  # fmt: skip

    def detect(self, image: np.ndarray, queries: list[str], model: str = "owlv2", threshold: float = 0.1) -> list[dict]:
        """Open-vocabulary detection: [{"label", "score", "box": [x0, y0, x1, y1]}, ...] (pixels).
        Models: robot.models()["vision"]."""
        r = self._call("detect", image=self._put(np.asarray(image, dtype=np.uint8)), queries=list(queries), model=model,
                       threshold=float(threshold))  # fmt: skip
        return r["detections"]

    def segment(self, image: np.ndarray, boxes: list | None = None, points: list | None = None,
                point_labels: list | None = None, model: str = "sam2") -> np.ndarray:
        """SAM 2 masks: one (H, W) bool mask per box, or one for the point prompt (points [[x, y], ...] with
        labels 1 = on the object, 0 = not)."""
        r = self._call("segment", image=self._put(np.asarray(image, dtype=np.uint8)), boxes=boxes, points=points,
                       point_labels=point_labels, model=model)  # fmt: skip
        return self._get(r["masks"]).astype(bool)

    # -- GPT-6 Astra (in an Astra lab: the robot's only model) --------------------------------

    def astra(self, prompt: str, images: list[np.ndarray] | None = None, schema: dict | None = None, effort: str = "low",
              tier: str = "default", n: int = 1, instructions: str | None = None, session: str | None = None) -> dict:
        """Ask GPT-6 Astra and wait for the answer: astra_start(...) then astra_wait(ticket)."""
        return self.astra_wait(self.astra_start(prompt, images, schema, effort, tier, n, instructions, session))

    def astra_start(self, prompt: str, images: list[np.ndarray] | None = None, schema: dict | None = None, effort: str = "low",
                    tier: str = "default", n: int = 1, instructions: str | None = None, session: str | None = None) -> int:
        """Send a request to GPT-6 Astra (through Codex, with no tools: it can only look and answer) and return a
        ticket at once; the robot keeps going meanwhile. images: RGB uint8 arrays (at most 6). schema: a JSON schema
        the answer must follow. effort: low | medium | high | xhigh. tier: default | fast | priority. n: send that many
        identical requests at once and take the first answer (1..4). instructions: replaces the model's system
        instructions. session: a name; calls with the same name continue one conversation (n=1)."""
        names = [self._put(np.asarray(im, dtype=np.uint8)) for im in images or []]
        return self._call("astra_start", prompt=prompt, images=names, schema=schema, effort=effort, tier=tier, n=n,
                          instructions=instructions, session=session)["ticket"]  # fmt: skip

    def astra_poll(self, ticket: int) -> dict | None:
        """The answer if it has arrived (in world time), else None. Doesn't wait."""
        r = self._call("astra_poll", ticket=ticket)
        return r if r.get("ready") else None

    def astra_wait(self, ticket: int) -> dict:
        """Wait for the answer (the world runs on meanwhile, the robot holding its last command). Returns {"ok", "text",
        "json" (parsed when a schema was given), "failure" (why, when not ok), "latency_s", "usage"}."""
        return self._call("astra_wait", ticket=ticket)

    def models(self) -> dict:
        """{"llm": {name: {"input_per_mtok", "output_per_mtok"}}, "vision": [names]}"""
        return self._call("models")["models"]

    # -- bookkeeping -----------------------------------------------------------------------

    def budget(self) -> dict:
        """Remaining: control_steps, think_s, model_usd (this episode)."""
        return self._call("budget")["budget"]

    def log(self, *msg) -> None:
        """A line in the run's skill log (the researcher's view; not scored)."""
        self._call("log", text=" ".join(str(m) for m in msg))

    def show(self, name: str, image: np.ndarray) -> None:
        """save_image, and on a real robot the image also opens on the operator's computer (e.g. to check a
        target before confirming the move)."""
        self._call("show", name=str(name), image=self._put(np.asarray(image)))

    def save_image(self, name: str, image: np.ndarray) -> None:
        """Keep an image (RGB uint8, or a mask) with the run, for debugging."""
        self._call("save_image", name=str(name), image=self._put(np.asarray(image)))


def _with_root(p: dict) -> dict:
    """proprio plus flange_pose_root: the flange in the root frame (odom's z is the root's, so a planar change)."""
    x, y, yaw = p["base_pose_odom"]
    T = np.eye(4)
    T[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    T[:2, 3] = [x, y]
    p["flange_pose_root"] = (np.linalg.inv(T) @ np.array(p["flange_pose_odom"])).tolist()
    return p


def _with_root_steps(r: dict) -> dict:
    if "proprio" in r:
        r["proprio"] = _with_root(r["proprio"])
    return r


def _guard(allow: list[str], deny: list[str]) -> None:
    """No network, no subprocesses, no reading or writing protected paths (the simulator, the
    harness, other runs, keys). Paths under `allow` are always fine. An audit hook cannot be removed."""
    allow = [os.path.realpath(a) for a in allow if a]
    deny = [os.path.realpath(d) for d in deny if d]

    def blocked(path) -> bool:
        try:
            p = os.path.realpath(os.fsdecode(path))
        except (TypeError, ValueError):
            return False
        if any(p == a or p.startswith(a + os.sep) for a in allow):
            return False
        return any(p == d or p.startswith(d + os.sep) for d in deny) or os.path.basename(p) == ".env"

    def deny_(msg):
        sys.__stderr__.write(f"[skill sandbox] blocked: {msg}\n")
        raise PermissionError(f"skill sandbox: {msg}")

    def hook(event, args):
        if event.startswith("socket.") or event in ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "pty.spawn"):
            deny_(f"{event} is not allowed")
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)) and blocked(args[0]):
            deny_(f"{args[0]!r} is outside the skill's files")
        if event in ("os.listdir", "os.scandir", "glob.glob", "shutil.copyfile", "os.rename", "os.remove") and args and \
                isinstance(args[0], (str, bytes, os.PathLike)) and blocked(args[0]):  # fmt: skip
            deny_(f"{args[0]!r} is outside the skill's files")

    sys.addaudithook(hook)


def _main() -> None:
    """Entry point: python -I skill_api.py <skill_dir> <xchg_dir>. The skill's prints go to its log (stderr)."""
    skill_dir, xchg = Path(sys.argv[1]), Path(sys.argv[2])
    req_out, resp_in = sys.stdout, sys.stdin
    sys.stdout = sys.stderr
    sys.path.insert(0, str(skill_dir))
    os.chdir(xchg)
    sep = os.pathsep
    _guard([str(skill_dir), str(xchg), sys.prefix, sys.base_prefix, os.path.dirname(__file__)] + os.environ.get("SKILL_ALLOW", "").split(sep),
           os.environ.get("SKILL_DENY", "").split(sep))
    robot = Robot(xchg, req_out, resp_in)
    t0 = time.time()
    # Arguments from the caller (the task, or an agent: skill.yaml `args`), if any: run(robot, **args). What run returns
    # (JSON-able: e.g. {"ok": True, ...}) goes back to the caller with "done".
    args = json.loads((xchg / "args.json").read_text()) if (xchg / "args.json").exists() else {}
    returned, error = None, None
    try:
        import main as skill  # noqa: PLC0415

        returned = skill.run(robot, **args) if args else skill.run(robot)
    except EpisodeOver as e:
        print(f"[skill_api] episode over: {e}", file=sys.stderr)
    except Exception as e:  # noqa: BLE001  (the skill's own failure: its traceback goes to its log)
        traceback.print_exc()
        error = f"{type(e).__name__}: {e}"[:300]
    finally:
        try:
            json.dumps(returned)
        except (TypeError, ValueError):
            returned = {"unserializable": repr(returned)[:500]}
        try:
            robot._call("done", wall_s=time.time() - t0, returned=returned, error=error)
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    _main()
