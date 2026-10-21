"""The robot as a skill sees it. A skill is a Python package with `main.py` defining
`run(robot: Robot) -> None`; it runs in its own sandboxed process (no simulator, no
network, no API keys) and talks to the robot through this object only.

Frames: odom = the robot root (arm base frame) where the episode started, fixed for
the episode: +x forward, +y left, +z up, meters. The flange is the gripper's mounting
flange with DROID's axes: local +X along the fingers (approach), Y across the pads.

Control: `act(rows)` runs one 15 Hz control step per row. A row is an absolute
whole-body target [x, y, z, qw, qx, qy, qz, gripper, base_x, base_y, base_yaw]:
the flange pose in odom, gripper 0 open / 1 closed, and where the robot root should
be (x, y, yaw in odom). Each step the base drives toward its target (at most 0.5 m/s,
1 rad/s) and the arm takes one IK step toward the flange target (at most about
0.02 m and 0.1 rad per 20 Hz tick), re-solved against where the base really is.

Time: the world does not wait for you. While your process computes, or waits on an
llm/detect/segment call, the simulation keeps running for that long, the robot holding
its last command (the base still driving to its last target, the gripper as it was),
before your next request is served. An episode is 90 s of world time (1350 steps of
15 Hz), thinking included, and is scored on that time (`robot_time_s`). You may compute
in a thread while `act()` executes motion: that overlap is free, as on a real robot.
Calls to the robot are served one at a time. Budgets (see `robot.budget()`): control
steps (world time) left, think seconds, model dollars.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
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
        """Cheap state: step, sim_time_s, flange_pose_odom (4x4), base_pose_odom [x, y, yaw],
        gripper_closed_cmd, gripper_closure (0 open .. 1 closed), gripper_width_m, arm_q, arm_q_cmd,
        base_in_contact."""
        return self._call("proprio")["proprio"]

    def observe(self, cameras: tuple[str, ...] | None = None) -> dict:
        """proprio plus, per camera (all of them by default): rgb (360x640x3 uint8), depth (360x640
        float32, meters along the optical axis, 0 where nothing), K (3x3 intrinsics), cam_to_odom (4x4,
        OpenCV axes: x right, y down, z forward). The wrist camera rides on the flange, the others on
        the base. The robot's camera names are in AGENTS.md."""
        r = self._call("observe", cameras=list(cameras) if cameras else None)
        cams = {}
        for name, c in r["cameras"].items():
            cams[name] = {"rgb": self._get(c["rgb"]), "depth": self._get(c["depth"]), "K": np.array(c["K"]),
                          "cam_to_odom": np.array(c["cam_to_odom"])}  # fmt: skip
        return {"proprio": r["proprio"], "cameras": cams}

    # -- acting ----------------------------------------------------------------------

    def act(self, rows) -> dict:
        """Run each 11-D row for one 15 Hz step (at most 150 rows per call). Returns
        {"proprio": ..., "steps": [{"pos_error_m", "rot_error_rad", "base_error_m", "ik_ok"}, ...]}."""
        rows = np.asarray(rows, dtype=float).reshape(-1, 11)
        return self._call("act", rows=rows.tolist())

    # -- models (metered; see budget()) ------------------------------------------------

    def llm(self, model: str, prompt: str, images: list[np.ndarray] | None = None, max_output_tokens: int = 1024,
            thinking_budget: int | None = 0, response_schema: dict | None = None, temperature: float | None = None) -> dict:
        """A Gemini call. images: RGB uint8 arrays (sent as PNG, in order, after the prompt).
        Returns {"text", "cost_usd", "latency_s", "input_tokens", "output_tokens"}. Models: robot.models()."""
        names = [self._put(np.asarray(im, dtype=np.uint8)) for im in images or []]
        return self._call("llm", model=model, prompt=prompt, images=names, max_output_tokens=max_output_tokens,
                          thinking_budget=thinking_budget, response_schema=response_schema, temperature=temperature)  # fmt: skip

    def detect(self, image: np.ndarray, queries: list[str], model: str = "owlv2", threshold: float = 0.1) -> list[dict]:
        """Open-vocabulary detection on the GPU: [{"label", "score", "box": [x0, y0, x1, y1]}, ...] (pixels).
        Models: robot.models()["vision"]."""
        r = self._call("detect", image=self._put(np.asarray(image, dtype=np.uint8)), queries=list(queries), model=model,
                       threshold=float(threshold))  # fmt: skip
        return r["detections"]

    def segment(self, image: np.ndarray, boxes: list | None = None, points: list | None = None,
                point_labels: list | None = None, model: str = "sam2") -> np.ndarray:
        """SAM 2 masks on the GPU: one (H, W) bool mask per box, or one for the point prompt
        (points [[x, y], ...] with labels 1 = on the object, 0 = not)."""
        r = self._call("segment", image=self._put(np.asarray(image, dtype=np.uint8)), boxes=boxes, points=points,
                       point_labels=point_labels, model=model)  # fmt: skip
        return self._get(r["masks"]).astype(bool)

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

    def save_image(self, name: str, image: np.ndarray) -> None:
        """Keep an image (RGB uint8, or a mask) with the run, for debugging."""
        self._call("save_image", name=str(name), image=self._put(np.asarray(image)))


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
    """Entry point: python -m fridge_robot <skill_dir> <xchg_dir>. The skill's prints go to its log (stderr)."""
    skill_dir, xchg = Path(sys.argv[1]), Path(sys.argv[2])
    req_out, resp_in = sys.stdout, sys.stdin
    sys.stdout = sys.stderr
    sys.path.insert(0, str(skill_dir))
    os.chdir(xchg)
    sep = os.pathsep
    _guard([str(skill_dir), str(xchg), sys.prefix, sys.base_prefix, os.path.dirname(__file__)] + os.environ.get("FRIDGE_ALLOW", "").split(sep),
           os.environ.get("FRIDGE_DENY", "").split(sep))
    robot = Robot(xchg, req_out, resp_in)
    t0 = time.time()
    try:
        import main as skill  # noqa: PLC0415

        skill.run(robot)
    except EpisodeOver as e:
        print(f"[fridge_robot] episode over: {e}", file=sys.stderr)
    finally:
        try:
            robot._call("done", wall_s=time.time() - t0)
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    _main()
