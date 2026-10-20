"""Astranomy's Direct Policy on the cup scenes: stateless API calls to GPT-6 Astra,
each returning one emit_eef_actions chunk of 3-15 absolute flange poses.

    python -m experiments.astranomy.direct --scenes 0-2 [--model openai/gpt-6-astra] [--effort medium]

As in the paper: observations at t-14, t-7 and t (three RGB views each, 384 x
216, plus measured state), no context kept between calls, the base + gripper +
camera geometry prompt, medium reasoning. The client normalizes quaternions and
limits each row to 0.02 m and 0.1 rad from the previous one; every row runs for
one 15 Hz control step. The episode ends when the cup is lifted (RoboLab's
role here) or the budget runs out.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from experiments.astranomy.host import REPO, Host, R_of, quat_wxyz, write_json
from experiments.astranomy.prompts import direct_prompt
from experiments.ortho_grasp import scenes as scene_set
from experiments.ortho_grasp.query.models import OPENROUTER, _http, _or_headers, load_env

INSTRUCTION = "Pick up the cup."
HISTORY = (14, 7, 0)  # control steps back: t-14, t-7, t
MAX_STEPS = 600  # 40 s at 15 Hz
MAX_CALLS = 80
TOOL = {
    "type": "function",
    "function": {
        "name": "emit_eef_actions",
        "description": "Emit 3 to 15 consecutive absolute 8D EEF actions [x, y, z, qw, qx, qy, qz, gripper] for the Robotiq base_link in the DROID robot-root frame.",
        "parameters": {"type": "object", "properties": {"actions": {"type": "array", "minItems": 3, "maxItems": 15,
            "items": {"type": "array", "minItems": 8, "maxItems": 8, "items": {"type": "number"}}}}, "required": ["actions"]},
    },
}  # fmt: skip
LABELS = {"over_shoulder_left_camera": "left shoulder", "over_shoulder_right_camera": "right shoulder", "wrist_cam": "wrist"}


def _png(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def user_message(host: Host) -> list[dict]:
    now = host.step_count
    content = [{"type": "text", "text": f"Task: {INSTRUCTION}"}]
    seen = set()
    for back in HISTORY:
        obs = host.at(now - back)
        if obs.step in seen:
            continue
        seen.add(obs.step)
        label = "t" if back == 0 else f"t-{back}"
        content.append({"type": "text", "text": f"Observation {label} (control step {obs.step}, sim time {obs.time - host.history[0].time:.2f} s). "
                                                f"measured_state = {json.dumps(obs.state)}"})  # fmt: skip
        for name, img in obs.images.items():
            content.append({"type": "text", "text": f"{LABELS[name]} camera ({name}), {label}:"})
            content.append({"type": "image_url", "image_url": {"url": _png(img)}})
    return content


def clamp_rows(rows: list[list[float]], start: np.ndarray) -> list[list[float]]:
    """Normalize quaternions; each row within 0.02 m and 0.1 rad of the previous (the first of the measured pose)."""
    out, prev_p, prev_R = [], start[:3, 3].copy(), start[:3, :3].copy()
    for r in rows:
        p, q, g = np.asarray(r[:3], float), np.asarray(r[3:7], float), float(r[7])
        if np.linalg.norm(q) < 1e-9:
            q = np.array(quat_wxyz(prev_R))
        R = R_of(q)
        dp = p - prev_p
        if np.linalg.norm(dp) > 0.02:
            p = prev_p + dp * 0.02 / np.linalg.norm(dp)
        rot = Rotation.from_matrix(R @ prev_R.T).as_rotvec()
        if np.linalg.norm(rot) > 0.1:
            R = Rotation.from_rotvec(rot * 0.1 / np.linalg.norm(rot)).as_matrix() @ prev_R
        out.append([*p.tolist(), *quat_wxyz(R), 1.0 if g >= 0.5 else 0.0])
        prev_p, prev_R = p, R
    return out


def call(model: str, effort: str, system: str, content: list[dict]) -> tuple[list | None, dict]:
    body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            "tools": [TOOL], "tool_choice": {"type": "function", "function": {"name": "emit_eef_actions"}},
            "reasoning": {"effort": effort}, "usage": {"include": True}, "max_tokens": 16000}  # fmt: skip
    t0 = time.time()
    resp = _http(f"{OPENROUTER}/chat/completions", body, _or_headers())
    meta = {"latency_s": time.time() - t0, "usage": resp.get("usage"), "model": resp.get("model")}
    msg = ((resp.get("choices") or [{}])[0]).get("message") or {}
    for tc in msg.get("tool_calls") or []:
        if tc.get("function", {}).get("name") == "emit_eef_actions":
            try:
                rows = json.loads(tc["function"]["arguments"]).get("actions")
                if isinstance(rows, list) and rows and all(isinstance(r, list) and len(r) == 8 for r in rows):
                    return rows, meta
            except (json.JSONDecodeError, AttributeError):
                pass
    meta["bad_reply"] = (msg.get("content") or "")[:500]
    return None, meta


def run_episode(host: Host, scene, model: str, effort: str, out: Path) -> dict:
    system = direct_prompt()
    host.reset(scene)
    t0 = time.time()
    calls, cost, steps_log = 0, 0.0, []
    success = False
    log = open(out / "calls.jsonl", "w")
    while host.step_count < MAX_STEPS and calls < MAX_CALLS and not success:
        rows, meta = call(model, effort, system, user_message(host))
        calls += 1
        cost += float((meta.get("usage") or {}).get("cost") or 0.0)
        if rows is None:
            log.write(json.dumps({"call": calls, "step": host.step_count, **meta, "actions": None}) + "\n")
            continue
        rows = rows[:15]
        executed = clamp_rows(rows, host.flange_pose())
        log.write(json.dumps({"call": calls, "step": host.step_count, **meta, "actions": rows, "executed": executed}) + "\n")
        log.flush()
        for row in executed:
            host.step(row)
            host.observe()
            if host.success():
                success = True
                break
            if host.step_count >= MAX_STEPS:
                break
        print(f"  call {calls:2d}: {len(rows):2d} actions, step {host.step_count:3d}, gripper {'closed' if executed[-1][7] else 'open'}, "
              f"cup bottom {host.cup_bottom() - host.cfg['counter_z']:+.3f} m, {meta['latency_s']:.0f} s, ${cost:.2f}", flush=True)  # fmt: skip
    log.close()
    video = host.save_video()
    result = {"scene": scene.name, "cup_model": scene.cup_model, "success": success, "calls": calls, "control_steps": host.step_count,
              "sim_time_s": host.step_count / 15.0, "wall_time_s": time.time() - t0, "cost_usd": cost, "model": model, "effort": effort,
              "cup_lift_m": host.cup_bottom() - host.cfg["counter_z"], "video": str(video) if video else None}  # fmt: skip
    write_json(out / "result.json", result)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Astranomy Direct Policy on the cup scenes.")
    ap.add_argument("--scenes", default="0")
    ap.add_argument("--model", default="openai/gpt-6-astra")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--out", type=Path, default=REPO / "experiments" / "astranomy" / "results")
    args = ap.parse_args()
    load_env(REPO / ".env")
    from experiments.ortho_grasp.run import parse_range

    scenes = scene_set.load(REPO / "experiments" / "ortho_grasp" / "results" / "scenes.json")
    picked = [scenes[i] for i in parse_range(args.scenes, len(scenes))]
    tag = args.model.split("/")[-1]
    host = None
    results = []
    try:
        for s in picked:
            out = args.out / f"direct_{tag}" / s.name
            if host is None:
                host = Host(out)
            host.out = out
            out.mkdir(parents=True, exist_ok=True)
            print(f"{s.name} ({s.cup_model}):", flush=True)
            r = run_episode(host, s, args.model, args.effort, out)
            results.append(r)
            print(f"{s.name}: {'SUCCESS' if r['success'] else 'failure'} after {r['calls']} calls, {r['control_steps']} steps, ${r['cost_usd']:.2f}", flush=True)
    finally:
        if host is not None:
            host.close()
    print(f"direct {tag}: {sum(r['success'] for r in results)}/{len(results)} succeeded, ${sum(r['cost_usd'] for r in results):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
