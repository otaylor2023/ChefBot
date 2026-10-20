"""E3: execute the resolved grasp in RoboCasa (spec §14).

From the fixed base pose: the arm from its ready pose to pre-grasp along a
collision-free joint path (the repo's planner, around what the depth cameras
see), a straight line to the grasp, close, then lift 12 cm (T1) or the
model's lift waypoint (T3); hold 2 s. The gripper opens only to the
pre-grasp width (spec §13.5), held there by the experiment env.

enclosed: after closing, both fingers touch the cup (ground-truth contacts).
lifted:   the cup's bottom >= 5 cm above the counter after the lift and 2 s
          later, still held.
"""

from __future__ import annotations

import json
import threading

import imageio.v3 as iio
import numpy as np

from chefbot.envs.robocasa.objects import object_points
from chefbot.robot.skills.common import follow_path

from experiments.ortho_grasp.frames import MAX_OPEN, pose_to_world

LIFTED = 0.05
HOLD_S = 2.0
CLIP_CAMERA = "robot0_agentview_left"
CLIP_EVERY = 2  # control ticks per frame


def run(cfg: dict, scenes, world_lazy, run_dir_of, load_geometry, repeat: int = 0) -> list[dict]:
    w = world_lazy()
    rows = []
    for scene in scenes:
        out = run_dir_of(cfg, scene, repeat)
        resolved = json.loads((out / "resolved.json").read_text()) if (out / "resolved.json").exists() else None
        metrics = json.loads((out / "metrics.json").read_text()) if (out / "metrics.json").exists() else {}
        if not resolved or not resolved["ok"]:
            row = {"enclosed": False, "lifted": False, "e3_reason": "not_resolved"}
        else:
            row = execute(w, scene, resolved, cfg, out, load_geometry(cfg, scene))
        metrics.update(row)
        (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str))
        rows.append(metrics)
        print(f"e3 {scene.name}: enclosed {row['enclosed']} lifted {row['lifted']} {row.get('e3_reason') or ''}", flush=True)
    print(f"e3: lifted {sum(bool(r.get('lifted')) for r in rows)}/{len(rows)}", flush=True)
    return rows


def _arm_pose(w, pose_robot):
    return w.arm.frames.pose_to_arm_frame(w.base, pose_to_world(np.asarray(pose_robot), w.base))


def execute(w, scene, resolved: dict, cfg: dict, out, geometry) -> dict:
    env, arm, stack = w.env, w.arm.arm, w.arm
    preempt = threading.Event()
    w.apply(scene)
    env.set_arm_joints(stack.ready)
    env.wait_for_next_tick(0.2)
    cup = w.cup_root(scene)
    frames: list[np.ndarray] = []
    tick = [0]
    original_tick = env.wait_for_next_tick

    def recording_tick(dt):
        original_tick(dt)
        tick[0] += 1
        if tick[0] % CLIP_EVERY == 0:
            frames.append(w.render(CLIP_CAMERA, 480, 360))

    env.wait_for_next_tick = recording_tick
    try:
        return _execute(w, env, arm, stack, cup, resolved, preempt, geometry)
    finally:
        env.wait_for_next_tick = original_tick
        env.gripper_hold = False
        if frames:
            iio.imwrite(out / "exec.mp4", np.stack(frames), fps=20 // CLIP_EVERY, codec="libx264")


def _execute(w, env, arm, stack, cup, resolved, preempt, geometry) -> dict:
    waypoints = resolved["waypoints"]
    pre, grasp = np.array(waypoints[0]["pose"]), np.array(waypoints[1]["pose"])
    after = [np.array(wp["pose"]) for wp in waypoints[2:]]
    workspace = stack.workspace()  # what the depth cameras see, the cup included
    q_pre = _free_ik(w, workspace, _arm_pose(w, pre))
    if q_pre is None:
        return {"enclosed": False, "lifted": False, "e3_reason": "no_free_ik_pre"}
    # Part-open to the pre-grasp width (fully open, the far finger can hit a cup's inside wall).
    opening = min(MAX_OPEN, float(resolved["width"]) + 0.01)
    if opening < MAX_OPEN - 0.002:
        env.set_gripper(True)
        for _ in range(100):
            env.wait_for_next_tick(0.05)
            if env.get_state().gripper_width <= opening:
                break
        env.gripper_hold = True
    path = workspace.path(arm.current_target(), q_pre)
    if path is None:
        return {"enclosed": False, "lifted": False, "e3_reason": "no_path_to_pregrasp"}
    contacts_before = env.contact_ticks
    result = follow_path(stack, path, preempt)
    approach_contact = env.contact_ticks > contacts_before
    if not result.ok:
        return {"enclosed": False, "lifted": False, "e3_reason": f"to_pregrasp:{result.status}", "approach_contact": approach_contact}
    line = arm.plan_line(arm.current_target(), _arm_pose(w, grasp))
    if line is None:
        return {"enclosed": False, "lifted": False, "e3_reason": "no_line_to_grasp", "approach_contact": approach_contact}
    contacts_before = env.contact_ticks
    arm.follow(line, preempt, expect_contact=True)
    line_contact = env.contact_ticks > contacts_before
    env.gripper_hold = False
    w.arm.gripper.close(preempt)
    enclosed = cup in env._grasped_roots()
    for target in after or [_lifted(grasp)]:
        line = arm.plan_line(arm.current_target(), _arm_pose(w, target))
        if line is None:
            break
        arm.follow(line, preempt, expect_contact=True)
    lifted_now = _bottom(env, cup) >= geometry.counter_z + LIFTED
    for _ in range(int(HOLD_S / 0.05)):
        env.wait_for_next_tick(0.05)
    lifted = lifted_now and _bottom(env, cup) >= geometry.counter_z + LIFTED and cup in env._grasped_roots()
    return {"enclosed": bool(enclosed), "lifted": bool(lifted), "e3_reason": None if lifted else ("not_enclosed" if not enclosed else "dropped_or_not_lifted"),
            "approach_contact": bool(approach_contact), "line_contact": bool(line_contact),
            "cup_bottom_after": float(_bottom(env, cup))}  # fmt: skip


def _free_ik(w, workspace, pose_arm: np.ndarray, tries: int = 40) -> np.ndarray | None:
    """An IK solution for `pose_arm` with the arm clear of what the cameras see:
    seeded from `ready`, then from random configurations (fixed seed)."""
    rng = np.random.default_rng(0)
    limits = np.asarray(w.arm.ik.joint_limits, dtype=float)
    seeds = [w.arm.ready] + [rng.uniform(limits[:, 0], limits[:, 1]) for _ in range(tries)]
    for seed in seeds:
        q = w.arm.ik.solve_pose(pose_arm, seed=seed)
        if q is not None and workspace.free(q):
            return q
    return None


def _lifted(pose: np.ndarray) -> np.ndarray:
    out = pose.copy()
    out[2, 3] += 0.12
    return out


def _bottom(env, root: int) -> float:
    with env.lock:
        return float(object_points(env, root)[:, 2].min())

