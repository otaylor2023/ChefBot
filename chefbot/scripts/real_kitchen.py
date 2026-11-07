"""The real kitchen for an agent's host-v1 skills (skills/*/host_v1): the base driven to the map's places and to
counter stances by the robot layer (scripts/real_nav.py, robot/nav/goal.py counter_stance), and things found with the
cameras and the vision models, placed in the mocap world the map is in. Runs on the laptop, in the agent's process,
on the same real host (environments/real/host.py) the sandboxed skills run their episodes on.

    kitchen = RealKitchen(host, REPO / "maps/src_kitchen", vision, out)
    kitchen.go_to("island_north")          # A* on the map to a place, the footprint clear, mocap live
    kitchen.find("tomato")                 # {found, world: [x, y, z], on: "island", camera, ...}
    kitchen.approach("tomato")             # to the stance in front of where it was found, square to that counter

A base move first stops the host's control (it holds the flange still in odom, which would pull the arm back as the
base drives), holds the arm at its joints, drives, and starts the host's control again where the base ended: the
next skill's odom starts there.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from chefbot.interface.types import Pose2D

FIND_CAMERAS = ("pole", "wrist")
FIND_THRESHOLD = 0.3


def _planar(p: Pose2D) -> np.ndarray:
    T = np.eye(4)
    c, s = math.cos(p.yaw), math.sin(p.yaw)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:2, 3] = [p.x, p.y]
    return T


class RealKitchen:
    def __init__(self, host, map_dir: Path, vision, out: Path, speed: float = 0.1, yaw_speed: float = 0.3,
                 confirm: bool = False, log=print):  # fmt: skip
        from chefbot.scripts.real_map import load as load_map

        self.host, self.env = host, host.raw_env
        self.map_dir, self.vision, self.out = Path(map_dir), vision, Path(out)
        self.grid, self.places = load_map(self.map_dir)
        self.spec = yaml.safe_load((self.map_dir / "spec.yaml").read_text())
        self.speed, self.yaw_speed, self.confirm, self.log = float(speed), float(yaw_speed), confirm, log
        self.found: dict[str, dict] = {}  # what find() last saw, by what was asked for
        self.n = 0

    def move_to(self, out: Path) -> None:
        self.out = Path(out)

    # -- where things are ---------------------------------------------------------------

    def where(self) -> dict:
        loc = self.env.localization()
        p = loc.mocap_pose
        return {"live": bool(loc.live), "x": None if p is None else round(p.x, 3), "y": None if p is None else round(p.y, 3),
                "yaw_deg": None if p is None else round(math.degrees(p.yaw), 1), "near": self._nearest_place(p)}  # fmt: skip

    def _nearest_place(self, p: Pose2D | None) -> str | None:
        if p is None or not self.places:
            return None
        return min(self.places, key=lambda n: math.hypot(self.places[n]["x"] - p.x, self.places[n]["y"] - p.y))

    def counters(self) -> dict[str, tuple]:
        """The map's counters (box obstacles that are fixtures): fixture name -> (centre, size, yaw rad, obstacle)."""
        out = {}
        for o in self.spec.get("obstacles") or []:
            if "box" in o and o.get("fixture"):
                b = o["box"]
                out[o["fixture"]] = (tuple(b["center"]), tuple(b["size"]), math.radians(b.get("yaw_deg", 0.0)), o["name"])
        return out

    def counter_at(self, xy, margin: float = 0.10) -> str | None:
        """The counter whose top `xy` (world) is over, within `margin`, or None."""
        for name, (c, size, yaw, _) in self.counters().items():
            d = np.asarray(xy[:2], dtype=float) - np.asarray(c)
            u = np.array([math.cos(yaw), math.sin(yaw)])
            v = np.array([-u[1], u[0]])
            if abs(d @ u) <= size[0] / 2 + margin and abs(d @ v) <= size[1] / 2 + margin:
                return name
        return None

    def _world_from_odom(self) -> np.ndarray:
        """The tie (robot/localization.py's rule, from one state): world = mocap * odom^-1."""
        loc = self.env.localization()
        if not loc.live or loc.mocap_pose is None or loc.odom_pose is None:
            raise RuntimeError("no live mocap pose: can't place what the cameras see in the map")
        return _planar(loc.mocap_pose) @ np.linalg.inv(_planar(loc.odom_pose))

    # -- the base -------------------------------------------------------------------------

    def _robot(self):
        from chefbot.scripts.real_nav import make_robot

        robot = make_robot(self.env, self.grid, speed=self.speed, yaw_speed=self.yaw_speed, margin=0.10, settle_s=0.2,
                           still_mps=0.01, allow_arm_env=True, turn_in_room=True)  # fmt: skip
        robot._cfg = replace(robot._cfg, transient_retries=2)  # a link stall or a brief mocap dropout: the same goal again
        return robot

    def drive(self, goal: Pose2D, approach_m: float = 0.0, label: str = "goal") -> dict:
        """The base to `goal` (world) on the map; the host's control stopped while it drives, started again after."""
        from chefbot.scripts.real_nav import go_planned

        self.n += 1
        log_dir = self.out / f"base{self.n:02d}_{label}"
        log_dir.mkdir(parents=True, exist_ok=True)
        if self.host.dry_run:
            r = {"ok": True, "reason": "dry run: planned, not driven", "moved": False}
            (log_dir / "drive.json").write_text(json.dumps(r | {"goal": [goal.x, goal.y, goal.yaw]}, indent=2))
            return r
        robot = self._robot()

        def confirm(prompt):
            if self.confirm:
                input(prompt)

        self.host.halt()  # the arm holds its joints (the host's control would hold the flange still in odom)
        try:
            r = go_planned(self.env, goal, self.grid, robot=robot, log_dir=log_dir, confirm=confirm, approach_m=approach_m)
        finally:
            self.host.start_here()  # control again; the next skill's odom starts where the base is
        (log_dir / "drive.json").write_text(json.dumps(r, indent=2, default=str))
        return r

    def ready(self) -> dict:
        """The arm to the ready position (interface/workspace.py READY: tucked, the hand forward and down, where the
        pick skills start and the base drives from), by the host's joint path around the robot's body; the gripper as
        it is (a held thing stays held)."""
        from chefbot.interface.workspace import READY

        q0 = np.asarray(self.env.get_state().arm_q, dtype=float)
        far = float(np.degrees(np.abs(np.asarray(READY) - q0).max()))
        if self.host.dry_run:
            return {"ok": True, "reason": f"dry run: not moved ({far:.0f} deg from the ready position)"}
        self.host.halt()  # the host's control would fight the move
        try:
            self.host.go_ready(READY, confirm=self.confirm)
        except Exception as e:  # noqa: BLE001  (SafetyStop: no clear path)
            return {"ok": False, "reason": f"{type(e).__name__}: {e}"}
        finally:
            self.host.start_here()
        err = float(np.degrees(np.abs(np.asarray(self.env.get_state().arm_q) - np.asarray(READY)).max()))
        return {"ok": err < 3.0, "reason": f"at the ready position ({err:.1f} deg off)" if err < 3.0 else f"stopped {err:.1f} deg off the ready position",
                "gripper_width_m": round(float(self.env.get_state().gripper_width), 4)}  # fmt: skip

    def go_to(self, place: str) -> dict:
        if place not in self.places:
            return {"ok": False, "reason": f"no place {place!r}; places: {', '.join(sorted(self.places))}"}
        p = self.places[place]
        goal = Pose2D(float(p["x"]), float(p["y"]), math.radians(float(p.get("yaw_deg", 0.0))))
        r = self.drive(goal, float(p.get("approach_m", 0.0)), place)
        return {"ok": bool(r.get("ok")), "reason": r.get("reason"), "at": self.where(),
                "final_error_m": r.get("final_error_m")}  # fmt: skip

    def approach(self, object: str | None = None, x: float | None = None, y: float | None = None) -> dict:
        """To the stance for working on a counter target: where find() last saw `object`, or (x, y) in the world.
        Square to the counter it is on, its centre 0.45 m out from the edge, the target straight ahead."""
        from chefbot.robot.nav.goal import counter_stance

        if x is None or y is None:
            seen = self.found.get(object or "")
            if seen is None:
                return {"ok": False, "reason": f"no {object!r} found yet: find it first (or give x, y)"}
            x, y = seen["world"][:2]
        name = self.counter_at((x, y))
        if name is None:
            return {"ok": False, "reason": f"({x:.2f}, {y:.2f}) is on no counter of the map"}
        robot = self._robot()
        pose, info = counter_stance(robot, self.counters()[name], (x, y), approach=0.25)
        if pose is None:
            return {"ok": False, "reason": f"no stance at the {name} for ({x:.2f}, {y:.2f}): {info}"}
        r = self.drive(pose, 0.25, f"{name}_{info['side']}")
        return {"ok": bool(r.get("ok")), "reason": r.get("reason"), "counter": name, "side": info["side"],
                "target_inset_m": round(info["inset_m"], 3), "at": self.where(), "final_error_m": r.get("final_error_m")}  # fmt: skip

    # -- looking --------------------------------------------------------------------------

    def _vision(self, op: str, image: np.ndarray, **args) -> dict:
        d = self.out / "vision"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{time.time_ns()}.npy"
        np.save(path, image)
        return self.vision(op, image=str(path), **args)

    def find(self, object: str, cameras=FIND_CAMERAS) -> dict:
        """`object` in the cameras' view (Grounding DINO, then SAM 2's mask, then the depth under it): where it is in the
        world (the map's frame, by mocap), which counter it is on, and how sure. Nothing moves."""
        import cv2

        if self.vision is None:
            return {"found": False, "reason": "no vision server (VISION_URL)"}
        W = self._world_from_odom()
        self.n += 1
        d = self.out / f"find{self.n:02d}_{object.replace(' ', '_')}"
        d.mkdir(parents=True, exist_ok=True)
        seen = []
        for cam in cameras:
            if cam not in self.env.camera_mounts:
                continue
            try:
                f = self.env.get_camera_rgbd(cam)
            except Exception as e:  # noqa: BLE001  (a camera off its USB)
                self.log(f"[find] {cam}: {type(e).__name__}: {e}")
                continue
            cv2.imwrite(str(d / f"{cam}.jpg"), cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR))
            dets = self._vision("detect", f.rgb, queries=[object], model="grounding-dino", threshold=FIND_THRESHOLD)["detections"]
            dets.sort(key=lambda x: x["score"], reverse=True)
            if not dets or f.depth is None:
                continue
            mask = np.load(self._vision("segment", f.rgb, boxes=[dets[0]["box"]], model="sam2")["masks"])[0]
            cv2.imwrite(str(d / f"{cam}_mask.png"), mask.astype(np.uint8) * 255)
            depth = np.asarray(f.depth, dtype=float)
            good = mask & (depth > 0.1) & (depth < 3.0)
            if good.sum() < 20:
                continue
            v, u = np.nonzero(good)
            K = np.asarray(f.intrinsics, dtype=float)
            z = depth[v, u]
            pc = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z, np.ones_like(z)])
            pw = (W @ np.asarray(f.cam_to_world) @ pc)[:3].T  # camera -> odom (the env's frame) -> world
            p = np.median(pw, axis=0)
            p[2] = float(np.percentile(pw[:, 2], 95))  # its top
            seen.append({"camera": cam, "score": round(float(dets[0]["score"]), 3), "world": [round(float(c), 3) for c in p],
                         "distance_m": round(float(np.median(z)), 2), "label": dets[0].get("label")})  # fmt: skip
        if not seen:
            return {"found": False, "reason": f"no {object} seen by {', '.join(cameras)}", "at": self.where()}
        best = max(seen, key=lambda s: s["score"])
        out = {"found": True, "object": object, **best, "on": self.counter_at(best["world"]), "seen_by": seen, "at": self.where()}
        self.found[object] = out
        (d / "found.json").write_text(json.dumps(out, indent=2))
        return out


class HostSkills:
    """host-v1 skills as an agent's tools: each call is the skill's run(kitchen, **args) in this process; its result
    goes to the LLM as is, ok/found false as a failed call."""

    def __init__(self, kitchen: RealKitchen):
        self.kitchen = kitchen

    def move_to(self, out: Path) -> None:
        self.kitchen.move_to(out)

    def tool(self, skill) -> tuple[dict, callable]:
        from chefbot.interface.types import ToolResult
        from chefbot.runtime.agent_skills import _signature

        if skill.interface != "host-v1":
            raise ValueError(f"skill {skill.path} is {skill.interface}, not host-v1")
        run = skill.function()

        def handler(**args) -> ToolResult:
            said = f"{skill.name}({', '.join(f'{k}={json.dumps(v)}' for k, v in args.items())})"
            try:
                r = run(self.kitchen, **args)
            except Exception as e:  # noqa: BLE001  (the robot's stops come back as a failed call)
                return ToolResult(False, "skill_error", f"{said}: {type(e).__name__}: {e}", {"skill": skill.path})
            ok = bool(r.get("ok", r.get("found", True)))
            return ToolResult(ok, None if ok else "skill_failed", f"{said}: {json.dumps(r, default=str)}",
                              {"skill": skill.path, "returned": r})  # fmt: skip

        handler.__name__ = skill.name
        handler.__signature__ = _signature(skill)
        return skill.schema, handler


def main() -> None:
    """One host-v1 skill on the real robot, as the agent would call it (launch.sh on the robot, up.sh here):

        python -m chefbot.scripts.real_kitchen find tomato        # nothing moves: where it is on the map
        python -m chefbot.scripts.real_kitchen ready              # the arm to READY (Enter first)
        python -m chefbot.scripts.real_kitchen go_to island_south # Enter first
        python -m chefbot.scripts.real_kitchen approach tomato    # finds it, then drives to its stance (Enter first)
        python -m chefbot.scripts.real_kitchen where
    """
    import argparse
    import os
    from datetime import datetime

    from chefbot import embodiments
    from chefbot.environments.real.host import RealHost
    from chefbot.runtime import models as M

    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("find", "ready", "go_to", "approach", "where"))
    ap.add_argument("arg", nargs="?", default=None, help="the object (find, approach) or the place (go_to)")
    ap.add_argument("--map", type=Path, default=repo / "maps" / "src_kitchen_work")
    ap.add_argument("--dry-run", action="store_true", help="plan, drive and move nothing")
    a = ap.parse_args()
    M.load_env(repo / ".env")
    out = repo / "runs" / "real_kitchen" / f"{datetime.now():%Y%m%d_%H%M%S}_{a.what}"
    robot = embodiments.load("tidybot_fr3")
    host = RealHost(robot, robot.speed("slow"), out, dry_run=a.dry_run, confirm=True, cameras=["pole", "wrist"])
    url = os.environ.get("VISION_URL")
    k = RealKitchen(host, a.map, M.Vision(url) if url else None, out, confirm=True)
    try:
        if a.what == "where":
            r = k.where()
        elif a.what == "find":
            r = k.find(a.arg)
        elif a.what == "ready":
            r = k.ready()
        elif a.what == "go_to":
            r = k.go_to(a.arg)
        else:
            f = k.find(a.arg)
            print(json.dumps(f, indent=2))
            r = k.approach(a.arg) if f.get("found") else {"ok": False, "reason": "not found"}
        print(json.dumps(r, indent=2, default=str))
        print(f"files: {out}")
    finally:
        host.close()


if __name__ == "__main__":
    main()
