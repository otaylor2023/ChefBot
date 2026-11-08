"""The real kitchen for an agent's host-v1 skills (skills/*/host_v1): the base driven to the map's places by the
robot layer (scripts/real_nav.py), the arm to the ready position, and things found with the
cameras and the vision models, placed in the mocap world the map is in. Runs on the laptop, in the agent's process,
on the same real host (environments/real/host.py) the sandboxed skills run their episodes on.

    kitchen = RealKitchen(host, REPO / "maps/src_kitchen", vision, out)
    kitchen.go_to("island_north")          # A* on the map to a place, the footprint clear, mocap live
    kitchen.find("tomato")                 # {found, world: [x, y, z], on: "island", camera, ...}
    kitchen.explore("full")                # the sweep's stops, a look at each: the object map (perception/kitchen_map.py)
    kitchen.go_to("tomato")                # not a place: the map's tomato, parked for by the pick rule

The object map (what is on the counters, where) lives here on the laptop: each look's frames go through the vision
server's detector (YOLOE by default) into it, it is saved as objects.json in the run's folder after every look, and
the agent's kitchen map (agent/world/map.py) gets its objects. A LiveFeed (scripts/live_view.py) shows it all as it
happens. The options are switches so live runs can compare them (KitchenOptions): what to capture (stops / drives /
both), which cameras, which detector, when to name (end / background / off).

A base move first stops the host's control (it holds the flange still in odom, which would pull the arm back as the
base drives), holds the arm at its joints, drives, and starts the host's control again where the base ended: the
next skill's odom starts there.
"""

from __future__ import annotations

import json
import math
import time
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import yaml

from chefbot.interface.types import Pose2D

FIND_CAMERAS = ("pole", "wrist")
FIND_THRESHOLD = 0.3
REPO = Path(__file__).resolve().parents[2]
VOCAB = REPO / "chefbot" / "perception" / "inventory.txt"
STOP_APPROACH_M = 0.0  # one smooth path straight to the stance, as real_nav park (its --approach default; the proven parks)


@dataclass
class KitchenOptions:
    """The object map's switches (each run logs its timings; the live tests pick)."""

    cameras: tuple = ("pole",)  # which cameras a look takes frames from
    detector: str = "yoloe"  # yoloe (masks, one pass for the whole vocabulary) | grounding-dino | owlv2 (boxes, then SAM 2)
    threshold: float = 0.15
    capture: str = "stops"  # stops | drive | both: frames at the sweep's stops, while it drives (capture_hz), or both
    capture_hz: float = 1.0
    naming: str = "end"  # end (after a sweep) | background (each new object as it comes) | off
    vlm: str = "gemini-3.5-flash-lite"
    vocab: Path = VOCAB
    layout: Path | None = None  # a closed list (perception/layouts/*.yaml): its prompts, counts and names, not vocab
    save_frames: bool = True  # every frame's color + depth in the run's folder (frames/<camera>/), for replays and training

    def load_layout(self):
        from chefbot.perception.kitchen_map import Layout

        return Layout.load(self.layout) if self.layout else None

    def queries(self) -> list[str]:
        from chefbot.perception.detect_frames import load_classes

        L = self.load_layout()
        return L.queries() if L else load_classes(str(self.vocab)) or []


def _planar(p: Pose2D) -> np.ndarray:
    T = np.eye(4)
    c, s = math.cos(p.yaw), math.sin(p.yaw)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:2, 3] = [p.x, p.y]
    return T


class RealKitchen:
    def __init__(self, host, map_dir: Path, vision, out: Path, speed: float = 0.1, yaw_speed: float = 0.3,
                 confirm: bool = False, log=print, options: KitchenOptions | None = None, objects=None, world_map=None,
                 live=None):  # fmt: skip
        """`objects`: a KitchenMap to go on from (default: a new one, made on the first look); `world_map`: the agent's
        kitchen map, given the objects after each look; `live`: a LiveFeed (scripts/live_view.py)."""
        from chefbot.scripts.real_map import load as load_map

        self.host, self.env = host, host.raw_env
        self.map_dir, self.vision, self.out = Path(map_dir), vision, Path(out)
        self.grid, self.places = load_map(self.map_dir)
        self.spec = yaml.safe_load((self.map_dir / "spec.yaml").read_text())
        self.speed, self.yaw_speed, self.confirm, self.log = float(speed), float(yaw_speed), confirm, log
        self.found: dict[str, dict] = {}  # what find() last saw, by what was asked for
        self.n = 0
        self.options = options or KitchenOptions()
        self.objects, self.world_map, self.live = objects, world_map, live
        self._naming: threading.Thread | None = None

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

    def _arm_offset(self) -> np.ndarray:
        """What the env's camera poses leave out, odom from odom: the env places the arm base by the embodiment's mount
        (config/robot_tidybot_fr3.yaml arm_mount: right over the base pose, 0.335 m up, the sim base's), but the base
        pose is the base's turning centre (mocap calibrated to it by a spin, onboard/config.yaml), with the arm 0.1825 m
        ahead of it and its plate 0.44 m up (interface/workspace.py Body: the plate's centre, Jack's measurements). The
        cameras and the arm share the env's frame, so the skills are right in it; a position for the map is not. As a
        4x4 on the base pose: base * shift * base^-1."""
        from chefbot import embodiments

        e = embodiments.load("tidybot_fr3")
        bx, by = e.body.base_xy  # the base centre in the arm base frame: the arm is at -base_xy from the centre
        m = np.asarray(e.kin.mount_xyz, dtype=float)
        shift = np.eye(4)
        shift[:3, 3] = [-bx - m[0], -by - m[1], e.body.plate_height - m[2]]
        s = self.env.get_state().base_pose  # odom
        B = _planar(s)
        return B @ shift @ np.linalg.inv(B)

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
                           still_mps=0.01, allow_arm_env=True, turn_in_room=True,
                           arrive_hold=True)  # fmt: skip  (inside the tolerances: the goal frozen in odometry, no re-aiming)
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
        """The base to a place of the map, or, if `place` names none, in front of that object of the object map."""
        if place not in self.places:
            if self.objects is not None:
                return self.go_to_object(place)
            return {"ok": False, "reason": f"no place {place!r} (places: {', '.join(sorted(self.places))}) and no object map yet (explore first)"}
        p = self.places[place]
        goal = Pose2D(float(p["x"]), float(p["y"]), math.radians(float(p.get("yaw_deg", 0.0))))
        self._target({"name": place, "kind": "place", "stance": [goal.x, goal.y, goal.yaw]})
        r = self._timed_drive(goal, float(p.get("approach_m", 0.0)), place)
        return {"ok": bool(r.get("ok")), "reason": r.get("reason"), "at": self.where(),
                "final_error_m": r.get("final_error_m")}  # fmt: skip

    # -- the object map: parking for a thing, exploring -----------------------------------------------------------

    def stance_for(self, xy, counter: str) -> dict:
        """Where the base parks to reach `xy` on the box obstacle `counter`: the pick rule's stance on a side it can
        stand at (perception/goto.py park_target; deep or end targets move to the nearest legal spot)."""
        from chefbot.perception.goto import Floor, park_target
        from chefbot.perception.kitchen_map import counter_box

        box = counter_box(self.map_dir / "spec.yaml", counter)
        pt = park_target(*box[:3], list(xy)[:2], floor=Floor.from_spec(self.map_dir / "spec.yaml"))
        if "stance" not in pt:
            raise ValueError(f"no stance on {counter}'s {pt['side']} side")
        return pt

    def go_to_object(self, name: str) -> dict:
        """The base in front of the object map's `name` ("tomato", "tomato 2"), squared up to its counter's side."""
        if self.objects is None:
            return {"ok": False, "reason": "no object map yet (explore first)"}
        try:
            o = self.objects.resolve(name)
            pt = self.stance_for(o["xyz"], o["counter"])
        except (LookupError, ValueError) as e:
            return {"ok": False, "reason": str(e).strip("'\"")}
        sx, sy, syaw = pt["stance"]
        goal = Pose2D(float(sx), float(sy), math.radians(float(syaw)))
        note = f"on {o.get('fixture') or o['counter']} at ({o['xyz'][0]:.2f}, {o['xyz'][1]:.2f}), seen {o['n_views']} times; park on its {pt['side']} side"
        if not pt["within_rule"]:
            note += f" ({pt['reach_extra_m']:.2f} m past the pick rule: untrained reach)"
        self._target({"name": o["name"], "kind": "object", "xy": o["xyz"][:2], "stance": [goal.x, goal.y, goal.yaw],
                      "on": o.get("fixture"), "note": note})  # fmt: skip
        r = self._timed_drive(goal, STOP_APPROACH_M, o["name"].replace(" ", "_"))
        end = self.env.localization().mocap_pose
        reached = None if end is None else {"error_m": round(math.hypot(goal.x - end.x, goal.y - end.y), 3),
                                            "yaw_error_deg": round(math.degrees(math.remainder(goal.yaw - end.yaw, 2 * math.pi)), 1)}  # fmt: skip
        out = {"ok": bool(r.get("ok")), "reason": r.get("reason"), "object": o["name"], "on": o.get("fixture"),
               "object_xyz": o["xyz"], "side": pt["side"], "within_rule": pt["within_rule"],
               "reach_extra_m": pt["reach_extra_m"], "stance": [round(goal.x, 3), round(goal.y, 3), round(math.degrees(goal.yaw), 1)],
               "reached": reached, "at": self.where()}  # fmt: skip
        self._target(dict(self.live.state.get("target") or {}, result=f"{out['reason']}" + (f", {reached['error_m'] * 100:.0f} cm off the stance" if reached else ""))
                     if self.live else None)  # fmt: skip
        self._note("go_to", out)
        return out

    def sweep_stops(self, route: str = "full", sweep_file: Path | None = None) -> list[dict]:
        """The route's stops (maps/<map>/sweep.yaml): each {counter, side, x, target, stance} or {..., error}."""
        from chefbot.perception.kitchen_map import counter_box
        from chefbot.robot.nav.goal import TRAINED_INSET_M, counter_faces

        doc = yaml.safe_load(Path(sweep_file or self.map_dir / "sweep.yaml").read_text())
        if route not in doc["routes"]:
            raise ValueError(f"no route {route!r} (have: {', '.join(doc['routes'])})")
        inset = sum(TRAINED_INSET_M) / 2
        out = []
        for st in doc["routes"][route]:
            box = counter_box(self.map_dir / "spec.yaml", st["counter"])
            face = next(f for f in counter_faces(*box[:3]) if f["name"] == st["side"])
            edge = np.asarray(box[0], dtype=float) + face["n"] * (face["half_n"] - inset)
            along = (float(st["x"]) - edge[0]) / face["t"][0] if abs(face["t"][0]) > 0.5 else 0.0
            tgt = edge + face["t"] * along
            stop = dict(st, target=[round(float(tgt[0]), 3), round(float(tgt[1]), 3)])
            try:
                stop["stance"] = self.stance_for(stop["target"], st["counter"])["stance"]
            except ValueError as e:
                stop["error"] = str(e)
            out.append(stop)
        return out

    def explore(self, route: str = "south") -> dict:
        """Drive the route's stops and look at each (and/or while driving: options.capture), then name what was
        found (options.naming). The object map is saved after every look; the result has the timings by phase."""
        o = self.options
        stops = self.sweep_stops(route)
        t_all = time.time()
        rows = []
        self._status(f"exploring: route {route}, {len(stops)} stops")
        for i, st in enumerate(stops):
            label = f"stop{i + 1}_{st['counter'][:12]}_{st['side']}"
            if "error" in st:
                rows.append({"stop": i + 1, "ok": False, "reason": st["error"]})
                self._note("stop", rows[-1])
                continue
            sx, sy, syaw = st["stance"]
            goal = Pose2D(float(sx), float(sy), math.radians(float(syaw)))
            self._target({"name": f"stop {i + 1}/{len(stops)}", "kind": "sweep", "xy": st["target"],
                          "stance": [goal.x, goal.y, goal.yaw], "on": st["counter"], "note": f"{st['side']} side, x {st['x']}"})  # fmt: skip
            grab = _DriveCapture(self, o.capture_hz) if o.capture in ("drive", "both") else None
            try:
                r = self._timed_drive(goal, STOP_APPROACH_M, label)
            finally:
                if grab is not None:
                    grab.stop()
            row = {"stop": i + 1, "ok": bool(r.get("ok")), "reason": r.get("reason"), "drive_s": round(float(r.get("seconds") or 0.0), 2)}
            if grab is not None:
                row["drive_frames"] = grab.n
            if r.get("ok") and o.capture in ("stops", "both"):
                row["look"] = self.look()
            rows.append(row)
            self._note("stop", row)
            if not r.get("ok") and r.get("moved") is not False and not self.host.dry_run:
                break  # stopped part-way (e-stop, lost mocap): the rest of the route waits for Jack
        if self.objects is not None:
            merged = self.objects.tidy()
            if merged:
                self._event(f"{merged} split objects merged")
        naming = self.name_objects() if o.naming == "end" else self._wait_naming() if o.naming == "background" else None
        objs = self._sync()
        total = round(time.time() - t_all, 1)
        sums = {k: round(sum(float(t["s"]) for t in (self.live.state["timings"] if self.live else []) if t["what"] == k), 2)
                for k in ("drive", "frame", "detect", "map", "naming")}  # fmt: skip
        out = {"ok": all(r["ok"] for r in rows), "route": route, "stops": rows, "objects": [x["name"] for x in objs],
               "n_objects": len(objs), "seconds": total, "naming": naming, "options": _opts(o)}  # fmt: skip
        if self.live:
            out["seconds_by_phase"] = sums
        self._status(f"explored: {len(objs)} objects in {total:.0f} s")
        (self.out / "explore.json").write_text(json.dumps(out, indent=2, default=str))
        return out

    def look(self, cameras=None) -> dict:
        """The cameras' frames now into the object map (nothing moves). Returns what each saw and the map's size."""
        if self.vision is None:
            return {"ok": False, "reason": "no vision server (VISION_URL)"}
        km = self._object_map()
        W = self._world_from_odom() @ self._arm_offset()
        seen = {}
        for cam in cameras or self.options.cameras:
            if cam not in self.env.camera_mounts:
                continue
            seen[cam] = self._frame_into_map(cam, W)
        self._sync()
        if self.options.naming == "background":
            self._name_in_background()
        return {"ok": True, "cameras": seen, "entries": len(km.mm.entries)}

    def _frame_into_map(self, cam: str, W: np.ndarray, phase: str = "stop") -> dict:
        t0 = time.time()
        f = None
        for attempt in range(2):  # a camera that idled out can miss its first frame (11-08: the pole's, 5 s): once more
            try:
                f = self.env.get_camera_rgbd(cam)
                break
            except Exception as e:  # noqa: BLE001  (a camera off its USB)
                self._note("camera", {"camera": cam, "attempt": attempt + 1, "error": f"{type(e).__name__}: {e}"})
                self._event(f"{cam}: no frame ({type(e).__name__})" + (", trying again" if attempt == 0 else ""))
                err = f"{type(e).__name__}: {e}"
        if f is None:
            return {"error": err}
        t_frame = time.time()
        self._timing("frame", t_frame - t0, camera=cam, phase=phase)
        dets = self._detect(f.rgb)
        t_det = time.time()
        self._timing("detect", t_det - t_frame, camera=cam, model=self.options.detector, n=len(dets))
        if self.live:
            self.live.frame(cam, f.rgb, dets)
        if f.depth is None:
            return {"detections": len(dets), "error": "no depth"}
        T_wc = W @ np.asarray(f.cam_to_world)
        base = self.env.localization().mocap_pose
        r = self._object_map().add_frame(f.rgb, f.depth, f.intrinsics, T_wc, dets, t=t0, camera=cam, base_pose=base)
        if self.options.save_frames:
            self._save_frame(cam, f, T_wc, dets, t0, phase, base)
        self._timing("map", time.time() - t_det, camera=cam, new=len(r["new"]))
        for name in r["new"]:
            self._event(f"new: {name} ({cam})")
        return {"detections": len(dets), "new": r["new"]}

    def _save_frame(self, cam, f, T_wc, dets, t, phase, base) -> None:
        """frames/<camera>/NNNNNN_color.jpg + _depth.png (uint16 mm) and a frames.jsonl row (pose, K, detections)."""
        import cv2

        d = self.out / "frames" / cam
        d.mkdir(parents=True, exist_ok=True)
        n = sum(1 for _ in d.glob("*_color.jpg"))
        cv2.imwrite(str(d / f"{n:06d}_color.jpg"), cv2.cvtColor(np.asarray(f.rgb, np.uint8), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(d / f"{n:06d}_depth.png"), np.clip(np.nan_to_num(np.asarray(f.depth, float)) * 1000, 0, 65535).astype(np.uint16))
        row = {"n": n, "t": t, "phase": phase, "T_wc": np.round(T_wc, 6).tolist(), "K": np.asarray(f.intrinsics).tolist(),
               "base": None if base is None else [base.x, base.y, base.yaw], "dets": dets}  # fmt: skip
        with open(d / "frames.jsonl", "a") as fh:
            fh.write(json.dumps(row, default=float) + "\n")

    def _detect(self, rgb: np.ndarray) -> list[dict]:
        """The detector's detections with mask outlines ("poly"): YOLOE's own masks; Grounding DINO's or OWLv2's
        boxes through SAM 2."""
        o = self.options
        dets = self._vision("detect", rgb, queries=o.queries(), model=o.detector, threshold=o.threshold)["detections"]
        if o.detector == "yoloe" or not dets:
            return dets
        import cv2

        dets = dets[:20]
        masks = np.load(self._vision("segment", rgb, boxes=[d["box"] for d in dets], model="sam2")["masks"])
        for d, m in zip(dets, masks):
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            d["poly"] = max(cs, key=cv2.contourArea).reshape(-1, 2).tolist() if cs else []
        return dets

    def _object_map(self):
        if self.objects is None:
            from chefbot.perception.kitchen_map import KitchenMap

            self.objects = KitchenMap(self.map_dir / "spec.yaml", log=self.log, layout=self.options.load_layout())
        return self.objects

    def name_objects(self) -> dict | None:
        """Name every object not named yet, now (the VLM, once each)."""
        if self.objects is None:
            return None
        n = len(self.objects.unnamed())
        if not n:
            return {"named": 0, "seconds": 0.0}
        self._status(f"naming {n} objects ({self.options.vlm})")
        try:
            r = self.objects.name_new(model=self.options.vlm, hint=self.options.queries())
        except Exception as e:  # noqa: BLE001  (no key, no network: the detector's names stand)
            self._note("naming", {"error": f"{type(e).__name__}: {e}"})
            return {"named": 0, "error": f"{type(e).__name__}: {e}"}
        self._timing("naming", r["seconds"], n=r["named"], model=r["model"])
        for i, ans in r["answers"].items():
            if ans.get("error"):
                self._event(f"naming #{i}: {ans['error']}")
        self._sync()
        return {k: v for k, v in r.items() if k != "answers"}

    def _name_in_background(self) -> None:
        if self._naming is not None and self._naming.is_alive():
            return  # the next look starts another for what came since
        self._naming = threading.Thread(target=self.name_objects, name="naming", daemon=True)
        self._naming.start()

    def _wait_naming(self) -> dict | None:
        if self._naming is not None:
            self._naming.join()
        return self.name_objects()  # what the last background round missed

    def _sync(self) -> list[dict]:
        """The object map out: objects.json in the run's folder, the live page, the agent's kitchen map."""
        if self.objects is None:
            return []
        objs = self.objects.objects()
        self.objects.save(self.out / "objects.json")
        if self.live:
            self.live.objects(self.objects.objects(include_tentative=True))
        if self.world_map is not None:
            from chefbot.agent.world.map import MapEntry

            names = set()
            for o in objs:
                if o["status"] == "rejected" or o["xyz"] is None:
                    continue
                n = o["name"].replace(" ", "_")
                names.add(n)
                self.world_map.put(MapEntry(n, (float(o["xyz"][0]), float(o["xyz"][1])), kind="object", inside=o.get("fixture") or None))
            for e in self.world_map.list():
                if e.kind == "object" and e.name not in names:
                    self.world_map.remove(e.name)
        return objs

    # -- the run's record and its live page ---------------------------------------------------------------------

    def _timed_drive(self, goal: Pose2D, approach_m: float, label: str) -> dict:
        self._status(f"driving to {label}")
        r = self.drive(goal, approach_m, label)
        if self.live:
            self.live.set(path=[[round(float(p[0]), 3), round(float(p[1]), 3)] for p in (r.get("path") or [])][:400])
        self._timing("drive", float(r.get("seconds") or 0.0), to=label, ok=bool(r.get("ok")),
                     error_cm=None if r.get("final_error_m") is None else round(100 * float(r["final_error_m"]), 1))  # fmt: skip
        self._event(f"drive to {label}: {r.get('reason')}")
        return r

    def _status(self, text: str) -> None:
        self.log(f"[kitchen] {text}")
        if self.live:
            self.live.set(status=text)
            self.live.event(text)

    def _event(self, text: str) -> None:
        if self.live:
            self.live.event(text)

    def _timing(self, what: str, seconds: float, **info) -> None:
        if self.live:
            self.live.timing(what, seconds, **info)
        self.out.mkdir(parents=True, exist_ok=True)
        with open(self.out / "timings.jsonl", "a") as f:
            f.write(json.dumps({"t": round(time.time(), 3), "what": what, "s": round(float(seconds), 4), **info}, default=str) + "\n")

    def _target(self, target: dict | None) -> None:
        if self.live:
            self.live.set(target=target)

    def _note(self, what: str, row: dict) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        with open(self.out / "kitchen_log.jsonl", "a") as f:
            f.write(json.dumps({"t": round(time.time(), 3), "what": what, **row}, default=str) + "\n")

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
        W = self._world_from_odom() @ self._arm_offset()  # the cameras' odom points, the arm where it is, to the map
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
            pw = (W @ np.asarray(f.cam_to_world) @ pc)[:3].T  # camera -> odom (the env's frame) -> the arm's true place -> world
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


def _opts(o: KitchenOptions) -> dict:
    return {"cameras": list(o.cameras), "detector": o.detector, "capture": o.capture, "capture_hz": o.capture_hz,
            "naming": o.naming, "vlm": o.vlm, "vocab": str(o.vocab), "layout": str(o.layout) if o.layout else None}  # fmt: skip


class _DriveCapture:
    """Frames into the object map while the base drives, at most `hz` a second (the pole camera's: the wrist's view
    swings with the base). Each frame placed with the tie as it is when the frame comes (mocap and odometry move
    together), so a fast turn blurs it: that is what the live tests measure, with the drive's holds."""

    def __init__(self, kitchen: RealKitchen, hz: float, camera: str = "pole"):
        self.k, self.period, self.camera, self.n = kitchen, 1.0 / max(float(hz), 0.05), camera, 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="drive-capture", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.period):
            if self.k.vision is None or self.camera not in self.k.env.camera_mounts:
                return
            try:
                W = self.k._world_from_odom() @ self.k._arm_offset()
                self.k._frame_into_map(self.camera, W, phase="drive")
                self.n += 1
            except Exception as e:  # noqa: BLE001  (no mocap for a moment: skip the frame, never the drive)
                self.k._note("drive_capture", {"error": f"{type(e).__name__}: {e}"})

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=15.0)


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
        python -m chefbot.scripts.real_kitchen where
        python -m chefbot.scripts.real_kitchen look               # nothing moves: the cameras' frames into the object map
        python -m chefbot.scripts.real_kitchen explore --route full   # Enter before each stop: the sweep, the map
        python -m chefbot.scripts.real_kitchen go_to tomato --objects runs/real_kitchen/<explore run>/objects_memory.json
        python -m chefbot.scripts.real_kitchen session --objects ...  # one connection: type explore / go_to X / look

    The live page: python -m chefbot.scripts.live_view (http://127.0.0.1:8780), fed by --live (default runs/live).
    """
    import argparse
    import os
    from datetime import datetime

    from chefbot import embodiments
    from chefbot.environments.real.host import RealHost
    from chefbot.runtime import models as M

    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("find", "ready", "go_to", "where", "look", "explore", "session"))
    ap.add_argument("arg", nargs="?", default=None, help="the object (find), the place or object (go_to)")
    ap.add_argument("--map", type=Path, default=repo / "maps" / "src_kitchen_work")
    ap.add_argument("--dry-run", action="store_true", help="plan, drive and move nothing")
    ap.add_argument("--objects", type=Path, default=None, help="an object map to go on from (a run's objects_memory.json)")
    ap.add_argument("--route", default="south", help="explore: the route in the map's sweep.yaml (full, south, stove)")
    ap.add_argument("--cameras", nargs="+", default=["pole"], help="the cameras a look takes frames from (pole, wrist)")
    ap.add_argument("--detector", default="yoloe", choices=("yoloe", "grounding-dino", "owlv2"))
    ap.add_argument("--capture", default="stops", choices=("stops", "drive", "both"), help="explore: frames at the stops, while driving, or both")
    ap.add_argument("--capture-hz", type=float, default=1.0, help="frames a second while driving (--capture drive / both)")
    ap.add_argument("--naming", default="end", choices=("end", "background", "off"))
    ap.add_argument("--vlm", default="gemini-3.5-flash-lite", help="the naming model (gemini-..., or an ollama model)")
    ap.add_argument("--vocab", type=Path, default=VOCAB, help="the detector's text prompts and the naming hint")
    ap.add_argument("--layout", type=Path, default=None,
                    help="a closed list of what is out (chefbot/perception/layouts/*.yaml): only those, at most their counts")
    ap.add_argument("--live", type=Path, default=repo / "runs" / "live", help="the live page's folder (scripts/live_view.py)")
    ap.add_argument("--no-live", action="store_true")
    ap.add_argument("--base-speed", default="max", help="the base's speed profile (default max: the robot's cap, 0.25 m/s, 0.8 rad/s)")
    ap.add_argument("--arm-speed", default="slow", help="the arm's speed profile (ready)")
    a = ap.parse_args()
    M.load_env(repo / ".env")
    out = repo / "runs" / "real_kitchen" / f"{datetime.now():%Y%m%d_%H%M%S}_{a.what}"
    robot = embodiments.load("tidybot_fr3")
    profile = robot.speed(arm=a.arm_speed, base=a.base_speed)
    host = RealHost(robot, profile, out, dry_run=a.dry_run, confirm=True, cameras=["pole", "wrist"])
    url = os.environ.get("VISION_URL")
    opts = KitchenOptions(cameras=tuple(a.cameras), detector=a.detector, capture=a.capture, capture_hz=a.capture_hz,
                          naming=a.naming, vlm=a.vlm, vocab=a.vocab, layout=a.layout)  # fmt: skip
    objects = None
    if a.objects is not None:
        from chefbot.perception.kitchen_map import KitchenMap

        objects = KitchenMap.load(a.map / "spec.yaml", a.objects, layout=opts.load_layout())
    live = None
    if not a.no_live:
        from chefbot.scripts.live_view import LiveFeed

        live = LiveFeed(a.live, host.raw_env, log_dir=out)
    k = RealKitchen(host, a.map, M.Vision(url) if url else None, out, confirm=True, options=opts, objects=objects, live=live,
                    speed=profile.base_max_vel[0], yaw_speed=profile.base_max_vel[2])  # fmt: skip
    print(f"[kitchen] base {profile.base_max_vel[0]:.2f} m/s, {profile.base_max_vel[2]:.2f} rad/s ({a.base_speed})")
    if objects is not None:
        k._sync()

    def one(what, arg):
        if what == "where":
            return k.where()
        if what == "find":
            return k.find(arg)
        if what == "ready":
            return k.ready()
        if what == "look":
            return k.look()
        if what == "explore":
            return k.explore(arg or a.route)
        if what == "go_to":
            return k.go_to(arg)
        return {"ok": False, "reason": f"what? {what} (explore [route] / look / go_to X / where / objects / quit)"}

    status = "finished"
    try:
        if a.what != "session":
            r = one(a.what, a.arg)
            print(json.dumps(r, indent=2, default=str))
        else:  # one connection, many commands: explore once, then go_to what it found
            while True:
                try:
                    line = input("kitchen> ").strip()
                except EOFError:
                    break
                if line in ("quit", "exit", "q"):
                    break
                if not line:
                    continue
                what, _, arg = line.partition(" ")
                if what == "objects":
                    for o in (k.objects.objects() if k.objects else []):
                        print(f"  {o['name']:24s} {o.get('fixture') or o['counter']:14s} {o['status']:8s} views {o['n_views']}")
                    continue
                print(json.dumps(one(what, arg.strip() or None), indent=2, default=str))
        print(f"files: {out}")
    except BaseException:
        status = "stopped"
        raise
    finally:
        if live is not None:
            live.close(status)
        host.close()


if __name__ == "__main__":
    main()
