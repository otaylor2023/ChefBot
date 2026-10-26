"""look_closely: an exploration tool that looks at one mapped object with the wrist camera from close by.

The robot stands in front of the object (the explorer drives there first), lifts the arm so the
wrist camera is about 0.3 m from the object, looking down at it with the object above the gripper
in the picture, and runs the detector on that picture (VisionDetector.look_with). The object's
position is the robot's own estimate (where it last saw it); nothing else is read. If the view is
blocked (no arm pose reaches it, or the object is still partly hidden), it tries from the left and
from the right.
"""

from __future__ import annotations

import math
import threading

import numpy as np

from chefbot.interface.types import ToolResult
from chefbot.interface.types import Detection, MotionResult, MotionStatus
from chefbot.robot.skills.common import follow_path, to_ready

WRIST_CAMERA = "robot0_eye_in_hand"

SCHEMA = {
    "type": "function",
    "function": {
        "name": "look_closely",
        "description": "Look at a mapped object with the wrist camera from about 0.3 m (stand in front of it "
                       "first: go_to it with distance 0.65). Updates its name and position from the close view.",
        "parameters": {
            "type": "object",
            "properties": {"place": {"type": "string", "description": "the object's map name"}},
            "required": ["place"],
        },
    },
}


class CloseLook:
    def __init__(self, robot, env, nav_tools, detector, where, config: dict | None = None):
        """`where(place)`: where the robot last saw that object (3D), or None."""
        cfg = config or {}
        self._robot, self._env, self._tools, self._detector, self._where = robot, env, nav_tools, detector, where
        self._distance = float(cfg.get("close_distance", 0.32))  # m, camera to object
        self._elevation = math.radians(float(cfg.get("close_elevation_deg", 50)))  # looking down this much
        self._dip = math.radians(float(cfg.get("close_dip_deg", 28)))  # aim below it: the object sits above the gripper
        self._sides = [math.radians(a) for a in cfg.get("close_sides_deg", [0, 30, -30])]
        self._move_occluders = bool(cfg.get("move_occluders", False))
        self._map = None  # the robot's start map (fixtures), for where to set an object down
        self.stats = {"looks": 0, "views": 0, "no_arm_pose": 0, "still_occluded": 0,
                      "moved": 0, "put_back": 0, "failed_grasps": 0, "failed_places": 0, "no_free_spot": 0}  # fmt: skip
        self.moves: list[dict] = []
        self._tcp_to_cam: np.ndarray | None = None

    def look_closely(self, place: str) -> ToolResult:
        target = self._where(place)
        if target is None:
            return ToolResult(False, "unknown_place", f'"{place}" has not been seen yet: nothing to look at.')
        target = np.asarray(target, dtype=float)
        stack = self._robot._arm
        self.stats["looks"] += 1
        tried, result, views = 0, None, []
        for side in self._sides:
            q = self._arm_pose(stack, target, side)
            if q is None:
                continue
            tried += 1
            if self._run(lambda preempt: self._move(stack, q, preempt)).status != MotionStatus.SUCCESS:
                self._run(lambda preempt: self._back(stack, preempt))
                continue
            self.stats["views"] += 1
            with self._detector.viewing(self._render, WRIST_CAMERA):  # detect() as usual, through the wrist camera
                result = self._tools.detect()
            views.append(self._view_record(stack, side))
            self._run(lambda preempt: self._back(stack, preempt))
            clear = self._detector.seen_clearly(target)
            if clear:
                break
            self.stats["still_occluded"] += clear is False  # partly hidden still: try another side
        if not tried:
            self.stats["no_arm_pose"] += 1
            return ToolResult(False, "unreachable", f"look_closely({place}): no arm pose puts the wrist camera near it from here.")
        if result is None:
            return ToolResult(False, "blocked", f"look_closely({place}): the arm could not get there without touching something.")
        if self._move_occluders and self._detector.seen_clearly(target) is False:
            again = self._move_occluder(place, target, stack)
            if again is not None:
                views += again.data.pop("close_views", [])
                result = again
        # For the replay (rootcause.py): where the arm was for each close view, and the frame it logged.
        result.data["close_views"] = views
        result.data["target"] = [round(float(v), 4) for v in target]
        result.message = f"look_closely({place}): " + result.message
        return result

    # -- moving what is in the way ------------------------------------------------

    def _move_occluder(self, place: str, target: np.ndarray, stack):
        """Pick the mapped object in front of the target, set it down on a free spot nearby, look
        again, and put it back where it was. Grasp and placement from the robot's own depth only."""
        from chefbot.explore.move_aside import VisionGraspProposer, VisionPlacementProposer, free_spots

        base = self._robot.get_base_pose()
        eye = np.array([base.x, base.y])
        occluder = self._occluder(target, eye)
        if occluder is None:
            return None
        label, where = occluder
        fixture = self._fixture_under(where)
        if fixture is None:
            return None
        keep_away = [p for _, p in self._detector.tracked()] + [target]
        from chefbot.explore.move_aside import own_points, surface_height

        level = surface_height(own_points(self._env, where, 0.25), where)
        spots = free_spots(self._env, fixture, eye, keep_away, (eye, target), level=level)
        record = {"target": place, "moved": label, "from": [round(float(v), 3) for v in where]}
        self.moves.append(record)
        if not spots:
            self.stats["no_free_spot"] += 1
            record["outcome"] = "no free spot"
            return None
        robot = self._robot
        saved = robot._grasp_proposer, robot._placement_proposer
        robot._grasp_proposer = VisionGraspProposer(self._env, lambda: (robot.get_base_pose().x, robot.get_base_pose().y))
        robot._placement_proposer = VisionPlacementProposer(self._env, stack.tcp_world)
        try:
            robot.align_to_point(where)  # within the arm's reach of it
            robot.wait(60.0)
            robot.pick(Detection(label, where, 0.9))
            if robot.wait(120.0).status != MotionStatus.SUCCESS:
                self.stats["failed_grasps"] += 1
                record["outcome"] = "grasp failed"
                self._run(lambda preempt: self._back(stack, preempt))
                return None
            spot = spots[0]
            robot.place(Detection("counter", spot, 0.9))
            if robot.wait(120.0).status != MotionStatus.SUCCESS:
                self.stats["failed_places"] += 1
                record["outcome"] = "could not set it down"
                return None
            self.stats["moved"] += 1
            record["to"] = [round(float(v), 3) for v in spot]
            result = self.look_closely_once(place, target, stack)
            # Put it back where it was.
            robot.align_to_point(np.array([spot[0], spot[1], where[2]]))
            robot.wait(60.0)
            robot.pick(Detection(label, np.array([spot[0], spot[1], where[2]]), 0.9))
            if robot.wait(120.0).status == MotionStatus.SUCCESS:
                surface = np.array([where[0], where[1], spot[2]])
                robot.place(Detection("counter", surface, 0.9))
                if robot.wait(120.0).status == MotionStatus.SUCCESS:
                    self.stats["put_back"] += 1
                    record["outcome"] = "moved, looked, put back"
                else:
                    self.stats["failed_places"] += 1
                    record["outcome"] = "moved, looked, could not put it back down"
            else:
                self.stats["failed_grasps"] += 1
                record["outcome"] = "moved, looked, could not pick it up again"
            return result
        finally:
            robot._grasp_proposer, robot._placement_proposer = saved

    def look_closely_once(self, place: str, target: np.ndarray, stack):
        """One close look, straight on."""
        q = self._arm_pose(stack, target, 0.0)
        if q is None or self._run(lambda preempt: self._move(stack, q, preempt)).status != MotionStatus.SUCCESS:
            self._run(lambda preempt: self._back(stack, preempt))
            return None
        self.stats["views"] += 1
        with self._detector.viewing(self._render, WRIST_CAMERA):
            result = self._tools.detect()
        result.data["close_views"] = [dict(self._view_record(stack, 0.0), moved_aside=True)]
        self._run(lambda preempt: self._back(stack, preempt))
        result.message = f"look_closely({place}) with what was in front moved aside: " + result.message
        return result

    def _occluder(self, target: np.ndarray, eye: np.ndarray):
        """The mapped object nearest the line of sight between the robot and the target, in front of it."""
        best, best_d = None, 0.1
        to = target[:2] - eye
        for label, p in self._detector.tracked():
            rel = p[:2] - eye
            t = float(rel @ to / max(float(to @ to), 1e-9))
            if not 0.3 < t < 1.0 or float(np.linalg.norm(p[:2] - target[:2])) < 0.04:
                continue  # behind it, beside the robot, or the target itself
            if float(np.linalg.norm(p[:2] - target[:2])) > 0.35:
                continue
            d = float(np.linalg.norm(rel - t * to))
            if d < best_d:
                best, best_d = (label, np.asarray(p, dtype=float)), d
        return best

    def _fixture_under(self, where: np.ndarray):
        for e in self._tools._map.list():
            if e.kind != "fixture" or e.size is None or e.facing is None or e.z is not None:
                continue
            fx, fy = e.facing
            dx, dy = float(where[0]) - e.location[0], float(where[1]) - e.location[1]
            if abs(dx * -fy + dy * fx) <= e.size[0] / 2 and abs(dx * fx + dy * fy) <= e.size[1] / 2:
                return e
        return None

    def _view_record(self, stack, side: float) -> dict:
        q = self._env.get_state().arm_q
        return {"q": [round(float(v), 5) for v in q], "side_deg": round(math.degrees(side)),
                "frame": getattr(self._detector, "_calls", None)}

    # -- arm --------------------------------------------------------------------

    def _arm_pose(self, stack, target: np.ndarray, side: float):
        """Joint angles putting the wrist camera `distance` from the target, `side` to the left or right
        of straight on, looking down `elevation` with its axis dipped below the target; or None."""
        base = self._robot.get_base_pose()
        yaw = math.atan2(target[1] - base.y, target[0] - base.x) + side
        el = self._elevation
        u = np.array([math.cos(yaw) * math.cos(el), math.sin(yaw) * math.cos(el), -math.sin(el)])  # camera to target
        x = np.cross(u, [0.0, 0.0, 1.0])
        x /= np.linalg.norm(x)  # the camera's right
        axis = u * math.cos(self._dip) - np.cross(x, u) * math.sin(self._dip)
        cam = np.eye(4)
        cam[:3, 0], cam[:3, 2], cam[:3, 3] = x, -axis, target - self._distance * u  # MuJoCo cameras look along -z
        cam[:3, 1] = np.cross(cam[:3, 2], x)
        tcp = cam @ np.linalg.inv(self._tcp_camera(stack))
        q = stack.ik.solve_pose(stack.to_arm(tcp), seed=stack.ready)
        return q

    def _tcp_camera(self, stack) -> np.ndarray:
        """The wrist camera's pose in the TCP frame (fixed: it is mounted on the hand)."""
        if self._tcp_to_cam is None:
            m, d = self._env.model, self._env.data
            with self._env.lock:
                i = m.camera(WRIST_CAMERA).id
                cam = np.eye(4)
                cam[:3, :3], cam[:3, 3] = d.cam_xmat[i].reshape(3, 3), d.cam_xpos[i]
            self._tcp_to_cam = np.linalg.inv(stack.tcp_world()) @ cam
        return self._tcp_to_cam

    def _move(self, stack, q, preempt: threading.Event) -> MotionResult:
        path = stack.workspace().path(stack.arm.current_target(), q)
        if path is None:
            return MotionResult(MotionStatus.UNREACHABLE, self._robot.get_base_pose(), "no collision-free way there")
        r = follow_path(stack, path, preempt)
        return MotionResult(MotionStatus.SUCCESS if r.ok else r.status, self._robot.get_base_pose(), r.message)

    def _back(self, stack, preempt: threading.Event) -> MotionResult:
        r = to_ready(stack, stack.workspace(), preempt)
        if not r.ok:  # the camera's view of the world can block the way back: straight back then
            r = stack.arm.move_joints(stack.ready, preempt)
        return MotionResult(MotionStatus.SUCCESS if r.ok else r.status, self._robot.get_base_pose(), r.message)

    def _run(self, fn) -> MotionResult:
        self._robot._motion.start(fn)
        return self._robot.wait(60.0)

    # -- camera -----------------------------------------------------------------

    def _render(self):
        """The wrist camera's image, and the robot's own pixels in it (from its own geometry)."""
        env = self._env
        rgbd = env.get_camera_rgbd(WRIST_CAMERA)
        h, w = rgbd.depth.shape

        def seg():
            with env.lock:
                return env.suite.sim.render(camera_name=WRIST_CAMERA, width=w, height=h, segmentation=True)

        import mujoco

        seg_ids = env._render_thread.submit(seg).result()[::-1]
        geom = seg_ids[..., 0] == mujoco.mjtObj.mjOBJ_GEOM
        mask = geom & np.isin(seg_ids[..., 1], np.array(sorted(env.robot_geoms)))
        return rgbd, mask
