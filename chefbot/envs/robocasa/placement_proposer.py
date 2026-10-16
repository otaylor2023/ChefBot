"""GroundTruthPlacementProposer: where to put the held object down, from the
simulator's geometry (MANIP_SPEC §6.2). Stands in for a learned placement model."""

from __future__ import annotations

import math

import numpy as np

from chefbot.envs.robocasa.detector import GroundTruthDetector, named_fixtures
from chefbot.envs.robocasa.env import RoboCasaEnv
from chefbot.envs.robocasa.objects import find_object, object_points
from chefbot.envs.robocasa.spawn import _footprint, _ray_down
from chefbot.interface.proposers import PlacementProposer
from chefbot.interface.types import Detection, HeldObject, Placement

_PAD = 0.02  # free border kept around the object's footprint
_LATTICE = 0.05  # spacing of candidate spots on a fixture's top
_MAX_SPOTS = 10
_REACH = (0.45, 0.95)  # candidate spots lie this far from the base center
_FLATNESS = 0.008
# Other gripper headings to offer if the current one cannot reach, nearest first.
_YAWS_DEG = (0, 45, -45, 90, -90, 135, -135, 180)


class GroundTruthPlacementProposer(PlacementProposer):
    def __init__(self, env: RoboCasaEnv, detector: GroundTruthDetector):
        self._env = env
        self._detector = detector

    def propose(self, surface: Detection, held: HeldObject) -> list[Placement]:
        env, cfg = self._env, self._env.config
        with env.lock:
            grasped = env.grasped_bodies()
            if not grasped:
                return []
            body = next(iter(grasped))
            points = object_points(env, body)
            tcp = env.tcp_pose()
            base = env.get_state().base_pose
            middle = (points[:, :2].min(axis=0) + points[:, :2].max(axis=0)) / 2
            radius = float(np.linalg.norm(points[:, :2] - middle, axis=1).max())
            bottom = float(points[:, 2].min())
            spots = self._spots(surface, np.array([base.x, base.y]), radius, body)

        placements = []
        # Where the object sits relative to the grip point, and how high the grip point is above its bottom.
        offset = np.append(middle - tcp[:3, 3][:2], 0.0)
        grip_height = tcp[2, 3] - bottom
        # Whether the arm can get the object down onto a spot around what is near
        # it (a wall cabinet) is the robot layer's call, from its depth camera.
        for rank, (x, y, z) in enumerate(spots[:_MAX_SPOTS], start=1):
            release_height = z + cfg.place_clearance + grip_height
            for turn, yaw in enumerate(_YAWS_DEG):
                c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
                rotate = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
                pose = np.eye(4)
                pose[:3, :3] = rotate @ tcp[:3, :3]
                pose[:3, 3] = np.array([x, y, release_height]) - rotate @ offset
                placements.append(Placement(pose, cfg.preplace_offset, 1.0 - 0.05 * rank - 0.005 * turn))
        return sorted(placements, key=lambda p: -p.score)

    def _spots(self, surface: Detection, robot_xy: np.ndarray, radius: float, held_body: int) -> list[tuple]:
        """(x, y, surface height) to set the object's middle down on, best first."""
        env = self._env
        ignore = int(env.model.body_rootid[held_body])
        position = np.asarray(surface.position_world, dtype=float)
        item = self._detector.locate(surface)

        if item is not None and item.kind == "fixture":
            # A whole fixture top: any flat, free spot within reach. The detection
            # points at the part of the top the robot aligned to, so spots nearest
            # that point come first: the object goes down where the robot is facing.
            fixture = next(f for f in named_fixtures(env) if f.sim_name == item.fixture)
            box = fixture.box
            width, depth = float(np.linalg.norm(box.width)), float(np.linalg.norm(box.depth))
            spots = []
            for u in np.arange(radius + _PAD, width - radius - _PAD + 1e-9, _LATTICE) / width:
                for v in np.arange(radius + _PAD, depth - radius - _PAD + 1e-9, _LATTICE) / depth:
                    point = box.point(u, v, 1.0)
                    distance = float(np.linalg.norm(point[:2] - robot_xy))
                    if not _REACH[0] <= distance <= _REACH[1]:
                        continue
                    z = self._flat_height(point[0], point[1], radius + _PAD, box.top + 0.25, fixture.root, ignore)
                    if z is not None and abs(z - box.top) <= 0.03:
                        from_target = float(np.linalg.norm(point[:2] - position[:2]))
                        spots.append((from_target, (float(point[0]), float(point[1]), z)))
            return [spot for _, spot in sorted(spots)]

        # A burner, or an object to put things on or in (pan, plate, cutting board): its middle.
        target = find_object(env, surface)
        if target is not None:
            cloud = object_points(env, target)
            xy = (cloud[:, :2].min(axis=0) + cloud[:, :2].max(axis=0)) / 2
            start = float(cloud[:, 2].max()) + 0.05
        elif item is not None:
            xy, start = position[:2], position[2] + 0.25
        else:
            return []
        geom, z = self._ray(xy[0], xy[1], start, ignore)
        return [(float(xy[0]), float(xy[1]), z)] if geom >= 0 else []

    def _ray(self, x: float, y: float, z: float, ignore_root: int) -> tuple[int, float]:
        """First surface below (x, y, z) that is neither the robot nor the held object."""
        env = self._env
        for _ in range(16):
            geom, height = _ray_down(env, x, y, z)
            if geom < 0:
                return -1, -math.inf
            if geom not in env.robot_geoms and int(env.model.body_rootid[env.model.geom_bodyid[geom]]) != ignore_root:
                return geom, height
            z = height - 1e-4
        return -1, -math.inf

    def _flat_height(self, x, y, radius, from_z, root: str, ignore_root: int) -> float | None:
        env = self._env
        heights = []
        for px, py in _footprint(x, y, radius):
            geom, z = self._ray(px, py, from_z, ignore_root)
            if geom < 0 or env.root_body_name(int(env.model.geom_bodyid[geom])) != root:
                return None
            heights.append(z)
        return float(np.mean(heights)) if max(heights) - min(heights) <= _FLATNESS else None
