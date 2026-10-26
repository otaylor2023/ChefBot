"""Moving an object out of the way to see what is behind it, from the robot's own sensing only.

VisionGraspProposer: the ground-truth proposer's grasp geometry (top-down, front, side), but on
points from the robot's own depth cameras around the detected position instead of the object's
mesh. A camera sees an object's near half only, so the cloud is completed by mirroring it about
its middle along the line of sight (right for round and box-like things).

VisionPlacementProposer: sets the held object down at one given spot, its height from the robot's
depth points there; where the held object hangs below the gripper comes from the grasp's box.

free_spots: places on a fixture's top (from the robot's start map) with nothing on them in the
robot's depth images, away from what it has mapped.
"""

from __future__ import annotations

import math

import numpy as np

from chefbot.envs.robocasa.grasp_proposer import GroundTruthGraspProposer
from chefbot.interface.proposers import PlacementProposer
from chefbot.interface.types import Detection, HeldObject, Placement

CAMERAS = ("robot0_frontview", "robot0_agentview_left", "robot0_agentview_right")
_OBJECT_RADIUS = 0.09  # m: depth points this close (xy) to the detection are the object's...
_RING = (0.11, 0.2)  # ...and these, further out, the surface it stands on
_ABOVE = 0.01  # m above the surface: the object
_YAWS_DEG = (0, 45, -45, 90, -90)
# The object's box (Grasp.object_box, what the arm's collision checks leave out) is this much bigger
# than its points: from depth alone its far side is a guess, and the unseen rest must not count as in the way.
_BOX_PAD = 0.03


def own_points(env, center: np.ndarray, radius: float, cameras=CAMERAS) -> np.ndarray:
    """World points (n, 3) from the robot's depth cameras within `radius` (xy) of `center`."""
    out = []
    for cam in cameras:
        rgbd = env.get_camera_rgbd(cam)
        h, w = rgbd.depth.shape
        v, u = np.mgrid[0:h:2, 0:w:2]
        z = rgbd.depth[::2, ::2]
        ok = np.isfinite(z) & (z > 0.05) & (z < 3.0)
        k = rgbd.intrinsics
        pts = np.stack([(u[ok] - k[0, 2]) / k[0, 0] * z[ok], (v[ok] - k[1, 2]) / k[1, 1] * z[ok], z[ok], np.ones(ok.sum())], axis=1)
        world = (pts @ np.asarray(rgbd.cam_to_world, dtype=float).T)[:, :3]
        near = np.linalg.norm(world[:, :2] - center[:2], axis=1) <= radius
        out.append(world[near])
    return np.concatenate(out) if out else np.zeros((0, 3))


def surface_height(points: np.ndarray, center: np.ndarray) -> float | None:
    """The height of what the object at `center` stands on: the densest level among the points in a
    ring around it, below it (a top seen from above is many points at one height; a wall or the
    counter's front face spreads over many)."""
    d = np.linalg.norm(points[:, :2] - center[:2], axis=1)
    ring = points[(d >= _RING[0]) & (d <= _RING[1]) & (points[:, 2] > 0.3) & (points[:, 2] <= center[2] + 0.02)]
    return flat_level(ring[:, 2])


def flat_level(z: np.ndarray, step: float = 0.01) -> float | None:
    """The height most of `z` sits at (1 cm bins), refined to their median; None if too few."""
    if len(z) < 20:
        return None
    bins = np.floor(z / step).astype(int)
    values, counts = np.unique(bins, return_counts=True)
    peak = values[np.argmax(counts)]
    near = z[np.abs(bins - peak) <= 1]
    return float(np.median(near))


class VisionGraspProposer(GroundTruthGraspProposer):
    def __init__(self, env, robot_xy):
        """`robot_xy()`: where the robot's base is (its own odometry)."""
        super().__init__(env)
        self._robot_xy = robot_xy
        self.last_points: np.ndarray | None = None

    def propose(self, target: Detection) -> list:
        center = np.asarray(target.position_world, dtype=float)
        cloud = own_points(self._env, center, _RING[1])
        surface = surface_height(cloud, center)
        if surface is None:
            return []
        d = np.linalg.norm(cloud[:, :2] - center[:2], axis=1)
        # Above the surface and not far above the detected point (a wall cabinet overhead is not it).
        points = cloud[(d <= _OBJECT_RADIUS) & (cloud[:, 2] > surface + _ABOVE) & (cloud[:, 2] < center[2] + 0.2)]
        if len(points) < 20:
            return []
        robot_xy = np.asarray(self._robot_xy(), dtype=float)
        points = _mirrored(points, robot_xy)
        self.last_points = points
        grasps = self._top_down(points) + self._front(points, robot_xy) + self._side(points, robot_xy)
        box = np.array([points.min(axis=0), points.max(axis=0)])
        box[0, 2] = surface  # it stands on the surface
        box += [[-_BOX_PAD, -_BOX_PAD, 0.005], [_BOX_PAD, _BOX_PAD, _BOX_PAD]]
        for grasp in grasps:
            grasp.object_box = box
            if grasp.pregrasp_offset <= 0.0:
                grasp.pregrasp_offset = self._env.config.pregrasp_offset
        return sorted(grasps, key=lambda g: -g.score)


def _mirrored(points: np.ndarray, robot_xy: np.ndarray) -> np.ndarray:
    """The seen (near) half and its mirror image about the object's middle along the line of sight."""
    xy = points[:, :2]
    view = xy.mean(axis=0) - robot_xy
    view /= np.linalg.norm(view)
    across = np.array([-view[1], view[0]])
    a = xy @ across
    radius = (a.max() - a.min()) / 2
    near = float((xy @ view).min())
    middle = near + radius  # the middle is half its width behind its nearest point
    along = xy @ view
    mirror = points.copy()
    mirror[:, :2] += np.outer(2 * (middle - along), view)
    return np.concatenate([points, mirror])


class VisionPlacementProposer(PlacementProposer):
    def __init__(self, env, tcp_world):
        """`tcp_world()`: the gripper's pose (4x4, world), from the robot's own joints."""
        self._env = env
        self._tcp_world = tcp_world

    def propose(self, surface: Detection, held: HeldObject) -> list[Placement]:
        spot = np.asarray(surface.position_world, dtype=float)  # x, y and the surface height
        tcp = self._tcp_world()
        cfg = self._env.config
        if held.box_in_tcp is not None:
            corners = held.box_in_tcp @ tcp[:3, :3].T + tcp[:3, 3]
            middle = (corners[:, :2].min(axis=0) + corners[:, :2].max(axis=0)) / 2
            grip_height = tcp[2, 3] - float(corners[:, 2].min())
        else:
            middle, grip_height = tcp[:2, 3], 0.05
        offset = np.append(middle - tcp[:2, 3], 0.0)
        out = []
        for turn, yaw in enumerate(_YAWS_DEG):
            c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            rotate = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            pose = np.eye(4)
            pose[:3, :3] = rotate @ tcp[:3, :3]
            pose[:3, 3] = np.array([spot[0], spot[1], spot[2] + cfg.place_clearance + grip_height]) - rotate @ offset
            out.append(Placement(pose, cfg.preplace_offset, 1.0 - 0.01 * turn))
        return out


def free_spots(env, fixture, base_xy: np.ndarray, keep_away: list[np.ndarray], line: tuple[np.ndarray, np.ndarray],
               level: float | None = None, radius: float = 0.07, reach=(0.5, 0.9)) -> list[np.ndarray]:  # fmt: skip
    """(x, y, surface height) spots on `fixture`'s top (a map entry: location, facing, size) with
    nothing on them in the depth images, at `level` (the height the moved object stood at), at least
    0.15 m from everything in `keep_away`, best first:
    furthest from the line of sight `line` (camera xy, target xy), then nearest the robot."""
    loc, (fx, fy), (width, depth) = np.asarray(fixture.location, float), fixture.facing, fixture.size
    along, out_dir = np.array([-fy, fx]), np.array([fx, fy])
    cloud = own_points(env, np.append(base_xy, 0.0), reach[1] + 0.2)
    spots = []
    for a in np.arange(-width / 2 + radius, width / 2 - radius + 1e-9, 0.05):
        for b in np.arange(-depth / 2 + radius + 0.02, depth / 2 - radius + 1e-9, 0.05):  # clear of the wall behind
            xy = loc + a * along + b * out_dir
            if not reach[0] <= float(np.linalg.norm(xy - base_xy)) <= reach[1]:
                continue
            if any(float(np.linalg.norm(xy - np.asarray(k)[:2])) < 0.15 for k in keep_away):
                continue
            d = np.linalg.norm(cloud[:, :2] - xy, axis=1)
            here = cloud[(d <= radius) & (cloud[:, 2] > 0.3)]
            top = flat_level(here[:, 2])
            if level is not None and top is not None and abs(top - level) > 0.02:
                continue  # on something else (an appliance standing on the counter)
            if top is None or (np.abs(here[:, 2] - top) <= 0.01).mean() < 0.5:
                continue  # not seen well enough to call it free
            if ((here[:, 2] > top + 0.012) & (here[:, 2] < top + 0.4)).mean() > 0.03:
                continue  # something on it
            spots.append((-_from_line(xy, *line), float(np.linalg.norm(xy - base_xy)), np.array([xy[0], xy[1], top])))
    return [s for *_, s in sorted(spots, key=lambda s: (round(s[0], 2), s[1]))]


def _from_line(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b[:2] - a[:2]
    t = float(np.clip((p - a[:2]) @ ab / max(float(ab @ ab), 1e-9), 0.0, 1.0))
    return float(np.linalg.norm(p - (a[:2] + t * ab)))
