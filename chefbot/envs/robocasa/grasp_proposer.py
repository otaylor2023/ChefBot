"""GroundTruthGraspProposer: grasps computed from the simulator's geometry.

Stands in for a learned grasp model (MANIP_SPEC §6.1). Three kinds of grasp,
all from the object's collision geometry as a point cloud:

  top-down   over the middle, fingers across the narrowest side
  side       for tall objects: from the robot's side at mid-height, gripper angled down
  front      the same approach for anything on a shelf, backing straight out afterwards
  handle     a thin part sticking out one end (a pan's handle)
  rim        the wall of an open container (a bowl), fingers astride it

Handle and rim grasps go beyond the spec; they cover objects too wide for the
gripper to span. A from-above grasp is shortened, or dropped in favor of the
others, when something overhead (a wall cabinet) leaves the arm no room.
"""

from __future__ import annotations

import math

import numpy as np

from chefbot.envs.robocasa.env import RoboCasaEnv
from chefbot.envs.robocasa.objects import ARM_ABOVE_TCP, ceiling_above, find_object, object_points
from chefbot.envs.robocasa.spawn import _ray_up
from chefbot.interface.proposers import GraspProposer
from chefbot.interface.types import Detection, Grasp

STROKE = 0.085  # 2F-85: widest gap between the pads
_FIT_MARGIN = 0.004  # an object must be this much narrower than the stroke
_FINGERTIP = 0.019  # the pads reach this far beyond the grip point
_TABLE_CLEARANCE = 0.004  # fingertips stay this far above what the object stands on
_BAND = 0.015  # half-height of the slice of the object the fingers close on
_HANDLE_MAX_WIDTH = 0.05
_HANDLE_MIN_LENGTH = 0.05
_SLICE = 0.01  # slices along an object's long axis when looking for a handle
_MIN_WALL = 0.002
_RIM_DEPTH = 0.015  # grip point this far below a container's rim
_RIM_ANGLES_DEG = (0, 25, -25, 50, -50, 80, -80)  # around the rim, starting on the robot's side
_DOWN = np.array([0.0, 0.0, -1.0])
_OPEN_ABOVE = 0.35  # less free height than this above an object: it is on a shelf
_FULL_LIFT = 0.12  # a shorter lift than this is spelled out in the grasp
# Angled down first: an object held with a level gripper cannot be set down at
# counter height afterwards (the arm's wrist limit), one held at an angle can.
_FRONT_PITCH_DEG = (35, 20, 0)
_FRONT_RETREAT = 0.16  # back out this far after grasping on a shelf


def tcp_pose(position: np.ndarray, approach: np.ndarray, closing: np.ndarray) -> np.ndarray:
    """TCP pose with z along `approach` and x along `closing` (made perpendicular to it)."""
    z = approach / np.linalg.norm(approach)
    x = closing - (closing @ z) * z
    x /= np.linalg.norm(x)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = x, np.cross(z, x), z, position
    return pose


def extent(points: np.ndarray, direction: np.ndarray) -> tuple[float, float]:
    along = points @ direction
    return float(along.min()), float(along.max())


class GroundTruthGraspProposer(GraspProposer):
    def __init__(self, env: RoboCasaEnv):
        self._env = env

    def propose(self, target: Detection) -> list[Grasp]:
        env, cfg = self._env, self._env.config
        with env.lock:
            body = find_object(env, target)
            if body is None:
                return []
            points = object_points(env, body)
            base = env.get_state().base_pose
            if len(points) == 0:
                return []
            headroom = self._headroom(points)
        robot_xy = np.array([base.x, base.y])
        from_above = [] if headroom < _OPEN_ABOVE else self._top_down(points) + self._side(points, robot_xy)
        if not from_above and headroom >= _OPEN_ABOVE:
            # Too wide to span: look for something thinner to hold on to.
            from_above = self._handle(points) or self._rim(points, robot_xy)
        with env.lock:
            grasps = [g for g in from_above if self._fit_under(g, body, float(points[:, 2].max()))]
        if not grasps and (headroom < _OPEN_ABOVE or from_above):
            # On a shelf, or under something the arm would hit coming down
            # (a wall cabinet): reach in from the front instead.
            grasps = self._front(points, robot_xy, headroom)
        for grasp in grasps:
            if grasp.pregrasp_offset <= 0.0:
                grasp.pregrasp_offset = cfg.pregrasp_offset
        return sorted(grasps, key=lambda g: -g.score)

    def _headroom(self, points: np.ndarray) -> float:
        """Free height above the object."""
        top = points[points[:, 2].argmax()]
        middle = (points[:, :2].min(axis=0) + points[:, :2].max(axis=0)) / 2
        return min(_ray_up(self._env, middle[0], middle[1], top[2] + 0.005), 10.0)

    def _fit_under(self, grasp: Grasp, body: int, top: float) -> bool:
        """Shorten a from-above grasp's approach and lift so the wrist stays under
        whatever is over the grasp point (a wall cabinet); False if there is not
        even room to get the fingers over the object."""
        if grasp.tcp_pose_world[2, 2] > -0.5:
            return True  # not coming down from above
        own = {int(self._env.model.body_rootid[body])}
        ceiling = ceiling_above(self._env, grasp.tcp_pose_world[:2, 3], top - 0.01, own)
        if math.isinf(ceiling):
            return True
        grip = float(grasp.tcp_pose_world[2, 3])
        travel = ceiling - ARM_ABOVE_TCP - grip
        if travel < (top - grip) + _FINGERTIP + 0.01:
            return False
        offset = self._env.config.pregrasp_offset
        if travel < offset:
            grasp.pregrasp_offset = travel
        if travel < _FULL_LIFT:
            grasp.retreat = np.array([0.0, 0.0, travel])
        return True

    # -- grasp kinds ----------------------------------------------------------

    def _front(self, points: np.ndarray, robot_xy: np.ndarray, headroom: float) -> list[Grasp]:
        """For an object with something over it: reach in from the robot's side,
        then lift it slightly and back straight out."""
        toward = (points[:, :2].min(axis=0) + points[:, :2].max(axis=0)) / 2 - robot_xy
        toward /= np.linalg.norm(toward)
        lift = min(0.02, max(0.0, headroom - 0.05))
        return self._angled(points, robot_xy, retreat=np.append(-_FRONT_RETREAT * toward, lift), score=0.9)

    def _angled(self, points: np.ndarray, robot_xy: np.ndarray, retreat, score: float) -> list[Grasp]:
        """Grasps around the object's middle, approaching from the robot's side
        with the gripper angled down a little or level; fingers close sideways."""
        bottom, top = extent(points, np.array([0.0, 0.0, 1.0]))
        z = max((bottom + top) / 2, bottom + 0.012)
        band = points[np.abs(points[:, 2] - z) <= _BAND]
        if len(band) < 3:
            band = points
        middle = (band[:, :2].min(axis=0) + band[:, :2].max(axis=0)) / 2
        toward = middle - robot_xy
        toward /= np.linalg.norm(toward)
        closing = np.array([-toward[1], toward[0]])
        low, high = extent(band[:, :2], closing)
        if high - low > STROKE - _FIT_MARGIN:
            return []
        center = middle + ((low + high) / 2 - middle @ closing) * closing
        grasps = []
        for rank, pitch in enumerate(_FRONT_PITCH_DEG):
            angle = math.radians(pitch)
            approach = np.append(math.cos(angle) * toward, -math.sin(angle))
            for grasp in _both_ways(
                np.array([center[0], center[1], z]), approach, np.append(closing, 0.0), high - low, score - 0.05 * rank
            ):
                grasp.retreat = retreat
                grasps.append(grasp)
        return grasps

    def _grip_height(self, bottom: float, top: float) -> float:
        """Height of the grip point for a top-down grasp: fingertips about 60% of
        the way down a small object, `grasp_depth` below the top of a tall one."""
        height = top - bottom
        depth = min(0.6 * height - _FINGERTIP, self._env.config.grasp_depth)
        return max(top - depth, bottom + _FINGERTIP + _TABLE_CLEARANCE)

    def _top_down(self, points: np.ndarray) -> list[Grasp]:
        bottom, top = extent(points, np.array([0.0, 0.0, 1.0]))
        z = self._grip_height(bottom, top)
        band = points[np.abs(points[:, 2] - z) <= max(_BAND, _FINGERTIP)]
        if len(band) < 3:
            return []
        narrow = _narrow_axis(band[:, :2])
        low, high = extent(band[:, :2], narrow)
        if high - low > STROKE - _FIT_MARGIN:
            return []
        across = np.array([-narrow[1], narrow[0]])
        a_low, a_high = extent(band[:, :2], across)
        center = (low + high) / 2 * narrow + (a_low + a_high) / 2 * across
        return _both_ways(np.array([center[0], center[1], z]), _DOWN, np.append(narrow, 0.0), high - low, 1.0)

    def _side(self, points: np.ndarray, robot_xy: np.ndarray) -> list[Grasp]:
        """For tall objects: around the middle, from the robot's side."""
        bottom, top = extent(points, np.array([0.0, 0.0, 1.0]))
        if top - bottom < self._env.config.side_grasp_min_height:
            return []
        return self._angled(points, robot_xy, retreat=None, score=0.6)

    def _handle(self, points: np.ndarray) -> list[Grasp]:
        """A run of thin slices at one end of the object's long axis."""
        long_axis = _long_axis(points[:, :2])
        across = np.array([-long_axis[1], long_axis[0]])
        along = points[:, :2] @ long_axis
        edges = np.arange(along.min(), along.max() + _SLICE, _SLICE)
        thin = []
        for start in edges[:-1]:
            piece = points[(along >= start) & (along < start + _SLICE)]
            thin.append(len(piece) >= 2 and np.ptp(piece[:, :2] @ across) <= _HANDLE_MAX_WIDTH)
        best: tuple[int, int] | None = None
        for end in (0, len(thin) - 1):  # a handle starts at an end of the object
            step, i = (1, 0) if end == 0 else (-1, len(thin) - 1)
            count = 0
            while 0 <= i < len(thin) and thin[i]:
                count, i = count + 1, i + step
            if count * _SLICE >= _HANDLE_MIN_LENGTH and (best is None or count > best[1]):
                best = (end, count)
        if best is None or best[1] == len(thin):
            return []
        end, count = best
        # Hold the handle two thirds of the way out from the body.
        offset = (count * _SLICE) * (1 / 3)
        at = along.min() + offset if end == 0 else along.max() - offset
        piece = points[np.abs(along - at) <= _SLICE]
        low, high = extent(piece[:, :2], across)
        bottom, top = float(piece[:, 2].min()), float(piece[:, 2].max())
        center = at * long_axis + (low + high) / 2 * across
        z = max(top - 0.6 * (top - bottom), float(points[:, 2].min()) + _FINGERTIP + _TABLE_CLEARANCE)
        return _both_ways(np.array([center[0], center[1], z]), _DOWN, np.append(across, 0.0), high - low, 0.8)

    def _rim(self, points: np.ndarray, robot_xy: np.ndarray) -> list[Grasp]:
        """Fingers astride the wall of an open container, nearest the robot first."""
        top, bottom = float(points[:, 2].max()), float(points[:, 2].min())
        if top - bottom < _RIM_DEPTH + _FINGERTIP + _TABLE_CLEARANCE:
            return []  # too shallow to have a wall the fingers can get astride
        rim = points[points[:, 2] >= top - 0.01]
        middle = (points[:, :2].min(axis=0) + points[:, :2].max(axis=0)) / 2
        toward_robot = math.atan2(robot_xy[1] - middle[1], robot_xy[0] - middle[0])
        grasps = []
        for rank, offset in enumerate(_RIM_ANGLES_DEG):
            angle = toward_robot + math.radians(offset)
            radial = np.array([math.cos(angle), math.sin(angle)])
            tangent = np.array([-radial[1], radial[0]])
            rel = rim[:, :2] - middle
            near = rim[(np.abs(rel @ tangent) <= 0.012) & (rel @ radial > 0)]
            if len(near) == 0:
                continue
            outer = float(np.max((near[:, :2] - middle) @ radial))
            # The wall: everything near the top within finger reach inward of the outer edge.
            rel_all = points[:, :2] - middle
            wall = points[
                (np.abs(rel_all @ tangent) <= 0.012)
                & (rel_all @ radial > outer - STROKE)
                & (points[:, 2] >= top - _RIM_DEPTH - _FINGERTIP)
            ]
            inner = float(np.min((wall[:, :2] - middle) @ radial))
            thickness = outer - inner
            if thickness > STROKE - 0.02 or thickness < _MIN_WALL:
                continue
            center = middle + (outer + inner) / 2 * radial
            position = np.array([center[0], center[1], top - _RIM_DEPTH])
            grasps += _both_ways(position, _DOWN, np.append(radial, 0.0), thickness, 0.7 - 0.02 * rank)
        return grasps


def _both_ways(position, approach, closing, width: float, score: float) -> list[Grasp]:
    """The grasp and its mirror image (gripper turned half a turn about the approach axis)."""
    return [
        Grasp(tcp_pose(position, approach, closing), 0.0, float(width), score),
        Grasp(tcp_pose(position, approach, -closing), 0.0, float(width), score - 0.01),
    ]


def _principal_axes(xy: np.ndarray) -> np.ndarray:
    """Unit vectors (rows) of a horizontal point set's principal axes, longest first."""
    centered = xy - xy.mean(axis=0)
    _, vectors = np.linalg.eigh(centered.T @ centered)
    return vectors[:, ::-1].T


def _long_axis(xy: np.ndarray) -> np.ndarray:
    return _principal_axes(xy)[0]


def _narrow_axis(xy: np.ndarray) -> np.ndarray:
    """The horizontal direction along which the points are narrowest (searched, not just PCA)."""
    best, best_width = None, math.inf
    for angle in np.linspace(0.0, math.pi, 36, endpoint=False):
        direction = np.array([math.cos(angle), math.sin(angle)])
        width = float(np.ptp(xy @ direction))
        if width < best_width:
            best, best_width = direction, width
    return best
