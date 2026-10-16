"""GroundTruthGraspProposer: grasps computed from the simulator's geometry.

Stands in for a learned grasp model (MANIP_SPEC §6.1). Grasps come from the
object's collision geometry as a point cloud, and nothing else:

  top-down   over the middle, fingers across the narrowest side
  front      from the robot's side at mid-height, gripper angled down, backing
             straight out afterwards (for under a shelf or a wall cabinet)
  side       for tall objects: the same, lifting straight up afterwards
  handle     a thin part sticking out one end (a pan's handle): found by
             handles.detect_handle, held where chefbot.grasping.handle says
  rim        the wall of an open container (a bowl), fingers astride it

Handle and rim grasps go beyond the spec; they cover objects too wide for the
gripper to span. What is around the object (a shelf above it, a wall cabinet)
is not looked at here: the robot layer checks each grasp against its depth
camera's view and takes the best one the arm can get to.
"""

from __future__ import annotations

import math

import numpy as np

from chefbot.envs.robocasa.env import RoboCasaEnv
from chefbot.envs.robocasa.handles import detect_handle
from chefbot.envs.robocasa.objects import find_object, object_points
from chefbot.interface.proposers import GraspProposer
from chefbot.interface.types import Detection, Grasp
from chefbot.grasping.handle import GRIP_FROM_BODY, handle_grasps, tcp_pose

STROKE = 0.085  # 2F-85: widest gap between the pads
_FIT_MARGIN = 0.004  # an object must be this much narrower than the stroke
_FINGERTIP = 0.019  # the pads reach this far beyond the grip point
_TABLE_CLEARANCE = 0.004  # fingertips stay this far above what the object stands on
_BAND = 0.015  # half-height of the slice of the object the fingers close on
_MIN_WALL = 0.002
_HANDLE_GRIPS = (GRIP_FROM_BODY, 0.09, 0.13)  # m from where the handle leaves the body
_RIM_DEPTH = 0.015  # grip point this far below a container's rim
_RIM_ANGLES_DEG = (0, 25, -25, 50, -50, 80, -80)  # around the rim, starting on the robot's side
_DOWN = np.array([0.0, 0.0, -1.0])
_BOX_PAD = 0.01  # the object's box (Grasp.object_box) is this much bigger than its points
# Angled down first: an object held with a level gripper cannot be set down at
# counter height afterwards (the arm's wrist limit), one held at an angle can.
_FRONT_PITCH_DEG = (35, 20, 0)
_FRONT_RETREAT = 0.16  # back out this far after a front grasp
_FRONT_LIFT = 0.02  # and lift this much while doing it


def extent(points: np.ndarray, direction: np.ndarray) -> tuple[float, float]:
    along = points @ direction
    return float(along.min()), float(along.max())


class GroundTruthGraspProposer(GraspProposer):
    def __init__(self, env: RoboCasaEnv):
        self._env = env

    def propose(self, target: Detection) -> list[Grasp]:
        """Every kind of grasp the object's shape allows, best first. Whether the
        arm can get to one around what is near the object (a shelf above, a wall
        cabinet) is the robot layer's call, from its depth camera."""
        env, cfg = self._env, self._env.config
        with env.lock:
            body = find_object(env, target)
            if body is None:
                return []
            points = object_points(env, body)
            base = env.get_state().base_pose
        if len(points) == 0:
            return []
        robot_xy = np.array([base.x, base.y])
        grasps = self._top_down(points) + self._front(points, robot_xy) + self._side(points, robot_xy)
        if not grasps:
            # Too wide to span: look for something thinner to hold on to.
            grasps = self._handle(points) or self._rim(points, robot_xy)
        box = np.array([points.min(axis=0), points.max(axis=0)])
        box += [[-_BOX_PAD] * 3, [_BOX_PAD] * 3]
        for grasp in grasps:
            grasp.object_box = box
            if grasp.pregrasp_offset <= 0.0:
                grasp.pregrasp_offset = cfg.pregrasp_offset
        return sorted(grasps, key=lambda g: -g.score)

    # -- grasp kinds ----------------------------------------------------------

    def _front(self, points: np.ndarray, robot_xy: np.ndarray) -> list[Grasp]:
        """Reach in from the robot's side, then lift it slightly and back straight
        out: for an object with something over it (a shelf, a wall cabinet)."""
        toward = (points[:, :2].min(axis=0) + points[:, :2].max(axis=0)) / 2 - robot_xy
        toward /= np.linalg.norm(toward)
        return self._angled(points, robot_xy, retreat=np.append(-_FRONT_RETREAT * toward, _FRONT_LIFT), score=0.9)

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
        """Find the handle (perception), then decide where to hold it (grasp planning)."""
        handle = detect_handle(points)
        if handle is None:
            return []
        # Near the body first (least leverage), then further out: food in a pan can lean
        # over the rim onto the base of the handle.
        return [
            grasp
            for k, out in enumerate(_HANDLE_GRIPS)
            for grasp in handle_grasps(points, handle, _FINGERTIP + _TABLE_CLEARANCE, out, 0.8 - 0.1 * k)
        ]

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


def _narrow_axis(xy: np.ndarray) -> np.ndarray:
    """The horizontal direction along which the points are narrowest (searched, not just PCA)."""
    best, best_width = None, math.inf
    for angle in np.linspace(0.0, math.pi, 36, endpoint=False):
        direction = np.array([math.cos(angle), math.sin(angle)])
        width = float(np.ptp(xy @ direction))
        if width < best_width:
            best, best_width = direction, width
    return best
