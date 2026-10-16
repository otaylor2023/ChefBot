"""The arm's collision model, and the world it moves in as seen by a depth camera.

The arm is a chain of capsules on the FR3 kinematics model (the IK model, so
"collision-free" and "reachable" agree), the gripper three boxes. The world is an octree of the points a
depth image sees, minus the robot itself and minus whatever the skill is about
to touch on purpose (the object being picked, the one being put down). Space
the camera cannot see counts as free.

`Workspace` is what the skills use: is this configuration free, is this
straight-line motion free, and a collision-free joint path between two
configurations (straight if possible, planned otherwise).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import mujoco
import numpy as np

from chefbot.interface.types import RGBD, Pose2D

try:  # optional: without them there is no world model and arm moves go unchecked
    import coal
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:  # pragma: no cover - depends on the install
    coal = ob = og = None

# Capsules between frames of the FR3 model: (from body, to body or site, radius).
# The wrist ones are fat: joint 6's housing sticks out about 9 cm from the
# line between the joints.
_LINKS = (
    ("fr3_link2", "fr3_link3", 0.075),
    ("fr3_link3", "fr3_link4", 0.07),
    ("fr3_link4", "fr3_link5", 0.07),
    ("fr3_link5", "fr3_link7", 0.09),
    ("fr3_link7", "attachment_site", 0.08),
)
# The 2F-85, open, as boxes in the TCP frame (x: the fingers' closing direction,
# y: across the fingers, z: the approach direction; TCP at the origin), measured
# on the sim gripper: (z from, z to, half-width in x, half-width in y).
_GRIPPER_BODY = (-0.156, -0.10, 0.065, 0.04)
_KNUCKLES = (-0.10, -0.045, 0.075, 0.02)
# The pads are left out of the straight-line checks of the final approach and
# lift, where touching the object and what it stands on is expected (and caught
# by contact sensing).
_PADS = (-0.045, 0.02, 0.06, 0.015)
_FLOOR = 0.02  # depth points lower than this (world z) are the floor
_STEP = 0.02  # rad: joint-space spacing of collision checks along a motion


@dataclass
class Capsule:
    p0: np.ndarray
    p1: np.ndarray
    radius: float

    def distance(self, points: np.ndarray) -> np.ndarray:
        """Distance (N,) from each point to the capsule's surface (negative inside)."""
        return distance_to_segment(points, self.p0, self.p1) - self.radius

    def collision_object(self):
        axis = self.p1 - self.p0
        length = float(np.linalg.norm(axis))
        z = axis / length if length > 1e-9 else np.array([0.0, 0.0, 1.0])
        x = np.cross(z, [1.0, 0.0, 0.0])
        if np.linalg.norm(x) < 1e-6:
            x = np.cross(z, [0.0, 1.0, 0.0])
        x /= np.linalg.norm(x)
        rotation = np.column_stack([x, np.cross(z, x), z])
        shape = coal.Capsule(self.radius, max(length, 1e-4))
        return coal.CollisionObject(shape, coal.Transform3s(rotation, (self.p0 + self.p1) / 2))


@dataclass
class Box:
    center: np.ndarray
    rotation: np.ndarray  # (3, 3): the box's axes as columns
    half: np.ndarray  # (3,)

    def distance(self, points: np.ndarray) -> np.ndarray:
        """Distance (N,) from each point to the box (0 inside)."""
        local = np.abs((points - self.center) @ self.rotation)
        return np.linalg.norm(np.maximum(local - self.half, 0.0), axis=1)

    def collision_object(self):
        shape = coal.Box(*(2.0 * self.half))
        return coal.CollisionObject(shape, coal.Transform3s(self.rotation, self.center))


class ArmCollisionModel:
    """The arm as capsules and the gripper as boxes, for a joint configuration,
    in the arm base frame. Owns its MjData."""

    def __init__(self, model: mujoco.MjModel, tcp_site: str = "tcp"):
        self._model, self._data = model, mujoco.MjData(model)
        self._links = [(self._frame(a), self._frame(b), r) for a, b, r in _LINKS]
        self._tcp = model.site(tcp_site).id

    def _frame(self, name: str) -> tuple[str, int]:
        try:
            return "site", self._model.site(name).id
        except KeyError:
            return "body", self._model.body(name).id

    def _position(self, frame: tuple[str, int]) -> np.ndarray:
        kind, i = frame
        return (self._data.site_xpos if kind == "site" else self._data.xpos)[i].copy()

    def shapes(self, q: np.ndarray, pads: bool = True) -> list[Capsule | Box]:
        self._data.qpos[: len(q)] = q
        mujoco.mj_kinematics(self._model, self._data)
        out: list[Capsule | Box] = [Capsule(self._position(a), self._position(b), r) for a, b, r in self._links]
        tcp, rotation = self._data.site_xpos[self._tcp].copy(), self._data.site_xmat[self._tcp].reshape(3, 3).copy()
        for z0, z1, hx, hy in (_GRIPPER_BODY, _KNUCKLES) + ((_PADS,) if pads else ()):
            center = tcp + rotation[:, 2] * (z0 + z1) / 2
            out.append(Box(center, rotation, np.array([hx, hy, (z1 - z0) / 2])))
        return out


def distance_to_segment(points: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    d = p1 - p0
    t = np.clip(((points - p0) @ d) / max(float(d @ d), 1e-12), 0.0, 1.0)
    return np.linalg.norm(points - (p0 + t[:, None] * d), axis=1)


def depth_points(rgbd: RGBD, max_depth: float = 3.0) -> np.ndarray:
    """World-frame points (N, 3) of a depth image."""
    h, w = rgbd.depth.shape
    v, u = np.mgrid[0:h, 0:w]
    z = rgbd.depth.reshape(-1).astype(float)
    ok = (z > 0.05) & (z < max_depth)
    k = rgbd.intrinsics
    x = (u.reshape(-1) - k[0, 2]) * z / k[0, 0]
    y = (v.reshape(-1) - k[1, 2]) * z / k[1, 1]
    camera = np.stack([x, y, z, np.ones_like(z)], axis=1)[ok]
    return (camera @ np.asarray(rgbd.cam_to_world, dtype=float).T)[:, :3]


class Workspace:
    """Collision queries for one skill, against one snapshot of the world (arm base frame)."""

    def __init__(
        self,
        collision: ArmCollisionModel | None = None,
        points: np.ndarray | None = None,
        voxel: float = 0.02,
        joint_limits: np.ndarray | None = None,
        timeout_s: float = 5.0,
    ):
        self._collision = collision
        self.points = np.zeros((0, 3)) if points is None else np.asarray(points, dtype=float)
        self._limits = joint_limits
        self._timeout = timeout_s
        self._world = None
        if collision is not None and coal is not None and len(self.points):
            octree = coal.makeOctree(np.ascontiguousarray(self.points), voxel)
            self._world = coal.CollisionObject(octree, coal.Transform3s())
        self._request = coal.CollisionRequest() if coal is not None else None

    @classmethod
    def unchecked(cls) -> "Workspace":
        """No world model: everything is free and every path is straight."""
        return cls()

    @property
    def checked(self) -> bool:
        return self._world is not None

    # -- queries ----------------------------------------------------------

    def free(self, q: np.ndarray, pads: bool = True) -> bool:
        if self._world is None:
            return True
        for shape in self._collision.shapes(np.asarray(q, dtype=float), pads):
            if coal.collide(shape.collision_object(), self._world, self._request, coal.CollisionResult()):
                return False
        return True

    def line_free(self, waypoints: list[np.ndarray]) -> bool:
        """A straight-line approach or retreat (from `plan_line`): finger pads excluded."""
        return all(self.free(q, pads=False) for q in waypoints)

    def segment_free(self, a: np.ndarray, b: np.ndarray) -> bool:
        """The joint-space straight move from `a` to `b`, gripper included."""
        if self._world is None:
            return True
        steps = max(1, int(np.ceil(np.abs(b - a).max() / _STEP)))
        return all(self.free(a + (b - a) * t) for t in np.linspace(0.0, 1.0, steps + 1))

    def path(self, start: np.ndarray, goal: np.ndarray) -> list[np.ndarray] | None:
        """Joint waypoints from `start` to `goal`: straight when that is free,
        else planned. None if no path was found in time."""
        start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
        if self.segment_free(start, goal):
            return [start, goal]
        if not self.free(goal):
            return None
        if not self.free(start):
            # Starting in contact with what the camera saw (the arm resting
            # against something): nothing to plan from, so go straight as before.
            return [start, goal]
        return self._plan(start, goal)

    def _plan(self, start: np.ndarray, goal: np.ndarray) -> list[np.ndarray] | None:
        if og is None or self._limits is None:
            return None
        n = len(start)
        space = ob.RealVectorStateSpace(n)
        bounds = ob.RealVectorBounds(n)
        for i, (low, high) in enumerate(self._limits):
            bounds.setLow(i, float(low))
            bounds.setHigh(i, float(high))
        space.setBounds(bounds)
        setup = og.SimpleSetup(space)
        setup.setStateValidityChecker(lambda state: self.free(np.array([state[i] for i in range(n)])))
        info = setup.getSpaceInformation()
        # As a fraction of the space's extent: about _STEP rad between checks.
        info.setStateValidityCheckingResolution(_STEP / max(space.getMaximumExtent(), 1e-9))
        a, b = space.allocState(), space.allocState()
        for i in range(n):
            a[i], b[i] = float(start[i]), float(goal[i])
        setup.setStartAndGoalStates(a, b)
        setup.setPlanner(og.RRTConnect(info))
        deadline = time.perf_counter() + self._timeout
        if not setup.solve(self._timeout) or not setup.haveExactSolutionPath():
            return None
        setup.simplifySolution(max(0.1, deadline - time.perf_counter()))
        path = setup.getSolutionPath()
        waypoints = [np.array([path.getState(k)[i] for i in range(n)]) for k in range(path.getStateCount())]
        waypoints[0], waypoints[-1] = start, goal
        return waypoints


def snapshot(
    rgbd: RGBD,
    base_pose: Pose2D,
    arm_base_in_world: np.ndarray,
    q: np.ndarray,
    collision: ArmCollisionModel,
    self_margin: float,
    base_box: tuple[float, float, float],
    ignore_boxes: list[np.ndarray] = (),
) -> np.ndarray:
    """The obstacle points (arm base frame) of a depth image taken with the arm at `q`."""
    return filter_points(
        depth_points(rgbd), base_pose, arm_base_in_world, q, collision, self_margin, base_box, ignore_boxes
    )


def filter_points(
    world: np.ndarray,
    base_pose: Pose2D,
    arm_base_in_world: np.ndarray,
    q: np.ndarray,
    collision: ArmCollisionModel,
    self_margin: float,
    base_box: tuple[float, float, float],
    ignore_boxes: list[np.ndarray] = (),
) -> np.ndarray:
    """World-frame points (N, 3) seen with the arm at `q`, as obstacles in the arm base frame.

    Dropped: the arm itself (within `self_margin` of its shapes), the mobile base
    (`base_box`: length, width, height, centered on the base), the floor, and anything
    inside `ignore_boxes` (world-frame (2, 3) min / max corners: what the skill is
    about to touch on purpose)."""
    keep = world[:, 2] > _FLOOR
    c, s = np.cos(base_pose.yaw), np.sin(base_pose.yaw)
    local = (world[:, :2] - [base_pose.x, base_pose.y]) @ np.array([[c, -s], [s, c]])
    keep &= ~(
        (np.abs(local[:, 0]) <= base_box[0] / 2) & (np.abs(local[:, 1]) <= base_box[1] / 2) & (world[:, 2] <= base_box[2])
    )
    for box in ignore_boxes:
        box = np.asarray(box, dtype=float)
        keep &= ~np.all((world >= box[0]) & (world <= box[1]), axis=1)
    to_arm = np.linalg.inv(arm_base_in_world)
    points = world[keep] @ to_arm[:3, :3].T + to_arm[:3, 3]
    for shape in collision.shapes(np.asarray(q, dtype=float)):
        points = points[shape.distance(points) > self_margin]
    return points
