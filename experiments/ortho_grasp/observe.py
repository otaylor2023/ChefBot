"""Observation pipeline (spec §5): the cup's point clouds in the robot frame.

partial: ground-truth segmentation mask (stand-in for SAM) on the shoulder
         camera's depth, back-projected.
full:    points sampled on the cup's collision mesh surfaces (ground truth).
hulls:   the collision mesh's convex pieces as half-spaces, for "is this point
         inside cup material" (the evaluators use them; the cup is hollow).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import ConvexHull

from experiments.ortho_grasp.frames import to_robot
from experiments.ortho_grasp.sim import World

FULL_POINTS = 20000


@dataclass
class CupGeometry:
    full: np.ndarray  # (N, 3) robot frame, surface samples
    partial: np.ndarray  # (M, 3) robot frame, masked depth
    hulls: list[np.ndarray]  # each (F, 4): a x + b y + c z + d <= 0 inside (robot frame)
    counter_z: float
    rim_z: float
    bottom_z: float

    @property
    def box(self) -> np.ndarray:
        return np.array([self.full.min(axis=0), self.full.max(axis=0)])

    def inside(self, points: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """(N,) bool: inside any convex piece (shrunk by `margin`)."""
        points = np.atleast_2d(points)
        out = np.zeros(len(points), dtype=bool)
        for eq in self.hulls:
            out |= np.all(points @ eq[:, :3].T + eq[:, 3] <= -margin, axis=1)
        return out


def observe(world: World, scene, rng: np.random.Generator) -> CupGeometry:
    root = world.cup_root(scene)
    full, hulls = _mesh_samples(world, root, rng)
    partial = _masked_depth(world, root, world.env.config.shoulder_camera)
    return CupGeometry(full, partial, hulls, world.counter_z, float(full[:, 2].max()), float(full[:, 2].min()))


def _mesh_samples(world: World, root: int, rng: np.random.Generator) -> tuple[np.ndarray, list[np.ndarray]]:
    m, d = world.env.model, world.env.data
    tris, hulls = [], []
    for g in np.flatnonzero(m.body_rootid[m.geom_bodyid] == root):
        if not (m.geom_contype[g] or m.geom_conaffinity[g]) or m.geom_dataid[g] < 0:
            continue
        mesh = m.geom_dataid[g]
        va, vn = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
        fa, fn = m.mesh_faceadr[mesh], m.mesh_facenum[mesh]
        verts = m.mesh_vert[va : va + vn] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        verts = to_robot(verts, world.base)
        tris.append(verts[m.mesh_face[fa : fa + fn]])
        hulls.append(ConvexHull(verts).equations)
    tris = np.concatenate(tris)  # (T, 3, 3)
    area = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    pick = rng.choice(len(tris), size=FULL_POINTS, p=area / area.sum())
    u, v = rng.random((2, FULL_POINTS))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = tris[pick]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0]), hulls


def _masked_depth(world: World, root: int, camera: str) -> np.ndarray:
    rgbd = world.env.get_camera_rgbd(camera)
    h, w = rgbd.depth.shape
    seg = world.render(camera, w, h, segmentation=True)
    mask = binary_erosion(world.segment_roots(seg) == root)  # edge pixels mix in the depth behind the cup
    v, u = np.nonzero(mask & np.isfinite(rgbd.depth) & (rgbd.depth > 0))
    z = rgbd.depth[v, u]
    K = rgbd.intrinsics
    cam = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=1)
    T = rgbd.cam_to_world
    return to_robot(cam @ T[:3, :3].T + T[:3, 3], world.base)
