"""Ground-truth geometry of loose objects, shared by the proposers."""

from __future__ import annotations

import mujoco
import numpy as np

from chefbot.envs.robocasa.detector import scene_objects
from chefbot.envs.robocasa.env import RoboCasaEnv
from chefbot.interface.types import Detection

# A detection is matched to the sim object of the same label within this distance.
MATCH_RADIUS = 0.15
# With the gripper pointing down, the wrist reaches this far above the grip
# point and this far to the side of it (the last arm links, not the elbow).
ARM_ABOVE_TCP = 0.37
ARM_RADIUS = 0.065


def find_object(env: RoboCasaEnv, detection: Detection) -> int | None:
    """Body id of the loose object a detection refers to, or None."""
    position = np.asarray(detection.position_world, dtype=float)
    best, best_distance = None, MATCH_RADIUS
    for _, label, body in scene_objects(env):
        distance = float(np.linalg.norm(env.data.xpos[body] - position))
        if label == detection.label and distance <= best_distance:
            best, best_distance = body, distance
    return best


def body_geoms(env: RoboCasaEnv, body: int) -> list[int]:
    """Collision geoms of the object rooted at `body`."""
    model = env.model
    root = model.body_rootid[body]
    return [
        g
        for g in range(model.ngeom)
        if model.body_rootid[model.geom_bodyid[g]] == root and (model.geom_contype[g] or model.geom_conaffinity[g])
    ]


def object_points(env: RoboCasaEnv, body: int) -> np.ndarray:
    """World-frame points (N, 3) on the surface of an object's collision geometry."""
    model, data = env.model, env.data
    clouds = []
    for geom in body_geoms(env, body):
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[geom]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            vertices = model.mesh_vert[start : start + count]
            faces = model.mesh_face[model.mesh_faceadr[mesh] : model.mesh_faceadr[mesh] + model.mesh_facenum[mesh]]
            local = np.concatenate([vertices, vertices[faces].mean(axis=1)])
        else:
            # Primitives: a lattice over the bounding box is close enough.
            center, half = model.geom_aabb[geom, :3], model.geom_aabb[geom, 3:]
            axes = [center[i] + np.linspace(-half[i], half[i], 7) for i in range(3)]
            local = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        clouds.append(local @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom])
    return np.concatenate(clouds) if clouds else np.zeros((0, 3))


def ceiling_above(env: RoboCasaEnv, xy: np.ndarray, z: float, ignore_roots: set[int] = frozenset()) -> float:
    """Height of the lowest thing above `z` over the point `xy` or within an
    arm's width of it (inf if open). Uses bounding boxes, so thin things like
    cabinet handles count. The robot and `ignore_roots` (root body ids) do not."""
    import math

    model, data = env.model, env.data
    other = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    other &= ~np.isin(model.body_rootid[model.geom_bodyid], list(ignore_roots))
    other[list(env.robot_geoms)] = False
    frames = data.geom_xmat.reshape(-1, 3, 3)
    center = data.geom_xpos + np.einsum("gij,gj->gi", frames, model.geom_aabb[:, :3])
    half = np.einsum("gij,gj->gi", np.abs(frames), model.geom_aabb[:, 3:])
    near = np.all(np.abs(center[:, :2] - np.asarray(xy, dtype=float)) <= half[:, :2] + ARM_RADIUS, axis=1)
    bottoms = center[:, 2] - half[:, 2]
    over = bottoms[other & near & (bottoms >= z)]
    return float(over.min()) if len(over) else math.inf
