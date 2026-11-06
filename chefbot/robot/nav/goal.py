"""Goal handling: snapping goals out of collision and choosing standoff poses."""

from __future__ import annotations

import math
from typing import Callable, Protocol

import numpy as np

from chefbot.interface.types import Pose2D
from chefbot.robot.nav.astar import AStarPlanner
from chefbot.robot.nav.grid import OccupancyGrid


class Reachability(Protocol):
    def is_reachable(self, base_pose: Pose2D, target_world: np.ndarray) -> bool: ...


def snap_to_free(grid: OccupancyGrid, pose: Pose2D, snap_radius: float) -> Pose2D | None:
    """`pose` if it is collision-free, else the nearest free cell center within
    `snap_radius` (yaw kept), else None."""
    if grid.is_free(pose.x, pose.y):
        return pose
    cell = grid.nearest_free_cell(pose.x, pose.y, snap_radius)
    if cell is None:
        return None
    x, y = grid.cell_to_world(cell)
    return Pose2D(x, y, pose.yaw)


def standoff_candidates(
    target_xy: tuple[float, float], r_min: float, r_max: float, rings: int, angles: int
) -> list[Pose2D]:
    """Base poses on rings around the target, each facing it."""
    radii = np.linspace(r_min, r_max, rings) if rings > 1 else [r_min]
    poses = []
    for radius in radii:
        for k in range(angles):
            angle = 2.0 * math.pi * k / angles
            x = target_xy[0] + radius * math.cos(angle)
            y = target_xy[1] + radius * math.sin(angle)
            poses.append(Pose2D(x, y, math.atan2(target_xy[1] - y, target_xy[0] - x)))
    return poses


def select_standoff(
    planner: AStarPlanner,
    start: Pose2D,
    target_world: np.ndarray,
    candidates: list[Pose2D],
    reachability: Reachability,
    should_abort: Callable[[], bool] = lambda: False,
    closeness: float = 0.0,
) -> Pose2D | None:
    """The collision-free, arm-reachable candidate with the lowest path cost
    from `start` plus `closeness` x its distance to the target (m of path per m
    of distance: closer stances leave the arm more room), or None."""
    grid = planner.grid
    free = [c for c in candidates if grid.is_free(c.x, c.y)]
    costs = planner.cost_to((start.x, start.y), {grid.world_to_cell(c.x, c.y) for c in free})
    target = np.asarray(target_world, dtype=float)
    ranked = sorted(
        (costs[cell] + closeness * math.hypot(c.x - target[0], c.y - target[1]), i)
        for i, c in enumerate(free)
        if (cell := grid.world_to_cell(c.x, c.y)) in costs
    )
    # Path cost is cheap and IK is not, so check reachability in cost order.
    for _, i in ranked:
        if should_abort():
            return None
        if reachability.is_reachable(free[i], target_world):
            return free[i]
    return None


# The pick task's start rule (chefbot/tasks/pick_up.py), the stance the pick skills were trained on: the base
# square to the counter's edge, its centre 0.50-0.62 m out from it, the target in front of it 0.10-0.16 m in from the
# edge and at least 0.35 m from either end. Distances here stay inside that range with room for arrival error.
PARK_OUT_M = (0.56, 0.53, 0.59)  # preferred first: the middle of 0.50-0.62
PARK_END_MARGIN_M = 0.35
TRAINED_INSET_M = (0.10, 0.16)


def counter_faces(center, size, yaw: float) -> list[dict]:
    """The four sides of a box counter (centre, (length, depth), yaw rad): outward normal n, along-edge unit t,
    half-length along the edge, half-depth to the edge, and a name by the normal's world direction."""
    c, s = math.cos(yaw), math.sin(yaw)
    ex, ey = np.array([c, s]), np.array([-s, c])
    L, W = float(size[0]), float(size[1])
    faces = []
    for n, t, half_along, half_n in ((ex, ey, W / 2, L / 2), (-ex, -ey, W / 2, L / 2), (ey, -ex, L / 2, W / 2),
                                     (-ey, ex, L / 2, W / 2)):  # fmt: skip
        name = ("east" if n[0] > 0 else "west") if abs(n[0]) >= abs(n[1]) else ("north" if n[1] > 0 else "south")
        faces.append({"name": name, "n": n, "t": t, "half_along": half_along, "half_n": half_n})
    return faces


def counter_park_candidates(center, size, yaw: float, target_xy, out_m=PARK_OUT_M, end_margin: float = PARK_END_MARGIN_M,
                            max_inset: float = 0.25) -> list[tuple[Pose2D, dict]]:  # fmt: skip
    """Base poses for working on `target_xy` (world) on a box counter, by the start rule: for each side the target is
    at most `max_inset` in from and at least `end_margin` from that side's ends, the base centre right in front of the
    target, `out_m` from the edge, facing the edge squarely. Each with what it is: side, the target's inset and along
    position, the distance out."""
    centre = np.asarray(center, dtype=float)
    r = np.asarray(target_xy, dtype=float) - centre
    out = []
    for f in counter_faces(center, size, yaw):
        inset = f["half_n"] - float(r @ f["n"])
        along = float(r @ f["t"])
        if not (0.0 <= inset <= max_inset) or abs(along) > f["half_along"] - end_margin:
            continue
        for d in out_m:
            p = centre + f["t"] * along + f["n"] * (f["half_n"] + d)
            yaw_facing = math.atan2(-f["n"][1], -f["n"][0])
            out.append((Pose2D(float(p[0]), float(p[1]), yaw_facing),
                        {"side": f["name"], "inset_m": inset, "along_m": along, "out_m": float(d),
                         "trained": TRAINED_INSET_M[0] <= inset <= TRAINED_INSET_M[1]}))  # fmt: skip
    return out


def park_report(center, size, yaw: float, side: str, base: Pose2D, target_xy) -> dict:
    """Where a base ended against the start rule, for the side it parked at: its centre's distance out from the edge,
    its heading off square, and the target's offset along the edge from straight ahead and inset."""
    f = next(f for f in counter_faces(center, size, yaw) if f["name"] == side)
    centre = np.asarray(center, dtype=float)
    rb, rt = np.array([base.x, base.y]) - centre, np.asarray(target_xy, dtype=float) - centre
    square = math.atan2(-f["n"][1], -f["n"][0])
    return {"out_m": float(rb @ f["n"]) - f["half_n"], "yaw_off_square_deg": math.degrees(wrap_yaw(base.yaw - square)),
            "target_lateral_m": float((rt - rb) @ f["t"]), "target_inset_m": f["half_n"] - float(rt @ f["n"])}  # fmt: skip


def wrap_yaw(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def counter_stance(robot, box, target, out_m=None, approach: float = 0.0, end_margin=None):
    """The stance for a counter target by the start rule (counter_park_candidates: the box's sides, out_m from its
    edge, default 0.45 then 0.48) that `robot` (robot/robot.py Robot) finds clear for its whole footprint at the
    stance's heading, with its straight `approach` clear, and cheapest to reach from where the base is
    (Robot.choose_stance): (pose, info) or (None, reason)."""
    kw = {} if end_margin is None else {"end_margin": end_margin}
    cands = counter_park_candidates(*box[:3], target, out_m=tuple(out_m or (0.45, 0.48)), **kw)
    if not cands:
        return None, "the target is not 0-0.25 m in from a side and >= 0.35 m from its ends: no stance by the rule"
    body = robot._space
    ok = [(p, i) for p, i in cands
          if body.at(p.yaw).segment_free((p.x - approach * math.cos(p.yaw), p.y - approach * math.sin(p.yaw)), (p.x, p.y))]  # fmt: skip
    pick, why = robot.choose_stance([p for p, _ in ok])
    if pick is None:
        return None, why
    return pick, next(i for p, i in ok if p is pick)
