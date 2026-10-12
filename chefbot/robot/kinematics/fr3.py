"""FR3 arm model (MuJoCo Menagerie kinematics) and position-only IK via mink."""

from __future__ import annotations

from pathlib import Path

import mink
import mujoco
import numpy as np
import qpsolvers

FR3_MJCF = Path(__file__).resolve().parent / "assets" / "fr3_kinematics.xml"
TCP_SITE = "tcp"
HOME_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
# Shoulder (joint 2) position in the arm base frame, and a generous bound on
# shoulder -> flange distance, used to reject far targets without running IK.
_SHOULDER = np.array([0.0, 0.0, 0.333])
_MAX_SHOULDER_TO_FLANGE = 0.95
# IK restarts: home, then the elbow tucked and extended.
_SEEDS = (
    HOME_Q,
    np.array([0.0, 0.6, 0.0, -1.2, 0.0, 1.8, -0.7853]),
    np.array([0.0, -0.6, 0.0, -2.4, 0.0, 1.8, -0.7853]),
)


def load_fr3(tcp_offset: float) -> mujoco.MjModel:
    """The FR3 with a `tcp` site `tcp_offset` beyond the flange along its z-axis."""
    spec = mujoco.MjSpec.from_file(str(FR3_MJCF))
    flange = next(s for s in spec.sites if s.name == "attachment_site")
    link7 = next(b for b in spec.bodies if b.name == "fr3_link7")
    link7.add_site(name=TCP_SITE, pos=np.asarray(flange.pos) + [0.0, 0.0, tcp_offset])
    return spec.compile()


def _pick_solver() -> str:
    for name in ("daqp", "quadprog", "osqp", "proxqp"):
        if name in qpsolvers.available_solvers:
            return name
    raise RuntimeError("no QP solver available for IK; install qpsolvers[daqp]")


class FR3PositionIK:
    """Position-only IK for the `tcp` site. Owns its MjData; never touches sim state."""

    def __init__(
        self,
        tcp_offset: float,
        pos_tolerance: float = 0.01,
        max_iters: int = 150,
        dt: float = 0.1,
        damping: float = 1e-3,
    ):
        self.model = load_fr3(tcp_offset)
        self._tcp_offset = tcp_offset
        self._pos_tolerance = pos_tolerance
        self._max_iters = max_iters
        self._dt = dt
        self._damping = damping
        self._solver = _pick_solver()
        self._configuration = mink.Configuration(self.model)
        self._task = mink.FrameTask(
            frame_name=TCP_SITE, frame_type="site", position_cost=1.0, orientation_cost=0.0
        )
        self._limits = [mink.ConfigurationLimit(self.model)]
        self._site_id = self.model.site(TCP_SITE).id

    def tcp_position(self, q: np.ndarray) -> np.ndarray:
        self._configuration.update(q)
        return self._configuration.data.site_xpos[self._site_id].copy()

    def solve(self, target: np.ndarray) -> np.ndarray | None:
        """Joint angles placing the tcp at `target` (arm base frame), or None."""
        target = np.asarray(target, dtype=float)
        if np.linalg.norm(target - _SHOULDER) > _MAX_SHOULDER_TO_FLANGE + self._tcp_offset:
            return None
        self._task.set_target(mink.SE3.from_translation(target))
        for seed in _SEEDS:
            q = seed.copy()
            # Turn joint 1 toward the target so each restart starts in the right half-space.
            q[0] = np.clip(np.arctan2(target[1], target[0]), *self.model.jnt_range[0])
            self._configuration.update(q)
            for _ in range(self._max_iters):
                error = self._configuration.data.site_xpos[self._site_id] - target
                if np.linalg.norm(error) <= self._pos_tolerance:
                    return self._configuration.q.copy()
                velocity = mink.solve_ik(
                    self._configuration,
                    [self._task],
                    self._dt,
                    solver=self._solver,
                    damping=self._damping,
                    limits=self._limits,
                )
                self._configuration.integrate_inplace(velocity, self._dt)
        return None
