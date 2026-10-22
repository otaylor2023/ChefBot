"""Arm backends for the real driver: joints in, joints out; kinematics on this side.

Every backend has `kin` (forward and inverse kinematics of the flange in the arm base frame, joint
limits, the gripper's full opening), `state()` -> (q, gripper width m), `command(q, gripper_width)`,
`reset(ready_q)`, `hold()` and `close()`. The flange is the fridge lab's: the gripper's mounting flange,
+x along the fingers, y across the pads (DROID axes).
"""

from __future__ import annotations

from multiprocessing.managers import BaseManager
from pathlib import Path

import mujoco
import numpy as np

MODELS = Path(__file__).resolve().parent / "models"


class Kinematics:
    """FK and damped least-squares IK of a MuJoCo arm's "flange" site, in the arm base frame."""

    def __init__(self, xml: Path, site: str = "flange", open_width: float = 0.088):
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self.site = self.model.site(site).id
        self.q_min, self.q_max = self.model.jnt_range[:, 0].copy(), self.model.jnt_range[:, 1].copy()
        self.open_width = open_width
        self.n = self.model.nq

    def fk(self, q) -> np.ndarray:
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)  # mj_jacSite needs it
        T = np.eye(4)
        T[:3, :3] = self.data.site_xmat[self.site].reshape(3, 3)
        T[:3, 3] = self.data.site_xpos[self.site]
        return T

    def ik(self, T: np.ndarray, seed, iters: int = 40, pos_tol: float = 2e-3, rot_tol: float = 2e-2) -> np.ndarray | None:
        """Joints that put the flange at T (near `seed`), or None if it can't be reached within tolerance."""
        q = np.array(seed, dtype=float)
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for _ in range(iters):
            F = self.fk(q)
            ep = T[:3, 3] - F[:3, 3]
            er = _rotvec(T[:3, :3] @ F[:3, :3].T)
            if np.linalg.norm(ep) < pos_tol and np.linalg.norm(er) < rot_tol:
                return q
            mujoco.mj_jacSite(self.model, self.data, jp, jr, self.site)
            J, e = np.vstack([jp, jr]), np.concatenate([ep, er])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.q_min, self.q_max)
        F = self.fk(q)
        ok = np.linalg.norm(T[:3, 3] - F[:3, 3]) < pos_tol and np.linalg.norm(_rotvec(T[:3, :3] @ F[:3, :3].T)) < rot_tol
        return q if ok else None


def _rotvec(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(R).as_rotvec()


class ArxArm:
    """The ARX5 L5 through real/nuc/arx_arm_server.py on the NUC."""

    def __init__(self, host: str, port: int, authkey: bytes):
        class _M(BaseManager):
            pass

        _M.register("Arm")
        m = _M(address=(host, port), authkey=authkey)
        try:
            m.connect()
        except ConnectionRefusedError as e:
            raise RuntimeError(f"no arm server at {host}:{port}: is real/nuc/arx_arm_server.py running on the NUC?") from e
        self.r = m.Arm()
        lim = self.r.limits()
        self.kin = Kinematics(MODELS / "arx5_L5.xml", open_width=lim["open_width"])
        # the SDK's limits, where tighter than the model's
        self.kin.q_min = np.maximum(self.kin.q_min, lim["q_min"])
        self.kin.q_max = np.minimum(self.kin.q_max, lim["q_max"])

    def state(self) -> tuple[np.ndarray, float]:
        s = self.r.state()
        return np.array(s["q"]), float(s["gripper_width"])

    def command(self, q, gripper_width: float) -> bool:
        return bool(self.r.command([float(v) for v in q], float(gripper_width)))

    def reset(self, ready_q=None) -> None:
        self.r.home()
        if ready_q is not None:
            self.r.move_to([float(v) for v in ready_q], 3.0)

    def hold(self) -> None:
        self.r.hold()

    def close(self) -> None:
        self.r.close()


class FakeArm:
    """No hardware: the ARX L5's kinematics, the joints reaching each command one control step later."""

    def __init__(self):
        self.kin = Kinematics(MODELS / "arx5_L5.xml")
        self.q = np.clip(np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0]), self.kin.q_min, self.kin.q_max)
        self.width, self._next = self.kin.open_width, None

    def state(self) -> tuple[np.ndarray, float]:
        if self._next is not None:
            self.q, self.width = self._next
        return self.q.copy(), self.width

    def command(self, q, gripper_width: float) -> bool:
        self._next = (np.clip(np.asarray(q, dtype=float), self.kin.q_min, self.kin.q_max), float(gripper_width))
        return True

    def reset(self, ready_q=None) -> None:
        if ready_q is not None:
            self.q = np.asarray(ready_q, dtype=float)
        self._next, self.width = None, self.kin.open_width

    def hold(self) -> None:
        self._next = None

    def close(self) -> None:
        pass


def make_arm(cfg: dict):
    kind = "fake" if cfg.get("dry_run") else cfg["arm"]["type"]
    if kind == "arx_l5":
        n = cfg["nuc"]
        return ArxArm(n["host"], int(n["arm_port"]), str(n["authkey"]).encode())
    if kind == "fake":
        return FakeArm()
    if kind == "yam":
        raise NotImplementedError("the YAM backend isn't written yet (i2rt's SDK; same interface as ArxArm)")
    raise ValueError(f"unknown arm type {kind!r}")
