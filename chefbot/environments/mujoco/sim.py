"""A plain-MuJoCo scene running: physics, plus the scene's dynamics at its control rate (reading the declared sensors,
setting the declared actuators and displays), with a timeline of what happened.

    sim = SceneSim(folder, params={"dwell_s": 3.0}, seed=0, extra=probe_xml, sandbox=prefix, python=...)
    sim.run(2.0, before_physics=probe.step)     # 2 s of world time
    sim.timeline                                # [{"t", "type", ...}, ...]: the dynamics' events
    sim.close()

The dynamics never see the simulator: only `sensed` (scene.py's sensors) each tick. Everything the caller adds
(`extra`: test probes, a robot) is part of the world they sense, as anything else would be.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

from chefbot.environments.mujoco import scene as S
from chefbot.environments.mujoco.dynamics import DynamicsProcess

G = 9.81


class SceneSim:
    def __init__(self, folder: Path, params: dict | None = None, seed: int = 0, extra: str = "", extra_assets: str = "",
                 extra_actuators: str = "", sandbox: list[str] | tuple = (), python: str | None = None, dynamics: bool = True):
        self.folder = Path(folder)
        self.ann = S.load_annotations(self.folder)
        self.params = {**self.ann["params"], **(params or {})}
        self.model = S.load_model(self.folder, extra, extra_assets, extra_actuators)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        m = self.model
        self.dt_control = 1.0 / float(self.ann["control_hz"])
        self.t = 0.0
        self.timeline: list[dict] = []
        self.display_state: dict[str, dict[str, str]] = {}
        self._sensors = {name: self._resolve_sensor(name, spec) for name, spec in self.ann["sensors"].items()}
        try:
            self._actuators = {a: m.actuator(a).id for a in self.ann["actuators"]}
        except KeyError as e:
            raise S.SceneError(f"annotations name an actuator the model lacks: {e}") from None
        self._displays = self._resolve_displays()
        self._next_control = 0.0
        self.dyn = DynamicsProcess(self.folder, self.ann, self.params, seed, python=python, prefix=sandbox) if dynamics and self.ann.get("dynamics") else None
        for d, fields in self._displays.items():
            for f, spec in fields.items():
                if spec["default"] is not None:
                    self._show(d, f, spec["default"])

    # -- sensors, actuators, displays -----------------------------------------------------

    def _resolve_sensor(self, name: str, spec: dict):
        m = self.model
        try:
            if "joint" in spec:
                j = m.joint(spec["joint"]).id
                return ("joint", int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j]))
            if "actuator_force" in spec:
                return ("actuator_force", m.actuator(spec["actuator_force"]).id)
            if "beam" in spec:
                a, b = spec["beam"]
                return ("beam", m.site(a).id, m.site(b).id)
            if "load" in spec:
                return ("load", m.body(spec["load"]).id)
        except KeyError as e:
            raise S.SceneError(f"sensor {name!r} names something the model lacks: {e}") from None
        raise S.SceneError(f"sensor {name!r}: one of joint, actuator_force, beam, load")

    def _resolve_displays(self) -> dict:
        m = self.model
        out = {}
        for d, spec in self.ann["displays"].items():
            out[d] = {}
            for f, fs in (spec.get("fields") or {}).items():
                states = {}
                for state, items in (fs.get("states") or {}).items():
                    resolved = []
                    for geom, look in items:
                        try:
                            g = m.geom(geom).id
                        except KeyError:
                            raise S.SceneError(f"display {d}.{f} state {state!r}: no geom {geom!r}") from None
                        mat = m.material(look["material"]).id if "material" in look else None
                        resolved.append((g, np.array(look["rgba"], float) if "rgba" in look else None, mat))
                    states[str(state)] = resolved
                geoms = {g for items in states.values() for g, _, _ in items}
                out[d][f] = {"states": states, "default": str(fs["default"]) if fs.get("default") is not None else None,
                             "own": {g: (m.geom_rgba[g].copy(), int(m.geom_matid[g])) for g in geoms}}  # fmt: skip
        return out

    def _show(self, display: str, field: str, state: str) -> None:
        spec = self._displays.get(display, {}).get(field)
        if spec is None or state not in spec["states"]:
            raise S.SceneError(f"no display state {display}.{field} = {state!r}")
        m = self.model
        for g, (rgba, mat) in spec["own"].items():
            m.geom_rgba[g], m.geom_matid[g] = rgba, mat
        for g, rgba, mat in spec["states"][state]:
            if rgba is not None:
                m.geom_rgba[g] = rgba
            if mat is not None:
                m.geom_matid[g] = mat
        self.display_state.setdefault(display, {})[field] = state

    def _beam_blocked(self, a: int, b: int) -> bool:
        """Anything that collides (contype/conaffinity set) on the line between the sites."""
        m, d = self.model, self.data
        p, q = d.site_xpos[a].copy(), d.site_xpos[b].copy()
        length = float(np.linalg.norm(q - p))
        if length < 1e-6:
            return False
        v = (q - p) / length
        start, travelled = p, 0.0
        gid = np.zeros(1, np.int32)
        for _ in range(16):
            dist = mujoco.mj_ray(m, d, start, v, None, 1, -1, gid)
            if dist < 0 or travelled + dist >= length - 1e-4:
                return False
            g = int(gid[0])
            if m.geom_contype[g] or m.geom_conaffinity[g]:
                return True
            travelled += dist + 1e-4
            start = p + v * travelled
        return False

    def _load_kg(self, body: int) -> float:
        """What presses down on the body through its contacts, in kg (a load cell under it)."""
        m, d = self.model, self.data
        f6 = np.zeros(6)
        down = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if body not in (b1, b2) or b1 == b2:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            world = f6[:3] @ c.frame.reshape(3, 3)  # the force geom1 exerts on geom2, in world coordinates
            on_body = world if b2 == body else -world
            down -= on_body[2]
        return max(0.0, down / G)

    def sense(self) -> dict:
        d = self.data
        out = {}
        for name, s in self._sensors.items():
            if s[0] == "joint":
                out[name] = {"q": float(d.qpos[s[1]]), "v": float(d.qvel[s[2]])}
            elif s[0] == "actuator_force":
                out[name] = float(d.actuator_force[s[1]])
            elif s[0] == "beam":
                out[name] = self._beam_blocked(s[1], s[2])
            elif s[0] == "load":
                out[name] = round(self._load_kg(s[1]), 3)
        return out

    def apply(self, commands: dict) -> None:
        for a, v in (commands.get("ctrl") or {}).items():
            if a not in self._actuators:
                raise S.SceneError(f"the dynamics set an actuator it may not: {a!r}")
            self.data.ctrl[self._actuators[a]] = float(v)
        for disp, fields in (commands.get("displays") or {}).items():
            for f, state in fields.items():
                if self.display_state.get(disp, {}).get(f) != str(state):
                    self._show(disp, f, str(state))
        for e in commands.get("events") or []:
            if isinstance(e, dict):
                self.timeline.append({"t": round(self.t, 3), **e})

    # -- time -----------------------------------------------------------------------------------

    def control(self) -> None:
        if self.dyn is not None:
            self.apply(self.dyn.step(self.t, self.sense()))

    def run(self, seconds: float, before_physics: Callable[["SceneSim"], None] | None = None) -> None:
        """Step `seconds` of world time; the dynamics run every control period; `before_physics(self)` every physics step."""
        m, d = self.model, self.data
        end = self.t + seconds
        while self.t < end - 1e-9:
            if self.t >= self._next_control - 1e-9:
                self.control()
                self._next_control += self.dt_control
            if before_physics is not None:
                before_physics(self)
            mujoco.mj_step(m, d)
            self.t += m.opt.timestep

    def close(self) -> None:
        if self.dyn is not None:
            self.dyn.close()
            self.dyn = None
