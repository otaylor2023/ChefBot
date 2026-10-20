"""Orthographic plots of the cup in the robot frame (spec §6).

Every image has a PlotSpec (saved next to it as JSON), the single source of
truth for world <-> pixel: the plot area is a square of `area` pixels at
(`left`, `top`), showing [h0, h1] x [v0, v1] meters, equal aspect, the same
meters per pixel in both views. Horizontal is robot x in both views (the robot
is to the left); vertical is robot y (top view, looking down) or z (side view,
looking along +y).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrow, Rectangle  # noqa: E402

from experiments.ortho_grasp.frames import FINGER, FLANGE, GRIPPER_BODY, KNUCKLES, MAX_OPEN  # noqa: E402

VIEWS = {"top": ("x", "y"), "side": ("x", "z")}
TITLES = {"top": "TOP VIEW (looking down)", "side": "SIDE VIEW (looking along +y)"}
AXIS_LABELS = {"x": "x (m) - robot frame, forward", "y": "y (m) - left", "z": "z (m) - up, from the floor"}


@dataclass
class PlotSpec:
    view: str  # "top" | "side"
    axes: tuple[str, str]  # (horizontal, vertical) robot-frame axes
    h_range: tuple[float, float]  # meters shown, horizontal
    v_range: tuple[float, float]  # meters shown, vertical
    image_size: tuple[int, int]  # (width, height) px
    left: int  # plot area, px from the image's left edge
    top: int  # px from the image's top edge
    area: int  # plot area side, px

    @property
    def m_per_px(self) -> float:
        return (self.h_range[1] - self.h_range[0]) / self.area

    @property
    def world_to_pixel(self) -> np.ndarray:
        """(2, 3) affine: [u, v] = A @ [h, v_m, 1] (u right, v down, origin top-left)."""
        s = 1.0 / self.m_per_px
        return np.array([[s, 0.0, self.left - s * self.h_range[0]], [0.0, -s, self.top + s * self.v_range[1]]])

    @property
    def pixel_to_world(self) -> np.ndarray:
        A = np.vstack([self.world_to_pixel, [0.0, 0.0, 1.0]])
        return np.linalg.inv(A)[:2]

    def to_pixel(self, hv: np.ndarray) -> np.ndarray:
        hv = np.atleast_2d(np.asarray(hv, dtype=float))
        return hv @ self.world_to_pixel[:, :2].T + self.world_to_pixel[:, 2]

    def to_world(self, uv: np.ndarray) -> np.ndarray:
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        return uv @ self.pixel_to_world[:, :2].T + self.pixel_to_world[:, 2]

    def project(self, points: np.ndarray) -> np.ndarray:
        """Robot-frame points (N, 3) -> this view's plot coordinates (N, 2)."""
        idx = ["xyz".index(a) for a in self.axes]
        return np.atleast_2d(points)[:, idx]

    def inside(self, hv: np.ndarray) -> np.ndarray:
        hv = np.atleast_2d(hv)
        return (hv[:, 0] >= self.h_range[0]) & (hv[:, 0] <= self.h_range[1]) & (hv[:, 1] >= self.v_range[0]) & (hv[:, 1] <= self.v_range[1])

    def save(self, path: Path) -> None:
        d = asdict(self)
        d["world_to_pixel"] = self.world_to_pixel.tolist()
        d["pixel_to_world"] = self.pixel_to_world.tolist()
        d["m_per_px"] = self.m_per_px
        Path(path).write_text(json.dumps(d, indent=1))

    @staticmethod
    def load(path: Path) -> "PlotSpec":
        d = json.loads(Path(path).read_text())
        keys = ("view", "axes", "h_range", "v_range", "image_size", "left", "top", "area")
        return PlotSpec(**{k: (tuple(d[k]) if isinstance(d[k], list) else d[k]) for k in keys})


def specs_for(box: np.ndarray, margin: float, size: int = 1024) -> dict[str, PlotSpec]:
    """Both views around a robot-frame box (2, 3), the same meters per pixel."""
    lo, hi = np.asarray(box[0]) - margin, np.asarray(box[1]) + margin
    span = float((hi - lo).max())
    middle = (lo + hi) / 2
    left, top = round(0.08 * size), round(0.05 * size)
    area = round(0.66 * size)
    out = {}
    for view, (h, v) in VIEWS.items():
        hi_, vi = "xyz".index(h), "xyz".index(v)
        out[view] = PlotSpec(view, (h, v), (middle[hi_] - span / 2, middle[hi_] + span / 2),
                             (middle[vi] - span / 2, middle[vi] + span / 2), (size, size), left, top, area)  # fmt: skip
    return out


# -- drawing -------------------------------------------------------------------------


def _figure(spec: PlotSpec):
    w, h = spec.image_size
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([spec.left / w, 1 - (spec.top + spec.area) / h, spec.area / w, spec.area / h])
    ax.set_xlim(*spec.h_range)
    ax.set_ylim(*spec.v_range)
    ax.set_aspect("equal", adjustable="box")
    major = 0.05
    ax.set_xticks(np.arange(np.ceil(spec.h_range[0] / major) * major, spec.h_range[1] + 1e-9, major))
    ax.set_yticks(np.arange(np.ceil(spec.v_range[0] / major) * major, spec.v_range[1] + 1e-9, major))
    ax.set_xticks(np.arange(np.ceil(spec.h_range[0] / 0.01) * 0.01, spec.h_range[1] + 1e-9, 0.01), minor=True)
    ax.set_yticks(np.arange(np.ceil(spec.v_range[0] / 0.01) * 0.01, spec.v_range[1] + 1e-9, 0.01), minor=True)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    ax.tick_params(labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=90)
    ax.grid(which="minor", color="#e4e4e4", linewidth=0.5, zorder=0)
    ax.grid(which="major", color="#a0a0a0", linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel(AXIS_LABELS[spec.axes[0]], fontsize=10)
    ax.set_ylabel(AXIS_LABELS[spec.axes[1]], fontsize=10)
    fig.text(spec.left / w, 1 - (spec.top - 8) / h, TITLES[spec.view], fontsize=13, weight="bold", va="bottom")
    # The robot is behind, toward -x.
    y0 = spec.v_range[0] + 0.06 * (spec.v_range[1] - spec.v_range[0])
    x0 = spec.h_range[0] + 0.03 * (spec.h_range[1] - spec.h_range[0])
    ax.annotate("robot", xy=(x0, y0), xytext=(x0 + 0.08 * (spec.h_range[1] - spec.h_range[0]), y0),
                arrowprops=dict(arrowstyle="->", color="#c03000"), color="#c03000", fontsize=10, va="center", zorder=6)  # fmt: skip
    return fig, ax


def _counter(ax, spec: PlotSpec, counter_z: float | None) -> None:
    if spec.view == "side" and counter_z is not None:
        ax.add_patch(Rectangle((spec.h_range[0], spec.v_range[0]), spec.h_range[1] - spec.h_range[0], counter_z - spec.v_range[0],
                               facecolor="#d8cbb0", edgecolor="#8a7a5a", zorder=1))  # fmt: skip
        ax.text(spec.h_range[1] - 0.01, counter_z - 0.01, "counter", ha="right", va="top", fontsize=10, color="#5a4a2a", zorder=6)


def _legend(fig, spec: PlotSpec, marks: str, color_range: tuple[float, float] | None) -> None:
    """The 2F-85 at the plot's scale, approach pointing down, and the points to mark."""
    w, h = spec.image_size
    x0 = spec.left + spec.area + 25
    width_px = w - x0 - 10
    mpp = spec.m_per_px
    height_px = int(np.ceil(0.23 / mpp))  # fingertips (-0.03) to above the flange (0.2 m)
    ax = fig.add_axes([x0 / w, 1 - (spec.top + height_px) / h, width_px / w, height_px / h])
    ax.set_xlim(-width_px * mpp / 2, width_px * mpp / 2)
    ax.set_ylim(-0.03, -0.03 + height_px * mpp)
    ax.set_aspect("equal")
    ax.axis("off")
    # Page y = -(TCP z): the flange at the top, fingertips at the bottom.
    def box(x_half, z0, z1, x_mid=0.0, **kw):
        ax.add_patch(Rectangle((x_mid - x_half, -z1), 2 * x_half, z1 - z0, **kw))
    box(GRIPPER_BODY[2], GRIPPER_BODY[0], GRIPPER_BODY[1], facecolor="#bbbbbb", edgecolor="k")
    box(KNUCKLES[2], KNUCKLES[0], KNUCKLES[1], facecolor="#d5d5d5", edgecolor="k")
    for side in (-1, 1):
        box(FINGER[2], FINGER[0], FINGER[1], x_mid=side * (MAX_OPEN / 2 + FINGER[2]), facecolor="#888888", edgecolor="k")
    ax.plot([-0.04, 0.04], [FLANGE, FLANGE], color="#c03000", lw=3)
    pts = {"A": (-MAX_OPEN / 2, 0.0), "B": (MAX_OPEN / 2, 0.0), "F": (0.0, FLANGE)}
    for name, (x, y) in pts.items():
        if name in marks:
            ax.plot(x, y, "o", color="#1060e0", ms=7, zorder=5)
            ax.text(x + (0.006 if name != "B" else -0.006), y + 0.008, name, color="#1060e0", fontsize=12, weight="bold",
                    ha="left" if name != "B" else "right")  # fmt: skip
    ax.plot(0, 0, "+", color="k", ms=8)
    ax.text(0, -0.012, "M", ha="center", va="top", fontsize=9)
    lines = ["2F-85 gripper, to scale", "(fully open: 85 mm)", "A, B: the two finger pads", "M: midpoint of A and B"]
    if "F" in marks:
        lines.append("F: center of the flange (top),")
        lines.append("   15.6 cm behind M")
    fig.text(x0 / w, 1 - (spec.top + height_px + 12) / h, "\n".join(lines), fontsize=8.5, va="top")
    if color_range is not None:
        cax = fig.add_axes([x0 / w, 1 - (spec.top + spec.area) / h, 0.025, 0.22])
        sm = matplotlib.cm.ScalarMappable(cmap="viridis", norm=matplotlib.colors.Normalize(*color_range))
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label("height (m): floor of cup -> rim", fontsize=9)
        cb.ax.tick_params(labelsize=8)


def draw(spec: PlotSpec, path: Path, points: np.ndarray | None = None, image: np.ndarray | None = None,
         counter_z: float | None = None, marks: str = "ABF", color_range=None, guides: dict | None = None,
         gripper_at: dict[str, np.ndarray] | None = None) -> None:  # fmt: skip
    """One view. V1: `points` (robot frame, N x 3) colored by height. V4:
    `image` (area x area RGB, the ortho render) under the grid instead.
    `guides`: {label: x} thin vertical lines (sequential side view)."""
    fig, ax = _figure(spec)
    _counter(ax, spec, counter_z)
    if image is not None:
        ax.imshow(image, extent=(*spec.h_range, *spec.v_range), origin="upper", zorder=2, interpolation="nearest")
        ax.set_axisbelow(False)
        ax.grid(which="major", color="#a0a0a0", linewidth=0.6, alpha=0.6, zorder=3)
    if points is not None and len(points):
        hv = spec.project(points)
        if spec.view == "top":
            order = np.argsort(points[:, 2])  # low to high: the rim on top
        else:
            order = np.argsort(-points[:, 1])  # far (high y) first, near drawn last
        size = 1.0 if len(points) > 5000 else 5.0
        lo, hi = color_range if color_range else (points[:, 2].min(), points[:, 2].max())
        ax.scatter(hv[order, 0], hv[order, 1], c=points[order, 2], cmap="viridis", vmin=lo, vmax=hi, s=size, linewidths=0, zorder=3)
    if gripper_at:  # a gripper pose drawn in gray: {"A0": xyz, "B0": xyz, "F0": xyz}
        P = {k: spec.project(np.asarray(v)[None])[0] for k, v in gripper_at.items()}
        names = list(P)
        a, b, f = (P[n] for n in names[:3])
        m = (a + b) / 2
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#888888", lw=2.5, zorder=5)
        ax.plot([m[0], f[0]], [m[1], f[1]], color="#888888", lw=2.5, zorder=5)
        for n, q in P.items():
            ax.plot(*q, "o", color="#666666", ms=7, zorder=6)
            ax.text(q[0] + 0.004, q[1] + 0.004, n, color="#555555", fontsize=11, weight="bold", zorder=6)
    for label, x in (guides or {}).items():
        ax.axvline(x, color="#e03070", lw=0.8, zorder=4)
        ax.text(x, spec.v_range[1], f" {label}", color="#e03070", fontsize=10, va="top", ha="left", zorder=6)
    _legend(fig, spec, marks, color_range if points is not None else None)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    spec.save(Path(path).with_suffix(".json"))


MARK_COLORS = {"A": "#e0201a", "B": "#1a5fe0", "F": "#f08c00", "F*": "#18a33a", "M": "#555555"}


def overlay(spec: PlotSpec, background: Path, path: Path, marks: dict[str, np.ndarray], ok: bool = True, note: str = "") -> None:
    """The input plot with the model's marks (name -> plot coords), each name its own
    color (A red, B blue, F orange, the corrected flange F* green, M gray; waypoint
    marks like "A2" take their letter's color)."""
    img = plt.imread(background)
    w, h = spec.image_size
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    for name, hv in marks.items():
        u, v = spec.to_pixel(np.asarray(hv))[0]
        color = MARK_COLORS.get(name, MARK_COLORS.get(name.rstrip("0123456789"), "#ff00aa"))
        ax.plot(u, v, "x" if name != "M" else "+", color=color, ms=13, mew=2.8)
        ax.text(u + 7, v - 7, name, color=color, fontsize=14, weight="bold")
    if note:
        ax.text(10, h - 10, note, fontsize=10, va="bottom", color="k", bbox=dict(facecolor="white", alpha=0.85))
    for i, k in enumerate(k for k in MARK_COLORS if any(n == k or n.rstrip("0123456789") == k for n in marks)):
        ax.text(spec.left + spec.area + 25 + 40 * i, spec.top + spec.area - 10, k, color=MARK_COLORS[k], fontsize=14, weight="bold", va="bottom")
    fig.savefig(path, dpi=100)
    plt.close(fig)
