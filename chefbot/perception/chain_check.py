"""Check the pole camera -> world chain on recorded frames against the kitchen map, no detector needed
(docs/memory-map-next.md gap 1):

  heights  the counter tops come out at their Motive heights (island 0.91, stove counter 0.83 m ...)
  edges    the points at a counter's top height land inside its box (a shifted chain spills them over an edge)

and search the arm-base offset (dx along the heading, dy, dyaw of the mount) that fits best.

    python -m chefbot.perception.chain_check STREAM/p2_layout3 --config CAL.yaml --spec maps/src_kitchen/spec.yaml
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from chefbot.perception.pole_geometry import ARM_IN_BASE, Pass, load_counters, load_pole_T, mount, planar, to_world


def frame_points(p: Pass, frames, stride: int):
    """Per usable frame: (base pose, color-camera points)."""
    out = []
    for f in frames:
        pose = p.pose(f)
        if pose is None:
            continue
        out.append((pose, p.camera.depth_points(p.depth(f), stride=stride, max_m=3.5)))
    return out


def score(data, T_mount, counters, top_tol=0.03, band=0.15):
    """Heights and edge fit for one chain. Per counter: the mode of point heights in its box (vs height_m), and of
    the points within top_tol of its top within 0.4 m of the box, the fraction inside the box."""
    per = {c.name: {"z": [], "in": 0, "near": 0} for c in counters}
    for pose, pc in data:
        w = to_world(planar(*pose) @ T_mount, pc)
        for c in counters:
            near = c.contains(w[:, :2], margin=0.4)
            if not near.any():
                continue
            wn = w[near]
            inside = c.contains(wn[:, :2])
            zin = wn[inside, 2]
            per[c.name]["z"].append(zin[np.abs(zin - c.height) < band])
            top = np.abs(wn[:, 2] - c.height) < top_tol
            per[c.name]["near"] += int(top.sum())
            per[c.name]["in"] += int((top & inside).sum())
    rep = {}
    for c in counters:
        z = np.concatenate(per[c.name]["z"]) if per[c.name]["z"] else np.zeros(0)
        if len(z) < 200:
            continue
        hist, edges = np.histogram(z, bins=np.arange(c.height - band, c.height + band + 1e-9, 0.005))
        k = int(np.argmax(hist))
        rep[c.name] = {"height_m": c.height, "top_mode_m": round(float((edges[k] + edges[k + 1]) / 2), 3),
                       "n_points": int(len(z)),
                       "inside_frac": round(per[c.name]["in"] / max(1, per[c.name]["near"]), 3)}  # fmt: skip
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("passes", nargs="+")
    ap.add_argument("--config", required=True, help="environments/real/config.yaml with the pole T to check")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--frames", type=int, default=12, help="frames per pass, spread evenly")
    ap.add_argument("--stride", type=int, default=4, help="depth pixel stride")
    ap.add_argument("--search", action="store_true", help="grid-search dx, dy, dyaw of the mount")
    ap.add_argument("--mount", type=float, nargs=6, metavar=("DX", "DY", "DZ", "DYAW", "DPITCH", "DROLL"),
                    help="also score this mount (m, deg), e.g. a fit from other passes")
    a = ap.parse_args(argv)

    T_pole = load_pole_T(a.config)
    counters = [c for c in load_counters(a.spec) if c.height < 1.2]  # work tops (not the fridge)
    data = []
    for root in a.passes:
        p = Pass(root)
        good = [f for f in p.frames if f.get("depth_valid", 1) > 0.3]
        pick = [good[i] for i in np.linspace(0, len(good) - 1, min(a.frames, len(good))).astype(int)]
        data += frame_points(p, pick, a.stride)
        print(f"{root}: {len(p.frames)} frames, {len(p.track.t)} mocap samples, using {len(pick)}")

    def summary(rep):
        dz = [abs(r["top_mode_m"] - r["height_m"]) for r in rep.values()]
        fr = [r["inside_frac"] for r in rep.values()]
        return (float(np.mean(dz)) if dz else 9.0), (float(np.mean(fr)) if fr else 0.0)

    def objective(x):
        rep = score(data, mount(T_pole, x), counters)
        dz, fr = summary(rep)
        return fr - 2.0 * dz, dz, fr, rep

    starts = {"arm at the turning centre": [0.0, 0.0, ARM_IN_BASE[2], 0, 0, 0],
              "assumed plate centre": [ARM_IN_BASE[0], ARM_IN_BASE[1], ARM_IN_BASE[2], 0, 0, 0]}  # fmt: skip
    if a.mount:
        starts["given --mount"] = list(a.mount)
    for label, x in starts.items():
        _, dz, fr, rep = objective(x)
        print(f"\n{label}: x={x}  mean|dz|={dz:.3f} m  mean inside={fr:.3f}")
        print(json.dumps(rep, indent=1))

    if a.search:  # coordinate descent from the assumed mount, steps halving
        x = list(starts["assumed plate centre"])
        steps = [0.05, 0.05, 0.02, 2.0, 1.0, 1.0]
        best = objective(x)
        for _ in range(6):
            for k in range(len(x)):
                for sgn in (1, -1):
                    while True:
                        y = list(x)
                        y[k] += sgn * steps[k]
                        r = objective(y)
                        if r[0] > best[0] + 1e-4:
                            x, best = y, r
                        else:
                            break
            steps = [s / 2 for s in steps]
        _, dz, fr, rep = best
        print("\nfit: dx={:.3f} dy={:.3f} dz={:.3f} m, dyaw={:.2f} dpitch={:.2f} droll={:.2f} deg".format(*x))
        print(f"  mean|dz|={dz:.3f} m  mean inside={fr:.3f}")
        print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
