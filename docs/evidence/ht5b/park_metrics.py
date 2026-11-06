"""Summarize real_nav park sessions from their logs: per park and totals."""
import glob, json, sys
import numpy as np
CX, CY, HX, HY = 1.49, 2.06, 1.13, 0.48          # counter box
BX, BY = (-2.00, 3.92), (-0.05, 4.32)             # map bounds (walls at xmax, ymax)
PX, PY = (0.097 + 0.462) / 2, 0.241               # plate half-extents (body: plate)
edge = np.array([[sx * PX, t] for sx in (-1, 1) for t in np.linspace(-PY, PY, 25)] + [[t, sy * PY] for sy in (-1, 1) for t in np.linspace(-PX, PX, 25)])
def box_d(x, y): return np.hypot(np.maximum(np.abs(x - CX) - HX, 0), np.maximum(np.abs(y - CY) - HY, 0))
rows = []
for sess in sys.argv[1:]:
    for d in sorted(glob.glob(sess + "/[0-9][0-9]_*")):
        pj = json.load(open(d + "/park.json"))
        if not pj.get("ok"):
            continue
        S = [json.loads(l) for f in sorted(glob.glob(d + "/leg*.jsonl")) for l in open(f)]
        drive = [s for s in S if s["phase"] == "drive"]
        P = np.array([s["pose"] for s in S]); c, s_ = np.cos(P[:, 2]), np.sin(P[:, 2])
        plate = min(float(box_d(x + ci * edge[:, 0] - si * edge[:, 1], y + si * edge[:, 0] + ci * edge[:, 1]).min()) for (x, y, _), ci, si in zip(P, c, s_))
        wall = float(min((BX[1] - P[:, 0]).min(), (BY[1] - P[:, 1]).min()))
        dev = max(abs(s["off_line"]) for s in drive)
        dist = np.array([s["dist"] for s in drive]); t = np.array([s["wall"] for s in drive])
        near = np.nonzero(dist < 0.05)[0]
        settle = float(t[-1] - t[near[0]]) if len(near) else float("nan")
        r = pj["rule"]
        rows.append(dict(name=d.split("/")[-2][-6:] + "/" + d.split("/")[-1], s=pj["seconds"], path=pj["path_m"] or 0, err=pj["final_error_m"] * 1000,
                         yaw=abs(pj["final_yaw_error_deg"]), out=r["out_m"], lat=r["target_lateral_m"] * 1000, sq=r["yaw_off_square_deg"],
                         stops=len(pj["stops"]) - 1, dev=dev * 1000, settle=settle, plate=plate * 1000, wall=wall, turn=pj["turn_taken_deg"], asked=pj["turn_asked_deg"]))
print(f"{'park':28s} {'s':>5s} {'path':>5s} {'m/s':>5s} {'err':>5s} {'yaw':>5s} {'out':>6s} {'lat':>5s} {'stops':>5s} {'dev':>4s} {'5cm→end':>7s} {'plate':>5s} {'wall-ctr':>8s} {'turn':>9s}")
for r in rows:
    print(f"{r['name']:28s} {r['s']:5.1f} {r['path']:5.2f} {r['path']/r['s']:5.3f} {r['err']:5.1f} {r['yaw']:5.2f} {r['out']:6.3f} {r['lat']:+5.0f} {r['stops']:5d} {r['dev']:4.0f} {r['settle']:7.2f} {r['plate']:5.0f} {r['wall']:8.3f} {r['turn']:+4.0f}/{r['asked']:+4.0f}")
a = lambda k: np.array([r[k] for r in rows])
print(f"TOTAL {len(rows)} parks: {a('s').sum():.1f} s for {a('path').sum():.2f} m; err max {a('err').max():.1f} mm, yaw max {a('yaw').max():.2f} deg;"
      f" intermediate stops {a('stops').sum()}; dev max {a('dev').max():.0f} mm; 5cm->end median {np.median(a('settle')):.2f} s;"
      f" plate>=counter min {a('plate').min():.0f} mm; base centre to wall min {a('wall').min():.3f} m")
