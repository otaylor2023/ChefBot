"""A live page of the real kitchen run, to watch from the laptop or a phone beside the robot: the floor map (bounds,
counters, no-go, places), the base where mocap has it, the objects the map holds, the go_to target and its stance,
the cameras' last frames with their detections, the timings, and what happened. Read-only: nothing on the page
moves the robot.

The run (scripts/real_kitchen.py with --live DIR) writes DIR/state.json about 4 times a second and DIR/<camera>.jpg
per look (LiveFeed below); this serves DIR:

    python -m chefbot.scripts.live_view --dir runs/live --map maps/src_kitchen_work      # http://127.0.0.1:8780
    python -m chefbot.scripts.live_view ... --host 0.0.0.0     # a phone on the same network: http://<laptop ip>:8780

Each run also keeps DIR's log (events.jsonl: every event and timing) in its own folder, for after.
Standard library + numpy (+ OpenCV for the frames' boxes), so it runs in the host env.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import yaml

# -- the writer (in the run's process) ---------------------------------------------------------------------------------


class LiveFeed:
    """What the page shows, written to `dir` by the run: set() fields, event() lines, timing() rows, frame() images;
    the base's pose (env.localization(), cached by the env: no request) every `period_s` by a thread."""

    def __init__(self, dir, env=None, period_s: float = 0.25, log_dir: Path | None = None):
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.env, self.period_s = env, float(period_s)
        self.log_path = Path(log_dir) / "events.jsonl" if log_dir else None
        self.state: dict = {"run_start": time.time(), "status": "starting", "events": [], "timings": [], "objects": [],
                            "target": None, "path": None, "frames": {}, "cards": []}  # fmt: skip
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="live-feed", daemon=True)
        self._thread.start()

    def set(self, **kw) -> None:
        with self.lock:
            self.state.update(kw)

    def event(self, text: str, **info) -> None:
        row = {"t": round(time.time(), 2), "text": text, **info}
        with self.lock:
            self.state["events"] = (self.state["events"] + [row])[-40:]
        self._log("event", row)

    def timing(self, what: str, seconds: float, **info) -> None:
        row = {"t": round(time.time(), 2), "what": what, "s": round(float(seconds), 3), **info}
        with self.lock:
            self.state["timings"] = (self.state["timings"] + [row])[-60:]
        self._log("timing", row)

    def objects(self, objs: list[dict]) -> None:
        keep = ("name", "label", "det_label", "xyz", "last_xyz", "counter", "fixture", "status", "n_views", "tentative", "extent")
        with self.lock:
            self.state["objects"] = [{k: o.get(k) for k in keep} for o in objs]

    def card(self, card: dict) -> None:
        """A go_to to review on the page (its pictures already in the folder): right object? right spot?"""
        with self.lock:
            self.state["cards"] = (self.state["cards"] + [card])[-60:]
        self._log("card", card)

    def frame(self, camera: str, rgb: np.ndarray, dets: list[dict] | None = None) -> None:
        """The camera's last frame with its detections drawn, as <camera>.jpg (atomic)."""
        import cv2

        im = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        for d in dets or []:
            x0, y0, x1, y1 = (int(v) for v in d["box"])
            cv2.rectangle(im, (x0, y0), (x1, y1), (255, 0, 255), 2)
            cv2.putText(im, f"{d['label']} {d['score']:.2f}", (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        h, w = im.shape[:2]
        if w > 960:
            im = cv2.resize(im, (960, int(h * 960 / w)))
        tmp = self.dir / f".{camera}.jpg"
        cv2.imwrite(str(tmp), im, [cv2.IMWRITE_JPEG_QUALITY, 80])
        tmp.replace(self.dir / f"{camera}.jpg")
        with self.lock:
            self.state["frames"][camera] = {"t": round(time.time(), 2), "n_dets": len(dets or [])}

    def _log(self, kind: str, row: dict) -> None:
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(json.dumps({"kind": kind, **row}, default=float) + "\n")

    def _pose(self) -> dict:
        if self.env is None:
            return {}
        try:
            loc = self.env.localization()
            p = loc.mocap_pose
            return {"live": bool(getattr(loc, "live", False)), "pose": None if p is None else [round(p.x, 3), round(p.y, 3), round(p.yaw, 4)],
                    "mocap_age_s": getattr(loc, "mocap_age_s", None)}  # fmt: skip
        except Exception as e:  # noqa: BLE001  (the page says so; the run goes on)
            return {"live": False, "pose": None, "error": f"{type(e).__name__}: {e}"}

    def write(self) -> None:
        with self.lock:
            d = dict(self.state, robot=self._pose(), now=time.time())
            body = json.dumps(d, default=float)
        tmp = self.dir / ".state.json"
        tmp.write_text(body)
        tmp.replace(self.dir / "state.json")

    def _loop(self) -> None:
        while not self._stop.wait(self.period_s):
            try:
                self.write()
            except Exception:  # noqa: BLE001,S110  (a full disk or a closing env: the page goes stale, the run doesn't stop)
                pass

    def close(self, status: str = "finished") -> None:
        self.set(status=status)
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.write()


# -- the page ----------------------------------------------------------------------------------------------------------


def map_json(map_dir: Path) -> dict:
    """The map for the page: bounds, box obstacles (with fixture), heading zones, places, sweep stops."""
    from chefbot.scripts.real_map import load as load_map

    spec = yaml.safe_load((map_dir / "spec.yaml").read_text())
    boxes = [{"name": o["name"], "fixture": o.get("fixture"), "center": o["box"]["center"], "size": o["box"]["size"],
              "yaw": math.radians(o["box"].get("yaw_deg", 0.0)), "height": o.get("height_m")}
             for o in spec.get("obstacles") or [] if "box" in o]  # fmt: skip
    _, places = load_map(map_dir)
    return {"bounds": spec["bounds"], "boxes": boxes,
            "places": {n: [p["x"], p["y"], math.radians(p.get("yaw_deg", 0.0))] for n, p in places.items()}}  # fmt: skip


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kitchen Live</title>
<style>
:root{--bg:#f6f5f2;--panel:#fff;--ink:#1d1d1b;--mute:#6b6a66;--line:#d9d6cf;--counter:#cfc7b8;--nogo:#f2d4cf;
--robot:#1f6feb;--obj:#1a7f37;--lost:#9a9a96;--target:#d4380d;--accent:#8250df}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--panel:#1e1e1c;--ink:#ecebe7;--mute:#9b9a95;--line:#3a3935;
--counter:#4a453c;--nogo:#4d2b27;--robot:#58a6ff;--obj:#3fb950;--lost:#6e6e6a;--target:#ff7a45;--accent:#b392f0}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 -apple-system,system-ui,sans-serif}
header{display:flex;gap:12px;align-items:baseline;flex-wrap:wrap;padding:10px 16px;border-bottom:1px solid var(--line)}
h1{font-size:16px;margin:0}.pill{padding:2px 8px;border-radius:10px;background:var(--line);font-size:12px}
.pill.ok{background:var(--obj);color:#fff}.pill.bad{background:var(--target);color:#fff}
main{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:12px;padding:12px 16px}
@media (max-width:820px){main{grid-template-columns:1fr}}
section{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px;min-width:0}
h2{font-size:13px;margin:0 0 6px;color:var(--mute);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
canvas{width:100%;display:block}.frames img{width:100%;border-radius:4px;display:block;margin-bottom:6px}
table{width:100%;border-collapse:collapse;font-size:12px}td,th{padding:2px 4px;border-bottom:1px solid var(--line);text-align:left}
td.n{text-align:right;font-variant-numeric:tabular-nums}.mute{color:var(--mute)}.ev{font-size:12px;max-height:220px;overflow:auto}
.right{display:flex;flex-direction:column;gap:12px}
.card{border-top:1px solid var(--line);padding:8px 0}.card img{width:49%;border-radius:4px}.card .pics{display:flex;gap:2%}
.btn{border:1px solid var(--line);background:var(--bg);color:var(--ink);border-radius:6px;padding:4px 10px;margin:2px;font:inherit;cursor:pointer}
.btn.on.yes{background:var(--obj);color:#fff}.btn.on.no{background:var(--target);color:#fff}
</style></head><body>
<header><h1>Kitchen live</h1><span id="status" class="pill">…</span><span id="mocap" class="pill">mocap …</span>
<span id="pose" class="mute"></span><span id="stale" class="mute"></span></header>
<main>
<div style="display:flex;flex-direction:column;gap:12px">
<section><h2>Floor</h2><canvas id="map" height="560"></canvas>
<div class="mute" style="font-size:12px">blue: the base (arrow = where the arm faces) · green: objects · grey ring: lost (last seen) ·
orange: go_to target and its stance · dots: places</div></section>
<section><h2>Objects</h2><table id="objs"></table></section>
<section><h2>go_to checks: review each</h2><div id="score" class="mute"></div><div id="cards"></div></section>
</div>
<div class="right">
<section><h2>Target</h2><div id="target" class="mute">none</div></section>
<section class="frames"><h2>Cameras</h2><div id="frames"></div></section>
<section><h2>Timings</h2><table id="tim"></table></section>
<section><h2>Events</h2><div id="ev" class="ev"></div></section>
</div></main>
<script>
let MAP=null, S=null;
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
async function j(u){const r=await fetch(u,{cache:"no-store"});if(!r.ok)throw new Error(r.status);return r.json()}
function box(ctx,T,c,s,yaw,fill,stroke){const [cx,cy]=c,[l,w]=s,co=Math.cos(yaw),si=Math.sin(yaw);
 const pts=[[-l/2,-w/2],[l/2,-w/2],[l/2,w/2],[-l/2,w/2]].map(([a,b])=>T(cx+co*a-si*b,cy+si*a+co*b));
 ctx.beginPath();pts.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.closePath();
 if(fill){ctx.fillStyle=fill;ctx.fill()}if(stroke){ctx.strokeStyle=stroke;ctx.stroke()}}
function draw(){const cv=document.getElementById("map"),ctx=cv.getContext("2d");if(!MAP)return;
 const W=cv.clientWidth,H=Math.round(W*0.85);cv.width=W*devicePixelRatio;cv.height=H*devicePixelRatio;cv.style.height=H+"px";
 ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);ctx.clearRect(0,0,W,H);
 // only the kitchen part with counters (the bounds reach far north): fit the fixtures' boxes plus a margin
 let xs=[],ys=[];MAP.boxes.forEach(b=>{if(!b.fixture||b.fixture==="counter")return;xs.push(b.center[0]-b.size[0]/2-.6,b.center[0]+b.size[0]/2+.6);ys.push(b.center[1]-1.6,b.center[1]+1.6)});
 if(S&&S.robot&&S.robot.pose){xs.push(S.robot.pose[0]-.5,S.robot.pose[0]+.5);ys.push(S.robot.pose[1]-.5,S.robot.pose[1]+.5)}
 const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys),k=Math.min(W/(x1-x0),H/(y1-y0));
 const T=(x,y)=>[(x-x0)*k+(W-(x1-x0)*k)/2,H-((y-y0)*k+(H-(y1-y0)*k)/2)];
 const [[bx0,by0],[bx1,by1]]=MAP.bounds;ctx.strokeStyle=css("--line");ctx.setLineDash([4,4]);
 ctx.strokeRect(...T(bx0,by1),(bx1-bx0)*k,(by1-by0)*k);ctx.setLineDash([]);
 MAP.boxes.forEach(b=>{const no=b.name.startsWith("nogo");box(ctx,T,b.center,b.size,b.yaw,no?css("--nogo"):css("--counter"),css("--line"));
  if(b.fixture){const p=T(...b.center);ctx.fillStyle=css("--mute");ctx.font="11px system-ui";ctx.textAlign="center";ctx.fillText(b.fixture,p[0],p[1]+4)}});
 Object.entries(MAP.places).forEach(([n,p])=>{const q=T(p[0],p[1]);ctx.fillStyle=css("--mute");ctx.beginPath();ctx.arc(q[0],q[1],2.5,0,7);ctx.fill();
  ctx.font="10px system-ui";ctx.textAlign="left";ctx.fillText(n,q[0]+4,q[1]-3)});
 if(!S)return;
 if(S.path&&S.path.length>1){ctx.strokeStyle=css("--robot");ctx.globalAlpha=.4;ctx.lineWidth=2;ctx.beginPath();
  S.path.forEach((p,i)=>{const q=T(p[0],p[1]);i?ctx.lineTo(...q):ctx.moveTo(...q)});ctx.stroke();ctx.globalAlpha=1;ctx.lineWidth=1}
 const t=S.target;if(t){if(t.stance){const [sx,sy,syaw]=t.stance;ctx.setLineDash([5,3]);box(ctx,T,[sx+Math.cos(syaw)*0,sy],[0.56,0.48],syaw,null,css("--target"));ctx.setLineDash([])}
  if(t.xy){const q=T(...t.xy);ctx.strokeStyle=css("--target");ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(q[0]-6,q[1]-6);ctx.lineTo(q[0]+6,q[1]+6);ctx.moveTo(q[0]+6,q[1]-6);ctx.lineTo(q[0]-6,q[1]+6);ctx.stroke();ctx.lineWidth=1}}
 (S.objects||[]).forEach(o=>{const p=o.xyz||o.last_xyz;if(!p)return;const q=T(p[0],p[1]);const lost=!o.xyz;
  ctx.strokeStyle=lost?css("--lost"):css("--obj");ctx.fillStyle=ctx.strokeStyle;ctx.lineWidth=o.tentative?1:2;ctx.beginPath();ctx.arc(q[0],q[1],5,0,7);
  if(!lost&&!o.tentative)ctx.fill();else ctx.stroke();ctx.lineWidth=1;ctx.font="11px system-ui";ctx.textAlign="left";ctx.fillStyle=css("--ink");ctx.fillText(o.name,q[0]+7,q[1]+4)});
 const r=S.robot;if(r&&r.pose){const [x,y,yaw]=r.pose;box(ctx,T,[x,y],[0.56,0.48],yaw,css("--robot")+"33",css("--robot"));
  const a=T(x,y),b=T(x+Math.cos(yaw)*.45,y+Math.sin(yaw)*.45);ctx.strokeStyle=css("--robot");ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();ctx.lineWidth=1}}
function render(){if(!S)return;const st=document.getElementById("status");st.textContent=S.status;
 st.className="pill "+(/fail|error|stopp/.test(S.status)?"bad":/finish|done|reached/.test(S.status)?"ok":"");
 const r=S.robot||{},m=document.getElementById("mocap");m.textContent=r.live?"mocap live":"mocap NOT live";m.className="pill "+(r.live?"ok":"bad");
 document.getElementById("pose").textContent=r.pose?`x ${r.pose[0].toFixed(2)}  y ${r.pose[1].toFixed(2)}  yaw ${(r.pose[2]*180/Math.PI).toFixed(0)}°`:"";
 const age=Date.now()/1000-S.now;document.getElementById("stale").textContent=age>2?`page ${age.toFixed(0)} s old (is the run going?)`:"";
 const t=S.target;document.getElementById("target").innerHTML=t?`<b>${esc(t.name)}</b> ${esc(t.on||"")}<br>${esc(t.note||"")}${t.result?`<br>result: ${esc(t.result)}`:""}`:"none";
 const objs=(S.objects||[]).filter(o=>o.status!=="rejected");
 document.getElementById("objs").innerHTML="<tr><th>name</th><th>on</th><th class=n>views</th><th>detector said</th></tr>"+
  objs.map(o=>`<tr><td>${esc(o.name)}${o.xyz?"":" <span class=mute>(lost)</span>"}${o.tentative?" <span class=mute>?</span>":""}</td><td>${esc(o.fixture||o.counter)}</td><td class=n>${o.n_views}</td><td class=mute>${esc(o.det_label)}</td></tr>`).join("");
 document.getElementById("frames").innerHTML=Object.entries(S.frames||{}).map(([c,f])=>`<div class=mute>${esc(c)} · ${f.n_dets} detections · ${(Date.now()/1000-f.t).toFixed(0)} s ago</div><img src="${esc(c)}.jpg?t=${f.t}">`).join("")||"<span class=mute>no frames yet</span>";
 document.getElementById("tim").innerHTML="<tr><th>what</th><th class=n>s</th><th></th></tr>"+(S.timings||[]).slice(-14).reverse().map(x=>{
  const {t:_t,what,s,...rest}=x;return `<tr><td>${esc(what)}</td><td class=n>${s.toFixed(2)}</td><td class=mute>${esc(Object.entries(rest).map(([k,v])=>k+" "+v).join(" · "))}</td></tr>`}).join("");
 document.getElementById("ev").innerHTML=(S.events||[]).slice().reverse().map(e=>`<div><span class=mute>${new Date(e.t*1000).toLocaleTimeString()}</span> ${esc(e.text)}</div>`).join("");
 draw()}
let REV={};
async function loadRev(){try{REV=await j("reviews.json")}catch(e){REV={}}}
async function review(id,key,val){const r=Object.assign({},REV[id]||{});r[key]=val;REV[id]=r;renderCards(true);
 await fetch("review",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(Object.assign({id},r))})}
let lastCards="";
function renderCards(force){const cs=(S&&S.cards)||[];const key=cs.map(c=>c.id).join()+JSON.stringify(REV);if(!force&&key===lastCards)return;lastCards=key;
 const b=(id,k,v,label)=>`<button class="btn ${(REV[id]||{})[k]===v?"on "+(v?"yes":"no"):""}" onclick="review('${id}','${k}',${v})">${label}</button>`;
 document.getElementById("cards").innerHTML=cs.slice().reverse().map(c=>{const s=c.sees||{};const sw=x=>x===null||x===undefined?"out of view":x?"sees it":"doesn't";
  return `<div class=card><b>${esc(c.object)}</b> <span class=mute>${esc(c.on||"")} · parked ${c.error_cm??"?"} cm off · pole ${sw(s.pole)} · wrist ${sw(s.wrist)} · map after: ${esc(c.map_after)}</span>
  <div class=pics>${Object.entries(c.pictures||{}).map(([cam,f])=>`<img title="${esc(cam)}" src="${esc(f)}">`).join("")}</div>
  <div>right object? ${b(c.id,"object_ok",true,"yes")}${b(c.id,"object_ok",false,"no")} &nbsp; right spot? ${b(c.id,"spot_ok",true,"yes")}${b(c.id,"spot_ok",false,"no")}</div></div>`}).join("")||"<span class=mute>none yet</span>";
 const rv=cs.map(c=>REV[c.id]||{}),n=x=>rv.filter(r=>r[x]===true).length,m=x=>rv.filter(r=>r[x]!==undefined).length;
 document.getElementById("score").textContent=cs.length?`${cs.length} go_tos · right object ${n("object_ok")}/${m("object_ok")} · right spot ${n("spot_ok")}/${m("spot_ok")} (of those reviewed)`:""}
async function tick(){try{S=await j("state.json")}catch(e){}render();renderCards(false);setTimeout(tick,500)}
(async()=>{MAP=await j("map.json");await loadRev();tick();addEventListener("resize",draw)})();
</script></body></html>
"""


def serve(dir: Path, map_dir: Path, host: str = "127.0.0.1", port: int = 8780) -> None:
    dir = Path(dir).resolve()
    dir.mkdir(parents=True, exist_ok=True)
    mapj = json.dumps(map_json(Path(map_dir))).encode()

    reviews = dir / "reviews.jsonl"
    lock = threading.Lock()

    def current_reviews() -> dict:
        out: dict = {}
        if reviews.exists():
            for line in reviews.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    out[r["id"]] = {k: v for k, v in r.items() if k not in ("id", "t")}
        return out

    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802  (a review: {id, object_ok, spot_ok, note}; appended, the last per id stands)
            if self.path.split("?")[0] != "/review":
                self.send_error(404)
                return
            try:
                r = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                assert isinstance(r.get("id"), str)
            except Exception:  # noqa: BLE001
                self.send_error(400)
                return
            with lock, open(reviews, "a") as f:
                f.write(json.dumps({"t": time.time(), **r}) + "\n")
            self.send_response(204)
            self.end_headers()

        def do_GET(self):  # noqa: N802
            name = self.path.split("?")[0].lstrip("/") or "index.html"
            if name == "index.html":
                body, ctype = PAGE.encode(), "text/html; charset=utf-8"
            elif name == "map.json":
                body, ctype = mapj, "application/json"
            elif name == "reviews.json":
                body, ctype = json.dumps(current_reviews()).encode(), "application/json"
            elif name == "state.json" or (name.endswith(".jpg") and "/" not in name):
                p = dir / name
                if not p.exists():
                    self.send_error(404)
                    return
                body, ctype = p.read_bytes(), "application/json" if name.endswith(".json") else "image/jpeg"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    print(f"kitchen live page: http://{host if host != '0.0.0.0' else '<this computer>'}:{port}  (from {dir})", flush=True)
    ThreadingHTTPServer((host, port), H).serve_forever()


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=repo / "runs" / "live")
    ap.add_argument("--map", type=Path, default=repo / "maps" / "src_kitchen_work")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8780)
    a = ap.parse_args()
    serve(a.dir, a.map, a.host, a.port)


if __name__ == "__main__":
    main()
