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
*{box-sizing:border-box}html,body{height:100%}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.35 -apple-system,system-ui,sans-serif;display:flex;flex-direction:column}
header{display:flex;gap:14px;align-items:center;flex-wrap:wrap;padding:8px 16px;border-bottom:1px solid var(--line)}
.cmd{font:600 20px/1.2 ui-monospace,SFMono-Regular,Menlo,monospace}.status{font-size:16px}
.pill{padding:2px 8px;border-radius:10px;background:var(--line);font-size:12px}.pill.ok{background:var(--obj);color:#fff}.pill.bad{background:var(--target);color:#fff}
main{flex:1;min-height:0;display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.25fr);gap:10px;padding:10px 16px}
.col{display:flex;flex-direction:column;gap:10px;min-height:0}
section{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px;min-width:0;min-height:0;display:flex;flex-direction:column}
h2{font-size:12px;margin:0 0 4px;color:var(--mute);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
#mapbox{flex:1}canvas{width:100%;height:100%;display:block;min-height:0}
.cams{display:grid;grid-template-columns:1fr 1fr;gap:8px}.cams figure{margin:0;min-width:0}.cams img{width:100%;aspect-ratio:4/3;object-fit:cover;border-radius:4px;background:var(--line);display:block}
figcaption{font-size:12px;color:var(--mute)}
.below{display:grid;grid-template-columns:1fr 1fr;gap:10px;flex:1;min-height:0}
table{width:100%;border-collapse:collapse;font-size:12px}td,th{padding:1px 4px;border-bottom:1px solid var(--line);text-align:left}
td.n{text-align:right;font-variant-numeric:tabular-nums}.mute{color:var(--mute)}.scroll{overflow:auto;min-height:0;flex:1}
.check{font-size:15px}.yes{color:var(--obj);font-weight:600}.no{color:var(--target);font-weight:600}
@media (max-width:900px){body{display:block}main,.below{grid-template-columns:1fr}#mapbox{height:60vh}}
</style></head><body>
<header><span class="cmd" id="cmd">…</span><span class="status" id="status"></span><span id="mocap" class="pill">mocap …</span>
<span id="pose" class="mute"></span><span id="stale" class="mute"></span><a href="grade" style="color:inherit;margin-left:auto" class="mute">grade go_tos →</a></header>
<main>
<div class="col">
<section id="mapbox"><h2>Kitchen map: base (blue, arrow = arm) · objects (green) · go_to target (orange)</h2><canvas id="map"></canvas></section>
</div>
<div class="col">
<section><h2>Cameras</h2><div class="cams"><figure><img id="img_pole" alt=""><figcaption id="cap_pole">pole</figcaption></figure>
<figure><img id="img_wrist" alt=""><figcaption id="cap_wrist">wrist</figcaption></figure></div></section>
<section><h2>Last go_to check</h2><div id="check" class="check mute">none yet</div></section>
<div class="below">
<section><h2>Objects on the map</h2><div class="scroll"><table id="objs"></table></div></section>
<section><h2>Timings · events</h2><div class="scroll"><table id="tim"></table><div id="ev" style="font-size:12px;margin-top:6px"></div></div></section>
</div></div></main>
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
 const W=cv.clientWidth,H=cv.clientHeight||400;cv.width=W*devicePixelRatio;cv.height=H*devicePixelRatio;
 ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);ctx.clearRect(0,0,W,H);
 let xs=[],ys=[];MAP.boxes.forEach(b=>{if(!b.fixture||!["island","stove_counter"].includes(b.fixture))return;xs.push(b.center[0]-b.size[0]/2-.35,b.center[0]+b.size[0]/2+.35);ys.push(b.center[1]-b.size[1]/2-.35,b.center[1]+b.size[1]/2+.35)});
 if(S&&S.robot&&S.robot.pose){xs.push(S.robot.pose[0]-.45,S.robot.pose[0]+.45);ys.push(S.robot.pose[1]-.45,S.robot.pose[1]+.45)}
 const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys),k=Math.min(W/(x1-x0),H/(y1-y0));
 const T=(x,y)=>[(x-x0)*k+(W-(x1-x0)*k)/2,H-((y-y0)*k+(H-(y1-y0)*k)/2)];
 MAP.boxes.forEach(b=>{const no=b.name.startsWith("nogo");box(ctx,T,b.center,b.size,b.yaw,no?css("--nogo"):css("--counter"),css("--line"));
  if(b.fixture){const p=T(...b.center);ctx.fillStyle=css("--mute");ctx.font="12px system-ui";ctx.textAlign="center";ctx.fillText(b.fixture.replace("_"," "),p[0],p[1]+4)}});
 if(!S)return;
 if(S.path&&S.path.length>1){ctx.strokeStyle=css("--robot");ctx.globalAlpha=.45;ctx.lineWidth=2;ctx.beginPath();
  S.path.forEach((p,i)=>{const q=T(p[0],p[1]);i?ctx.lineTo(...q):ctx.moveTo(...q)});ctx.stroke();ctx.globalAlpha=1;ctx.lineWidth=1}
 const t=S.target;if(t){if(t.stance){ctx.setLineDash([5,3]);box(ctx,T,[t.stance[0],t.stance[1]],[0.56,0.48],t.stance[2],null,css("--target"));ctx.setLineDash([])}
  if(t.xy){const q=T(...t.xy);ctx.strokeStyle=css("--target");ctx.lineWidth=3;ctx.beginPath();ctx.arc(q[0],q[1],11,0,7);ctx.stroke();ctx.lineWidth=1}}
 (S.objects||[]).forEach(o=>{if(["rejected","extra","moved"].includes(o.status))return;const p=o.xyz||o.last_xyz;if(!p)return;const q=T(p[0],p[1]);const lost=!o.xyz;
  ctx.strokeStyle=lost?css("--lost"):css("--obj");ctx.fillStyle=ctx.strokeStyle;ctx.lineWidth=2;ctx.beginPath();ctx.arc(q[0],q[1],5,0,7);
  if(!lost&&!o.tentative)ctx.fill();else ctx.stroke();ctx.lineWidth=1;ctx.font=(o.tentative?"10px":"12px")+" system-ui";ctx.textAlign="left";
  ctx.fillStyle=o.tentative?css("--mute"):css("--ink");ctx.fillText(o.name+(lost?" (lost)":""),q[0]+7,q[1]+4)});
 const r=S.robot;if(r&&r.pose){const [x,y,yaw]=r.pose;box(ctx,T,[x,y],[0.56,0.48],yaw,css("--robot")+"33",css("--robot"));
  const a=T(x,y),b=T(x+Math.cos(yaw)*.45,y+Math.sin(yaw)*.45);ctx.strokeStyle=css("--robot");ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();ctx.lineWidth=1}}
const shown={};
function render(){if(!S)return;
 document.getElementById("cmd").textContent=S.command?"> "+S.command:"";
 document.getElementById("status").textContent=S.status||"";
 const r=S.robot||{},m=document.getElementById("mocap");m.textContent=r.live?"mocap live":"mocap NOT live";m.className="pill "+(r.live?"ok":"bad");
 document.getElementById("pose").textContent=r.pose?`x ${r.pose[0].toFixed(2)}  y ${r.pose[1].toFixed(2)}  yaw ${(r.pose[2]*180/Math.PI).toFixed(0)}°`:"";
 const age=Date.now()/1000-S.now;document.getElementById("stale").textContent=age>2?`page ${age.toFixed(0)} s old`:"";
 for(const cam of ["pole","wrist"]){const f=(S.frames||{})[cam];if(f&&shown[cam]!==f.t){shown[cam]=f.t;document.getElementById("img_"+cam).src=`${cam}.jpg?t=${f.t}`}
  document.getElementById("cap_"+cam).textContent=f?`${cam} · ${f.n_dets} detections · ${(Date.now()/1000-f.t).toFixed(0)} s ago`:`${cam} · no frame yet`}
 const c=(S.cards||[]).slice(-1)[0];
 if(c){const v=(c.vlm||{}).wrist;document.getElementById("check").innerHTML=`<b>go_to ${esc(c.object)}</b> on ${esc(c.on||"")} · parked <b>${c.error_cm??"?"} cm</b> off`+
  (v?` · wrist camera (Gemini): <span class="${v.match?"yes":"no"}">${esc(v.label||v.error)} ${v.match?"✓":"✗"}</span>`:"")+` · pole ${c.sees&&c.sees.pole?'<span class="yes">sees it</span>':'<span class="mute">no detection</span>'}`;document.getElementById("check").classList.remove("mute")}
 const objs=(S.objects||[]).filter(o=>!["rejected","extra","moved"].includes(o.status)).sort((a,b)=>(a.tentative-b.tentative)||a.name.localeCompare(b.name));
 document.getElementById("objs").innerHTML="<tr><th>object</th><th>on</th><th class=n>views</th></tr>"+
  objs.map(o=>`<tr class="${o.tentative?"mute":""}"><td>${esc(o.name)}${o.xyz?"":" <span class=mute>(lost)</span>"}</td><td>${esc((o.fixture||o.counter||"").replace("_"," "))}</td><td class=n>${o.n_views}</td></tr>`).join("");
 document.getElementById("tim").innerHTML=(S.timings||[]).slice(-8).reverse().map(x=>`<tr><td>${esc(x.what)}</td><td class=n>${x.s.toFixed(2)} s</td><td class=mute>${esc(x.to||x.camera||"")}</td></tr>`).join("");
 document.getElementById("ev").innerHTML=(S.events||[]).slice(-8).reverse().map(e=>`<div><span class=mute>${new Date(e.t*1000).toLocaleTimeString()}</span> ${esc(e.text)}</div>`).join("");
 draw()}
async function tick(){try{S=await j("state.json")}catch(e){}render();setTimeout(tick,400)}
(async()=>{MAP=await j("map.json");tick();addEventListener("resize",draw)})();
</script></body></html>
"""


GRADE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Grade go_tos</title>
<style>
:root{--bg:#f6f5f2;--panel:#fff;--ink:#1d1d1b;--mute:#6b6a66;--line:#d9d6cf;--yes:#1a7f37;--no:#cf222e}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--panel:#1e1e1c;--ink:#ecebe7;--mute:#9b9a95;--line:#3a3935;--yes:#3fb950;--no:#ff7b72}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.4 -apple-system,system-ui,sans-serif}
header{position:sticky;top:0;background:var(--bg);padding:10px 16px;border-bottom:1px solid var(--line);display:flex;gap:16px;flex-wrap:wrap;align-items:baseline;z-index:2}
h1{font-size:17px;margin:0}.mute{color:var(--mute)}main{padding:12px 16px;display:flex;flex-direction:column;gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px}.card.cur{outline:3px solid #8250df}
.card.done{opacity:.6}.pics{display:flex;gap:8px}.pics img{width:50%;min-width:0;border-radius:4px}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:8px}
.btn{border:2px solid var(--line);background:var(--bg);color:var(--ink);border-radius:8px;padding:8px 18px;font:inherit;font-weight:600;cursor:pointer}
.btn.on.yes{background:var(--yes);border-color:var(--yes);color:#fff}.btn.on.no{background:var(--no);border-color:var(--no);color:#fff}
</style></head><body>
<header><h1>Grade the go_tos</h1><span id="score"></span>
<span class="mute">keys: 1/2 right object yes/no · 3/4 right spot yes/no · j/k next/previous · saved on each click</span></header>
<main id="list"></main>
<script>
let C=[],R={},cur=0;
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
async function j(u){const r=await fetch(u,{cache:"no-store"});return r.json()}
async function set(i,k,v){const c=C[i];const r=Object.assign({},R[c.id]||{});r[k]=v;R[c.id]=r;
 await fetch("review",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(Object.assign({id:c.id,object:c.object,run:c.run},r))});
 if(r.object_ok!==undefined&&r.spot_ok!==undefined&&i===cur&&cur<C.length-1)cur++;draw()}
function draw(){const b=(i,k,v,l)=>{const on=(R[C[i].id]||{})[k]===v;return `<button class="btn ${on?"on "+(v?"yes":"no"):""}" onclick="set(${i},'${k}',${v})">${l}</button>`};
 document.getElementById("list").innerHTML=C.map((c,i)=>{const r=R[c.id]||{},done=r.object_ok!==undefined&&r.spot_ok!==undefined;const s=c.sees||{};
  const sw=x=>x===null||x===undefined?"out of view":x?"detects it":"no detection";
  return `<div class="card ${i===cur?"cur":""} ${done?"done":""}" id="c${i}"><b>${i+1}. ${esc(c.object)}</b> <span class=mute>${esc(c.on||"")} · ${esc(c.run)} · parked ${c.error_cm??"?"} cm off · pole ${sw(s.pole)} · wrist ${sw(s.wrist)}${c.vlm&&c.vlm.wrist?` · Gemini (wrist): <b>${esc(c.vlm.wrist.label||c.vlm.wrist.error)}</b> ${c.vlm.wrist.match?"✓":"✗"}`:""}</span>
  <div class=pics>${Object.entries(c.pictures||{}).map(([cam,f])=>`<img title="${esc(cam)}: the circle is where the map puts it" src="${esc(f)}">`).join("")}</div>
  <div class=row>Right object at the circle? ${b(i,"object_ok",true,"Yes")}${b(i,"object_ok",false,"No")} &nbsp; Base parked in front of it? ${b(i,"spot_ok",true,"Yes")}${b(i,"spot_ok",false,"No")}</div></div>`}).join("")||"<span class=mute>no go_to cards yet</span>";
 const rv=C.map(c=>R[c.id]||{}),y=k=>rv.filter(r=>r[k]===true).length,n=k=>rv.filter(r=>r[k]!==undefined).length;
 document.getElementById("score").innerHTML=`<b>${C.length}</b> go_tos · graded ${rv.filter(r=>r.object_ok!==undefined&&r.spot_ok!==undefined).length} · right object <b>${y("object_ok")}/${n("object_ok")}</b> · right spot <b>${y("spot_ok")}/${n("spot_ok")}</b>`;
 const el=document.getElementById("c"+cur);if(el)el.scrollIntoView({block:"center",behavior:"smooth"})}
addEventListener("keydown",e=>{if(!C.length)return;const m={"1":["object_ok",true],"2":["object_ok",false],"3":["spot_ok",true],"4":["spot_ok",false]}[e.key];
 if(m)set(cur,...m);else if(e.key==="j"){cur=Math.min(C.length-1,cur+1);draw()}else if(e.key==="k"){cur=Math.max(0,cur-1);draw()}});
(async()=>{C=await j("cards.json");R=await j("reviews.json");cur=Math.max(0,C.findIndex(c=>!(R[c.id]&&R[c.id].object_ok!==undefined&&R[c.id].spot_ok!==undefined)));draw()})();
</script></body></html>
"""


def all_cards(runs: Path, pictures: Path) -> list[dict]:
    """Every go_to card of every run (runs/real_kitchen/*/events.jsonl), oldest first, those whose pictures still exist."""
    out = []
    for ev in sorted(Path(runs).glob("*/events.jsonl")):
        for line in ev.read_text().splitlines():
            if '"kind": "card"' not in line:
                continue
            c = json.loads(line)
            if all((pictures / f).exists() for f in (c.get("pictures") or {}).values()):
                out.append(dict(c, run=ev.parent.name))
    return sorted(out, key=lambda c: c.get("t", 0))


def serve(dir: Path, map_dir: Path, host: str = "127.0.0.1", port: int = 8780, runs: Path | None = None) -> None:
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
            elif name == "grade":
                body, ctype = GRADE.encode(), "text/html; charset=utf-8"
            elif name == "cards.json":
                body, ctype = json.dumps(all_cards(runs or dir.parent / "real_kitchen", dir), default=float).encode(), "application/json"
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
    ap.add_argument("--runs", type=Path, default=repo / "runs" / "real_kitchen", help="the runs whose go_to cards /grade lists")
    a = ap.parse_args()
    serve(a.dir, a.map, a.host, a.port, a.runs.resolve())


if __name__ == "__main__":
    main()
