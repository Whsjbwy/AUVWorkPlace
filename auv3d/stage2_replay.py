"""Self-contained offline replay of recorded REMUS experiment data.

Canvas projection is intentionally dependency-free. Display downsampling never
participates in simulation, prediction validation, observation or collision.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _indices(count, maximum):
    if count <= maximum:
        return np.arange(count)
    return np.unique(np.r_[np.arange(0, count, int(np.ceil((count - 1) / (maximum - 1)))), count - 1])


def latest_at_or_before(records, stamp):
    """The replay's causal display rule; never select a future snapshot."""
    selected = None
    for record in records:
        if record["time_s"] <= stamp + 1e-9:
            selected = record
        else:
            break
    return selected


def build_replay_payload(result):
    states = np.asarray(result["states"], dtype=float)
    goal = np.asarray(result["goal"], dtype=float)
    if states.ndim != 2 or states.shape[1] != 16 or not len(states) or not np.isfinite(states).all():
        raise ValueError("offline REMUS replay requires finite [time,eta6,nu6,actual3] rows")
    if np.any(np.diff(states[:, 0]) < 0) or goal.shape != (3,) or not np.isfinite(goal).all():
        raise ValueError("replay requires ordered actual times and a finite NED goal")
    if result["summary"].get("backend") != "remus6dof":
        raise ValueError("REMUS replay cannot label a simplified backend as six-DOF execution")
    terrain = result["terrain"]
    if not np.isfinite(np.r_[terrain.xs, terrain.ys, terrain.depths.ravel()]).all():
        raise ValueError("replay terrain must be finite")
    xi, yi = _indices(len(terrain.xs), 40), _indices(len(terrain.ys), 40)
    plans = []
    for row in result.get("selected_plans", []):
        values = np.asarray(row["states"], dtype=float)
        if values.ndim != 2 or values.shape[1] != 16 or not len(values) or not np.isfinite(values).all():
            raise ValueError("replay predictions require finite original REMUS trajectory rows")
        if np.any(np.diff(values[:, 0]) < 0) or values[0, 0] < -1e-9:
            raise ValueError("prediction offsets must be ordered and nonnegative")
        plans.append({"time_s": float(row["time_s"]), "feasible": bool(row["feasible"]),
                      "states": values[_indices(len(values), 180), :7].tolist()})
    snapshots = []
    for row in result.get("obstacles", []):
        positions = np.asarray(row["positions"], dtype=float).reshape(-1, 3)
        radii = np.asarray(row["radii"], dtype=float).reshape(-1)
        ids = np.asarray(row["ids"])
        if len(positions) != len(radii) or len(ids) != len(radii) or not np.isfinite(positions).all() or not np.isfinite(radii).all() or np.any(radii <= 0):
            raise ValueError("invalid recorded obstacle snapshot")
        snapshots.append({"time_s": float(row["time_s"]), "positions": positions.tolist(),
                          "radii": radii.tolist(), "ids": ids.tolist()})
    events = [{"time_s": float(e["time_s"]), "kind": str(e["kind"]),
               "position": np.asarray(e["position"], dtype=float).tolist(), "reason": str(e.get("reason", ""))}
              for e in result.get("events", [])]
    for records in (plans, snapshots, events):
        if any(not np.isfinite(row["time_s"]) for row in records) or any(a["time_s"] > b["time_s"] for a, b in zip(records, records[1:])):
            raise ValueError("recorded replay times must be finite and ordered")
    payload = {
        "backend": "original REMUS six-DOF actual execution",
        "scenario": str(result["summary"].get("scenario", "")),
        "status": str(result["summary"].get("status", "unknown")),
        "safety_validation_status":str(result["summary"].get("safety_validation_status", "not reported")),
        "avoidance_acceptance_passed":result["summary"].get("avoidance_acceptance_passed"),
        "actual": states[_indices(len(states), 2400), :7].tolist(),
        "actual_original_count": len(states), "goal": goal.tolist(),
        "terrain": {"xs": terrain.xs[xi].tolist(), "ys": terrain.ys[yi].tolist(),
                    "depths": terrain.depths[np.ix_(yi, xi)].tolist()},
        "plans": plans, "obstacles": snapshots, "events": events,
        "display_note": "Offline display only. Actual line stops at replay time; obstacle snapshot and selected plan use latest timestamp <= replay time. Forecast line is model prediction. Geometry is downsampled with final points preserved; collision checks used full physical data.",
        "coordinates": "NED: North, East, depth positive down",
    }
    # Reject nonfinite event or other fields instead of producing invalid JSON.
    json.dumps(payload, allow_nan=False)
    return payload


def export_stage2_replay(result, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists; choose a new replay path")
    data = json.dumps(build_replay_payload(result), ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_HTML.replace("__REPLAY_DATA__", data), encoding="utf-8")
    return str(path)


_HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>REMUS six-DOF offline replay</title><style>
body{margin:0;background:#0a1420;color:#e5eef8;font:14px system-ui,sans-serif}header{padding:14px 20px;background:#132336}h1{font-size:19px;margin:0 0 7px}p{margin:5px 0;color:#bbcbdd}#controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:12px 20px}button{background:#244566;color:white;border:1px solid #5d829f;border-radius:5px;padding:8px 13px;cursor:pointer}input{flex:1;min-width:200px}canvas{display:block;width:100%;height:65vh;touch-action:none;cursor:grab}#info,#legend{padding:8px 20px;line-height:1.6}#legend span{display:inline-block;margin-right:20px}.actual{color:#51d9ff}.plan{color:#ffb65a}.obstacle{color:#e88ba2}.event{color:#ff6269}small{color:#abc0d4}#events{white-space:pre-wrap;padding:4px 20px 16px;max-height:100px;overflow:auto}</style>
<header><h1 id="title"></h1><p>Drag to rotate · wheel to zoom · slider or Play to replay · NED depth increases downward</p><small id="note"></small></header>
<div id="controls"><button id="play">Play</button><input id="time" type="range" min="0" value="0" step="1"><button id="reset">Reset view</button><label>Speed <select id="speed"><option value="1">1×</option><option value="5" selected>5×</option><option value="20">20×</option></select></label></div>
<canvas id="view"></canvas><div id="legend"><span class="actual">━ REMUS actual six-DOF</span><span class="plan">┄ Current REMUS DWA forecast</span><span class="obstacle">● Recorded dynamic obstacles</span><span class="event">× Emergency / collision</span><span>▧ Seabed · Start / Goal</span></div><div id="info"></div><div id="events"></div>
<script id="replay-data" type="application/json">__REPLAY_DATA__</script><script>
'use strict';
const data=JSON.parse(document.getElementById('replay-data').textContent),canvas=document.getElementById('view'),ctx=canvas.getContext('2d'),slider=document.getElementById('time'),info=document.getElementById('info');
document.getElementById('title').textContent='REMUS six-DOF · '+data.scenario+' · navigation: '+data.status+' · safety: '+data.safety_validation_status;
document.getElementById('note').textContent=data.display_note;
slider.max=data.actual.length-1;
const tx=data.terrain.xs,ty=data.terrain.ys,tz=data.terrain.depths;
const zmax=Math.max(...tz.flat(),...data.actual.map(r=>r[3])),center=[(tx[0]+tx.at(-1))/2,(ty[0]+ty.at(-1))/2,zmax/2];
const span=Math.max(tx.at(-1)-tx[0],ty.at(-1)-ty[0],zmax);
let yaw=-.7,pitch=.8,zoom=1.1,drag=null,playing=false,playTime=0,lastFrame=0;
function latest(records,t){let found=null;for(const r of records){if(r.time_s<=t+1e-9)found=r;else break;}return found;}
function project(p){const n=p[0]-center[0],e=p[1]-center[1],z=-(p[2]-center[2]);const x=Math.cos(yaw)*n-Math.sin(yaw)*e,y=Math.sin(yaw)*n+Math.cos(yaw)*e;const scale=Math.min(canvas.width,canvas.height)*.82/span*zoom;return [canvas.width/2+x*scale,canvas.height/2-(Math.cos(pitch)*z-Math.sin(pitch)*y)*scale,scale];}
function line(points,color,width=1,dash=[]){if(!points.length)return;ctx.beginPath();for(let i=0;i<points.length;i++){const q=project(points[i]);i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]);}ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.stroke();ctx.setLineDash([]);}
function dot(p,r,color,label){const q=project(p);ctx.beginPath();ctx.arc(q[0],q[1],Math.max(3,r*q[2]),0,Math.PI*2);ctx.fillStyle=color;ctx.fill();if(label){ctx.font='12px system-ui';ctx.fillStyle='#e4f0ff';ctx.fillText(label,q[0]+7,q[1]-6);}}
function draw(){ctx.clearRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#0a1420';ctx.fillRect(0,0,canvas.width,canvas.height);
for(let j=0;j<ty.length;j++)line(tx.map((x,i)=>[x,ty[j],tz[j][i]]),'#547c78',.75);
for(let i=0;i<tx.length;i++)line(ty.map((y,j)=>[tx[i],y,tz[j][i]]),'#547c78',.75);
const index=Number(slider.value),row=data.actual[index],t=row[0],ob=latest(data.obstacles,t),plan=latest(data.plans,t);
if(ob)ob.positions.forEach((p,i)=>dot(p,ob.radii[i],'#ba6679aa','ID '+ob.ids[i]));
if(plan)line(plan.states.map(r=>r.slice(1,4)),plan.feasible?'#ffb65a':'#ff6269',2,[7,4]);
line(data.actual.slice(0,index+1).map(r=>r.slice(1,4)),'#51d9ff',3);
dot(data.actual[0].slice(1,4),1,'#68df9d','Start');dot(data.goal,1.5,'#f4df76','Goal');dot(row.slice(1,4),1.3,'#51d9ff','REMUS actual');
const visibleEvents=data.events.filter(e=>e.time_s<=t+1e-9);visibleEvents.forEach(e=>{const p=project(e.position);ctx.strokeStyle='#ff6269';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(p[0]-5,p[1]-5);ctx.lineTo(p[0]+5,p[1]+5);ctx.moveTo(p[0]+5,p[1]-5);ctx.lineTo(p[0]-5,p[1]+5);ctx.stroke();});
const origin=[tx[0],ty[0],0];line([origin,[tx[0]+span*.14,ty[0],0]],'#c0d1e3',2);line([origin,[tx[0],ty[0]+span*.14,0]],'#c0d1e3',2);line([origin,[tx[0],ty[0],span*.14]],'#c0d1e3',2);dot([tx[0]+span*.14,ty[0],0],0,'#c0d1e3','N');dot([tx[0],ty[0]+span*.14,0],0,'#c0d1e3','E');dot([tx[0],ty[0],span*.14],0,'#c0d1e3','Depth +');
info.textContent='Actual t='+t.toFixed(2)+' s · N/E/D '+row.slice(1,4).map(x=>x.toFixed(2)).join(' / ')+' m · Roll/Pitch/Yaw '+row.slice(4,7).map(x=>(x*180/Math.PI).toFixed(1)).join(' / ')+' deg · Obstacle snapshot '+(ob?ob.time_s.toFixed(2)+' s ('+ob.positions.length+' objects)':'none yet')+' · Selected forecast issued '+(plan?plan.time_s.toFixed(2)+' s':'none yet')+' · Events '+visibleEvents.length;
document.getElementById('events').textContent=visibleEvents.slice(-5).map(e=>e.time_s.toFixed(2)+' s '+e.kind+': '+e.reason).join('\n');}
function resize(){const rect=canvas.getBoundingClientRect();canvas.width=Math.max(400,Math.round(rect.width));canvas.height=Math.max(300,Math.round(rect.height));draw();}window.addEventListener('resize',resize);
canvas.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);});canvas.addEventListener('pointermove',e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*.008;pitch=Math.max(-1.4,Math.min(1.4,pitch+(e.clientY-drag[1])*.008));drag=[e.clientX,e.clientY];draw();});canvas.addEventListener('pointerup',()=>drag=null);canvas.addEventListener('pointercancel',()=>drag=null);canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.2,Math.min(12,zoom*Math.exp(-e.deltaY*.001)));draw();},{passive:false});
slider.addEventListener('input',()=>{playTime=data.actual[Number(slider.value)][0];draw();});document.getElementById('reset').onclick=()=>{yaw=-.7;pitch=.8;zoom=1.1;draw();};document.getElementById('play').onclick=()=>{playing=!playing;if(playing&&Number(slider.value)>=data.actual.length-1){slider.value=0;playTime=data.actual[0][0];}document.getElementById('play').textContent=playing?'Pause':'Play';};
function frame(now){if(playing&&lastFrame){playTime+=Math.min(.1,(now-lastFrame)/1000)*Number(document.getElementById('speed').value);let i=Number(slider.value);while(i<data.actual.length-1&&data.actual[i+1][0]<=playTime)i++;slider.value=i;if(i===data.actual.length-1){playing=false;document.getElementById('play').textContent='Play';}draw();}lastFrame=now;requestAnimationFrame(frame);}resize();requestAnimationFrame(frame);
</script></html>'''
