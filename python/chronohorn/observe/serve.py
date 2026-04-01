"""Compact visualization server for chronohorn runtime state.

Designed for a small browser pane (square format, ~500px).
Tabs: Curves | Frontier | Fleet | Efficiency
Polls /api/status every 20s. No dependencies beyond stdlib.
"""
from __future__ import annotations

import json
import glob
import argparse
import subprocess
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Sequence


def _load_all_results(result_dir: str = "out/results") -> list[dict[str, Any]]:
    results = []
    for p in sorted(glob.glob(f"{result_dir}/*.json")):
        try:
            d = json.loads(Path(p).read_text())
            d["_source"] = p
            d["_name"] = Path(p).stem
            results.append(d)
        except (json.JSONDecodeError, OSError):
            pass
    return results


def _probe_fleet() -> dict[str, Any]:
    """Quick SSH probe of slop boxes. Returns running containers."""
    fleet = {}
    for host in ("slop-01", "slop-02"):
        try:
            out = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3", host,
                 'sudo docker ps --format "{{.Names}}|{{.Status}}" 2>/dev/null'],
                capture_output=True, text=True, timeout=5,
            )
            containers = []
            for line in out.stdout.strip().splitlines():
                if "|" in line:
                    name, status = line.split("|", 1)
                    containers.append({"name": name.replace("chronohorn-", ""), "status": status})
            fleet[host] = {"containers": containers, "online": True}
        except Exception:
            fleet[host] = {"containers": [], "online": False}
    return fleet


def _build_api_data(result_dir: str = "out/results") -> dict[str, Any]:
    results = _load_all_results(result_dir)

    # Group by config prefix (strip step suffix like -1000s, -5000s, -s5200)
    groups: dict[str, list[dict]] = {}
    for r in results:
        name = r["_name"]
        # Normalize: strip trailing step patterns
        prefix = name
        for pat in ("-1000s", "-5000s", "-10000s", "-s5200", "-s10000", "-seed43", "-seed44"):
            prefix = prefix.replace(pat, "")
        groups.setdefault(prefix, []).append(r)

    # Build learning curves (merged across step horizons per config)
    curves = {}
    for prefix, group in groups.items():
        points = []
        for r in sorted(group, key=lambda x: x.get("config", {}).get("steps", 0)):
            probes = r.get("training", {}).get("probes", [])
            bpb = r.get("model", {}).get("test_bpb")
            steps = r.get("config", {}).get("steps")
            for p in probes:
                step = p.get("step", 0)
                pbpb = p.get("bpb") or p.get("test_bpb")
                if pbpb and step:
                    points.append({"step": step, "bpb": round(pbpb, 4)})
            if bpb and steps:
                points.append({"step": steps, "bpb": round(bpb, 4)})
        # Deduplicate by step
        seen = set()
        unique = []
        for p in sorted(points, key=lambda x: x["step"]):
            if p["step"] not in seen:
                seen.add(p["step"])
                unique.append(p)
        if unique:
            curves[prefix] = unique

    # Leaderboard (best per config, not per run)
    leaderboard = []
    for r in results:
        bpb = r.get("model", {}).get("test_bpb")
        if bpb:
            steps = r.get("config", {}).get("steps")
            perf = r.get("training", {}).get("performance", {})
            tflops_s = perf.get("estimated_sustained_tflops", 0)
            elapsed = perf.get("elapsed_sec", 0)
            leaderboard.append({
                "name": r["_name"],
                "bpb": round(bpb, 4),
                "steps": steps,
                "tflops": round(tflops_s * elapsed, 1) if tflops_s and elapsed else 0,
            })
    leaderboard.sort(key=lambda x: x["bpb"])

    # Efficiency: marginal gain per TFLOP from last two probes
    efficiency = []
    for r in results:
        name = r["_name"]
        bpb = r.get("model", {}).get("test_bpb")
        probes = r.get("training", {}).get("probes", [])
        perf = r.get("training", {}).get("performance", {})
        tflops_s = perf.get("estimated_sustained_tflops", 0)
        if bpb and len(probes) >= 2 and tflops_s > 0:
            p1 = probes[-2]
            p2 = probes[-1]
            b1 = p1.get("bpb") or p1.get("test_bpb") or 0
            b2 = p2.get("bpb") or p2.get("test_bpb") or 0
            s1 = p1.get("step", 0)
            s2 = p2.get("step", 0)
            if b1 > b2 and s2 > s1 and tflops_s > 0:
                dt = (s2 - s1) / tflops_s  # rough TFLOP estimate
                marginal = (b1 - b2) / max(dt, 1e-10)
                efficiency.append({
                    "name": name,
                    "bpb": round(bpb, 4),
                    "marginal": round(marginal, 6),
                    "steps": r.get("config", {}).get("steps"),
                })
    efficiency.sort(key=lambda x: -x["marginal"])

    # Fleet
    fleet = _probe_fleet()

    return {
        "n": len(results),
        "curves": curves,
        "board": leaderboard[:30],
        "eff": efficiency[:20],
        "fleet": fleet,
        "best": leaderboard[0] if leaderboard else None,
    }


_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>chronohorn</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'SF Mono','Fira Code','Menlo',monospace;background:#0a0a0a;color:#b0b0b0;overflow:hidden;width:100vw;height:100vh}
#root{display:flex;flex-direction:column;height:100vh;padding:6px}
#head{display:flex;justify-content:space-between;align-items:center;padding:2px 4px;flex-shrink:0}
#head h1{font-size:11px;color:#e0e0e0;letter-spacing:1px}
#head .ts{font-size:9px;color:#404040}
#tabs{display:flex;gap:1px;flex-shrink:0;margin:4px 0 2px}
.tab{font-size:10px;padding:3px 8px;background:#141414;border:1px solid #222;border-bottom:none;border-radius:3px 3px 0 0;cursor:pointer;color:#606060}
.tab.on{background:#1a1a1a;color:#c0c0c0;border-color:#333}
#body{flex:1;background:#111;border:1px solid #222;border-radius:0 3px 3px 3px;overflow:hidden;position:relative}
.pane{position:absolute;inset:0;overflow:auto;display:none;padding:4px}
.pane.on{display:block}
canvas{width:100%;height:100%;display:block}
table{width:100%;border-collapse:collapse;font-size:10px}
th{text-align:left;color:#505050;padding:2px 4px;border-bottom:1px solid #1e1e1e;font-weight:normal;font-size:9px;text-transform:uppercase;letter-spacing:.5px}
td{padding:2px 4px;border-bottom:1px solid #161616;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:140px}
.g{color:#4caf50}.b{color:#42a5f5}.y{color:#ffa726}.r{color:#ef5350}.d{color:#383838}
.dot{display:inline-block;width:6px;height:6px;border-radius:50%;margin-right:4px}
.dot.on{background:#4caf50}.dot.off{background:#333}
.host{font-size:10px;margin:6px 4px 2px;color:#707070}
.job{font-size:9px;padding:1px 4px 1px 14px;color:#909090}
.job .st{color:#505050;font-size:8px}
.empty{color:#303030;font-size:10px;padding:12px;text-align:center}
</style>
</head>
<body>
<div id="root">
<div id="head"><h1>CHRONOHORN</h1><span class="ts" id="ts">--</span></div>
<div id="tabs">
<div class="tab on" data-p="curves">curves</div>
<div class="tab" data-p="frontier">frontier</div>
<div class="tab" data-p="fleet">fleet</div>
<div class="tab" data-p="eff">efficiency</div>
</div>
<div id="body">
<div class="pane on" id="p-curves"><canvas id="cv"></canvas></div>
<div class="pane" id="p-frontier"><table><thead><tr><th>#</th><th>run</th><th>bpb</th><th>steps</th><th>TF</th></tr></thead><tbody id="tb-f"></tbody></table></div>
<div class="pane" id="p-fleet"><div id="fleet-content"></div></div>
<div class="pane" id="p-eff"><table><thead><tr><th>#</th><th>run</th><th>bpb</th><th>marginal</th><th>steps</th></tr></thead><tbody id="tb-e"></tbody></table></div>
</div>
</div>
<script>
const C=['#42a5f5','#66bb6a','#ffa726','#ef5350','#ab47bc','#26c6da','#d4e157','#ec407a','#8d6e63','#78909c','#ff7043','#5c6bc0','#4db6ac','#9ccc65','#7e57c2','#f06292'];
let curTab='curves';

document.querySelectorAll('.tab').forEach(t=>{
  t.onclick=()=>{
    document.querySelectorAll('.tab').forEach(x=>x.classList.remove('on'));
    document.querySelectorAll('.pane').forEach(x=>x.classList.remove('on'));
    t.classList.add('on');
    document.getElementById('p-'+t.dataset.p).classList.add('on');
    curTab=t.dataset.p;
    if(curTab==='curves'&&window._lastData) drawCurves(window._lastData);
  };
});

function drawCurves(data){
  const cv=document.getElementById('cv');
  const rect=cv.parentElement.getBoundingClientRect();
  cv.width=rect.width*2;cv.height=rect.height*2;
  const ctx=cv.getContext('2d');
  ctx.scale(2,2);
  const W=rect.width,H=rect.height;
  ctx.clearRect(0,0,W,H);

  const entries=Object.entries(data.curves||{}).filter(([_,v])=>v.length>1);
  if(!entries.length){ctx.fillStyle='#303030';ctx.font='11px monospace';ctx.fillText('no curves yet',W/2-40,H/2);return}

  let sMin=Infinity,sMax=0,bMin=Infinity,bMax=0;
  entries.forEach(([_,pts])=>pts.forEach(p=>{sMin=Math.min(sMin,p.step);sMax=Math.max(sMax,p.step);bMin=Math.min(bMin,p.bpb);bMax=Math.max(bMax,p.bpb)}));
  const m={l:32,r:6,t:6,b:16};
  const pw=W-m.l-m.r,ph=H-m.t-m.b;
  const lsMin=Math.log(sMin+1),lsMax=Math.log(sMax+1);
  const sx=s=>m.l+(Math.log(s+1)-lsMin)/(lsMax-lsMin)*pw;
  const bPad=0.03,bRange=bMax-bMin+bPad*2;
  const sy=b=>m.t+(1-(b-bMin+bPad)/bRange)*ph;

  // Grid
  ctx.strokeStyle='#1a1a1a';ctx.lineWidth=.5;
  ctx.fillStyle='#383838';ctx.font='8px monospace';
  [50,100,500,1000,5000,10000].forEach(s=>{if(s>=sMin&&s<=sMax){const x=sx(s);ctx.beginPath();ctx.moveTo(x,m.t);ctx.lineTo(x,H-m.b);ctx.stroke();ctx.fillText(s>=1000?(s/1000)+'k':s,x-6,H-m.b+10)}});
  const bStep=bRange>1?.2:.1;
  for(let b=Math.ceil(bMin/bStep)*bStep;b<=bMax;b+=bStep){const y=sy(b);ctx.beginPath();ctx.moveTo(m.l,y);ctx.lineTo(W-m.r,y);ctx.stroke();ctx.fillText(b.toFixed(1),1,y+3)}

  // Sort by final bpb (best on top / last drawn)
  const sorted=entries.map(([n,pts])=>[n,pts,pts[pts.length-1].bpb]).sort((a,b)=>b[2]-a[2]);

  // Draw curves
  let labelY={};
  sorted.forEach(([name,pts,_],ci)=>{
    pts.sort((a,b)=>a.step-b.step);
    ctx.strokeStyle=C[ci%C.length];ctx.lineWidth=1.2;
    ctx.beginPath();
    pts.forEach((p,i)=>{const x=sx(p.step),y=sy(p.bpb);i===0?ctx.moveTo(x,y):ctx.lineTo(x,y)});
    ctx.stroke();
    // Endpoint dot
    const last=pts[pts.length-1];
    const ex=sx(last.step),ey=sy(last.bpb);
    ctx.beginPath();ctx.arc(ex,ey,2,0,Math.PI*2);ctx.fillStyle=C[ci%C.length];ctx.fill();
  });

  // Labels: only top 8 by best final bpb, right-aligned, avoid overlap
  const top=sorted.slice(-8).reverse();
  let usedY=[];
  ctx.font='8px monospace';
  top.forEach(([name,pts,_],ci)=>{
    const last=pts[pts.length-1];
    let y=sy(last.bpb);
    // Avoid overlap
    for(const uy of usedY){if(Math.abs(y-uy)<10)y=uy+(y>uy?10:-10)}
    usedY.push(y);
    const short=name.length>20?name.slice(0,18)+'..':name;
    ctx.fillStyle=C[(sorted.length-8+ci)%C.length];
    ctx.fillText(short+' '+last.bpb.toFixed(3),m.l+4,y-3);
  });
}

function updateFrontier(data){
  const tb=document.getElementById('tb-f');
  while(tb.firstChild)tb.removeChild(tb.firstChild);
  (data.board||[]).forEach((r,i)=>{
    const tr=document.createElement('tr');
    const vals=[i+1,r.name,r.bpb.toFixed(4),r.steps||'?',r.tflops?r.tflops.toFixed(0):''];
    const cls=['d','',i<3?'g':'b','d','d'];
    vals.forEach((v,j)=>{const td=document.createElement('td');td.textContent=String(v);if(cls[j])td.className=cls[j];tr.appendChild(td)});
    tb.appendChild(tr);
  });
}

function updateFleet(data){
  const el=document.getElementById('fleet-content');
  while(el.firstChild)el.removeChild(el.firstChild);
  const fleet=data.fleet||{};
  let totalRunning=0;
  Object.entries(fleet).forEach(([host,info])=>{
    const h=document.createElement('div');h.className='host';
    const dot=document.createElement('span');dot.className='dot '+(info.online?'on':'off');
    h.appendChild(dot);h.appendChild(document.createTextNode(host));
    el.appendChild(h);
    (info.containers||[]).forEach(c=>{
      totalRunning++;
      const j=document.createElement('div');j.className='job';
      j.textContent=c.name;
      const st=document.createElement('span');st.className='st';st.textContent=' '+c.status;
      j.appendChild(st);el.appendChild(j);
    });
    if(!info.containers||!info.containers.length){
      const j=document.createElement('div');j.className='job d';j.textContent='idle';el.appendChild(j);
    }
  });
  const sum=document.createElement('div');
  sum.style.cssText='margin-top:8px;padding:4px;font-size:10px;color:#505050';
  sum.textContent=data.n+' results | '+totalRunning+' running';
  if(data.best)sum.textContent+=' | best: '+data.best.bpb.toFixed(4);
  el.appendChild(sum);
}

function updateEfficiency(data){
  const tb=document.getElementById('tb-e');
  while(tb.firstChild)tb.removeChild(tb.firstChild);
  (data.eff||[]).forEach((r,i)=>{
    const tr=document.createElement('tr');
    const vals=[i+1,r.name,r.bpb.toFixed(4),r.marginal.toFixed(4),r.steps||'?'];
    const cls=['d','',i<3?'y':'b','y','d'];
    vals.forEach((v,j)=>{const td=document.createElement('td');td.textContent=String(v);if(cls[j])td.className=cls[j];tr.appendChild(td)});
    tb.appendChild(tr);
  });
}

async function poll(){
  try{
    const r=await fetch('/api/status');
    const data=await r.json();
    window._lastData=data;
    if(curTab==='curves')drawCurves(data);
    updateFrontier(data);
    updateFleet(data);
    updateEfficiency(data);
    document.getElementById('ts').textContent=new Date().toLocaleTimeString();
  }catch(e){document.getElementById('ts').textContent='err: '+e.message}
}
poll();setInterval(poll,20000);
window.addEventListener('resize',()=>{if(window._lastData&&curTab==='curves')drawCurves(window._lastData)});
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    result_dir = "out/results"

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(_HTML.encode())
        elif self.path.startswith("/api/status"):
            data = _build_api_data(self.result_dir)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chronohorn observe serve")
    parser.add_argument("--port", type=int, default=7070)
    parser.add_argument("--result-dir", default="out/results")
    args = parser.parse_args(argv)

    Handler.result_dir = args.result_dir
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"chronohorn: http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0
