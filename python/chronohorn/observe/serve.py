"""Compact visualization server for chronohorn runtime state.

Tabs: Curves | Frontier | Fleet | Efficiency | Config | Manifests
Polls /api/status every 15s. No dependencies beyond stdlib.
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


def _load_manifests() -> list[dict[str, Any]]:
    manifests = []
    for p in sorted(glob.glob("manifests/frontier_*.jsonl")):
        try:
            lines = [l for l in Path(p).read_text().splitlines() if l.strip() and not l.startswith("#")]
            manifests.append({"name": Path(p).stem, "path": p, "jobs": len(lines)})
        except OSError:
            pass
    return manifests


def _build_api_data(result_dir: str = "out/results") -> dict[str, Any]:
    results = _load_all_results(result_dir)

    # Group by config prefix
    groups: dict[str, list[dict]] = {}
    for r in results:
        name = r["_name"]
        prefix = name
        for pat in ("-1000s", "-5000s", "-10000s", "-s5200", "-s10000", "-seed43", "-seed44"):
            prefix = prefix.replace(pat, "")
        groups.setdefault(prefix, []).append(r)

    # Merged learning curves
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
        seen = set()
        unique = []
        for p in sorted(points, key=lambda x: x["step"]):
            if p["step"] not in seen:
                seen.add(p["step"])
                unique.append(p)
        if unique:
            curves[prefix] = unique

    # Full leaderboard with config detail
    leaderboard = []
    for r in results:
        bpb = r.get("model", {}).get("test_bpb")
        if bpb:
            m = r.get("model", {})
            steps = r.get("config", {}).get("steps")
            perf = r.get("training", {}).get("performance", {})
            tflops_s = perf.get("estimated_sustained_tflops", 0)
            elapsed = perf.get("elapsed_sec", 0)
            toks = perf.get("tokens_per_second", 0)
            leaderboard.append({
                "name": r["_name"],
                "bpb": round(bpb, 4),
                "steps": steps,
                "tflops": round(tflops_s * elapsed, 1) if tflops_s and elapsed else 0,
                "tok_s": round(toks),
                "tf_s": round(tflops_s, 3),
                "wall": round(elapsed),
                "mb": round(m.get("payload_mb_est", 0), 1) if m.get("payload_mb_est") else None,
                "params": m.get("params"),
                "scale": m.get("scale"),
                "readout": m.get("linear_readout_kind"),
                "seq": r.get("config", {}).get("train", {}).get("seq_len") if isinstance(r.get("config", {}).get("train"), dict) else None,
                "osc": m.get("oscillatory_frac"),
                "window": m.get("local_window"),
                "overfit": round(m.get("overfit_pct", 0), 1) if m.get("overfit_pct") else None,
                "train_bpb": round(m.get("train_bpb", 0), 4) if m.get("train_bpb") else None,
            })
    leaderboard.sort(key=lambda x: x["bpb"])

    # Efficiency ranking
    efficiency = []
    for r in results:
        name = r["_name"]
        bpb = r.get("model", {}).get("test_bpb")
        probes = r.get("training", {}).get("probes", [])
        perf = r.get("training", {}).get("performance", {})
        tflops_s = perf.get("estimated_sustained_tflops", 0)
        elapsed = perf.get("elapsed_sec", 0)
        if bpb and len(probes) >= 2 and tflops_s > 0:
            p1 = probes[-2]
            p2 = probes[-1]
            b1 = p1.get("bpb") or p1.get("test_bpb") or 0
            b2 = p2.get("bpb") or p2.get("test_bpb") or 0
            s1 = p1.get("step", 0)
            s2 = p2.get("step", 0)
            if b1 > b2 and s2 > s1 and tflops_s > 0:
                dt_tflops = (s2 - s1) * (tflops_s / max(perf.get("tokens_per_second", 1), 1))
                marginal = (b1 - b2) / max(dt_tflops, 1e-10)
                total_tflops = tflops_s * elapsed if elapsed else 0
                efficiency.append({
                    "name": name,
                    "bpb": round(bpb, 4),
                    "marginal": round(marginal, 6),
                    "steps": r.get("config", {}).get("steps"),
                    "total_tf": round(total_tflops, 1),
                    "slope_alive": b2 < b1 - 0.001,
                })
    efficiency.sort(key=lambda x: -x["marginal"])

    # Config detail for each unique run
    configs = []
    for r in results:
        m = r.get("model", {})
        bpb = m.get("test_bpb")
        if not bpb:
            continue
        configs.append({
            "name": r["_name"],
            "bpb": round(bpb, 4),
            "scale": m.get("scale"),
            "readout": m.get("linear_readout_kind"),
            "depth": m.get("linear_readout_depth", 1),
            "experts": m.get("linear_readout_num_experts"),
            "window": m.get("local_window"),
            "osc": m.get("oscillatory_frac"),
            "hlmax": m.get("linear_half_life_max"),
            "gate": m.get("static_bank_gate"),
            "mix": m.get("mix_mode"),
            "proj": m.get("input_proj_scheme") if m.get("input_proj_scheme") != "random" else None,
            "sched": m.get("oscillatory_schedule") if m.get("oscillatory_schedule") != "logspace" else None,
            "params": m.get("params"),
            "mb": round(m.get("payload_mb_est", 0), 1) if m.get("payload_mb_est") else None,
        })
    configs.sort(key=lambda x: x["bpb"])

    return {
        "n": len(results),
        "curves": curves,
        "board": leaderboard[:30],
        "eff": efficiency[:25],
        "fleet": _probe_fleet(),
        "best": leaderboard[0] if leaderboard else None,
        "manifests": _load_manifests(),
        "configs": configs[:30],
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
#head .best{font-size:10px;color:#4caf50;margin-left:8px}
#head .ts{font-size:9px;color:#404040}
#tabs{display:flex;gap:1px;flex-shrink:0;margin:4px 0 2px}
.tab{font-size:9px;padding:3px 6px;background:#141414;border:1px solid #222;border-bottom:none;border-radius:3px 3px 0 0;cursor:pointer;color:#505050}
.tab.on{background:#1a1a1a;color:#c0c0c0;border-color:#333}
.tab .badge{font-size:8px;color:#404040;margin-left:2px}
#body{flex:1;background:#111;border:1px solid #222;border-radius:0 3px 3px 3px;overflow:hidden;position:relative}
.pane{position:absolute;inset:0;overflow:auto;display:none}
.pane.on{display:block}
.pane::-webkit-scrollbar{width:4px}
.pane::-webkit-scrollbar-thumb{background:#252525;border-radius:2px}
canvas{width:100%;height:100%;display:block}
table{width:100%;border-collapse:collapse;font-size:9px}
th{text-align:left;color:#454545;padding:2px 3px;border-bottom:1px solid #1e1e1e;font-weight:normal;font-size:8px;text-transform:uppercase;letter-spacing:.5px;position:sticky;top:0;background:#111}
td{padding:2px 3px;border-bottom:1px solid #151515;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:120px}
tr:hover td{background:#161616}
.g{color:#4caf50}.b{color:#42a5f5}.y{color:#ffa726}.r{color:#ef5350}.d{color:#333}.w{color:#888}
.dot{display:inline-block;width:5px;height:5px;border-radius:50%;margin-right:3px}
.dot.on{background:#4caf50}.dot.off{background:#333}
.host{font-size:9px;margin:6px 4px 2px;color:#606060;text-transform:uppercase;letter-spacing:.5px}
.job{font-size:9px;padding:1px 4px 1px 12px;color:#909090}
.job .st{color:#454545;font-size:8px}
.mf{font-size:9px;padding:3px 4px;display:flex;justify-content:space-between;border-bottom:1px solid #161616}
.mf .n{color:#909090}.mf .c{color:#42a5f5;font-size:8px}
.section{color:#454545;font-size:8px;text-transform:uppercase;letter-spacing:.5px;padding:6px 4px 2px;border-bottom:1px solid #1a1a1a}
</style>
</head>
<body>
<div id="root">
<div id="head"><div><h1 style="display:inline">CHRONOHORN</h1><span class="best" id="hbest"></span></div><span class="ts" id="ts">--</span></div>
<div id="tabs">
<div class="tab on" data-p="curves">curves<span class="badge" id="nc"></span></div>
<div class="tab" data-p="frontier">frontier</div>
<div class="tab" data-p="fleet">fleet</div>
<div class="tab" data-p="eff">bpb/tf</div>
<div class="tab" data-p="config">config</div>
<div class="tab" data-p="manifests">manifests</div>
</div>
<div id="body">
<div class="pane on" id="p-curves"><canvas id="cv"></canvas></div>
<div class="pane" id="p-frontier"><table><thead><tr><th>#</th><th>run</th><th>bpb</th><th>steps</th><th>TF</th><th>tok/s</th><th>wall</th><th>MB</th><th>overfit</th></tr></thead><tbody id="tb-f"></tbody></table></div>
<div class="pane" id="p-fleet"><div id="fleet-content"></div></div>
<div class="pane" id="p-eff"><table><thead><tr><th>#</th><th>run</th><th>bpb</th><th>marginal</th><th>TF total</th><th>alive</th></tr></thead><tbody id="tb-e"></tbody></table></div>
<div class="pane" id="p-config"><table><thead><tr><th>run</th><th>bpb</th><th>scale</th><th>readout</th><th>d</th><th>w</th><th>osc</th><th>mix</th><th>proj</th><th>sched</th><th>MB</th></tr></thead><tbody id="tb-c"></tbody></table></div>
<div class="pane" id="p-manifests"><div id="mf-content"></div></div>
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
    if(curTab==='curves'&&window._d)drawCurves(window._d);
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
  const m={l:30,r:4,t:4,b:14};
  const pw=W-m.l-m.r,ph=H-m.t-m.b;
  const lsMin=Math.log(sMin+1),lsMax=Math.log(sMax+1);
  const sx=s=>m.l+(Math.log(s+1)-lsMin)/(lsMax-lsMin)*pw;
  const bPad=0.03,bRange=bMax-bMin+bPad*2;
  const sy=b=>m.t+(1-(b-bMin+bPad)/bRange)*ph;
  ctx.strokeStyle='#1a1a1a';ctx.lineWidth=.5;
  ctx.fillStyle='#333';ctx.font='7px monospace';
  [50,100,500,1000,5000,10000].forEach(s=>{if(s>=sMin&&s<=sMax){const x=sx(s);ctx.beginPath();ctx.moveTo(x,m.t);ctx.lineTo(x,H-m.b);ctx.stroke();ctx.fillText(s>=1000?(s/1000)+'k':s,x-4,H-m.b+9)}});
  const bStep=bRange>1?.2:.1;
  for(let b=Math.ceil(bMin/bStep)*bStep;b<=bMax;b+=bStep){const y=sy(b);ctx.beginPath();ctx.moveTo(m.l,y);ctx.lineTo(W-m.r,y);ctx.stroke();ctx.fillText(b.toFixed(1),1,y+2)}
  // Reference lines
  ctx.setLineDash([3,3]);ctx.strokeStyle='#2a1a1a';ctx.lineWidth=.5;
  [1.898,2.078].forEach(ref=>{if(ref>=bMin&&ref<=bMax){const y=sy(ref);ctx.beginPath();ctx.moveTo(m.l,y);ctx.lineTo(W-m.r,y);ctx.stroke()}});
  ctx.setLineDash([]);
  const sorted=entries.map(([n,pts])=>[n,pts,pts[pts.length-1].bpb]).sort((a,b)=>b[2]-a[2]);
  sorted.forEach(([name,pts],ci)=>{
    pts.sort((a,b)=>a.step-b.step);
    ctx.strokeStyle=C[ci%C.length];ctx.lineWidth=1;ctx.globalAlpha=0.7;
    ctx.beginPath();pts.forEach((p,i)=>{const x=sx(p.step),y=sy(p.bpb);i===0?ctx.moveTo(x,y):ctx.lineTo(x,y)});ctx.stroke();
    const last=pts[pts.length-1];ctx.beginPath();ctx.arc(sx(last.step),sy(last.bpb),1.5,0,Math.PI*2);ctx.fillStyle=C[ci%C.length];ctx.fill();
    ctx.globalAlpha=1;
  });
  const top=sorted.slice(-10).reverse();
  let usedY=[];ctx.font='7px monospace';
  top.forEach(([name,pts],ci)=>{
    const last=pts[pts.length-1];let y=sy(last.bpb);
    for(const uy of usedY){if(Math.abs(y-uy)<8)y=uy+(y>uy?8:-8)}
    usedY.push(y);
    const short=name.length>22?name.slice(0,20)+'..':name;
    ctx.fillStyle=C[(sorted.length-10+ci)%C.length];
    ctx.fillText(short+' '+last.bpb.toFixed(3),m.l+2,y-2);
  });
}

function updateFrontier(data){
  const tb=document.getElementById('tb-f');
  while(tb.firstChild)tb.removeChild(tb.firstChild);
  (data.board||[]).forEach((r,i)=>{
    const tr=document.createElement('tr');
    [[i+1,'d'],[r.name,''],[r.bpb.toFixed(4),i<3?'g':'b'],[r.steps||'?','d'],
     [r.tflops?r.tflops.toFixed(0):'','d'],[r.tok_s||'','d'],
     [r.wall?r.wall+'s':'','d'],[r.mb||'','d'],
     [r.overfit!=null?r.overfit+'%':'','d']
    ].forEach(([v,c])=>{const td=document.createElement('td');td.textContent=String(v);if(c)td.className=c;tr.appendChild(td)});
    tb.appendChild(tr);
  });
}

function updateFleet(data){
  const el=document.getElementById('fleet-content');
  while(el.firstChild)el.removeChild(el.firstChild);
  const fleet=data.fleet||{};let totalRunning=0;
  Object.entries(fleet).forEach(([host,info])=>{
    const h=document.createElement('div');h.className='host';
    const dot=document.createElement('span');dot.className='dot '+(info.online?'on':'off');
    h.appendChild(dot);h.appendChild(document.createTextNode(host));el.appendChild(h);
    (info.containers||[]).forEach(c=>{
      totalRunning++;
      const j=document.createElement('div');j.className='job';j.textContent=c.name;
      const st=document.createElement('span');st.className='st';st.textContent=' '+c.status;
      j.appendChild(st);el.appendChild(j);
    });
    if(!info.containers||!info.containers.length){
      const j=document.createElement('div');j.className='job d';j.textContent='idle';el.appendChild(j);
    }
  });
  const s=document.createElement('div');s.className='section';
  s.textContent=data.n+' results | '+totalRunning+' gpu'+(totalRunning!==1?'s':'')+' active';
  if(data.best)s.textContent+=' | best '+data.best.bpb.toFixed(4);
  el.appendChild(s);
}

function updateEfficiency(data){
  const tb=document.getElementById('tb-e');
  while(tb.firstChild)tb.removeChild(tb.firstChild);
  (data.eff||[]).forEach((r,i)=>{
    const tr=document.createElement('tr');
    [[i+1,'d'],[r.name,''],[r.bpb.toFixed(4),i<3?'y':'b'],
     [r.marginal.toFixed(4),'y'],[r.total_tf||'','d'],
     [r.slope_alive?'yes':'flat',r.slope_alive?'g':'r']
    ].forEach(([v,c])=>{const td=document.createElement('td');td.textContent=String(v);if(c)td.className=c;tr.appendChild(td)});
    tb.appendChild(tr);
  });
}

function updateConfig(data){
  const tb=document.getElementById('tb-c');
  while(tb.firstChild)tb.removeChild(tb.firstChild);
  (data.configs||[]).forEach(r=>{
    const tr=document.createElement('tr');
    [[r.name,''],[r.bpb.toFixed(4),'b'],[r.scale||'','w'],[r.readout||'','w'],
     [r.depth||'','d'],[r.window||'','d'],[r.osc!=null?r.osc:'','d'],
     [r.mix||'','d'],[r.proj||'','y'],[r.sched||'','y'],[r.mb||'','d']
    ].forEach(([v,c])=>{const td=document.createElement('td');td.textContent=String(v!=null?v:'');if(c)td.className=c;tr.appendChild(td)});
    tb.appendChild(tr);
  });
}

function updateManifests(data){
  const el=document.getElementById('mf-content');
  while(el.firstChild)el.removeChild(el.firstChild);
  (data.manifests||[]).forEach(m=>{
    const d=document.createElement('div');d.className='mf';
    const n=document.createElement('span');n.className='n';n.textContent=m.name.replace('frontier_','');
    const c=document.createElement('span');c.className='c';c.textContent=m.jobs+' jobs';
    d.appendChild(n);d.appendChild(c);el.appendChild(d);
  });
}

async function poll(){
  try{
    const r=await fetch('/api/status');
    const data=await r.json();window._d=data;
    if(curTab==='curves')drawCurves(data);
    updateFrontier(data);updateFleet(data);updateEfficiency(data);
    updateConfig(data);updateManifests(data);
    document.getElementById('ts').textContent=new Date().toLocaleTimeString();
    document.getElementById('nc').textContent=Object.keys(data.curves||{}).length;
    const hb=document.getElementById('hbest');
    if(data.best)hb.textContent=data.best.bpb.toFixed(4)+' bpb';
  }catch(e){document.getElementById('ts').textContent='err'}
}
poll();setInterval(poll,15000);
window.addEventListener('resize',()=>{if(window._d&&curTab==='curves')drawCurves(window._d)});
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


def _find_chrome() -> str | None:
    import shutil
    candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        shutil.which("google-chrome-stable"),
        shutil.which("google-chrome"),
        shutil.which("chromium"),
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    return None


def _launch_chrome_app(port: int, width: int = 420, height: int = 520) -> subprocess.Popen | None:
    chrome = _find_chrome()
    if not chrome:
        return None
    return subprocess.Popen([
        chrome,
        f"--app=http://127.0.0.1:{port}",
        f"--window-size={width},{height}",
        "--disable-extensions",
        "--disable-sync",
        "--no-first-run",
        "--no-default-browser-check",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chronohorn observe serve")
    parser.add_argument("--port", type=int, default=7070)
    parser.add_argument("--result-dir", default="out/results")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open Chrome window")
    parser.add_argument("--width", type=int, default=420)
    parser.add_argument("--height", type=int, default=520)
    args = parser.parse_args(argv)

    Handler.result_dir = args.result_dir
    server = HTTPServer(("127.0.0.1", args.port), Handler)

    chrome_proc = None
    if not args.no_browser:
        chrome_proc = _launch_chrome_app(args.port, args.width, args.height)
        if chrome_proc:
            print(f"chronohorn: app window opened (pid {chrome_proc.pid})")
        else:
            print(f"chronohorn: chrome not found, open http://127.0.0.1:{args.port}")
    else:
        print(f"chronohorn: http://127.0.0.1:{args.port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if chrome_proc:
            chrome_proc.terminate()
    return 0
