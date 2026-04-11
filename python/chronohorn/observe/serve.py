"""Compact visualization server for chronohorn runtime state.

Tabs: Curves | Frontier | Fleet | Efficiency | Config | Manifests
Polls /api/status every 15s. No dependencies beyond stdlib.

The DB is always the source of truth. On startup, main() creates/opens
a ChronohornDB and rebuilds from archive if empty.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from chronohorn.fleet.hosts import DEFAULT_FLEET_HOSTS, probe_hosts
from chronohorn.service_log import configure_service_log, service_log

FLEET_HOSTS: tuple[str, ...] = DEFAULT_FLEET_HOSTS

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _probe_fleet(fleet_hosts: tuple[str, ...] | None = None) -> dict[str, Any]:
    fleet: dict[str, Any] = {}
    for info in probe_hosts(fleet_hosts or FLEET_HOSTS):
        host = str(info.get("host") or "")
        rows = []
        for row in info.get("container_rows") or []:
            name = str(row.get("name") or "")
            rows.append({"name": name.replace("chronohorn-", ""), "status": str(row.get("status") or "running")})
        fleet[host] = {"containers": rows, "online": bool(info.get("online", False))}
    return fleet


def _get_steps(r: dict) -> int | None:
    """Extract steps from result JSON, handling both config layouts."""
    cfg = r.get("config", {})
    # New layout: config.train.steps
    train = cfg.get("train")
    if isinstance(train, dict) and train.get("steps"):
        return int(train["steps"])
    # Old layout: config.steps
    if cfg.get("steps"):
        return int(cfg["steps"])
    # Also check training.performance.steps_completed
    perf_steps = r.get("training", {}).get("performance", {}).get("steps_completed")
    if perf_steps:
        return int(perf_steps)
    return None


def _get_train_field(r: dict, field: str) -> Any:
    """Extract a field from config.train or config."""
    cfg = r.get("config", {})
    train = cfg.get("train")
    if isinstance(train, dict) and train.get(field) is not None:
        return train[field]
    return cfg.get(field)


def _int6_mb(params: int | None) -> float | None:
    """Compute int6 artifact size from trainable param count."""
    if params is None:
        return None
    return round(params * 6 / 8 / 1024 / 1024, 2)


def _build_api_data(db) -> dict[str, Any]:
    """Build the canonical API payload from ChronohornDB.

    This is the single source of truth for the dashboard.
    """

    # Frontier (canonical shape from DB)
    board = db.frontier(30, trust="admissible")
    board_trust = "admissible"
    if not board:
        board = db.frontier(30, trust="provisional")
        board_trust = "provisional"

    # Curves from probes — single bulk query instead of N+1
    try:
        all_probes = db.query("SELECT name, step, bpb, tflops FROM probes ORDER BY name, step")
    except Exception as exc:
        service_log("observe.serve", "probes query failed", level="error", error=str(exc))
        all_probes = []

    from collections import defaultdict
    raw_curves: dict[str, list[dict]] = defaultdict(list)
    for p in all_probes:
        raw_curves[p["name"]].append({"step": p["step"], "bpb": p["bpb"], "tf": p.get("tflops") or 0})

    # Merge by prefix
    curves: dict[str, list[dict]] = {}
    for name, points in raw_curves.items():
        if len(points) < 2:
            continue
        prefix = name
        for pat in ("-1000s", "-5000s", "-10000s", "-s5200", "-s10000", "-seed43", "-seed44"):
            prefix = prefix.replace(pat, "")
        curves.setdefault(prefix, []).extend(points)

    # Deduplicate
    for prefix in curves:
        seen: set[int] = set()
        unique = []
        for p in sorted(curves[prefix], key=lambda x: x["step"]):
            if p["step"] not in seen and p.get("bpb"):
                seen.add(p["step"])
                unique.append(p)
        curves[prefix] = unique

    # Efficiency (canonical shape from DB)
    eff = db.marginal_rank(25)
    mutations = db.mutation_leaderboard(20, trust="all")

    # Fleet
    fleet_data = db.fleet_latest()
    fleet = {}
    for host, info in fleet_data.items():
        containers = info.get("containers", [])
        gpu = info.get("gpu", [])
        gpu_info = gpu[0] if isinstance(gpu, list) and gpu else {}
        fleet[host] = {
            "online": info.get("online", False),
            "gpu_util": gpu_info.get("util_pct", 0) if isinstance(gpu_info, dict) else 0,
            "gpu_mem": f"{gpu_info.get('mem_used_mb', 0)}/{gpu_info.get('mem_total_mb', 0)}" if isinstance(gpu_info, dict) else "",
            "containers": [{"name": c, "status": "running"} for c in containers] if isinstance(containers, list) else [],
        }

    # Best legal
    best = board[0] if board else None

    # Configs
    configs = []
    for entry in board[:30]:
        cfg_entry = {
            "name": entry.get("name"),
            "bpb": entry.get("bpb"),
            "family": entry.get("family", "unknown"),
            "steps": entry.get("steps"),
            "params": entry.get("params"),
            "int6_mb": entry.get("int6_mb"),
            "tok_s": entry.get("tok_s"),
            "illegal": entry.get("illegal"),
        }
        # Try to get config summary from adapter
        try:
            summary = db.config_summary(entry["name"])
            for k, v in summary.items():
                if v is not None:
                    cfg_entry.setdefault(k, v)
        except Exception as exc:
            service_log(
                "observe.serve",
                "config summary failed",
                level="error",
                name=entry.get("name"),
                error=str(exc),
            )
        # Enrich with config columns from the joined row
        cfg_json_str = entry.get("config_json")
        if cfg_json_str:
            try:
                import json as _json
                cfg_parsed = _json.loads(cfg_json_str) if isinstance(cfg_json_str, str) else cfg_json_str
                for k in ("substrate_mode", "readout", "linear_readout_kind", "readout_bands", "scale",
                          "state_dim", "num_heads", "local_window", "bands"):
                    if k in cfg_parsed and cfg_parsed[k] is not None:
                        cfg_entry.setdefault(k, cfg_parsed[k])
            except Exception:  # noqa: S110
                pass  # config enrichment is best-effort
        configs.append(cfg_entry)

    # Drain
    drain = db.drain_status()

    # Events
    events = db.events_recent(30)

    # Manifests
    manifests_raw = db.query("SELECT DISTINCT manifest, COUNT(*) as jobs FROM jobs GROUP BY manifest ORDER BY manifest")
    manifests = []
    for m in manifests_raw:
        if m["manifest"]:
            manifests.append({"name": Path(m["manifest"]).stem, "jobs": m["jobs"]})

    return {
        "n": db.result_count(),
        "curves": curves,
        "board": board,
        "board_trust": board_trust,
        "eff": eff,
        "mutations": mutations,
        "fleet": fleet,
        "best": best,
        "manifests": manifests,
        "configs": configs,
        "drain": drain,
        "events": events,
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
#status-bar{font-size:8px;color:#505050;padding:1px 4px;display:flex;gap:12px;flex-shrink:0}
#status-bar .active{color:#4caf50}
#status-bar .eta{color:#ffa726}
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
<div id="status-bar"><span id="sb-drain"></span><span id="sb-eta"></span><span id="sb-fleet"></span></div>
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
<div class="pane" id="p-frontier"><table><thead><tr><th>#</th><th>run</th><th>bpb</th><th>tok/s</th><th>steps</th><th>int6</th><th>eval</th><th>gpu</th><th>fc bpb</th><th>slope</th></tr></thead><tbody id="tb-f"></tbody></table></div>
<div class="pane" id="p-fleet"><div id="fleet-content"></div></div>
<div class="pane" id="p-eff"><table><thead><tr><th>#</th><th>run</th><th>bpb</th><th>ubpb/TF</th><th>TF</th><th>alive</th><th>fc bpb</th></tr></thead><tbody id="tb-e"></tbody></table></div>
<div class="pane" id="p-config"><table><thead><tr><th>run</th><th>bpb</th><th>substrate</th><th>readout</th><th>bands</th><th>scale</th><th>int6</th><th>tok/s</th></tr></thead><tbody id="tb-c"></tbody></table></div>
<div class="pane" id="p-manifests"><div id="mf-content"></div><div class="section" style="margin-top:8px">events</div><div id="ev-content" style="max-height:120px;overflow-y:auto"></div></div>
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
  const dpr=window.devicePixelRatio||1;
  cv.width=rect.width*dpr;cv.height=rect.height*dpr;
  const ctx=cv.getContext('2d');
  ctx.scale(dpr,dpr);
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
    const cidx=((sorted.length-top.length+ci)%C.length+C.length)%C.length;ctx.fillStyle=C[cidx];
    ctx.fillText(short+' '+last.bpb.toFixed(3),m.l+2,y-2);
  });
  // Dynamic reference: best result
  if(data.best&&data.best.bpb){
    const y=sy(data.best.bpb);
    if(y>m.t&&y<H-m.b){
      ctx.setLineDash([2,4]);ctx.strokeStyle='#2a3a2a';ctx.lineWidth=0.5;
      ctx.beginPath();ctx.moveTo(m.l,y);ctx.lineTo(W-m.r,y);ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle='#2a3a2a';ctx.font='6px monospace';
      ctx.fillText('best '+data.best.bpb.toFixed(3),W-m.r-50,y-2);
    }
  }
}

function updateFrontier(data){
  const tb=document.getElementById('tb-f');
  while(tb.firstChild)tb.removeChild(tb.firstChild);
  (data.board||[]).forEach((r,i)=>{
    const tr=document.createElement('tr');
    if(r.illegal){tr.style.opacity='0.35';tr.style.textDecoration='line-through'}
    const bpbClass=r.illegal?'r':(i<3?'g':'b');
    const toks=r.tok_s?Math.round(r.tok_s/1000)+'k':'';
    const evalB=r.eval_batches||'';
    const gpu=(r.accelerator_arch||'').replace('nvidia-','').replace('quadro-','Q:');
    const slope=r.slope?r.slope.toFixed(3):'';
    [[i+1,'d'],[r.name+(r.illegal?' !!':''),''],[r.bpb.toFixed(4),bpbClass],
     [toks,'w'],[r.steps||'?','w'],
     [r.int6_mb?r.int6_mb.toFixed(1):'','d'],
     [evalB,'d'],[gpu,'d'],
     [r.fc_bpb?r.fc_bpb.toFixed(4):'','y'],
     [slope,'d']
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
    h.appendChild(dot);
    const gpuInfo=info.gpu_util!=null?' gpu:'+info.gpu_util+'%':'';
    const memInfo=info.gpu_mem?' mem:'+info.gpu_mem+'MB':'';
    h.appendChild(document.createTextNode(host+gpuInfo+memInfo));el.appendChild(h);
    (info.containers||[]).forEach(c=>{
      totalRunning++;
      const j=document.createElement('div');j.className='job';j.textContent=c.name;
      const st=document.createElement('span');st.className='st';st.textContent=' '+c.status;
      j.appendChild(st);el.appendChild(j);
    });
    if(!info.containers||!info.containers.length){
      const j=document.createElement('div');j.className='job';
      j.textContent=info.gpu_util>0?'gpu active (no container)':'idle';
      j.className='job '+(info.gpu_util>0?'y':'d');
      el.appendChild(j);
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
    if(r.illegal){tr.style.opacity='0.35';tr.style.textDecoration='line-through'}
    [[i+1,'d'],[r.name+(r.illegal?' !!':''),''],[r.bpb.toFixed(4),r.illegal?'r':(i<3?'y':'b')],
     [r.marginal.toFixed(1),'y'],[r.total_tf||'','d'],
     [r.slope_alive?'yes':'flat',r.slope_alive?'g':'r'],
     [r.fc_bpb?r.fc_bpb.toFixed(4):'','d']
    ].forEach(([v,c])=>{const td=document.createElement('td');td.textContent=String(v);if(c)td.className=c;tr.appendChild(td)});
    tb.appendChild(tr);
  });
}

function updateConfig(data){
  const tb=document.getElementById('tb-c');
  while(tb.firstChild)tb.removeChild(tb.firstChild);
  (data.configs||[]).forEach(r=>{
    const tr=document.createElement('tr');
    if(r.illegal){tr.style.opacity='0.35';tr.style.textDecoration='line-through'}
    const sub=(r.substrate_mode||'frozen').replace('learnable_','L:').replace('gated_','G:');
    const ro=(r.readout||r.linear_readout_kind||'mlp').replace('routed_sqrelu_experts','exp').replace('tied_embed_readout','tied');
    const toks2=r.tok_s?Math.round(r.tok_s/1000)+'k':'';
    [[r.name+(r.illegal?' !!':''),''],[r.bpb?r.bpb.toFixed(4):'?',r.illegal?'r':'b'],
     [sub,'w'],[ro,'w'],[r.readout_bands||r.bands||'1','d'],
     [r.scale||'','d'],[r.int6_mb?r.int6_mb.toFixed(1):'','d'],[toks2,'d']
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

function updateEvents(data){
  const el=document.getElementById('ev-content');
  if(!el)return;
  while(el.firstChild)el.removeChild(el.firstChild);
  (data.events||[]).slice(-20).forEach(e=>{
    const d=document.createElement('div');d.className='job';
    const ts=new Date(e.ts*1000).toLocaleTimeString();
    d.textContent=ts+' '+e.event;
    if(e.data){const s=document.createElement('span');s.className='st';s.textContent=' '+JSON.stringify(JSON.parse(e.data||'{}'));d.appendChild(s)}
    el.appendChild(d);
  });
}

async function poll(){
  try{
    const r=await fetch('/api/status');
    const data=await r.json();window._d=data;
    if(curTab==='curves')drawCurves(data);
    updateFrontier(data);updateFleet(data);updateEfficiency(data);
    updateConfig(data);updateManifests(data);updateEvents(data);
    document.getElementById('ts').textContent=new Date().toLocaleTimeString();
    document.getElementById('nc').textContent=Object.keys(data.curves||{}).length;
    const hb=document.getElementById('hbest');
    if(data.best&&data.best.bpb!=null)hb.textContent=data.best.bpb.toFixed(4)+' bpb';
    // Status bar — canonical drain shape from DB
    const dr=data.drain||{};
    const p=dr.pending||0,r2=dr.running||0,c=dr.completed||0,t=dr.total||0;
    document.getElementById('sb-drain').innerHTML=c+'/'+t+' done, '+r2+' running, '+p+' pending';
    document.getElementById('sb-eta').innerHTML=dr.done?'complete':(p+r2>0?'<span class="eta">draining</span>':'idle');
    const fleet=data.fleet||{};
    let gpus=0,activeGpus=0;Object.values(fleet).forEach(h=>{gpus+=(h.containers||[]).length;if(h.gpu_util>0)activeGpus++});
    document.getElementById('sb-fleet').textContent=(activeGpus||gpus)+' gpu'+(gpus!==1?'s':'')+' active';
  }catch(e){document.getElementById('ts').textContent='err'}
}
poll();setInterval(poll,15000);
window.addEventListener('resize',()=>{if(window._d&&curTab==='curves')drawCurves(window._d)});
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    db = None  # Set at startup
    tool_server = None

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(_HTML.encode())
        elif self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        elif self.path.startswith("/api/status"):
            try:
                data = _build_api_data(self.db)
            except Exception as exc:
                data = {"error": str(exc), "n": 0, "curves": {}, "board": [], "eff": [], "fleet": {}, "best": None, "manifests": [], "configs": [], "drain": {}, "events": []}
            self._send_json(data)
        elif self.path == "/api/events":
            try:
                events = self.db.events_recent(30) if self.db else []
            except Exception as exc:
                service_log("observe.serve", "events query failed", level="error", error=str(exc))
                events = []
            self._send_json(events)
        elif self.path.startswith("/api/query"):
            from urllib.parse import parse_qs, urlparse
            params = parse_qs(urlparse(self.path).query)
            sql = params.get("sql", [""])[0]
            if not sql or not sql.strip().upper().startswith("SELECT"):
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Only SELECT queries allowed"}).encode())
                return
            if "limit" not in sql.lower():
                sql = sql.rstrip().rstrip(";") + " LIMIT 1000"
            if self.db is not None:
                try:
                    rows = self.db.query(sql)
                    self._send_json({"rows": rows, "count": len(rows)})
                except Exception as exc:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(exc)}).encode())
            else:
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "No database available"}).encode())
        elif self.path == "/api/tools":
            if self.tool_server:
                tools = self.tool_server.list_tools()
            else:
                tools = []
            self._send_json({"tools": [t["name"] for t in tools]})
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/action":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                req = json.loads(body)
                tool_name = req.get("tool", "")
                tool_args = req.get("args", {})
                if not self.tool_server:
                    result = {"error": "no tool server attached -- run via chronohorn runtime"}
                else:
                    result = self.tool_server.call_tool(tool_name, tool_args)
                self._send_json(result)
            except Exception as exc:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(exc)}).encode())
            return
        self.send_error(404)

    def do_OPTIONS(self):
        """Handle CORS preflight for action endpoint."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _send_json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

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

    # Always use DB
    from chronohorn.db import ChronohornDB
    db_path = Path(args.result_dir).parent / "chronohorn.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    configure_service_log(db_path.parent / "chronohorn.service.jsonl")
    db = ChronohornDB(str(db_path))

    # Rebuild from archive if DB is empty
    if db.result_count() == 0:
        count = db.rebuild_from_archive(args.result_dir)
        service_log("observe.serve", "rebuilt results from archive", results=count, result_dir=args.result_dir)

    # Also ingest manifests so drain status works
    manifests_dir = _PROJECT_ROOT / "manifests"
    if manifests_dir.exists():
        manifest_count = 0
        for p in sorted(manifests_dir.glob("frontier_*.jsonl")):
            try:
                manifest_count += db.ingest_manifest(str(p))
            except Exception as exc:
                service_log("observe.serve", "manifest ingest failed", level="error", manifest=p.name, error=str(exc))
        if manifest_count:
            service_log("observe.serve", "manifest jobs ingested", jobs=manifest_count)

    Handler.db = db

    from chronohorn.mcp import ToolServer
    Handler.tool_server = ToolServer(db=db)

    # Check port availability
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            s.connect(("127.0.0.1", args.port))
            service_log("observe.serve", "port already in use", level="error", port=args.port)
            return 1
    except (ConnectionRefusedError, OSError):
        pass  # port is free

    server = HTTPServer(("127.0.0.1", args.port), Handler)

    chrome_proc = None
    if not args.no_browser:
        chrome_proc = _launch_chrome_app(args.port, args.width, args.height)
        if chrome_proc:
            service_log("observe.serve", "app window opened", pid=chrome_proc.pid, port=args.port)
        else:
            service_log("observe.serve", "chrome not found", level="warning", url=f"http://127.0.0.1:{args.port}")
    else:
        service_log("observe.serve", "server ready", port=args.port, url=f"http://127.0.0.1:{args.port}")

    # Note: this server is localhost-only, so rate limiting and auth
    # (bugs #12, #13) are not needed for a single-user research tool.
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        service_log("observe.serve", "shutdown requested", level="warning")
    finally:
        if chrome_proc:
            chrome_proc.terminate()
        db.close()  # Flush writer thread and close connections
    return 0
