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


def _compute_drain_status(result_dir: str = "out/results") -> dict[str, Any]:
    """Compute overall drain status and ETA from results + manifests + fleet."""
    import math

    # Count results by wave prefix
    results = list(Path(result_dir).glob("*.json"))
    cs_done = sum(1 for r in results if r.stem.startswith("cs-"))
    ws_done = sum(1 for r in results if r.stem.startswith("ws-"))
    ex_done = sum(1 for r in results if r.stem.startswith("ex-"))

    # Count manifest totals
    manifest_counts = {}
    for p in Path("manifests").glob("frontier_*.jsonl"):
        lines = [l for l in p.read_text().splitlines() if l.strip() and not l.startswith("#")]
        manifest_counts[p.stem] = len(lines)
    cs_total = manifest_counts.get("frontier_compute_scaling", 0)
    ws_total = manifest_counts.get("frontier_workstream_matrix", 0)

    # Average walltime by step tier from completed results
    tier_times: dict[str, list[float]] = {"1k": [], "5k": [], "10k": []}
    for p in results:
        try:
            d = json.loads(p.read_text())
            steps = d.get("config", {}).get("train", {}).get("steps") or d.get("config", {}).get("steps") or 0
            wall = d.get("training", {}).get("performance", {}).get("elapsed_sec", 0)
            if steps and wall:
                tier = "1k" if steps <= 1500 else ("5k" if steps <= 6000 else "10k")
                tier_times[tier].append(wall)
        except (json.JSONDecodeError, OSError):
            pass

    avg_times = {t: (sum(v) / len(v) if v else 0) for t, v in tier_times.items()}

    # Estimate remaining time (assume balanced tier distribution, 2 GPU slots)
    cs_remaining = max(cs_total - cs_done, 0)
    ws_remaining = max(ws_total - ws_done, 0)
    total_remaining = cs_remaining + ws_remaining

    # Rough: 1/3 at each tier
    if total_remaining > 0 and any(avg_times.values()):
        avg_per_job = (avg_times.get("1k", 180) + avg_times.get("5k", 780) + avg_times.get("10k", 1800)) / 3
        eta_sec = total_remaining * avg_per_job / 2  # 2 GPU slots
    else:
        eta_sec = 0

    return {
        "cs": {"done": cs_done, "total": cs_total},
        "ws": {"done": ws_done, "total": ws_total},
        "ex": {"done": ex_done},
        "total_done": len(results),
        "remaining": total_remaining,
        "eta_min": round(eta_sec / 60),
        "eta_h": round(eta_sec / 3600, 1),
        "avg_1k": round(avg_times.get("1k", 0)),
        "avg_5k": round(avg_times.get("5k", 0)),
        "avg_10k": round(avg_times.get("10k", 0)),
    }


def _load_manifests() -> list[dict[str, Any]]:
    manifests = []
    for p in sorted(glob.glob("manifests/frontier_*.jsonl")):
        try:
            lines = [l for l in Path(p).read_text().splitlines() if l.strip() and not l.startswith("#")]
            manifests.append({"name": Path(p).stem, "path": p, "jobs": len(lines)})
        except OSError:
            pass
    return manifests


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
    if not params:
        return None
    return round(params * 6 / 8 / 1024 / 1024, 2)


def _is_illegal(r: dict[str, Any]) -> bool:
    """Detect results that leak future information (illegal for golf).

    NOTE: This is a filesystem-fallback version for the legacy _build_api_data path.
    The canonical illegal detection is db.record_result(), which writes the `illegal`
    column into the results table. The DB path reads that column directly.
    """
    name = r.get("_name", "")
    cfg = r.get("config", {})
    train = cfg.get("train", {}) if isinstance(cfg.get("train"), dict) else cfg
    # Check config if available
    patch_size = train.get("patch_size", 1)
    decoder = train.get("patch_causal_decoder", "NOT_SET")
    if patch_size > 1 and decoder in ("none", "NOT_SET"):
        return True
    # Heuristic: name contains "patch" but not "cpatch" (causal patch)
    # and the result has suspiciously low bpb (< 1.0 from a 2k-step run)
    if "patch" in name and "cpatch" not in name:
        bpb = r.get("model", {}).get("test_bpb", 99)
        steps = train.get("steps", 0)
        if bpb < 1.0 and steps <= 5000:
            return True
    return False


def _build_api_data(result_dir: str = "out/results", *, skip_fleet_probe: bool = False) -> dict[str, Any]:
    results = _load_all_results(result_dir)

    # Group by config prefix
    groups: dict[str, list[dict]] = {}
    for r in results:
        name = r["_name"]
        prefix = name
        for pat in ("-1000s", "-5000s", "-10000s", "-s5200", "-s10000", "-seed43", "-seed44"):
            prefix = prefix.replace(pat, "")
        groups.setdefault(prefix, []).append(r)

    # Merged learning curves with TFLOPs on each point
    curves = {}
    for prefix, group in groups.items():
        points = []
        for r in sorted(group, key=lambda x: _get_steps(x) or 0):
            probes = r.get("training", {}).get("probes", [])
            bpb = r.get("model", {}).get("test_bpb")
            steps = _get_steps(r)
            perf = r.get("training", {}).get("performance", {})
            step_flops = perf.get("train_step_flops_per_step_est", 0)
            for p in probes:
                step = p.get("step", 0)
                pbpb = p.get("bpb") or p.get("test_bpb")
                if pbpb and step:
                    tf = round(step * step_flops / 1e12, 1) if step_flops else 0
                    points.append({"step": step, "bpb": round(pbpb, 4), "tf": tf})
            if bpb and steps:
                tf = round(steps * step_flops / 1e12, 1) if step_flops else 0
                points.append({"step": steps, "bpb": round(bpb, 4), "tf": tf})
        seen = set()
        unique = []
        for p in sorted(points, key=lambda x: x["step"]):
            if p["step"] not in seen:
                seen.add(p["step"])
                unique.append(p)
        if unique:
            curves[prefix] = unique

    # Full leaderboard
    leaderboard = []
    for r in results:
        bpb = r.get("model", {}).get("test_bpb")
        if bpb:
            m = r.get("model", {})
            steps = _get_steps(r)
            perf = r.get("training", {}).get("performance", {})
            tflops_s = perf.get("estimated_sustained_tflops", 0)
            elapsed = perf.get("elapsed_sec", 0)
            toks = perf.get("tokens_per_second", 0)
            params = m.get("params")
            # Forecast data
            fc = r.get("forecast", {})
            proj = fc.get("projection", {})
            forecast_bpb = proj.get("forecast_metric_at_budget")
            marginal = proj.get("dbpb_dtotal_tflop")
            curve_r2 = proj.get("curve_model", {}).get("weighted_r2")

            illegal = _is_illegal(r)
            leaderboard.append({
                "name": r["_name"],
                "bpb": round(bpb, 4),
                "steps": steps,
                "tflops": round(tflops_s * elapsed, 1) if tflops_s and elapsed else 0,
                "tok_s": round(toks),
                "tf_s": round(tflops_s, 3),
                "wall": round(elapsed),
                "int6_mb": _int6_mb(params),
                "params": params,
                "scale": m.get("scale"),
                "readout": m.get("linear_readout_kind"),
                "seq": _get_train_field(r, "seq_len"),
                "batch": _get_train_field(r, "batch_size"),
                "osc": m.get("oscillatory_frac"),
                "window": m.get("local_window"),
                "overfit": round(m.get("overfit_pct", 0), 1) if m.get("overfit_pct") else None,
                "train_bpb": round(m.get("train_bpb"), 4) if m.get("train_bpb") else None,
                "lr": m.get("learning_rate"),
                "illegal": illegal,
                # Forecast
                "fc_bpb": round(forecast_bpb, 4) if forecast_bpb else None,
                "fc_marginal": round(marginal * 1e6, 2) if marginal else None,
                "fc_r2": round(curve_r2, 3) if curve_r2 else None,
            })
    leaderboard.sort(key=lambda x: x["bpb"])

    # Efficiency ranking
    efficiency = []
    for r in results:
        name = r["_name"]
        bpb = r.get("model", {}).get("test_bpb")
        steps = _get_steps(r)
        probes = r.get("training", {}).get("probes", [])
        perf = r.get("training", {}).get("performance", {})
        tflops_s = perf.get("estimated_sustained_tflops", 0)
        elapsed = perf.get("elapsed_sec", 0)
        fc = r.get("forecast", {})
        proj = fc.get("projection", {})
        if bpb and len(probes) >= 2 and tflops_s > 0:
            p1 = probes[-2]
            p2 = probes[-1]
            b1 = p1.get("bpb") or p1.get("test_bpb") or 0
            b2 = p2.get("bpb") or p2.get("test_bpb") or 0
            s1 = p1.get("step", 0)
            s2 = p2.get("step", 0)
            if b1 > b2 and s2 > s1:
                step_flops = perf.get("train_step_flops_per_step_est", 0)
                dt_tflops = (s2 - s1) * step_flops / 1e12 if step_flops else (s2 - s1) * tflops_s / max(perf.get("tokens_per_second", 1), 1)
                marginal = (b1 - b2) / max(dt_tflops, 1e-10)
                total_tflops = tflops_s * elapsed if elapsed else 0
                efficiency.append({
                    "name": name,
                    "bpb": round(bpb, 4),
                    "marginal": round(marginal * 1e6, 2),  # μbpb/TFLOP
                    "steps": steps,
                    "total_tf": round(total_tflops, 1),
                    "slope_alive": b2 < b1 - 0.001,
                    "fc_bpb": round(proj.get("forecast_metric_at_budget", 0), 4) if proj.get("forecast_metric_at_budget") else None,
                    "illegal": _is_illegal(r),
                })
    efficiency.sort(key=lambda x: -x["marginal"])

    # Config detail
    configs = []
    for r in results:
        m = r.get("model", {})
        bpb = m.get("test_bpb")
        if not bpb:
            continue
        configs.append({
            "name": r["_name"],
            "bpb": round(bpb, 4),
            "steps": _get_steps(r),
            "scale": m.get("scale"),
            "readout": m.get("linear_readout_kind"),
            "depth": m.get("linear_readout_depth", 1),
            "experts": m.get("linear_readout_num_experts"),
            "window": m.get("local_window"),
            "seq": _get_train_field(r, "seq_len"),
            "osc": m.get("oscillatory_frac"),
            "hlmax": m.get("linear_half_life_max"),
            "gate": m.get("static_bank_gate"),
            "mix": m.get("mix_mode"),
            "proj": m.get("input_proj_scheme") if m.get("input_proj_scheme") != "random" else None,
            "sched": m.get("oscillatory_schedule") if m.get("oscillatory_schedule") != "logspace" else None,
            "params": m.get("params"),
            "int6_mb": _int6_mb(m.get("params")),
            "lr": m.get("learning_rate"),
            "illegal": _is_illegal(r),
        })
    configs.sort(key=lambda x: x["bpb"])

    return {
        "n": len(results),
        "curves": curves,
        "board": leaderboard[:30],
        "eff": efficiency[:25],
        "fleet": {} if skip_fleet_probe else _probe_fleet(),
        "best": next((r for r in leaderboard if not r.get("illegal")), leaderboard[0] if leaderboard else None),
        "manifests": _load_manifests(),
        "configs": configs[:30],
        "drain": _compute_drain_status(result_dir),
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
<div class="pane" id="p-frontier"><table><thead><tr><th>#</th><th>run</th><th>bpb</th><th>steps</th><th>seq</th><th>TF</th><th>int6</th><th>fc bpb</th><th>fc R2</th><th>overfit</th></tr></thead><tbody id="tb-f"></tbody></table></div>
<div class="pane" id="p-fleet"><div id="fleet-content"></div></div>
<div class="pane" id="p-eff"><table><thead><tr><th>#</th><th>run</th><th>bpb</th><th>ubpb/TF</th><th>TF</th><th>alive</th><th>fc bpb</th></tr></thead><tbody id="tb-e"></tbody></table></div>
<div class="pane" id="p-config"><table><thead><tr><th>run</th><th>bpb</th><th>steps</th><th>s</th><th>readout</th><th>seq</th><th>w</th><th>osc</th><th>mix</th><th>proj</th><th>int6</th><th>lr</th></tr></thead><tbody id="tb-c"></tbody></table></div>
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
    if(r.illegal){tr.style.opacity='0.35';tr.style.textDecoration='line-through'}
    const bpbClass=r.illegal?'r':(i<3?'g':'b');
    [[i+1,'d'],[r.name+(r.illegal?' !!':''),''],[r.bpb.toFixed(4),bpbClass],
     [r.steps||'?','w'],[r.seq||'','d'],
     [r.tflops?r.tflops.toFixed(0):'','d'],
     [r.int6_mb?r.int6_mb.toFixed(1):'','d'],
     [r.fc_bpb?r.fc_bpb.toFixed(4):'','y'],
     [r.fc_r2?r.fc_r2.toFixed(3):'','d'],
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
    [[r.name+(r.illegal?' !!':''),''],[r.bpb.toFixed(4),r.illegal?'r':'b'],[r.steps||'','w'],[r.scale||'','w'],
     [r.readout||'','w'],[r.seq||'','w'],[r.window||'','d'],
     [r.osc!=null?r.osc:'','d'],[r.mix||'','d'],[r.proj||'','y'],
     [r.int6_mb?r.int6_mb.toFixed(1):'','d'],[r.lr||'','d']
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
    // Status bar — handle both runtime drain format and standalone format
    const dr=data.drain||{};
    if(dr.cs){
      const cs=dr.cs||{},ws=dr.ws||{};
      document.getElementById('sb-drain').innerHTML='cs <span class="active">'+(cs.done||0)+'/'+(cs.total||0)+'</span> ws <span class="active">'+(ws.done||0)+'/'+(ws.total||0)+'</span>';
      const eta=(dr.eta_h||0)>1?dr.eta_h+'h':(dr.eta_min||0)+'m';
      document.getElementById('sb-eta').innerHTML=(dr.remaining||0)>0?'<span class="eta">~'+eta+' left</span>':'done';
    } else if(dr.manifest){
      const p=dr.pending||0,r=dr.running||0,c=dr.completed||0;
      const mf=(dr.manifest||'').replace('frontier_','').replace('.jsonl','');
      document.getElementById('sb-drain').innerHTML=mf+' <span class="active">'+c+' done</span> '+r+' run '+p+' wait';
      document.getElementById('sb-eta').innerHTML=dr.done?'complete':(p+r>0?'<span class="eta">draining</span>':'idle');
    } else {
      document.getElementById('sb-drain').textContent=data.n+' results';
      document.getElementById('sb-eta').textContent='';
    }
    const fleet=data.fleet||{};
    let gpus=0;Object.values(fleet).forEach(h=>{gpus+=(h.containers||[]).length});
    document.getElementById('sb-fleet').textContent=gpus+' gpu'+(gpus!==1?'s':'')+' active';
  }catch(e){document.getElementById('ts').textContent='err'}
}
poll();setInterval(poll,15000);
window.addEventListener('resize',()=>{if(window._d&&curTab==='curves')drawCurves(window._d)});
</script>
</body>
</html>"""


def _build_api_data_from_db(db: "ChronohornDB") -> dict[str, Any]:
    """Build API data from ChronohornDB instead of filesystem scanning."""
    from chronohorn.db import ChronohornDB

    board = db.frontier(30)

    # Build curves from probes table
    # Get all unique result names, then fetch their probes
    all_results = db.query("SELECT DISTINCT name FROM probes")
    curves = {}
    for row in all_results:
        name = row["name"]
        points = db.learning_curve(name)
        if len(points) > 1:
            # Group by config prefix for merging
            prefix = name
            for pat in ("-1000s", "-5000s", "-10000s", "-s5200", "-s10000", "-seed43", "-seed44"):
                prefix = prefix.replace(pat, "")
            if prefix not in curves:
                curves[prefix] = []
            for p in points:
                curves[prefix].append({"step": p["step"], "bpb": round(p["bpb"], 4) if p["bpb"] else None})

    # Deduplicate curve points
    for prefix in curves:
        seen = set()
        unique = []
        for p in sorted(curves[prefix], key=lambda x: x["step"]):
            if p["step"] not in seen and p["bpb"]:
                seen.add(p["step"])
                unique.append(p)
        curves[prefix] = unique

    # Efficiency
    eff = db.marginal_rank(25)

    # Fleet
    fleet_data = db.fleet_latest()
    fleet = {}
    for host, info in fleet_data.items():
        containers = info.get("containers", [])
        fleet[host] = {
            "online": info.get("online", False),
            "containers": [{"name": c, "status": "running"} for c in containers] if isinstance(containers, list) else [],
        }

    # Events
    events = db.events_recent(30)

    # Best legal
    best = board[0] if board else None

    # Configs
    configs = db.query("""
        SELECT r.name, r.bpb, c.scale, c.readout, c.seq_len, c.num_blocks,
               c.substrate_mode, c.patch_size, c.patch_decoder, c.state_dim,
               c.local_window, c.oscillatory_frac, c.int6_mb, c.params,
               r.illegal
        FROM results r LEFT JOIN configs c ON r.config_id = c.id
        ORDER BY r.bpb LIMIT 30
    """)

    # Drain status from DB
    pending = db.query("SELECT COUNT(*) as c FROM jobs WHERE state = 'pending'")[0]["c"]
    running = db.query("SELECT COUNT(*) as c FROM jobs WHERE state IN ('dispatched', 'running')")[0]["c"]
    completed = db.query("SELECT COUNT(*) as c FROM jobs WHERE state = 'completed'")[0]["c"]

    # Manifests
    manifests = db.query("SELECT DISTINCT manifest, COUNT(*) as jobs FROM jobs GROUP BY manifest ORDER BY manifest")

    return {
        "n": db.result_count(),
        "curves": curves,
        "board": board,
        "eff": eff,
        "fleet": fleet,
        "best": best,
        "manifests": [{"name": m["manifest"], "jobs": m["jobs"]} for m in manifests if m["manifest"]],
        "configs": configs,
        "drain": {
            "pending": pending,
            "running": running,
            "completed": completed,
            "done": pending == 0 and running == 0,
        },
        "events": events,
    }


class Handler(BaseHTTPRequestHandler):
    result_dir = "out/results"
    tool_server = None  # Set by runtime to enable action endpoint
    db = None  # Set by runtime to enable DB-backed queries

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(_HTML.encode())
        elif self.path.startswith("/api/status"):
            if self.db is not None:
                data = _build_api_data_from_db(self.db)
            else:
                data = _build_api_data(self.result_dir, skip_fleet_probe=True)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
        elif self.path.startswith("/api/query"):
            from urllib.parse import parse_qs, urlparse
            params = parse_qs(urlparse(self.path).query)
            sql = params.get("sql", [""])[0]
            if not sql or not sql.strip().upper().startswith("SELECT"):
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Only SELECT queries allowed"}).encode())
                return
            if self.db is not None:
                try:
                    rows = self.db.query(sql)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"rows": rows, "count": len(rows)}).encode())
                except Exception as exc:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(exc)}).encode())
            else:
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "No database available"}).encode())
        elif self.path == "/api/tools":
            if self.tool_server:
                tools = self.tool_server.list_tools()
            else:
                tools = []
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"tools": [t["name"] for t in tools]}).encode())
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
                    result = {"error": "no tool server attached — run via chronohorn runtime"}
                else:
                    result = self.tool_server.call_tool(tool_name, tool_args)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
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
