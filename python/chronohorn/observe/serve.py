"""Lightweight visualization server for chronohorn runtime state.

Serves a single HTML page that polls /api/status and renders:
- Fleet panel: running/completed/blocked jobs
- Learning curves: bpb vs step for active configs
- Frontier leaderboard: best runs sorted by bpb
"""
from __future__ import annotations

import json
import glob
import argparse
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


def _build_api_status(result_dir: str = "out/results") -> dict[str, Any]:
    results = _load_all_results(result_dir)

    # Build learning curves
    curves = {}
    for r in results:
        name = r["_name"]
        probes = r.get("training", {}).get("probes", [])
        bpb = r.get("model", {}).get("test_bpb")
        steps = r.get("config", {}).get("train", {}).get("steps") if isinstance(r.get("config", {}).get("train"), dict) else r.get("config", {}).get("steps")
        perf = r.get("training", {}).get("performance", {})
        tflops_s = perf.get("estimated_sustained_tflops", 0)

        points = []
        for p in probes:
            step = p.get("step", 0)
            pbpb = p.get("bpb") or p.get("test_bpb")
            if pbpb and step:
                points.append({"step": step, "bpb": round(pbpb, 4)})
        if bpb and steps:
            points.append({"step": steps, "bpb": round(bpb, 4)})

        curves[name] = {
            "points": points,
            "final_bpb": round(bpb, 4) if bpb else None,
            "steps": steps,
            "tflops_s": round(tflops_s, 3) if tflops_s else 0,
        }

    # Build leaderboard
    leaderboard = []
    for r in results:
        bpb = r.get("model", {}).get("test_bpb")
        if bpb:
            steps = r.get("config", {}).get("train", {}).get("steps") if isinstance(r.get("config", {}).get("train"), dict) else r.get("config", {}).get("steps")
            leaderboard.append({
                "name": r["_name"],
                "bpb": round(bpb, 4),
                "steps": steps,
            })
    leaderboard.sort(key=lambda x: x["bpb"])

    return {
        "total_results": len(results),
        "curves": curves,
        "leaderboard": leaderboard[:20],
    }


def _escape(text: str) -> str:
    """Escape HTML special characters to prevent injection."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Chronohorn</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'SF Mono', 'Fira Code', monospace; background: #0a0a0a; color: #c0c0c0; padding: 16px; }
h1 { color: #e0e0e0; font-size: 16px; margin-bottom: 12px; }
h2 { color: #909090; font-size: 13px; margin: 12px 0 6px; text-transform: uppercase; letter-spacing: 1px; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.panel { background: #141414; border: 1px solid #252525; border-radius: 4px; padding: 12px; }
.full { grid-column: 1 / -1; }
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th { text-align: left; color: #707070; padding: 4px 8px; border-bottom: 1px solid #252525; }
td { padding: 4px 8px; border-bottom: 1px solid #1a1a1a; }
.bpb { color: #4fc3f7; font-weight: bold; }
.best { color: #66bb6a; }
.dim { color: #505050; }
canvas { width: 100%; height: 300px; }
#status { color: #505050; font-size: 11px; position: fixed; top: 8px; right: 16px; }
</style>
</head>
<body>
<h1>chronohorn</h1>
<div id="status">loading...</div>
<div class="grid">
  <div class="panel full">
    <h2>Learning Curves</h2>
    <canvas id="curves" width="1200" height="300"></canvas>
  </div>
  <div class="panel">
    <h2>Frontier</h2>
    <table><thead><tr><th>rank</th><th>name</th><th>bpb</th><th>steps</th></tr></thead>
    <tbody id="leaderboard"></tbody></table>
  </div>
  <div class="panel">
    <h2>Status</h2>
    <div id="summary"></div>
  </div>
</div>
<script>
const COLORS = ['#4fc3f7','#66bb6a','#ffa726','#ef5350','#ab47bc','#26c6da','#d4e157','#ec407a','#8d6e63','#78909c','#ff7043','#5c6bc0'];

function esc(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function drawCurves(data) {
  const canvas = document.getElementById('curves');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const entries = Object.entries(data.curves).filter(([_,v]) => v.points.length > 1);
  if (!entries.length) return;

  let minStep = Infinity, maxStep = 0, minBpb = Infinity, maxBpb = 0;
  for (const [_, v] of entries) {
    for (const p of v.points) {
      minStep = Math.min(minStep, p.step);
      maxStep = Math.max(maxStep, p.step);
      minBpb = Math.min(minBpb, p.bpb);
      maxBpb = Math.max(maxBpb, p.bpb);
    }
  }
  const pad = {l:50, r:20, t:10, b:30};
  const pw = W-pad.l-pad.r, ph = H-pad.t-pad.b;
  const sx = s => pad.l + (Math.log(s+1)-Math.log(minStep+1))/(Math.log(maxStep+1)-Math.log(minStep+1))*pw;
  const sy = b => pad.t + (1-(b-minBpb+0.02)/(maxBpb-minBpb+0.04))*ph;

  // Axes
  ctx.strokeStyle = '#252525'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, H-pad.b); ctx.lineTo(W-pad.r, H-pad.b); ctx.stroke();
  ctx.fillStyle = '#505050'; ctx.font = '10px monospace';
  for (const s of [100,1000,5000,10000]) {
    if (s >= minStep && s <= maxStep) {
      const x = sx(s);
      ctx.fillText(s>=1000?s/1000+'k':String(s), x-8, H-pad.b+14);
      ctx.beginPath(); ctx.moveTo(x,pad.t); ctx.lineTo(x,H-pad.b); ctx.strokeStyle='#1a1a1a'; ctx.stroke();
    }
  }
  for (let b = Math.ceil(minBpb*10)/10; b <= maxBpb; b += 0.1) {
    const y = sy(b);
    ctx.fillText(b.toFixed(1), 4, y+3);
    ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(W-pad.r,y); ctx.strokeStyle='#1a1a1a'; ctx.stroke();
  }

  // Curves
  let ci = 0;
  for (const [name, v] of entries) {
    const pts = v.points.sort((a,b) => a.step - b.step);
    ctx.strokeStyle = COLORS[ci % COLORS.length];
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    pts.forEach((p,i) => { const x=sx(p.step), y=sy(p.bpb); i===0?ctx.moveTo(x,y):ctx.lineTo(x,y); });
    ctx.stroke();
    // Label at end
    const last = pts[pts.length-1];
    ctx.fillStyle = COLORS[ci % COLORS.length];
    ctx.font = '9px monospace';
    const label = name.length > 25 ? name.slice(0,25)+'...' : name;
    ctx.fillText(label + ' ' + last.bpb.toFixed(3), sx(last.step)+4, sy(last.bpb)+3);
    ci++;
  }
}

function updateLeaderboard(data) {
  const tb = document.getElementById('leaderboard');
  while (tb.firstChild) tb.removeChild(tb.firstChild);
  data.leaderboard.forEach((r, i) => {
    const tr = document.createElement('tr');
    const tdRank = document.createElement('td');
    tdRank.className = 'dim';
    tdRank.textContent = String(i + 1);
    const tdName = document.createElement('td');
    tdName.textContent = r.name;
    const tdBpb = document.createElement('td');
    tdBpb.className = i < 3 ? 'best' : 'bpb';
    tdBpb.textContent = r.bpb.toFixed(4);
    const tdSteps = document.createElement('td');
    tdSteps.className = 'dim';
    tdSteps.textContent = r.steps ? String(r.steps) : '?';
    tr.appendChild(tdRank);
    tr.appendChild(tdName);
    tr.appendChild(tdBpb);
    tr.appendChild(tdSteps);
    tb.appendChild(tr);
  });
}

function updateSummary(data) {
  const el = document.getElementById('summary');
  while (el.firstChild) el.removeChild(el.firstChild);
  const pCount = document.createElement('p');
  pCount.textContent = data.total_results + ' results';
  el.appendChild(pCount);
  const best = data.leaderboard[0];
  if (best) {
    const pBest = document.createElement('p');
    pBest.textContent = 'best: ';
    const span = document.createElement('span');
    span.className = 'best';
    span.textContent = best.bpb.toFixed(4);
    pBest.appendChild(span);
    pBest.appendChild(document.createTextNode(' (' + best.name + ')'));
    el.appendChild(pBest);
  }
}

async function poll() {
  try {
    const r = await fetch('/api/status');
    const data = await r.json();
    drawCurves(data);
    updateLeaderboard(data);
    updateSummary(data);
    document.getElementById('status').textContent = 'updated ' + new Date().toLocaleTimeString();
  } catch(e) {
    document.getElementById('status').textContent = 'error: ' + e.message;
  }
}

poll();
setInterval(poll, 30000);
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
            data = _build_api_status(self.result_dir)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # suppress access logs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chronohorn observe serve")
    parser.add_argument("--port", type=int, default=7070)
    parser.add_argument("--result-dir", default="out/results")
    args = parser.parse_args(argv)

    Handler.result_dir = args.result_dir
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"chronohorn visualization: http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0
