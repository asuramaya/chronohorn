"""The big picture — one page: the frontier ladder, live runs, the machine.

The old observe/serve.py chased competition bpb across six tabs and a fleet
of hosts that no longer exists. This is its potato-first successor: one host,
one GPU, one question — where is the frontier and what is moving it right now.

Reads only what training already writes (result JSONs, probes.jsonl, fleet
logs, nvidia-smi); no DB, no dependencies beyond stdlib.

    python -m chronohorn.observe.big_picture --port 8321
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "out" / "results"
FLEET_LOGS = ROOT / "out" / "fleet"
STAGING = ROOT / "data" / "staging" / "pentad"

_PROGRESS_RE = re.compile(
    r"training progress .*?loss=(?P<loss>[\d.]+) step=(?P<step>\d+) .*?"
    r"tokens_per_second=(?P<tps>[\d.]+)")
_STEP_SUFFIX_RE = re.compile(r"_step\d+$")

# Static rungs — references that never move. Live rungs are computed.
STATIC_RUNGS = [
    (8.00, "untrained genome (seed-42 init)", "sanity floor"),
    (5.0, "marginal byte entropy", "order-0 reference"),
    (2.97, "frozen substrate + whitened kNN, zero training", "the innate ceiling"),
    (2.37, "order-5 n-gram floor", "classical reference"),
    (1.0, "Decepticon-1 target", "the north star"),
]


def _tail(path: Path, n_bytes: int = 8192) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(max(fh.seek(0, 2) - n_bytes, 0))
            return fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _cmdline_args() -> list[str]:
    """Command lines of live trainer processes (for --steps/--seq-len/--batch-size)."""
    try:
        out = subprocess.run(["pgrep", "-af", "train-causal-bank-torch|train_causal_bank_torch"],
                             capture_output=True, text=True, timeout=5).stdout
    except OSError:
        return []
    return [line for line in out.splitlines() if "--steps" in line]


def _flag(cmd: str, name: str) -> str | None:
    m = re.search(rf"--{name}\s+(\S+)", cmd)
    return m.group(1) if m else None


def _live_runs() -> list[dict]:
    now = time.time()
    cmds = _cmdline_args()
    runs = []
    for log in sorted(FLEET_LOGS.glob("*.log")):
        if now - log.stat().st_mtime > 180:
            continue
        name = log.stem
        m = None
        for line in reversed(_tail(log).splitlines()):
            m = _PROGRESS_RE.search(line)
            if m:
                break
        if not m:
            continue
        step, tps, loss = int(m["step"]), float(m["tps"]), float(m["loss"])
        cmd = next((c for c in cmds if f"{name}.json" in c), "")
        steps_total = int(_flag(cmd, "steps") or 0)
        seq = int(_flag(cmd, "seq-len") or 0)
        batch = int(_flag(cmd, "batch-size") or 0)
        eta_s = ((steps_total - step) * seq * batch / tps
                 if steps_total and seq and batch and tps else None)
        probe = None
        plines = _tail(RESULTS / f"{name}.probes.jsonl").strip().splitlines()
        curve = []
        for pl in plines[-40:]:
            try:
                p = json.loads(pl)
                curve.append({"step": p["step"], "bpb": round(p["bpb"], 4)})
            except (json.JSONDecodeError, KeyError):
                continue
        if curve:
            probe = curve[-1]
        runs.append({
            "name": name, "step": step, "steps_total": steps_total or None,
            "tok_s": round(tps), "loss": round(loss, 4),
            "eta_min": round(eta_s / 60, 1) if eta_s else None,
            "probe": probe, "curve": curve,
        })
    return runs


def _results() -> list[dict]:
    rows = []
    for p in RESULTS.glob("*.json"):
        if _STEP_SUFFIX_RE.search(p.stem) or p.stem.endswith(".probes"):
            continue
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        m = d.get("model") or {}
        ds = d.get("dataset") or {}
        # The ladder is BYTE bpb on the causal-bank family. sp1024 runs report
        # token-basis "bpb" (the graveyard's sub-1.78 confound) and polyhash
        # is a different organism entirely — both would fake a frontier.
        if m.get("architecture") != "causal_bank":
            continue
        if ds.get("tokenizer") not in (None, "bytes"):
            continue
        bpb = m.get("test_bpb")
        if not isinstance(bpb, (int, float)) or not 0.5 < bpb < 9:
            continue
        src = ds.get("source_path") or ""
        rows.append({
            "name": p.stem, "bpb": round(bpb, 4),
            "params": m.get("params"),
            "adaptive": m.get("adaptive_substrate"),
            "data": src.split("::")[0].strip().replace("data/roots/", "").split("/*")[0],
            "steps": (d.get("config") or {}).get("train", {}).get("steps"),
            "mtime": int(p.stat().st_mtime),
        })
    rows.sort(key=lambda r: r["mtime"], reverse=True)
    return rows


def _gpu() -> dict | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
        util, used, total = [float(x) for x in out[0].split(",")]
        return {"util_pct": util, "mem_used_mb": used, "mem_total_mb": total}
    except (OSError, ValueError, IndexError):
        return None


def _staging() -> list[dict]:
    now = time.time()
    rows = []
    if STAGING.is_dir():
        for raw in sorted(STAGING.glob("*.raw")):
            st = raw.stat()
            if now - st.st_mtime > 3600 and st.st_size >= 380_000_000:
                continue  # finished long ago — not worth a panel
            rows.append({"fold": raw.stem, "mb": round(st.st_size / 1e6),
                         "target_mb": 400, "live": now - st.st_mtime < 120})
    return rows


def build_status() -> dict:
    results = _results()
    frontier = min(results, key=lambda r: r["bpb"]) if results else None
    small = [r for r in results if isinstance(r.get("params"), int) and r["params"] < 8_000_000]
    efficiency = min(small, key=lambda r: r["bpb"]) if small else None
    return {
        "generated": int(time.time()),
        "live": _live_runs(),
        "gpu": _gpu(),
        "staging": _staging(),
        "frontier": frontier,
        "efficiency": efficiency,
        "static_rungs": [{"bpb": b, "what": w, "note": n} for b, w, n in STATIC_RUNGS],
        "results": results[:20],
    }


_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Daat — big picture</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root { --bg:#faf8f4; --ink:#23211c; --faint:#7a756a; --rule:#e2ddd2;
  --accent:#8a4b2d; --good:#2e6e4e; --card:#f1ede4;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace; }
@media (prefers-color-scheme: dark) { :root { --bg:#16150f; --ink:#e8e4da;
  --faint:#96917f; --rule:#33301f; --accent:#d98d5f; --good:#6fbf8f; --card:#201e15; } }
body { background:var(--bg); color:var(--ink);
  font:15px/1.5 Georgia,serif; max-width:52rem; margin:0 auto; padding:1.6rem 1rem 4rem; }
h1 { font-size:1.45rem; margin:0; } .sub { color:var(--faint); font-size:.85rem; margin-bottom:1.4rem; }
h2 { font-size:.78rem; text-transform:uppercase; letter-spacing:.12em; color:var(--accent);
  margin:1.8rem 0 .6rem; font-family:var(--mono); }
.card { background:var(--card); border-left:3px solid var(--accent); padding:.7rem .9rem;
  margin:.4rem 0; border-radius:0 4px 4px 0; }
.mono { font-family:var(--mono); font-variant-numeric:tabular-nums; }
.big { font-size:1.5rem; font-weight:700; color:var(--accent); }
.bar { height:6px; background:var(--rule); border-radius:3px; margin-top:.4rem; overflow:hidden; }
.bar>div { height:100%; background:var(--accent); }
.bar.g>div { background:var(--good); }
.faint { color:var(--faint); font-size:.83rem; }
table { border-collapse:collapse; width:100%; font-size:.85rem; }
td,th { padding:.35rem .5rem; border-top:1px solid var(--rule); text-align:left; }
th { color:var(--faint); font-family:var(--mono); font-size:.72rem; text-transform:uppercase;
  letter-spacing:.08em; border-top:none; }
td.n { font-family:var(--mono); font-variant-numeric:tabular-nums; }
.rung { display:grid; grid-template-columns:4.6rem 1fr; gap:.8rem; padding:.35rem .6rem;
  border-left:3px solid var(--rule); align-items:baseline; }
.rung .b { font-family:var(--mono); text-align:right; color:var(--faint); }
.rung.hot { border-left-color:var(--accent); background:var(--card); }
.rung.hot .b { color:var(--accent); font-weight:700; }
.rung.target { border-left-color:var(--good); } .rung.target .b { color:var(--good); font-weight:700; }
.spark { width:100%; height:44px; margin-top:.3rem; }
.overflow { overflow-x:auto; }
#stale { display:none; color:var(--accent); font-family:var(--mono); font-size:.8rem; }
</style></head><body>
<h1>Daat — big picture</h1>
<div class="sub">frontier · live runs · the machine — refreshes every 10s ·
<span id="ts" class="mono"></span> <span id="stale">STALE — server unreachable</span></div>
<div id="live-section"></div>
<div id="staging-section"></div>
<h2>The ladder</h2><div id="ladder"></div>
<h2>Recent results</h2><div class="overflow"><table id="results"></table></div>
<script>
const fmt = (x, d=0) => x == null ? "—" : Number(x).toFixed(d);
function spark(curve) {
  if (!curve || curve.length < 2) return "";
  const w = 560, h = 44, xs = curve.map(p => p.step), ys = curve.map(p => p.bpb);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const pts = curve.map(p => `${((p.step - x0) / (x1 - x0 || 1) * w).toFixed(1)},${(h - 4 - (p.bpb - y0) / (y1 - y0 || 1) * (h - 8)).toFixed(1)}`).join(" ");
  return `<svg class="spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
    <polyline points="${pts}" fill="none" stroke="var(--accent)" stroke-width="1.6"/></svg>`;
}
function render(s) {
  document.getElementById("ts").textContent = new Date(s.generated * 1000).toLocaleTimeString();
  let lv = "";
  if (s.live.length) {
    lv += "<h2>Live now</h2>";
    for (const r of s.live) {
      const pct = r.steps_total ? (100 * r.step / r.steps_total) : 0;
      lv += `<div class="card"><b>${r.name}</b>
        <span class="faint mono"> step ${r.step.toLocaleString()}${r.steps_total ? " / " + r.steps_total.toLocaleString() : ""}
        · ${r.tok_s.toLocaleString()} tok/s · loss ${fmt(r.loss, 3)}
        ${r.eta_min != null ? " · ~" + fmt(r.eta_min) + " min left" : ""}</span>
        ${r.probe ? `<div>held-out <span class="big">${fmt(r.probe.bpb, 3)}</span>
          <span class="faint mono">bpb @ step ${r.probe.step.toLocaleString()}</span></div>` : ""}
        ${spark(r.curve)}
        <div class="bar"><div style="width:${pct}%"></div></div></div>`;
    }
  } else { lv = "<h2>Live now</h2><div class='faint'>GPU idle — no active training run.</div>"; }
  if (s.gpu) lv += `<div class="faint mono">GPU ${fmt(s.gpu.util_pct)}% ·
    ${fmt(s.gpu.mem_used_mb / 1024, 1)} / ${fmt(s.gpu.mem_total_mb / 1024, 1)} GB</div>`;
  document.getElementById("live-section").innerHTML = lv;

  let st = "";
  if (s.staging.length) {
    st += "<h2>Dataset assembly</h2>";
    for (const f of s.staging) {
      st += `<div class="card"><b>${f.fold}</b> <span class="faint mono">${f.mb} / ${f.target_mb} MB
        ${f.live ? "· streaming" : "· idle"}</span>
        <div class="bar g"><div style="width:${Math.min(100, 100 * f.mb / f.target_mb)}%"></div></div></div>`;
    }
  }
  document.getElementById("staging-section").innerHTML = st;

  const rungs = [...s.static_rungs];
  if (s.frontier) rungs.push({bpb: s.frontier.bpb, what: `<b>THE FRONTIER — ${s.frontier.name}</b>`,
    note: `${(s.frontier.params / 1e6).toFixed(1)}M params · ${s.frontier.data}`, cls: "hot"});
  if (s.efficiency && (!s.frontier || s.efficiency.name !== s.frontier.name))
    rungs.push({bpb: s.efficiency.bpb, what: `efficiency arm — ${s.efficiency.name}`,
      note: `${(s.efficiency.params / 1e6).toFixed(1)}M params · ${s.efficiency.data}`});
  rungs.sort((a, b) => b.bpb - a.bpb);
  document.getElementById("ladder").innerHTML = rungs.map(r =>
    `<div class="rung ${r.cls || (r.bpb === 1.0 ? "target" : "")}"><span class="b">${fmt(r.bpb, r.bpb < 8 ? 3 : 2)}</span>
     <span>${r.what}<span class="faint" style="display:block">${r.note || ""}</span></span></div>`).join("");

  document.getElementById("results").innerHTML =
    "<tr><th>run</th><th>bpb</th><th>params</th><th>organ</th><th>data</th><th>steps</th></tr>" +
    s.results.map(r => `<tr><td>${r.name}</td><td class="n"><b>${fmt(r.bpb, 4)}</b></td>
      <td class="n">${r.params ? (r.params / 1e6).toFixed(1) + "M" : "—"}</td>
      <td>${r.adaptive === true ? "ON" : r.adaptive === false ? "off" : "—"}</td>
      <td>${r.data}</td><td class="n">${r.steps ? r.steps.toLocaleString() : "—"}</td></tr>`).join("");
}
async function tick() {
  try {
    const s = await (await fetch("/api/big")).json();
    document.getElementById("stale").style.display = "none";
    render(s);
  } catch (e) { document.getElementById("stale").style.display = "inline"; }
}
tick(); setInterval(tick, 10000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path.startswith("/api/big"):
            body = json.dumps(build_status()).encode()
            ctype = "application/json"
        else:
            body = _PAGE.encode()
            ctype = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # quiet
        pass


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="chronohorn observe big-picture")
    parser.add_argument("--port", type=int, default=8321)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"big picture at http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
