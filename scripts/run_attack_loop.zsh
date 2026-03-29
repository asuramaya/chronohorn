#!/bin/zsh
set -euo pipefail
unsetopt BGNICE

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
BIN="$ROOT_DIR/target/debug/chronohorn"

DATA_ROOT="@fineweb"
SLEEP_SEC=30
CYCLES=0
HEAVY_EVERY=3
TRAIN_TOKENS=1000000
HEAVY_TRAIN_TOKENS=2000000
TRIGRAM_BUCKETS=16384
SKIP_BUCKETS=16384
VAL_TOKENS=4096
MATCH_DEPTH_BASE=8
MATCH_DEPTH_ALT=12
TRAIN_STRIDE=4
CHUNK_SIZE=64
MAX_CHUNKS=8
K_VALUES=(12 16 20)
ORACLE_ATTACK_JSON="$REPO_ROOT/blinx/conker/out/blinx_oracle_attack_2026-03-28.json"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/out/attack_loops/$RUN_ID}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --sleep-sec)
      SLEEP_SEC="$2"
      shift 2
      ;;
    --cycles)
      CYCLES="$2"
      shift 2
      ;;
    --heavy-every)
      HEAVY_EVERY="$2"
      shift 2
      ;;
    --train-tokens)
      TRAIN_TOKENS="$2"
      shift 2
      ;;
    --heavy-train-tokens)
      HEAVY_TRAIN_TOKENS="$2"
      shift 2
      ;;
    --trigram-buckets)
      TRIGRAM_BUCKETS="$2"
      shift 2
      ;;
    --skip-buckets)
      SKIP_BUCKETS="$2"
      shift 2
      ;;
    --val-tokens)
      VAL_TOKENS="$2"
      shift 2
      ;;
    --oracle-attack-json)
      ORACLE_ATTACK_JSON="$2"
      shift 2
      ;;
    --out-root)
      OUT_ROOT="$2"
      shift 2
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$OUT_ROOT"
mkdir -p "$ROOT_DIR/out"

cargo build --manifest-path "$ROOT_DIR/Cargo.toml" >/dev/null

"$BIN" doctrine-json > "$OUT_ROOT/doctrine.json"
"$BIN" inspect-data-root "$DATA_ROOT" > "$OUT_ROOT/data_root.json"

if [[ -f "$ORACLE_ATTACK_JSON" ]]; then
  "$BIN" oracle-clean-summary "$ORACLE_ATTACK_JSON" 16 > "$OUT_ROOT/oracle_clean_summary.txt"
fi

SUMMARY_TSV="$OUT_ROOT/summary.tsv"
BEST_JSON="$OUT_ROOT/best.json"
BEST_TXT="$OUT_ROOT/best.txt"
LATEST_TXT="$OUT_ROOT/latest_cycle.txt"

if [[ ! -f "$SUMMARY_TSV" ]]; then
  printf "cycle\tlabel\tdirect_bpb\toracle_bpb\tmatch_bpb\tskip_bpb\tgate\tlambda\tfailures\tpath\n" > "$SUMMARY_TSV"
fi

extract_row() {
  local cycle="$1"
  local label="$2"
  local json_path="$3"
  python3 - "$cycle" "$label" "$json_path" <<'PY'
import json, pathlib, sys
cycle, label, path = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
d = json.loads(path.read_text())
r = d["report"] if "report" in d else d
aud = d.get("audit", {})
fails = sum(v.get("failure_count", 0) for v in aud.values()) if isinstance(aud, dict) else 0
print(
    "\t".join(
        [
            cycle,
            label,
            str(r.get("eval_bpb_direct")),
            str(r.get("eval_bpb_oracle")),
            str(r.get("eval_bpb_match")),
            str(r.get("eval_bpb_skip")),
            str(r.get("selected_runtime_gate")),
            str(r.get("selected_runtime_lambda")),
            str(fails),
            str(path),
        ]
    )
)
PY
}

promote_best() {
  python3 - "$SUMMARY_TSV" "$BEST_TXT" "$BEST_JSON" <<'PY'
import pathlib, shutil, sys
summary_path = pathlib.Path(sys.argv[1])
best_txt = pathlib.Path(sys.argv[2])
best_json = pathlib.Path(sys.argv[3])
lines = [line.strip().split("\t") for line in summary_path.read_text().splitlines() if line.strip()]
header, rows = lines[0], lines[1:]
if not rows:
    raise SystemExit(0)
rows.sort(key=lambda row: float(row[2]))
best = rows[0]
best_txt.write_text(
    "\n".join(
        [
            f"cycle: {best[0]}",
            f"label: {best[1]}",
            f"direct_bpb: {best[2]}",
            f"oracle_bpb: {best[3]}",
            f"match_bpb: {best[4]}",
            f"skip_bpb: {best[5]}",
            f"gate: {best[6]}",
            f"lambda: {best[7]}",
            f"failures: {best[8]}",
            f"path: {best[9]}",
        ]
    )
    + "\n"
)
src = pathlib.Path(best[9])
if src.exists():
    shutil.copy2(src, best_json)
PY
}

cycle=1
while :; do
  if [[ "$CYCLES" -gt 0 && "$cycle" -gt "$CYCLES" ]]; then
    break
  fi

  cycle_dir="$OUT_ROOT/cycle_$(printf '%04d' "$cycle")"
  mkdir -p "$cycle_dir"
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$cycle_dir/started_at_utc.txt"
  if [[ -f "$ORACLE_ATTACK_JSON" ]]; then
    "$BIN" oracle-clean-summary "$ORACLE_ATTACK_JSON" 16 > "$cycle_dir/oracle_clean_summary.txt"
  fi

  jobs=()

  for k in "${K_VALUES[@]}"; do
    json="$cycle_dir/matchskip_k${k}.json"
    log="$cycle_dir/matchskip_k${k}.log"
    (
      "$BIN" run-token-matchskip-bundle-json \
        "$DATA_ROOT" \
        "$TRAIN_TOKENS" \
        "$TRIGRAM_BUCKETS" \
        "$SKIP_BUCKETS" \
        "$VAL_TOKENS" \
        "$MATCH_DEPTH_BASE" \
        "$k" \
        "$TRAIN_STRIDE" \
        "$CHUNK_SIZE" \
        "$MAX_CHUNKS" > "$json"
    ) >"$log" 2>&1 &
    jobs+=($!)
  done

  depth_json="$cycle_dir/matchskip_depth${MATCH_DEPTH_ALT}_k16.json"
  depth_log="$cycle_dir/matchskip_depth${MATCH_DEPTH_ALT}_k16.log"
  (
    "$BIN" run-token-matchskip-bundle-json \
      "$DATA_ROOT" \
      "$TRAIN_TOKENS" \
      "$TRIGRAM_BUCKETS" \
      "$SKIP_BUCKETS" \
      "$VAL_TOKENS" \
      "$MATCH_DEPTH_ALT" \
      16 \
      "$TRAIN_STRIDE" \
      "$CHUNK_SIZE" \
      "$MAX_CHUNKS" > "$depth_json"
  ) >"$depth_log" 2>&1 &
  jobs+=($!)

  if [[ "$HEAVY_EVERY" -gt 0 && $((cycle % HEAVY_EVERY)) -eq 0 ]]; then
    heavy_json="$cycle_dir/matchskip_heavy_k16.json"
    heavy_log="$cycle_dir/matchskip_heavy_k16.log"
    (
      "$BIN" run-token-matchskip-bundle-json \
        "$DATA_ROOT" \
        "$HEAVY_TRAIN_TOKENS" \
        "$TRIGRAM_BUCKETS" \
        "$SKIP_BUCKETS" \
        "$VAL_TOKENS" \
        "$MATCH_DEPTH_BASE" \
        16 \
        "$TRAIN_STRIDE" \
        "$CHUNK_SIZE" \
        "$MAX_CHUNKS" > "$heavy_json"
    ) >"$heavy_log" 2>&1 &
    jobs+=($!)
  fi

  for job in "${jobs[@]}"; do
    wait "$job"
  done

  : > "$LATEST_TXT"
  for json in "$cycle_dir"/*.json; do
    if [[ -f "$json" ]]; then
      label="${json:t:r}"
      row="$(extract_row "$cycle" "$label" "$json")"
      printf "%s\n" "$row" | tee -a "$SUMMARY_TSV" >> "$LATEST_TXT"
    fi
  done

  promote_best
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$cycle_dir/finished_at_utc.txt"

  cycle=$((cycle + 1))
  if [[ "$CYCLES" -gt 0 && "$cycle" -gt "$CYCLES" ]]; then
    break
  fi
  sleep "$SLEEP_SEC"
done
