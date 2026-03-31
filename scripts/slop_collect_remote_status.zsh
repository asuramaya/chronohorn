#!/bin/zsh
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat <<'EOF'
usage: slop_collect_remote_status.zsh <host> <job_name> [job_name...]

Queries one slop host for Chronohorn run status and prints one JSON object per job.
EOF
  exit 1
fi

HOST="$1"
shift
SSH_ARGS=(-o BatchMode=yes -o ConnectTimeout=5)

function container_name_for_job() {
  local job_name="$1"
  python3 - "$job_name" <<'PY'
from __future__ import annotations
import sys

name = sys.argv[1]
safe = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in name)
print(f"chronohorn-{safe}", end="")
PY
}

typeset -a RUN_LINES
for job_name in "$@"; do
  RUN_LINES+=("$(printf '%s\t%s\t%s' "${job_name}" "$(container_name_for_job "${job_name}")" "/tmp/chronohorn-runs/${job_name}")")
done

RUN_SPEC="$(printf '%s\n' "${RUN_LINES[@]}")"

REMOTE_PAYLOAD=$(cat <<EOF
set -euo pipefail
docker_names="\$(sudo -n docker ps --format '{{.Names}}' || true)"
while IFS=\$'\\t' read -r name container run; do
  [[ -z "\$name" ]] && continue
  running=0
  if grep -Fx "\$container" <<<"\$docker_names" >/dev/null 2>&1; then
    running=1
  fi
  report="\$run/report.txt"
  log="\$run/job.log"
  python3 - "\$name" "${HOST}" "\$container" "\$run" "\$running" "\$report" "\$log" <<'PY'
from __future__ import annotations
import json
from pathlib import Path
import sys

name, host, container, run_dir, running, report_path, log_path = sys.argv[1:]
report = Path(report_path)
log = Path(log_path)
payload = {
    "name": name,
    "host": host,
    "container_name": container,
    "remote_run": run_dir,
    "running": running == "1",
    "report_exists": report.exists(),
    "log_exists": log.exists(),
    "report_size_bytes": report.stat().st_size if report.exists() else 0,
    "log_size_bytes": log.stat().st_size if log.exists() else 0,
    "report_last_line": report.read_text(encoding="utf-8", errors="replace").splitlines()[-1] if report.exists() and report.stat().st_size else "",
}
print(json.dumps(payload, sort_keys=True))
PY
done <<'RUNS'
${RUN_SPEC}
RUNS
EOF
)

ssh "${SSH_ARGS[@]}" "${HOST}" "/bin/bash -lc $(printf '%q' "${REMOTE_PAYLOAD}")"
