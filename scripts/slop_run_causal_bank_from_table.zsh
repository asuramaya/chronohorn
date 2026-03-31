#!/bin/zsh
set -euo pipefail

if [[ $# -lt 5 ]]; then
  cat <<'EOF'
usage: slop_run_causal_bank_from_table.zsh <host> <job_name> <checkpoint-path|bundle-dir> <summary.json> <artifact.bin> [threads] [val_tokens] [report_every]

Runs Chronohorn's artifact-backed causal-bank eval on a slop node inside a disposable Rust container.
The script:
  1. reuses or stages a keyed remote Chronohorn snapshot
  2. reuses cached FineWeb/assets/runtime image on the host
  3. builds Chronohorn once per snapshot key into cached target state
  4. launches the eval as a detached docker container

Outputs:
  - remote report: /tmp/chronohorn-runs/<job_name>/report.txt
  - remote stderr/stdout: /tmp/chronohorn-runs/<job_name>/job.log
EOF
  exit 1
fi

HOST="$1"
JOB_NAME="$2"
CHECKPOINT_SOURCE="$3"
SUMMARY_PATH="$4"
ARTIFACT_BIN="$5"
THREADS="${6:-12}"
VAL_TOKENS="${7:-62021846}"
REPORT_EVERY="${8:-1000000}"

SCRIPT_DIR="${0:A:h}"
CHRONOHORN_DIR="${SCRIPT_DIR:h}"
ROOT_DIR="${CHRONOHORN_DIR:h}"
DATA_ROOT_LOCAL="${CHRONOHORN_DIR}/data/roots/fineweb10B_sp1024"

REMOTE_CACHE="/tmp/chronohorn-cache"
REMOTE_RUN="/tmp/chronohorn-runs/${JOB_NAME}"
REMOTE_DATA="${REMOTE_CACHE}/data/roots/fineweb10B_sp1024"
REMOTE_ASSETS="${REMOTE_CACHE}/assets"
REMOTE_ASSET_BLOBS="${REMOTE_ASSETS}/by-sha256"
REMOTE_RUNTIME="${REMOTE_CACHE}/runtime"
IMAGE="chronohorn-rust-openblas:1.90-bookworm"
CONTAINER_NAME="chronohorn-${JOB_NAME//[^A-Za-z0-9_.-]/-}"
SSH_ARGS=(-o BatchMode=yes -o ConnectTimeout=5)
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || true)}"
if [[ -z "${PYTHON_BIN}" && -x /usr/bin/python3 ]]; then
  PYTHON_BIN="/usr/bin/python3"
fi
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "slop_run_causal_bank_from_table.zsh requires python3" >&2
  exit 1
fi

function compute_stage_key() {
  "${PYTHON_BIN}" - "$CHRONOHORN_DIR" <<'PY'
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
exclude_dirs = {".git", "target", "out", "__pycache__"}
exclude_suffixes = (".pyc", ".pyo", ".swp", ".tmp")
digest = hashlib.sha256()
digest.update(str(root).encode("utf-8"))
for path in sorted(root.rglob("*")):
    rel = path.relative_to(root)
    if any(part in exclude_dirs for part in rel.parts):
        continue
    if path.is_dir():
        continue
    if path.name.endswith(exclude_suffixes):
        continue
    stat = path.stat()
    digest.update(rel.as_posix().encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(stat.st_size).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(stat.st_mtime_ns).encode("ascii"))
    digest.update(b"\n")
print(digest.hexdigest()[:16], end="")
PY
}

STAGE_KEY="${CHRONOHORN_STAGE_KEY:-$(compute_stage_key)}"
REMOTE_SNAPSHOT_ROOT="${REMOTE_CACHE}/snapshots/${STAGE_KEY}"
REMOTE_CODE="${REMOTE_SNAPSHOT_ROOT}/chronohorn"
REMOTE_TARGET="${REMOTE_CACHE}/target/${STAGE_KEY}"
REMOTE_STAGE_READY="${REMOTE_SNAPSHOT_ROOT}/.chronohorn-stage-ready"
REMOTE_BUILD_READY="${REMOTE_TARGET}/.chronohorn-build-ready"
RUNTIME_DOCKERFILE="${REMOTE_RUNTIME}/Dockerfile.eval-openblas"
RUNTIME_READY="${REMOTE_RUNTIME}/.eval-openblas-ready"
DATA_READY="${REMOTE_DATA}/.chronohorn-data-ready"
FORCE_STAGE="${CHRONOHORN_FORCE_STAGE:-0}"
FORCE_RELAUNCH="${CHRONOHORN_FORCE_RELAUNCH:-0}"
REUSE_COMPLETED="${CHRONOHORN_REUSE_COMPLETED:-1}"
CONTAINER_TARGET="/cache/target/${STAGE_KEY}"

function remote_bash() {
  ssh "${SSH_ARGS[@]}" "${HOST}" "/bin/bash -lc $(printf '%q' "$1")"
}

function wait_for_remote_path_absent() {
  local path="$1"
  local attempt=0
  while remote_bash "[[ -e ${path:q} ]]" >/dev/null 2>&1; do
    attempt=$((attempt + 1))
    if (( attempt > 180 )); then
      echo "timed out waiting for remote lock to clear: ${HOST}:${path}" >&2
      exit 1
    fi
    sleep 1
  done
}

function acquire_remote_lock() {
  local lock_path="$1"
  local lock_parent="${lock_path:h}"
  local attempt=0
  while ! remote_bash "mkdir -p ${lock_parent:q} && mkdir ${lock_path:q}" >/dev/null 2>&1; do
    attempt=$((attempt + 1))
    if (( attempt > 180 )); then
      echo "timed out acquiring remote lock: ${HOST}:${lock_path}" >&2
      exit 1
    fi
    sleep 1
  done
}

function release_remote_lock() {
  local lock_path="$1"
  remote_bash "rm -rf ${lock_path:q}" >/dev/null 2>&1 || true
}

function compute_path_sha256() {
  local path="$1"
  "${PYTHON_BIN}" - "$path" <<'PY'
from __future__ import annotations
import hashlib
from pathlib import Path
import sys

path = Path(sys.argv[1]).resolve()
digest = hashlib.sha256()

if path.is_file():
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
elif path.is_dir():
    digest.update(b"dir\0")
    digest.update(path.name.encode("utf-8"))
    digest.update(b"\0")
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = child.relative_to(path).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        with child.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        digest.update(b"\0")
else:
    raise SystemExit(f"unsupported path for hashing: {path}")

print(digest.hexdigest(), end="")
PY
}

function remote_cache_container_path() {
  local remote_path="$1"
  if [[ "${remote_path}" == ${REMOTE_CACHE}/* ]]; then
    printf '/cache/%s' "${remote_path#${REMOTE_CACHE}/}"
  else
    echo "remote path is not under ${REMOTE_CACHE}: ${remote_path}" >&2
    exit 1
  fi
}

function ensure_remote_cached_input() {
  local local_path="$1"
  local sha256
  sha256="$(compute_path_sha256 "${local_path}")"
  local base_name
  base_name="$(basename "${local_path}")"
  local remote_path="${REMOTE_ASSET_BLOBS}/${sha256}-${base_name}"
  local remote_tmp="${remote_path}.partial"
  local lock_path="${REMOTE_ASSET_BLOBS}/.${sha256}.lock"
  if [[ -f "${local_path}" ]]; then
    if remote_bash "[[ -s ${remote_path:q} ]]" >/dev/null 2>&1; then
      echo "${remote_path}"
      return 0
    fi
  elif [[ -d "${local_path}" ]]; then
    if remote_bash "[[ -d ${remote_path:q} && -f ${remote_path:q}/.chronohorn-asset-ready ]]" >/dev/null 2>&1; then
      echo "${remote_path}"
      return 0
    fi
  else
    echo "input path does not exist: ${local_path}" >&2
    exit 1
  fi
  acquire_remote_lock "${lock_path}"
  if [[ -f "${local_path}" ]]; then
    if ! remote_bash "[[ -s ${remote_path:q} ]]" >/dev/null 2>&1; then
      remote_bash "mkdir -p ${REMOTE_ASSET_BLOBS:q}" >/dev/null
      rsync -a "${local_path}" "${HOST}:${remote_tmp}"
      remote_bash "mv -f ${remote_tmp:q} ${remote_path:q}" >/dev/null
    fi
  else
    if ! remote_bash "[[ -d ${remote_path:q} && -f ${remote_path:q}/.chronohorn-asset-ready ]]" >/dev/null 2>&1; then
      remote_bash "mkdir -p ${REMOTE_ASSET_BLOBS:q} && rm -rf ${remote_tmp:q} && mkdir -p ${remote_tmp:q}" >/dev/null
      rsync -a --delete "${local_path}/" "${HOST}:${remote_tmp}/"
      remote_bash "touch ${remote_tmp:q}/.chronohorn-asset-ready && rm -rf ${remote_path:q} && mv ${remote_tmp:q} ${remote_path:q}" >/dev/null
    fi
  fi
  release_remote_lock "${lock_path}"
  echo "${remote_path}"
}

function build_launch_meta_json() {
  python3 - "$JOB_NAME" "$HOST" "$CONTAINER_NAME" "$STAGE_KEY" "$REMOTE_RUN" \
    "$REMOTE_CHECKPOINT_SOURCE" "$REMOTE_SUMMARY_PATH" "$REMOTE_ARTIFACT_BIN" \
    "$THREADS" "$VAL_TOKENS" "$REPORT_EVERY" <<'PY'
from __future__ import annotations
import json
import sys

payload = {
    "job_name": sys.argv[1],
    "host": sys.argv[2],
    "container_name": sys.argv[3],
    "stage_key": sys.argv[4],
    "remote_run": sys.argv[5],
    "remote_checkpoint_path": sys.argv[6],
    "remote_summary_path": sys.argv[7],
    "remote_artifact_bin": sys.argv[8],
    "threads": int(sys.argv[9]),
    "val_tokens": int(sys.argv[10]),
    "report_every": int(sys.argv[11]),
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY
}

if [[ "${FORCE_RELAUNCH}" != "1" ]]; then
  if remote_bash "sudo -n docker ps --format '{{.Names}}' | grep -Fx ${CONTAINER_NAME:q} >/dev/null" >/dev/null 2>&1; then
    echo "reusing running container ${CONTAINER_NAME} on ${HOST}"
    echo "${CONTAINER_NAME}"
    echo "${REMOTE_RUN}"
    exit 0
  fi
  if [[ "${REUSE_COMPLETED}" == "1" ]] && remote_bash "[[ -s ${REMOTE_RUN:q}/report.txt ]]" >/dev/null 2>&1; then
    echo "reusing completed remote report for ${JOB_NAME} on ${HOST}"
    echo "${CONTAINER_NAME}"
    echo "${REMOTE_RUN}"
    exit 0
  fi
fi

remote_bash "mkdir -p \
  ${REMOTE_RUN:q} \
  ${REMOTE_DATA:q} \
  ${REMOTE_ASSETS:q} \
  ${REMOTE_ASSET_BLOBS:q} \
  ${REMOTE_RUNTIME:q} \
  ${(q)REMOTE_CACHE}/cargo/registry \
  ${(q)REMOTE_CACHE}/cargo/git \
  ${REMOTE_TARGET:q}" >/dev/null

if [[ "${FORCE_STAGE}" == "1" ]] || ! remote_bash "[[ -f ${REMOTE_STAGE_READY:q} ]]" >/dev/null 2>&1; then
  echo "staging chronohorn snapshot ${STAGE_KEY} to ${HOST}:${REMOTE_CODE}"
  acquire_remote_lock "${REMOTE_SNAPSHOT_ROOT}.lock"
  if [[ "${FORCE_STAGE}" == "1" ]] || ! remote_bash "[[ -f ${REMOTE_STAGE_READY:q} ]]" >/dev/null 2>&1; then
    remote_bash "mkdir -p ${REMOTE_CODE:q} && rm -f ${REMOTE_STAGE_READY:q}" >/dev/null
    rsync -a --delete \
      --exclude .git \
      --exclude target \
      --exclude out \
      "${CHRONOHORN_DIR}/" "${HOST}:${REMOTE_CODE}/"
    remote_bash "touch ${REMOTE_STAGE_READY:q}" >/dev/null
  fi
  release_remote_lock "${REMOTE_SNAPSHOT_ROOT}.lock"
else
  echo "reusing chronohorn snapshot ${STAGE_KEY} on ${HOST}"
fi

if [[ "${FORCE_STAGE}" == "1" ]] || ! remote_bash "[[ -f ${DATA_READY:q} ]]" >/dev/null 2>&1; then
  echo "syncing fineweb root to ${HOST}:${REMOTE_DATA}"
  acquire_remote_lock "${REMOTE_DATA}.lock"
  if [[ "${FORCE_STAGE}" == "1" ]] || ! remote_bash "[[ -f ${DATA_READY:q} ]]" >/dev/null 2>&1; then
    rsync -a "${DATA_ROOT_LOCAL}/" "${HOST}:${REMOTE_DATA}/"
    remote_bash "touch ${DATA_READY:q}" >/dev/null
  fi
  release_remote_lock "${REMOTE_DATA}.lock"
else
  echo "reusing fineweb root on ${HOST}:${REMOTE_DATA}"
fi

echo "reusing or seeding immutable inputs in ${HOST}:${REMOTE_ASSET_BLOBS}"
REMOTE_CHECKPOINT_SOURCE="$(ensure_remote_cached_input "${CHECKPOINT_SOURCE}")"
REMOTE_SUMMARY_PATH="$(ensure_remote_cached_input "${SUMMARY_PATH}")"
REMOTE_ARTIFACT_BIN="$(ensure_remote_cached_input "${ARTIFACT_BIN}")"
CONTAINER_DATA="/cache/data/roots/fineweb10B_sp1024"
CONTAINER_CHECKPOINT_SOURCE="$(remote_cache_container_path "${REMOTE_CHECKPOINT_SOURCE}")"
CONTAINER_SUMMARY_PATH="$(remote_cache_container_path "${REMOTE_SUMMARY_PATH}")"
CONTAINER_ARTIFACT_BIN="$(remote_cache_container_path "${REMOTE_ARTIFACT_BIN}")"

acquire_remote_lock "${REMOTE_RUNTIME}/eval-openblas.lock"
if ! remote_bash "sudo -n docker image inspect ${IMAGE:q} >/dev/null 2>&1" >/dev/null 2>&1; then
  echo "building cached runtime image ${IMAGE} on ${HOST}"
  remote_bash "$(cat <<EOF
set -euo pipefail
mkdir -p ${REMOTE_RUNTIME:q}
cat > ${RUNTIME_DOCKERFILE:q} <<'DOCKERFILE'
FROM rust:1.90-bookworm
RUN export DEBIAN_FRONTEND=noninteractive \\
 && apt-get update -qq \\
 && apt-get install -y --no-install-recommends libopenblas-dev \\
 && rm -rf /var/lib/apt/lists/*
DOCKERFILE
sudo -n docker build -t ${IMAGE:q} -f ${RUNTIME_DOCKERFILE:q} ${REMOTE_RUNTIME:q} >/dev/null
touch ${RUNTIME_READY:q}
EOF
)" >/dev/null
else
  remote_bash "touch ${RUNTIME_READY:q}" >/dev/null || true
fi
release_remote_lock "${REMOTE_RUNTIME}/eval-openblas.lock"

acquire_remote_lock "${REMOTE_TARGET}.lock"
if [[ "${FORCE_STAGE}" == "1" ]] || ! remote_bash "[[ -x ${REMOTE_TARGET:q}/release/chronohorn && -f ${REMOTE_BUILD_READY:q} ]]" >/dev/null 2>&1; then
  echo "building chronohorn once for snapshot ${STAGE_KEY} on ${HOST}"
  remote_bash "$(cat <<EOF
set -euo pipefail
rm -f ${REMOTE_BUILD_READY:q}
sudo -n docker run --rm \
  -e CARGO_HOME=/cache/cargo \
  -e CARGO_TARGET_DIR=${CONTAINER_TARGET:q} \
  -v ${REMOTE_SNAPSHOT_ROOT:q}:/snapshot \
  -v ${REMOTE_CACHE:q}:/cache \
  ${IMAGE:q} \
  bash -lc '
    set -euo pipefail
    export PATH=/usr/local/cargo/bin:\$PATH
    cd /snapshot/chronohorn
    cargo build --release
  ' >/dev/null
touch ${REMOTE_BUILD_READY:q}
EOF
)" >/dev/null
else
  echo "reusing cached chronohorn binary for snapshot ${STAGE_KEY} on ${HOST}"
fi
release_remote_lock "${REMOTE_TARGET}.lock"

LAUNCH_META_JSON="$(build_launch_meta_json)"
remote_bash "$(cat <<EOF
set -euo pipefail
cat > ${REMOTE_RUN:q}/launch_meta.json <<'JSON'
${LAUNCH_META_JSON}
JSON
EOF
)" >/dev/null

REMOTE_CMD=$(cat <<EOF
set -euo pipefail
sudo -n docker rm -f '${CONTAINER_NAME}' >/dev/null 2>&1 || true
nohup sudo -n docker run --rm --name '${CONTAINER_NAME}' \
  -e CHRONOHORN_THREADS='${THREADS}' \
  -e OPENBLAS_NUM_THREADS=4 \
  -e OMP_NUM_THREADS=4 \
  -v '${REMOTE_RUN}:/work' \
  -v '${REMOTE_CACHE}:/cache' \
  '${IMAGE}' \
  bash -lc '
    set -euo pipefail
    export PATH=/usr/local/cargo/bin:$PATH
    ${CONTAINER_TARGET}/release/chronohorn run-causal-bank-ngram-bulk-from-table \
      "${CONTAINER_CHECKPOINT_SOURCE}" \
      "${CONTAINER_SUMMARY_PATH}" \
      "${CONTAINER_DATA}" \
      "${CONTAINER_ARTIFACT_BIN}" \
      "${VAL_TOKENS}" \
      "${REPORT_EVERY}" \
      | tee /work/report.txt
  ' > '${REMOTE_RUN}/job.log' 2>&1 &
echo '${CONTAINER_NAME}'
echo '${REMOTE_RUN}'
EOF
)

echo "launching ${JOB_NAME} on ${HOST}"
LAUNCH_OUT="$(ssh "${SSH_ARGS[@]}" "${HOST}" "${REMOTE_CMD}")"
echo "${LAUNCH_OUT}"
echo
echo "monitor:"
echo "  ssh ${HOST} 'sudo -n docker logs -f ${CONTAINER_NAME}'"
echo "  ssh ${HOST} 'tail -f ${REMOTE_RUN}/job.log'"
echo
echo "collect:"
echo "  rsync -a ${HOST}:${REMOTE_RUN}/report.txt ${CHRONOHORN_DIR}/out/$(basename "${REMOTE_RUN}")-report.txt"
