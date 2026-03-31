#!/bin/zsh
set -euo pipefail

if [[ $# -lt 4 ]]; then
  cat <<'EOF'
usage: slop_build_oracle_budgeted_table.zsh <host> <job_name> <train_tokens> <profile> [threads] [report_every] [oracle_stride]

Builds an oracle-budgeted Chronohorn table artifact on a slop node inside a disposable Rust container.

Outputs:
  - remote artifact: /tmp/chronohorn-runs/<job_name>/<job_name>.bin
  - remote log:      /tmp/chronohorn-runs/<job_name>/job.log
  - remote report:   /tmp/chronohorn-runs/<job_name>/report.txt
EOF
  exit 1
fi

HOST="$1"
JOB_NAME="$2"
TRAIN_TOKENS="$3"
PROFILE="$4"
THREADS="${5:-12}"
REPORT_EVERY="${6:-5000}"
ORACLE_STRIDE="${7:-1}"

SCRIPT_DIR="${0:A:h}"
CHRONOHORN_DIR="${SCRIPT_DIR:h}"
DATA_ROOT_LOCAL="${CHRONOHORN_DIR}/data/roots/fineweb10B_sp1024"

REMOTE_CACHE="/tmp/chronohorn-cache"
REMOTE_RUN="/tmp/chronohorn-runs/${JOB_NAME}"
REMOTE_CODE="${REMOTE_RUN}/chronohorn"
REMOTE_DATA="${REMOTE_CACHE}/data/roots/fineweb10B_sp1024"
REMOTE_ARTIFACT="${REMOTE_RUN}/${JOB_NAME}.bin"
CONTAINER_NAME="chronohorn-${JOB_NAME//[^A-Za-z0-9_.-]/-}"
IMAGE="rust:1.90-bookworm"
CONTAINER_DATA="/cache/data/roots/fineweb10B_sp1024"
CONTAINER_ARTIFACT="/work/${JOB_NAME}.bin"

echo "syncing chronohorn tree to ${HOST}:${REMOTE_CODE}"
ssh "${HOST}" "mkdir -p '${REMOTE_CODE}' '${REMOTE_DATA}' '${REMOTE_CACHE}/cargo/registry' '${REMOTE_CACHE}/cargo/git' '${REMOTE_CACHE}/target'"
rsync -a --delete \
  --exclude .git \
  --exclude target \
  --exclude out \
  "${CHRONOHORN_DIR}/" "${HOST}:${REMOTE_CODE}/"

echo "syncing fineweb root to ${HOST}:${REMOTE_DATA}"
rsync -a "${DATA_ROOT_LOCAL}/" "${HOST}:${REMOTE_DATA}/"

REMOTE_CMD=$(cat <<EOF
set -euo pipefail
sudo -n docker rm -f '${CONTAINER_NAME}' >/dev/null 2>&1 || true
sudo -n docker image inspect '${IMAGE}' >/dev/null 2>&1 || sudo -n docker pull '${IMAGE}' >/dev/null
nohup sudo -n docker run --rm --name '${CONTAINER_NAME}' \
  -e CARGO_HOME=/cache/cargo \
  -e CARGO_TARGET_DIR=/cache/target \
  -e CHRONOHORN_THREADS='${THREADS}' \
  -v '${REMOTE_RUN}:/work' \
  -v '${REMOTE_CACHE}:/cache' \
  '${IMAGE}' \
  bash -lc '
    set -euo pipefail
    export PATH=/usr/local/cargo/bin:$PATH
    cd /work/chronohorn
    cargo build --release
    /cache/target/release/chronohorn build-causal-bank-ngram-oracle-budgeted-table \
      "${CONTAINER_DATA}" \
      "${CONTAINER_ARTIFACT}" \
      "${TRAIN_TOKENS}" \
      "${REPORT_EVERY}" \
      "${PROFILE}" \
      "${ORACLE_STRIDE}" \
      | tee /work/report.txt
  ' > '${REMOTE_RUN}/job.log' 2>&1 &
echo '${CONTAINER_NAME}'
echo '${REMOTE_RUN}'
echo '${REMOTE_ARTIFACT}'
EOF
)

echo "launching ${JOB_NAME} on ${HOST}"
LAUNCH_OUT="$(ssh "${HOST}" "${REMOTE_CMD}")"
echo "${LAUNCH_OUT}"
echo
echo "monitor:"
echo "  ssh ${HOST} 'sudo -n docker logs -f ${CONTAINER_NAME}'"
echo "  ssh ${HOST} 'tail -f ${REMOTE_RUN}/job.log'"
echo
echo "collect:"
echo "  rsync -a ${HOST}:${REMOTE_ARTIFACT} /tmp/"
echo "  rsync -a ${HOST}:${REMOTE_RUN}/report.txt ${CHRONOHORN_DIR}/out/$(basename "${REMOTE_RUN}")-report.txt"
