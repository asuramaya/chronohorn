"""Pull result JSONs from completed remote fleet jobs."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chronohorn.metrics import probe_bpb_from_row

from .validation import validate_job_name, validate_posix_path_within_any_root


@dataclass(frozen=True)
class PullResult:
    job_name: str
    success: bool
    local_path: Path | None = None
    skipped: bool = False
    ingested: bool = False
    error: str | None = None


DEFAULT_RESULT_DIR = Path("out/results")
_REMOTE_CHECKPOINT_DIR = "/data/chronohorn/checkpoints"


_DEFAULT_SHARTS_EXPORT_ROOT = Path("/Volumes/Sharts/heinrich")


def _resolve_checkpoint_export_dir() -> Path | None:
    """Resolve the shared checkpoint export directory.

    Checked in order:
    1. CHRONOHORN_CHECKPOINT_EXPORT_DIR env var (explicit override, any path)
    2. /Volumes/Sharts/heinrich/<most-recent-subdir> (macOS workstation auto-discover)
    3. None (disabled)

    Auto-discovery: when Sharts is mounted and contains a heinrich directory
    with per-session subdirectories (session10_byte_scaling, session11_*, etc.),
    use the most-recently-modified subdirectory. This means a freshly-started
    runtime daemon exports to the active session automatically — no env var
    fiddling required.
    """
    raw = os.environ.get("CHRONOHORN_CHECKPOINT_EXPORT_DIR", "")
    if raw:
        p = Path(raw)
        return p if p.is_dir() else None
    # Auto-discover on macOS workstations with Sharts mounted.
    if _DEFAULT_SHARTS_EXPORT_ROOT.is_dir():
        try:
            subdirs = [
                d for d in _DEFAULT_SHARTS_EXPORT_ROOT.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ]
            if subdirs:
                return max(subdirs, key=lambda d: d.stat().st_mtime)
        except OSError:
            pass
    return None


def _export_checkpoint(host: str, job_name: str, local_json: Path) -> None:
    """Copy result JSON + checkpoint(s) to the shared export directory. Best-effort.

    Pulls the end-of-training {job_name}.checkpoint.pt AND any periodic
    checkpoints {job_name}_step*.checkpoint.pt written by
    --save-checkpoint-every from the remote durable storage.
    """
    import sys
    export_dir = _resolve_checkpoint_export_dir()
    if export_dir is None:
        return
    # Copy local JSON
    dst_json = export_dir / f"{job_name}.json"
    if not dst_json.exists():
        try:
            import shutil
            shutil.copy2(local_json, dst_json)
        except Exception as exc:
            print(f"chronohorn: checkpoint export {job_name}.json failed: {exc}", file=sys.stderr)
    # Enumerate all checkpoint files for this job on the remote host, then
    # scp each one individually. SFTP-based scp in OpenSSH 9+ doesn't expand
    # remote globs; ls-then-scp works across scp protocol versions.
    list_cmd = (
        f"ls {shlex.quote(_REMOTE_CHECKPOINT_DIR)}/{shlex.quote(job_name)}.checkpoint.pt "
        f"{shlex.quote(_REMOTE_CHECKPOINT_DIR)}/{shlex.quote(job_name)}_step*.checkpoint.pt "
        f"2>/dev/null || true"
    )
    try:
        listing = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host, list_cmd],
            capture_output=True, text=True, timeout=30,
        )
        remote_files = [p for p in (listing.stdout or "").strip().split("\n") if p]
    except Exception as exc:
        print(f"chronohorn: checkpoint list on {host} failed: {exc}", file=sys.stderr)
        remote_files = []
    for remote_ckpt in remote_files:
        fname = os.path.basename(remote_ckpt)
        dst_ckpt = export_dir / fname
        if dst_ckpt.exists():
            continue
        try:
            subprocess.run(
                ["scp", f"{host}:{remote_ckpt}", str(dst_ckpt)],
                capture_output=True, timeout=300,
            )
        except Exception as exc:
            print(f"chronohorn: checkpoint export {fname} failed: {exc}", file=sys.stderr)
_ALLOWED_REMOTE_RESULT_ROOTS = ("/tmp/chronohorn-runs", "/data/chronohorn/out")


def _ssh_cat_file(host: str, remote_path: str) -> str:
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host,
         f"cat {shlex.quote(remote_path)}"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ssh cat failed: {result.stderr.strip()}")
    return result.stdout


def _validated_remote_result_run(remote_run: str) -> str:
    return validate_posix_path_within_any_root(
        remote_run,
        roots=_ALLOWED_REMOTE_RESULT_ROOTS,
        field_name="remote_run",
    )


def _ingest_local_result_artifact(*, local_path: Path, safe_name: str, db) -> bool:
    if db is None:
        return False
    try:
        payload = json.loads(local_path.read_text())
        db.record_result(safe_name, payload, json_archive=str(local_path))
        # Register local checkpoint if it exists alongside the result
        ckpt_path = local_path.with_suffix(".checkpoint.pt")
        if ckpt_path.exists():
            db._write(
                "UPDATE jobs SET checkpoint_path = ?, checkpoint_host = ? WHERE name = ?",
                (str(ckpt_path), "local", safe_name),
            )
        return True
    except (json.JSONDecodeError, OSError) as exc:
        import sys
        print(f"chronohorn: DB ingestion failed for {safe_name}: {exc}", file=sys.stderr)
        return False


def _persist_remote_checkpoints(host: str, remote_run: str, job_name: str) -> None:
    """Copy checkpoint and training_state files to persistent storage on the remote host.

    Covers:
      - End-of-training {job_name}.checkpoint.pt / .training_state.pt / .json
      - Periodic checkpoints {job_name}_step*.checkpoint.pt from --save-checkpoint-every
    """
    import sys

    for suffix in (".checkpoint.pt", ".training_state.pt", ".json"):
        src = f"{remote_run}/results/{job_name}{suffix}"
        dst = f"{_REMOTE_CHECKPOINT_DIR}/{job_name}{suffix}"
        cmd = f"mkdir -p {_REMOTE_CHECKPOINT_DIR} && test -f {shlex.quote(src)} && cp -n {shlex.quote(src)} {shlex.quote(dst)} 2>/dev/null && echo copied || true"
        try:
            subprocess.run(
                ["ssh", host, cmd],
                capture_output=True, text=True, timeout=120,
            )
        except Exception as exc:
            print(f"chronohorn: remote persist {job_name}{suffix} on {host} failed: {exc}", file=sys.stderr)
    # Periodic checkpoints — glob pattern for --save-checkpoint-every output.
    safe_job = shlex.quote(job_name)
    safe_remote = shlex.quote(remote_run)
    safe_dest = shlex.quote(_REMOTE_CHECKPOINT_DIR)
    glob_cmd = (
        f"mkdir -p {safe_dest} && "
        f"for f in {safe_remote}/results/{safe_job}_step*.checkpoint.pt; do "
        f"[ -f \"$f\" ] && cp -n \"$f\" {safe_dest}/ 2>/dev/null; "
        f"done; true"
    )
    try:
        subprocess.run(
            ["ssh", host, glob_cmd],
            capture_output=True, text=True, timeout=120,
        )
    except Exception as exc:
        print(f"chronohorn: periodic checkpoint persist for {job_name} on {host}: {exc}", file=sys.stderr)


def _pull_remote_probes(host: str, safe_remote_run: str, safe_name: str, db) -> None:
    """Pull .probes.jsonl from remote host and ingest new probes into DB."""
    if db is None:
        return
    probes_remote = f"{safe_remote_run}/results/{safe_name}.probes.jsonl"
    try:
        probes_text = _ssh_cat_file(host, probes_remote)
    except RuntimeError:
        return  # probes file may not exist on remote
    existing = {r["step"] for r in db.query(
        "SELECT step FROM probes WHERE name = ?", (safe_name,)
    )}
    for line in probes_text.strip().splitlines():
        try:
            p = json.loads(line)
        except json.JSONDecodeError:
            continue  # skip corrupt line, keep processing rest
        step = p.get("step")
        pbpb = probe_bpb_from_row(p)
        if step and step not in existing and pbpb:
            db.record_probe(
                safe_name,
                step,
                pbpb,
                loss=p.get("loss", p.get("eval_loss")),
                elapsed_sec=p.get("elapsed_sec"),
                train_elapsed_sec=p.get("train_elapsed_sec"),
            )
            existing.add(step)


def pull_remote_result(
    *,
    host: str,
    remote_run: str,
    job_name: str,
    local_out_dir: Path | None = None,
    db=None,
) -> PullResult:
    safe_name = validate_job_name(job_name)
    safe_remote_run = _validated_remote_result_run(remote_run)
    out_dir = local_out_dir or DEFAULT_RESULT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    local_path = out_dir / f"{safe_name}.json"

    if local_path.exists():
        ingested = _ingest_local_result_artifact(local_path=local_path, safe_name=safe_name, db=db)
        _pull_remote_probes(host, safe_remote_run, safe_name, db)
        return PullResult(
            job_name=safe_name,
            success=True,
            local_path=local_path,
            skipped=True,
            ingested=ingested,
        )

    remote_path = f"{safe_remote_run}/results/{safe_name}.json"
    # Fallback: durable checkpoint storage survives k8s TTL cleanup
    durable_path = f"{_REMOTE_CHECKPOINT_DIR}/{safe_name}.json"
    try:
        try:
            payload_text = _ssh_cat_file(host, remote_path)
        except RuntimeError:
            # Primary path gone (TTL cleaned). Try durable storage.
            payload_text = _ssh_cat_file(host, durable_path)
        json.loads(payload_text)  # validate JSON
        local_path.write_text(payload_text)
        ingested = _ingest_local_result_artifact(local_path=local_path, safe_name=safe_name, db=db)
        result = PullResult(job_name=safe_name, success=True, local_path=local_path, ingested=ingested)
        # Export to shared location (CHRONOHORN_CHECKPOINT_EXPORT_DIR, best-effort)
        _export_checkpoint(host, safe_name, local_path)
        # Persist checkpoints to durable storage on the remote host
        _persist_remote_checkpoints(host, safe_remote_run, safe_name)
        # Register checkpoint location in DB
        if db is not None:
            ckpt_remote = f"{_REMOTE_CHECKPOINT_DIR}/{safe_name}.checkpoint.pt"
            try:
                check = subprocess.run(
                    ["ssh", host, f"test -f {shlex.quote(ckpt_remote)} && echo yes"],
                    capture_output=True, text=True, timeout=10,
                )
                if "yes" in (check.stdout or ""):
                    db._write(
                        "UPDATE jobs SET checkpoint_path = ?, checkpoint_host = ? WHERE name = ?",
                        (ckpt_remote, host, safe_name),
                    )
            except Exception:  # noqa: S110
                pass  # checkpoint registration is non-fatal
        # Clean up the k8s job after successful pull — prevents completed jobs
        # from blocking GPU allocation on subsequent dispatches.
        if db is not None:
            try:
                job_row = db.query("SELECT runtime_job_name, runtime_namespace, cluster_gateway_host FROM jobs WHERE name = ?", (safe_name,))
                if job_row:
                    from chronohorn.fleet.k8s import delete_k8s_job
                    delete_k8s_job({
                        "name": safe_name,
                        "runtime_job_name": job_row[0].get("runtime_job_name"),
                        "runtime_namespace": job_row[0].get("runtime_namespace"),
                        "cluster_gateway_host": job_row[0].get("cluster_gateway_host"),
                    })
            except Exception:
                pass  # cleanup is best-effort
        _pull_remote_probes(host, safe_remote_run, safe_name, db)
        return result
    except Exception as exc:
        return PullResult(job_name=safe_name, success=False, error=str(exc))


def pull_all_completed_results(
    launch_records: list[dict[str, Any]],
    *,
    local_out_dir: Path | None = None,
    db=None,
) -> list[PullResult]:
    results = []
    for record in launch_records:
        host = record.get("host")
        remote_run = record.get("remote_run")
        name = record.get("name")
        if not host or not remote_run or not name:
            continue
        results.append(
            pull_remote_result(
                host=host,
                remote_run=remote_run,
                job_name=name,
                local_out_dir=local_out_dir,
                db=db,
            )
        )
    return results
