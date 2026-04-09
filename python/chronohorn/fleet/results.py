"""Pull result JSONs from completed remote fleet jobs."""
from __future__ import annotations

import json
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
        return True
    except (json.JSONDecodeError, OSError) as exc:
        import sys
        print(f"chronohorn: DB ingestion failed for {safe_name}: {exc}", file=sys.stderr)
        return False


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
        return PullResult(
            job_name=safe_name,
            success=True,
            local_path=local_path,
            skipped=True,
            ingested=_ingest_local_result_artifact(local_path=local_path, safe_name=safe_name, db=db),
        )

    remote_path = f"{safe_remote_run}/results/{safe_name}.json"
    try:
        payload_text = _ssh_cat_file(host, remote_path)
        json.loads(payload_text)  # validate JSON
        local_path.write_text(payload_text)
        ingested = _ingest_local_result_artifact(local_path=local_path, safe_name=safe_name, db=db)
        result = PullResult(job_name=safe_name, success=True, local_path=local_path, ingested=ingested)
        probes_remote = f"{safe_remote_run}/results/{safe_name}.probes.jsonl"
        try:
            probes_text = _ssh_cat_file(host, probes_remote)
            if db is not None:
                for line in probes_text.strip().splitlines():
                    p = json.loads(line)
                    pbpb = probe_bpb_from_row(p)
                    if pbpb and p.get("step"):
                        db.record_probe(
                            safe_name,
                            p["step"],
                            pbpb,
                            loss=p.get("loss", p.get("eval_loss")),
                            elapsed_sec=p.get("elapsed_sec"),
                        )
        except RuntimeError:
            pass  # probes file may not exist on remote
        except json.JSONDecodeError as exc:
            import sys
            print(f"chronohorn: corrupt probe data for {safe_name}: {exc}", file=sys.stderr)
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
