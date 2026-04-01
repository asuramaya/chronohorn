"""Pull result JSONs from completed remote fleet jobs."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PullResult:
    job_name: str
    success: bool
    local_path: Path | None = None
    skipped: bool = False
    error: str | None = None


DEFAULT_RESULT_DIR = Path("out/results")


def _ssh_cat_file(host: str, remote_path: str) -> str:
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host, f"cat {remote_path}"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ssh cat failed: {result.stderr.strip()}")
    return result.stdout


def pull_remote_result(
    *,
    host: str,
    remote_run: str,
    job_name: str,
    local_out_dir: Path | None = None,
) -> PullResult:
    out_dir = local_out_dir or DEFAULT_RESULT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    local_path = out_dir / f"{job_name}.json"

    if local_path.exists():
        return PullResult(job_name=job_name, success=True, local_path=local_path, skipped=True)

    remote_path = f"{remote_run}/results/{job_name}.json"
    try:
        payload_text = _ssh_cat_file(host, remote_path)
        json.loads(payload_text)  # validate JSON
        local_path.write_text(payload_text)
        return PullResult(job_name=job_name, success=True, local_path=local_path)
    except Exception as exc:
        return PullResult(job_name=job_name, success=False, error=str(exc))


def pull_all_completed_results(
    launch_records: list[dict[str, Any]],
    *,
    local_out_dir: Path | None = None,
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
            )
        )
    return results
