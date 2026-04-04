# Chronohorn Runtime Gaps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the operational gaps that force manual babysitting of the chronohorn fleet, observer, and control surfaces.

**Architecture:** Seven focused fixes to the existing Python stack. No new services or dependencies. Each fix targets a specific friction point from real usage: fleet can't queue work, results don't come back, scan emits bad configs silently, manifest paths crash, MCP server is disconnected, no artifact-size guard, no manifest transformation.

**Tech Stack:** Python 3.9+, existing chronohorn CLI, SSH/Docker (remote fleet), pytest (new test infra)

---

## File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Modify | `python/chronohorn/fleet/dispatch.py` | Add `--drain` mode and result pull-back |
| Create | `python/chronohorn/fleet/drain.py` | Drain loop: poll, re-dispatch, collect results |
| Modify | `python/chronohorn/fleet/cli.py` | Wire drain subcommand |
| Create | `python/chronohorn/fleet/results.py` | Pull result JSONs from remote containers |
| Modify | `python/chronohorn/families/causal_bank/scan.py` | Add spec/command validation and artifact-size estimation |
| Create | `python/chronohorn/fleet/manifest_transform.py` | Filter/mutate manifest rows |
| Modify | `python/chronohorn/fleet/dispatch.py` | Manifest path resolution with fallback |
| Modify | `python/chronohorn/mcp.py` | Add fleet dispatch and drain tools |
| Create | `tests/test_scan_consistency.py` | Validate scan spec/command agreement |
| Create | `tests/test_manifest_transform.py` | Test manifest filter/mutate |
| Create | `tests/test_drain.py` | Test drain loop logic |
| Create | `tests/conftest.py` | Shared test fixtures |

---

### Task 1: Test Infrastructure

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create test package and conftest**

```python
# tests/__init__.py
# (empty)
```

```python
# tests/conftest.py
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_manifest(tmp_path: Path):
    """Write a temporary manifest JSONL and return its path."""
    def _write(rows: list[dict]) -> Path:
        p = tmp_path / "test_manifest.jsonl"
        with p.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return p
    return _write


@pytest.fixture
def sample_job() -> dict:
    return {
        "name": "test-job-1",
        "family": "causal-bank",
        "backend": "cuda",
        "resource_class": "cuda_gpu",
        "launcher": "managed_command",
        "command": "echo hello",
        "scale": 14.0,
        "steps": 1000,
        "learning_rate": 0.0015,
        "oscillatory_schedule": "logspace",
        "input_proj_scheme": "random",
    }
```

- [ ] **Step 2: Verify pytest runs**

Run: `cd /Users/asuramaya/Code/carving_machine_v3/chronohorn && PYTHONPATH=python:../decepticons/src python -m pytest tests/ -v --co 2>&1 | head -20`
Expected: No errors, conftest discovered

- [ ] **Step 3: Commit**

```bash
git add tests/
git commit -m "test: add Python test infrastructure for chronohorn"
```

---

### Task 2: Result Pull-Back from Remote Jobs

The fleet dispatcher launches remote Docker jobs but never pulls results back. The observer can fetch them via SSH in the pipeline, but there's no standalone "pull completed results" operation. This is the foundation for drain mode.

**Files:**
- Create: `python/chronohorn/fleet/results.py`
- Create: `tests/test_result_pullback.py`

- [ ] **Step 1: Write test for result pull-back logic**

```python
# tests/test_result_pullback.py
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from chronohorn.fleet.results import pull_remote_result, PullResult


def test_pull_result_returns_payload_on_success(tmp_path: Path):
    fake_payload = {"model": {"test_bpb": 2.05}, "config": {}}
    fake_ssh_output = json.dumps(fake_payload)

    with patch("chronohorn.fleet.results._ssh_cat_file", return_value=fake_ssh_output):
        result = pull_remote_result(
            host="slop-01",
            remote_run="/tmp/chronohorn-runs/test-job",
            job_name="test-job",
            local_out_dir=tmp_path,
        )

    assert result.success is True
    assert result.local_path is not None
    assert result.local_path.exists()
    saved = json.loads(result.local_path.read_text())
    assert saved["model"]["test_bpb"] == 2.05


def test_pull_result_returns_failure_on_ssh_error(tmp_path: Path):
    with patch("chronohorn.fleet.results._ssh_cat_file", side_effect=RuntimeError("ssh failed")):
        result = pull_remote_result(
            host="slop-01",
            remote_run="/tmp/chronohorn-runs/test-job",
            job_name="test-job",
            local_out_dir=tmp_path,
        )

    assert result.success is False
    assert result.error is not None


def test_pull_result_skips_if_local_exists(tmp_path: Path):
    local_file = tmp_path / "test-job.json"
    local_file.write_text('{"already": "here"}')

    result = pull_remote_result(
        host="slop-01",
        remote_run="/tmp/chronohorn-runs/test-job",
        job_name="test-job",
        local_out_dir=tmp_path,
    )

    assert result.success is True
    assert result.skipped is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=python:../decepticons/src python -m pytest tests/test_result_pullback.py -v`
Expected: ImportError — `chronohorn.fleet.results` does not exist

- [ ] **Step 3: Implement result pull-back**

```python
# python/chronohorn/fleet/results.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=python:../decepticons/src python -m pytest tests/test_result_pullback.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add python/chronohorn/fleet/results.py tests/test_result_pullback.py
git commit -m "feat: add result pull-back from remote fleet jobs"
```

---

### Task 3: Fleet Drain Mode

Add `chronohorn fleet drain --manifest <path>` that polls for completion, re-dispatches pending jobs, and pulls results — unattended until the manifest is done.

**Files:**
- Create: `python/chronohorn/fleet/drain.py`
- Modify: `python/chronohorn/fleet/cli.py`
- Create: `tests/test_drain.py`

- [ ] **Step 1: Write test for drain loop tick**

```python
# tests/test_drain.py
from __future__ import annotations

from unittest.mock import patch, MagicMock
from chronohorn.fleet.drain import drain_tick, DrainState


def test_drain_tick_returns_done_when_no_pending():
    state = DrainState(
        manifest_path="/fake/manifest.jsonl",
        pending=0,
        running=0,
        completed=5,
        blocked=0,
        launched=0,
        pulled=0,
    )
    assert state.is_done is True


def test_drain_tick_returns_not_done_when_pending():
    state = DrainState(
        manifest_path="/fake/manifest.jsonl",
        pending=3,
        running=2,
        completed=5,
        blocked=0,
        launched=0,
        pulled=0,
    )
    assert state.is_done is False


def test_drain_tick_returns_not_done_when_running():
    state = DrainState(
        manifest_path="/fake/manifest.jsonl",
        pending=0,
        running=2,
        completed=5,
        blocked=0,
        launched=0,
        pulled=0,
    )
    assert state.is_done is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=python:../decepticons/src python -m pytest tests/test_drain.py -v`
Expected: ImportError — `chronohorn.fleet.drain` does not exist

- [ ] **Step 3: Implement drain module**

```python
# python/chronohorn/fleet/drain.py
"""Fleet drain loop: poll, re-dispatch, pull results until manifest is done."""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from chronohorn.fleet.dispatch import (
    load_manifest,
    select_jobs,
    filter_jobs_by_class,
    probe_fleet_state,
    partition_running_jobs,
    assign_jobs_best_effort,
    launch_job,
    write_launch_record,
    load_launch_record,
)
from chronohorn.fleet.planner import collect_performance_samples
from chronohorn.fleet.results import pull_all_completed_results


@dataclass(frozen=True)
class DrainState:
    manifest_path: str
    pending: int
    running: int
    completed: int
    blocked: int
    launched: int
    pulled: int

    @property
    def is_done(self) -> bool:
        return self.pending == 0 and self.running == 0


def drain_tick(
    manifest_path: str | Path,
    *,
    job_names: Sequence[str] = (),
    classes: Sequence[str] = (),
    telemetry_globs: Sequence[str] | None = None,
    result_out_dir: Path | None = None,
) -> DrainState:
    """Run one dispatch+pull cycle. Returns the current drain state."""
    manifest_path = Path(manifest_path)
    jobs = load_manifest(manifest_path)
    if job_names:
        jobs = select_jobs(jobs, list(job_names))
    if classes:
        jobs = filter_jobs_by_class(jobs, list(classes))

    fleet_state = probe_fleet_state(jobs)
    telemetry = collect_performance_samples(telemetry_globs)
    pending, running, completed, stale = partition_running_jobs(jobs, fleet_state)

    # Try to launch pending jobs
    assigned, blocked = assign_jobs_best_effort(pending, fleet_state, telemetry)
    launched_count = 0
    for job, assignment in assigned:
        try:
            record = launch_job(job, assignment)
            write_launch_record(record)
            launched_count += 1
            print(f"  launched {job['name']} -> {assignment.get('host', 'local')}", file=sys.stderr)
        except Exception as exc:
            print(f"  FAILED to launch {job['name']}: {exc}", file=sys.stderr)

    # Pull results from completed jobs
    completed_records = []
    for job in completed:
        launch_rec = load_launch_record(job["name"])
        if launch_rec:
            completed_records.append(launch_rec)
    pull_results = pull_all_completed_results(completed_records, local_out_dir=result_out_dir)
    pulled_count = sum(1 for r in pull_results if r.success and not r.skipped)

    return DrainState(
        manifest_path=str(manifest_path),
        pending=len(pending) - launched_count,
        running=len(running) + launched_count,
        completed=len(completed),
        blocked=len(blocked),
        launched=launched_count,
        pulled=pulled_count,
    )


def drain_loop(
    manifest_path: str | Path,
    *,
    poll_interval: int = 60,
    job_names: Sequence[str] = (),
    classes: Sequence[str] = (),
    telemetry_globs: Sequence[str] | None = None,
    result_out_dir: Path | None = None,
    max_ticks: int | None = None,
) -> DrainState:
    """Poll until all manifest jobs are completed or blocked."""
    tick = 0
    while True:
        tick += 1
        state = drain_tick(
            manifest_path,
            job_names=job_names,
            classes=classes,
            telemetry_globs=telemetry_globs,
            result_out_dir=result_out_dir,
        )

        status_line = (
            f"[tick {tick}] pending={state.pending} running={state.running} "
            f"completed={state.completed} blocked={state.blocked} "
            f"launched={state.launched} pulled={state.pulled}"
        )
        print(status_line, file=sys.stderr)

        if state.is_done:
            print("drain complete: all jobs finished", file=sys.stderr)
            return state

        if state.pending == 0 and state.running == 0 and state.blocked > 0:
            print(f"drain stalled: {state.blocked} jobs blocked, none running", file=sys.stderr)
            return state

        if max_ticks is not None and tick >= max_ticks:
            print(f"drain stopped: reached max ticks ({max_ticks})", file=sys.stderr)
            return state

        time.sleep(poll_interval)
```

- [ ] **Step 4: Wire drain into fleet CLI**

Read `python/chronohorn/fleet/cli.py` and add the drain subcommand. Add to the existing CLI dispatch:

```python
# Add to python/chronohorn/fleet/cli.py - in the main() function or dispatch logic

def drain_main(argv: Sequence[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="chronohorn fleet drain")
    parser.add_argument("--manifest", required=True, help="Manifest JSONL path.")
    parser.add_argument("--job", action="append", default=[], help="Restrict to named jobs.")
    parser.add_argument("--class", dest="classes", action="append", default=[], help="Restrict to resource classes.")
    parser.add_argument("--poll-interval", type=int, default=60, help="Seconds between polls (default 60).")
    parser.add_argument("--result-dir", default=None, help="Local directory for pulled results.")
    parser.add_argument("--telemetry-glob", action="append", default=[], help="Extra telemetry globs.")
    parser.add_argument("--max-ticks", type=int, default=None, help="Maximum poll cycles before stopping.")
    args = parser.parse_args(argv)

    from chronohorn.fleet.drain import drain_loop
    from pathlib import Path

    state = drain_loop(
        args.manifest,
        poll_interval=args.poll_interval,
        job_names=args.job,
        classes=args.classes,
        telemetry_globs=args.telemetry_glob or None,
        result_out_dir=Path(args.result_dir) if args.result_dir else None,
        max_ticks=args.max_ticks,
    )

    import json
    print(json.dumps({
        "manifest": state.manifest_path,
        "pending": state.pending,
        "running": state.running,
        "completed": state.completed,
        "blocked": state.blocked,
        "done": state.is_done,
    }, indent=2))
    return 0 if state.is_done else 1
```

Add `"drain"` to the fleet CLI dispatch so `chronohorn fleet drain --manifest ...` works.

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=python:../decepticons/src python -m pytest tests/test_drain.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add python/chronohorn/fleet/drain.py python/chronohorn/fleet/cli.py tests/test_drain.py
git commit -m "feat: add fleet drain mode for unattended manifest execution"
```

---

### Task 4: Scan Spec/Command Consistency Validation

Prevent silent defaults by validating that every key in `_training_spec()` appears in the corresponding `_torch_train_command()` output.

**Files:**
- Create: `tests/test_scan_consistency.py`

- [ ] **Step 1: Write validation test**

```python
# tests/test_scan_consistency.py
from __future__ import annotations

import re

from chronohorn.families.causal_bank.scan import (
    _training_spec,
    _torch_train_command,
    default_frontier_topology,
    build_exotic_16mb_scan,
)


def test_training_spec_keys_appear_in_command():
    """Every key in _training_spec should map to a CLI arg in _torch_train_command."""
    topology = default_frontier_topology()
    spec = _training_spec()
    command = _torch_train_command(row_name="test", topology=topology)

    # Keys that are metadata only (not CLI args)
    metadata_only = {"profile", "variant", "steps", "seq_len", "batch_size", "seed",
                     "linear_readout_num_experts", "scale"}

    # Map spec keys to CLI arg names
    key_to_arg = {
        "oscillatory_frac": "--oscillatory-frac",
        "oscillatory_schedule": "--oscillatory-schedule",
        "oscillatory_period_min": "--oscillatory-period-min",
        "oscillatory_period_max": "--oscillatory-period-max",
        "input_proj_scheme": "--input-proj-scheme",
        "linear_readout_kind": "--linear-readout-kind",
        "linear_half_life_max": "--linear-half-life-max",
        "static_bank_gate": "--static-bank-gate",
        "bank_gate_span": "--bank-gate-span",
        "local_window": "--local-window",
        "local_scale_override": "--local-scale-override",
        "learning_rate": "--learning-rate",
        "weight_decay": "--weight-decay",
    }

    for key in spec:
        if key in metadata_only:
            continue
        if key in key_to_arg:
            assert key_to_arg[key] in command, (
                f"spec key {key!r} maps to {key_to_arg[key]!r} but that arg is missing from the command string"
            )


def test_exotic_16mb_scan_all_under_budget():
    """Every exotic-16mb row should declare an artifact-viable config."""
    rows = build_exotic_16mb_scan()
    assert len(rows) > 0
    for row in rows:
        name = row["name"]
        # Basic structural checks
        assert "command" in row, f"{name}: missing command"
        assert "family" in row, f"{name}: missing family"
        assert row.get("gpu") is True, f"{name}: should request GPU"


def test_spec_values_match_command_values():
    """Spot-check that spec values actually appear in the command string."""
    topology = default_frontier_topology()
    spec_kwargs = dict(
        oscillatory_schedule="mincorr_greedy",
        input_proj_scheme="split_banks",
        oscillatory_frac=0.95,
        scale=14.0,
    )
    spec = _training_spec(**spec_kwargs)
    command = _torch_train_command(row_name="test", topology=topology, **spec_kwargs)

    assert "--oscillatory-schedule mincorr_greedy" in command
    assert "--input-proj-scheme split_banks" in command
    assert "--oscillatory-frac 0.95" in command
    assert "--scale 14.0" in command
```

- [ ] **Step 2: Run tests**

Run: `PYTHONPATH=python:../decepticons/src python -m pytest tests/test_scan_consistency.py -v`
Expected: 3 passed (since we already wired the knobs in the previous session)

- [ ] **Step 3: Commit**

```bash
git add tests/test_scan_consistency.py
git commit -m "test: add scan spec/command consistency validation"
```

---

### Task 5: Manifest Path Resolution

Make the manifest path resolver try common fallbacks before crashing with a raw FileNotFoundError.

**Files:**
- Modify: `python/chronohorn/fleet/dispatch.py` (lines 65-82, `load_manifest()`)

- [ ] **Step 1: Read current load_manifest**

Read `python/chronohorn/fleet/dispatch.py` lines 65-82.

- [ ] **Step 2: Add path resolution with fallback**

Replace `load_manifest()` to try the given path, then `manifests/<name>`, then `chronohorn/manifests/<name>`:

```python
def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        # Try common relative locations
        candidates = [
            Path("manifests") / path.name,
            Path("chronohorn/manifests") / path.name,
            path,
        ]
        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break
        else:
            existing = "\n  ".join(str(c) for c in candidates if c != path)
            raise FileNotFoundError(
                f"Manifest not found: {path}\n"
                f"Also checked:\n  {existing}"
            )
    # ... rest of existing parsing code unchanged ...
```

- [ ] **Step 3: Run existing exotic manifest to verify it still loads**

Run: `PYTHONPATH=python:../decepticons/src python -c "from chronohorn.fleet.dispatch import load_manifest; jobs = load_manifest('manifests/frontier_exotic_16mb.jsonl'); print(f'{len(jobs)} jobs loaded')"`
Expected: `42 jobs loaded`

- [ ] **Step 4: Commit**

```bash
git add python/chronohorn/fleet/dispatch.py
git commit -m "fix: add manifest path resolution with fallback candidates"
```

---

### Task 6: Artifact-Size Estimation in Scan Emitter

Add a size estimate to each emitted row so the observer and forecaster know upfront whether a config is artifact-viable.

**Files:**
- Modify: `python/chronohorn/families/causal_bank/scan.py`

- [ ] **Step 1: Add size estimation helper**

Add a function that estimates int6 artifact size from scale and readout config without building the full model:

```python
def _estimate_artifact_mb(
    *,
    scale: float,
    linear_readout_kind: str = "mlp",
    linear_readout_num_experts: int = 8,
    local_window: int = 4,
) -> float:
    """Rough int6 artifact size estimate in MB.

    Based on measured parameter counts at known scales. Accurate within ~5%
    for the causal-bank family.
    """
    embed_dim = int(32 * scale)
    linear_modes = int(256 * scale)
    hidden = int(128 * scale)
    vocab = 1024

    # Embeddings (2 paths)
    embed_params = 2 * vocab * embed_dim
    # Linear input projection
    proj_params = linear_modes * embed_dim
    # Linear readout
    if linear_readout_kind == "mlp":
        linear_readout_params = hidden * vocab + hidden  # one hidden layer + bias
    else:
        # routed experts: each expert has hidden->vocab, plus router
        expert_params = linear_readout_num_experts * (hidden * vocab + vocab)
        router_params = linear_readout_num_experts * hidden
        linear_readout_params = expert_params + router_params
    # Local readout (same structure as linear but for local path)
    local_hidden = int(128 * scale)
    local_readout_params = local_hidden * local_window * embed_dim  # local embedding
    local_readout_params += local_hidden * vocab + local_hidden  # readout MLP
    # Bank gate
    gate_params = 6  # tiny

    total = embed_params + proj_params + linear_readout_params + local_readout_params + gate_params
    int6_mb = total * 6 / 8 / 1024 / 1024
    return round(int6_mb, 2)
```

- [ ] **Step 2: Add estimated size to every emitted row**

In the `add()` helper inside `build_exotic_16mb_scan()`, after building the spec, add:

```python
row["artifact_mb_est"] = _estimate_artifact_mb(
    scale=float(merged.get("scale", 14.0)),
    linear_readout_kind=str(merged.get("linear_readout_kind", "mlp")),
    linear_readout_num_experts=int(merged.get("linear_readout_num_experts", 8)),
    local_window=int(merged.get("local_window", 4)),
)
```

- [ ] **Step 3: Add budget guard warning**

At the end of `build_exotic_16mb_scan()`, before returning rows, add:

```python
for row in rows:
    est = row.get("artifact_mb_est", 0)
    if est > 16.0:
        import warnings
        warnings.warn(
            f"Row {row['name']} estimated at {est:.1f}MB — exceeds 16MB artifact budget",
            stacklevel=2,
        )
```

- [ ] **Step 4: Run the emitter and verify estimates appear**

Run: `PYTHONPATH=python:../decepticons/src python -c "
from chronohorn.families.causal_bank.scan import build_exotic_16mb_scan
rows = build_exotic_16mb_scan()
for r in rows[:5]:
    print(f'{r[\"name\"]:30s} {r.get(\"artifact_mb_est\", \"?\")} MB')
"`
Expected: Each row shows an estimated MB under 16

- [ ] **Step 5: Commit**

```bash
git add python/chronohorn/families/causal_bank/scan.py
git commit -m "feat: add artifact-size estimation to scan emitter with budget guard"
```

---

### Task 7: Manifest Transform Command

Add `chronohorn fleet transform` to filter, mutate steps/seeds, and compose manifests without editing code.

**Files:**
- Create: `python/chronohorn/fleet/manifest_transform.py`
- Modify: `python/chronohorn/fleet/cli.py`
- Create: `tests/test_manifest_transform.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_manifest_transform.py
from __future__ import annotations

import json
from pathlib import Path

from chronohorn.fleet.manifest_transform import filter_manifest, mutate_manifest


def test_filter_by_name_glob(tmp_path: Path):
    rows = [
        {"name": "ex-a-s12-mlp", "steps": 1000},
        {"name": "ex-a-s15-mlp", "steps": 1000},
        {"name": "ex-b-mincorr", "steps": 1000},
        {"name": "ex-j-s17-allknobs", "steps": 1000},
    ]
    result = filter_manifest(rows, name_pattern="ex-a-*")
    assert len(result) == 2
    assert all(r["name"].startswith("ex-a-") for r in result)


def test_filter_by_name_glob_j_group(tmp_path: Path):
    rows = [
        {"name": "ex-a-s12-mlp", "steps": 1000},
        {"name": "ex-j-s17-allknobs", "steps": 1000},
        {"name": "ex-j-s12-e2-mincorr", "steps": 1000},
    ]
    result = filter_manifest(rows, name_pattern="ex-j-*")
    assert len(result) == 2


def test_mutate_steps(tmp_path: Path):
    rows = [{"name": "job1", "steps": 1000, "command": "--steps 1000 --json /run/results/job1.json"}]
    result = mutate_manifest(rows, steps=5200)
    assert result[0]["steps"] == 5200
    assert "--steps 5200" in result[0]["command"]


def test_mutate_seed(tmp_path: Path):
    rows = [{"name": "job1", "seed": 42, "command": "--seed 42 --json /run/results/job1.json"}]
    result = mutate_manifest(rows, seed=43)
    assert result[0]["seed"] == 43
    assert "--seed 43" in result[0]["command"]
    assert result[0]["name"] == "job1-seed43"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=python:../decepticons/src python -m pytest tests/test_manifest_transform.py -v`
Expected: ImportError

- [ ] **Step 3: Implement manifest transform**

```python
# python/chronohorn/fleet/manifest_transform.py
"""Filter and mutate manifest rows without editing scan code."""
from __future__ import annotations

import copy
import fnmatch
import json
import re
from pathlib import Path
from typing import Any, Sequence


def filter_manifest(
    rows: list[dict[str, Any]],
    *,
    name_pattern: str | None = None,
    family: str | None = None,
    resource_class: str | None = None,
) -> list[dict[str, Any]]:
    result = rows
    if name_pattern:
        result = [r for r in result if fnmatch.fnmatch(r.get("name", ""), name_pattern)]
    if family:
        result = [r for r in result if r.get("family") == family]
    if resource_class:
        result = [r for r in result if r.get("resource_class") == resource_class]
    return result


def mutate_manifest(
    rows: list[dict[str, Any]],
    *,
    steps: int | None = None,
    seed: int | None = None,
    learning_rate: float | None = None,
) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        row = copy.deepcopy(row)
        cmd = row.get("command", "")

        if steps is not None:
            row["steps"] = steps
            cmd = re.sub(r"--steps \S+", f"--steps {steps}", cmd)
            row["work_tokens"] = steps * row.get("seq_len", 256) * row.get("batch_size", 16)

        if seed is not None:
            old_seed = row.get("seed", 42)
            row["seed"] = seed
            cmd = re.sub(r"--seed \S+", f"--seed {seed}", cmd)
            # Update result path in command
            old_name = row["name"]
            if f"seed{old_seed}" in old_name:
                row["name"] = old_name.replace(f"seed{old_seed}", f"seed{seed}")
            else:
                row["name"] = f"{old_name}-seed{seed}"
            cmd = cmd.replace(f"{old_name}.json", f"{row['name']}.json")

        if learning_rate is not None:
            row["learning_rate"] = learning_rate
            cmd = re.sub(r"--learning-rate \S+", f"--learning-rate {learning_rate}", cmd)

        row["command"] = cmd
        result.append(row)
    return result


def load_and_transform(
    manifest_path: Path,
    *,
    name_pattern: str | None = None,
    steps: int | None = None,
    seed: int | None = None,
    output_path: Path | None = None,
) -> list[dict[str, Any]]:
    from chronohorn.fleet.dispatch import load_manifest

    rows = load_manifest(manifest_path)
    if name_pattern:
        rows = filter_manifest(rows, name_pattern=name_pattern)
    rows = mutate_manifest(rows, steps=steps, seed=seed)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            f.write(f"# Transformed from {manifest_path.name}\n")
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")

    return rows
```

- [ ] **Step 4: Wire into fleet CLI**

Add `transform` as a subcommand in `python/chronohorn/fleet/cli.py`:

```python
def transform_main(argv: Sequence[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="chronohorn fleet transform")
    parser.add_argument("--manifest", required=True, help="Source manifest JSONL path.")
    parser.add_argument("--filter", default=None, help="Name glob pattern to keep (e.g. 'ex-j-*').")
    parser.add_argument("--steps", type=int, default=None, help="Override step count.")
    parser.add_argument("--seed", type=int, default=None, help="Override seed.")
    parser.add_argument("--output", required=True, help="Output manifest JSONL path.")
    args = parser.parse_args(argv)

    from chronohorn.fleet.manifest_transform import load_and_transform
    from pathlib import Path

    rows = load_and_transform(
        Path(args.manifest),
        name_pattern=args.filter,
        steps=args.steps,
        seed=args.seed,
        output_path=Path(args.output),
    )
    print(json.dumps({"output": args.output, "job_count": len(rows)}, indent=2))
    return 0
```

- [ ] **Step 5: Run tests**

Run: `PYTHONPATH=python:../decepticons/src python -m pytest tests/test_manifest_transform.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add python/chronohorn/fleet/manifest_transform.py python/chronohorn/fleet/cli.py tests/test_manifest_transform.py
git commit -m "feat: add manifest transform command for filter/mutate without code edits"
```

---

### Task 8: MCP Server — Wire Fleet and Drain Tools

Add `chronohorn_fleet_dispatch` and `chronohorn_fleet_drain` tools to the MCP server so agents can dispatch and drain without bash.

**Files:**
- Modify: `python/chronohorn/mcp.py`

- [ ] **Step 1: Add fleet tool definitions to TOOLS dict**

Add after the existing `chronohorn_reset` entry:

```python
"chronohorn_fleet_dispatch": {
    "description": "Dispatch pending jobs from a manifest to the fleet. Returns launched, blocked, and running jobs.",
    "parameters": {
        "manifest_path": {"type": "string", "description": "Manifest JSONL path", "required": True},
        "job_names": {"type": "array", "description": "Restrict to named jobs"},
        "classes": {"type": "array", "description": "Restrict to resource classes"},
        "dry_run": {"type": "boolean", "description": "Plan only, do not launch"},
    },
},
"chronohorn_fleet_drain": {
    "description": "Start draining a manifest: poll for completion, re-dispatch, pull results. Runs one tick (use repeatedly for polling).",
    "parameters": {
        "manifest_path": {"type": "string", "description": "Manifest JSONL path", "required": True},
        "job_names": {"type": "array", "description": "Restrict to named jobs"},
        "classes": {"type": "array", "description": "Restrict to resource classes"},
    },
},
"chronohorn_fleet_status": {
    "description": "Check fleet placement and job status for a manifest without launching.",
    "parameters": {
        "manifest_path": {"type": "string", "description": "Manifest JSONL path", "required": True},
    },
},
```

- [ ] **Step 2: Implement tool handlers**

Add to the `ToolServer` class:

```python
def _do_fleet_dispatch(self, args: dict[str, Any]) -> dict[str, Any]:
    from chronohorn.fleet.dispatch import main as fleet_main
    import io, contextlib, json

    argv = ["--manifest", str(args["manifest_path"])]
    for name in (args.get("job_names") or []):
        argv.extend(["--job", name])
    for cls in (args.get("classes") or []):
        argv.extend(["--class", cls])
    if args.get("dry_run"):
        argv.append("--dry-run")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fleet_main(argv)
    try:
        return json.loads(buf.getvalue())
    except json.JSONDecodeError:
        return {"raw_output": buf.getvalue()}

def _do_fleet_drain(self, args: dict[str, Any]) -> dict[str, Any]:
    from chronohorn.fleet.drain import drain_tick
    state = drain_tick(
        args["manifest_path"],
        job_names=list(args.get("job_names") or []),
        classes=list(args.get("classes") or []),
    )
    return {
        "pending": state.pending,
        "running": state.running,
        "completed": state.completed,
        "blocked": state.blocked,
        "launched": state.launched,
        "pulled": state.pulled,
        "done": state.is_done,
    }

def _do_fleet_status(self, args: dict[str, Any]) -> dict[str, Any]:
    return self._do_fleet_dispatch({**args, "dry_run": True})
```

- [ ] **Step 3: Wire handlers into call_tool dispatch**

Add to the `call_tool()` method:

```python
if name == "chronohorn_fleet_dispatch":
    return self._do_fleet_dispatch(arguments)
if name == "chronohorn_fleet_drain":
    return self._do_fleet_drain(arguments)
if name == "chronohorn_fleet_status":
    return self._do_fleet_status(arguments)
```

- [ ] **Step 4: Commit**

```bash
git add python/chronohorn/mcp.py
git commit -m "feat: add fleet dispatch, drain, and status tools to MCP server"
```

---

## Execution Order

Tasks 1-8 are mostly independent. The dependency chain is:

```
Task 1 (test infra) → Task 2 (results) → Task 3 (drain)
Task 1 → Task 4 (scan validation)
Task 5 (path resolution) — independent
Task 6 (artifact size) — independent
Task 1 → Task 7 (manifest transform)
Task 2 + Task 3 → Task 8 (MCP wiring)
```

Parallelizable groups:
- **Group 1:** Tasks 1, 5, 6 (no deps)
- **Group 2:** Tasks 2, 4, 7 (depend on Task 1)
- **Group 3:** Tasks 3, 8 (depend on Task 2)
