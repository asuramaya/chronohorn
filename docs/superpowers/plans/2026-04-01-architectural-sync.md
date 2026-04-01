# OPC/Chronohorn Architectural Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 6 architectural inconsistencies between OPC (kernel) and Chronohorn (runtime), generalizing both systems toward abstraction rather than overcommitting to one model.

**Architecture:** Three kernel-side changes (OPC validation, memory protocol, config serialization) and three runtime-side changes (spec derivation, result caching, forecaster fallback). Each fix respects the boundary: OPC owns mechanisms and protocols, Chronohorn owns execution and policy.

**Tech Stack:** Python 3.9+, dataclasses, pytest

---

## File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Modify | `chronohorn: python/chronohorn/families/causal_bank/scan.py` | Derive command from spec (issue #3) |
| Modify | `chronohorn: python/chronohorn/fleet/forecast_results.py` | Fallback to manifest artifact_mb_est (issue #1) |
| Modify | `chronohorn: python/chronohorn/pipeline.py` | Check local results before SSH (issue #2) |
| Modify | `chronohorn: python/chronohorn/train/causal_bank_training_support.py` | Add input_proj_scheme to export substrate (issue #5) |
| Create | `opc: src/open_predictive_coder/memory_protocol.py` | Memory attachment protocol (issue #4) |
| Modify | `opc: src/open_predictive_coder/causal_bank.py` | Add memory_kind field to CausalBankConfig (issue #4) |
| Modify | `opc: src/open_predictive_coder/__init__.py` | Export memory protocol symbols |
| Modify | `chronohorn: python/chronohorn/train/causal_bank_training_primitives.py` | Add --memory-kind CLI arg (issue #4) |
| Create | `chronohorn: tests/test_spec_derivation.py` | Test spec-to-command derivation |
| Create | `chronohorn: tests/test_artifact_fallback.py` | Test forecaster artifact fallback |
| Create | `chronohorn: tests/test_result_cache.py` | Test local result cache |

---

### Task 1: Derive training command from spec dict

Eliminate the spec/command duplication in scan.py. The spec dict already has all the knobs and values — the command should be derived from it.

**Files:**
- Modify: `chronohorn: python/chronohorn/families/causal_bank/scan.py`
- Create: `chronohorn: tests/test_spec_derivation.py`

- [ ] **Step 1: Write test for spec-derived command**

```python
# tests/test_spec_derivation.py
from __future__ import annotations

from chronohorn.families.causal_bank.scan import (
    _training_spec,
    _torch_train_command,
    _command_from_spec,
    default_frontier_topology,
)


def test_command_from_spec_includes_all_keys():
    """Every spec key with a CLI mapping should appear in the derived command."""
    spec = _training_spec(
        oscillatory_schedule="mincorr_greedy",
        input_proj_scheme="split_banks",
        oscillatory_frac=0.95,
        scale=14.0,
        learning_rate=0.002,
    )
    topology = default_frontier_topology()
    command = _command_from_spec(spec, row_name="test", topology=topology)

    assert "--oscillatory-schedule mincorr_greedy" in command
    assert "--input-proj-scheme split_banks" in command
    assert "--oscillatory-frac 0.95" in command
    assert "--scale 14.0" in command
    assert "--learning-rate 0.002" in command


def test_command_from_spec_matches_legacy():
    """Derived command should produce equivalent flags to the old hand-built command."""
    topology = default_frontier_topology()
    spec = _training_spec()
    old_cmd = _torch_train_command(row_name="test", topology=topology)
    new_cmd = _command_from_spec(spec, row_name="test", topology=topology)

    # Both should contain the same core flags
    for flag in ["--scale", "--steps", "--seed", "--oscillatory-frac",
                 "--oscillatory-schedule", "--input-proj-scheme",
                 "--learning-rate", "--weight-decay", "--linear-readout-kind"]:
        assert flag in new_cmd, f"Missing {flag} in derived command"


def test_static_bank_gate_conditional():
    """--static-bank-gate should only appear when True."""
    spec_on = _training_spec(static_bank_gate=True)
    spec_off = _training_spec(static_bank_gate=False)
    topology = default_frontier_topology()

    cmd_on = _command_from_spec(spec_on, row_name="test", topology=topology)
    cmd_off = _command_from_spec(spec_off, row_name="test", topology=topology)

    assert "--static-bank-gate" in cmd_on
    assert "--static-bank-gate" not in cmd_off
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/asuramaya/Code/carving_machine_v3/chronohorn && PYTHONPATH=python:../open-predictive-coder/src python3 -m pytest tests/test_spec_derivation.py -v`
Expected: ImportError — `_command_from_spec` does not exist

- [ ] **Step 3: Implement _command_from_spec**

Add this function to `python/chronohorn/families/causal_bank/scan.py`, before `_torch_train_command`:

```python
# Mapping from _training_spec keys to CLI flags.
# Keys not in this map are either metadata-only or handled specially.
_SPEC_KEY_TO_FLAG: dict[str, str] = {
    "scale": "--scale",
    "steps": "--steps",
    "seq_len": "--seq-len",
    "batch_size": "--batch-size",
    "seed": "--seed",
    "variant": "--variant",
    "profile": "--profile",
    "linear_readout_kind": "--linear-readout-kind",
    "linear_readout_num_experts": "--linear-readout-num-experts",
    "linear_half_life_max": "--linear-half-life-max",
    "oscillatory_frac": "--oscillatory-frac",
    "oscillatory_schedule": "--oscillatory-schedule",
    "oscillatory_period_min": "--oscillatory-period-min",
    "oscillatory_period_max": "--oscillatory-period-max",
    "input_proj_scheme": "--input-proj-scheme",
    "local_window": "--local-window",
    "learning_rate": "--learning-rate",
    "weight_decay": "--weight-decay",
    "bank_gate_span": "--bank-gate-span",
    "local_scale_override": "--local-scale-override",
}

# Boolean flags (emit flag name when True, omit when False)
_SPEC_BOOL_FLAGS: dict[str, str] = {
    "static_bank_gate": "--static-bank-gate",
}

# Keys that are manifest metadata only (not CLI args)
_SPEC_METADATA_KEYS = frozenset({"profile"})  # profile IS a CLI arg, keep in _SPEC_KEY_TO_FLAG


def _command_from_spec(
    spec: dict[str, object],
    *,
    row_name: str,
    topology: "FrontierTopology",
    probe_policy: str = "explicit",
    probe_eval_batches: int = 8,
    probe_steps: str | None = None,
    final_eval_batches: int = 20,
    probe_geometric_start: int = 100,
    probe_geometric_ratio: float = 2.0,
    probe_micro_cutoff_step: int = 400,
    probe_standard_eval_batches: int | None = None,
    probe_micro_eval_batches: int | None = None,
    probe_promotion_eval_batches: int | None = None,
    probe_promotion_count: int = 1,
) -> str:
    """Build the training command string from a spec dict.

    Single source of truth: the spec dict drives both manifest metadata
    and the CLI command.  No need to maintain parallel parameter lists.
    """
    steps = spec.get("steps", 1000)
    probe_args = (
        f"--probe-steps {probe_steps or steps} --probe-eval-batches {probe_eval_batches} "
        if probe_policy == "explicit"
        else (
            _adaptive_probe_args(
                geometric_start=probe_geometric_start,
                geometric_ratio=probe_geometric_ratio,
                micro_cutoff_step=probe_micro_cutoff_step,
                standard_eval_batches=(
                    probe_standard_eval_batches
                    if probe_standard_eval_batches is not None
                    else probe_eval_batches
                ),
                micro_eval_batches=(
                    probe_micro_eval_batches
                    if probe_micro_eval_batches is not None
                    else max(1, min(probe_eval_batches, max(probe_eval_batches // 2, 1)))
                ),
                promotion_eval_batches=(
                    probe_promotion_eval_batches
                    if probe_promotion_eval_batches is not None
                    else max(probe_eval_batches * 2, probe_eval_batches)
                ),
                promotion_count=probe_promotion_count,
            )
            + " "
        )
    )

    # Build flag string from spec
    parts: list[str] = []
    for key, flag in _SPEC_KEY_TO_FLAG.items():
        value = spec.get(key)
        if value is None:
            continue
        parts.append(f"{flag} {value}")

    for key, flag in _SPEC_BOOL_FLAGS.items():
        if spec.get(key):
            parts.append(flag)

    flags = " ".join(parts)

    train_command = (
        "PYTHONPATH=python python -m chronohorn train train-causal-bank-torch "
        f"--data-root {topology.remote_data_root} "
        + flags + " "
        + probe_args
        + f"--final-eval-batches {final_eval_batches} "
        + f"--device cuda "
        + f"--json /run/results/{row_name}.json"
    )

    args = [
        'if ! python -c "import sentencepiece" >/dev/null 2>&1; then python -m pip install -q sentencepiece; fi',
        "mkdir -p /run/results",
        train_command,
    ]
    return "; ".join(args)
```

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=python:../open-predictive-coder/src python3 -m pytest tests/test_spec_derivation.py -v`
Expected: 3 passed

- [ ] **Step 5: Update build_exotic_16mb_scan to use _command_from_spec**

In the `add()` helper inside `build_exotic_16mb_scan()`, replace the call to `_torch_train_command` with `_command_from_spec`:

Change from:
```python
command = _torch_train_command(row_name=name, topology=active_topology, probe_policy="adaptive", ...)
```

To:
```python
command = _command_from_spec(
    merged_spec, row_name=name, topology=active_topology,
    probe_policy="adaptive",
    probe_geometric_start=50,
    probe_geometric_ratio=2.0,
    probe_micro_cutoff_step=200,
    probe_standard_eval_batches=4,
    probe_micro_eval_batches=2,
    probe_promotion_eval_batches=promo_batches,
    probe_promotion_count=2,
    final_eval_batches=final_batches,
)
```

Where `merged_spec = _training_spec(**merged)` is already computed.

Note: keep `_torch_train_command` for backward compatibility with the older regimes.

- [ ] **Step 6: Run all tests**

Run: `PYTHONPATH=python:../open-predictive-coder/src python3 -m pytest tests/ -v`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add python/chronohorn/families/causal_bank/scan.py tests/test_spec_derivation.py
git commit -m "feat: derive training command from spec dict (single source of truth)"
```

---

### Task 2: Forecaster artifact_mb_est fallback

When the forecaster has no actual quantization results (short pilots), fall back to the manifest's `artifact_mb_est` field to determine artifact viability.

**Files:**
- Modify: `chronohorn: python/chronohorn/fleet/forecast_results.py`
- Create: `chronohorn: tests/test_artifact_fallback.py`

- [ ] **Step 1: Write test**

```python
# tests/test_artifact_fallback.py
from __future__ import annotations

from chronohorn.fleet.forecast_results import _resolve_artifact_viable


def test_resolve_uses_forecast_when_available():
    assert _resolve_artifact_viable(
        forecast_artifact_viable=True,
        manifest_artifact_mb_est=None,
        artifact_limit_mb=16.0,
    ) is True


def test_resolve_uses_forecast_false():
    assert _resolve_artifact_viable(
        forecast_artifact_viable=False,
        manifest_artifact_mb_est=10.0,
        artifact_limit_mb=16.0,
    ) is False


def test_resolve_falls_back_to_manifest_estimate_under():
    assert _resolve_artifact_viable(
        forecast_artifact_viable=None,
        manifest_artifact_mb_est=10.0,
        artifact_limit_mb=16.0,
    ) is True


def test_resolve_falls_back_to_manifest_estimate_over():
    assert _resolve_artifact_viable(
        forecast_artifact_viable=None,
        manifest_artifact_mb_est=20.0,
        artifact_limit_mb=16.0,
    ) is False


def test_resolve_unknown_when_no_data():
    assert _resolve_artifact_viable(
        forecast_artifact_viable=None,
        manifest_artifact_mb_est=None,
        artifact_limit_mb=16.0,
    ) is None
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Add _resolve_artifact_viable to forecast_results.py**

Add near the top of the file (after imports):

```python
def _resolve_artifact_viable(
    *,
    forecast_artifact_viable: bool | None,
    manifest_artifact_mb_est: float | None,
    artifact_limit_mb: float,
) -> bool | None:
    """Resolve artifact viability with fallback to manifest size estimate.

    Priority: actual forecast result > manifest estimate > unknown.
    """
    if forecast_artifact_viable is not None:
        return forecast_artifact_viable
    if manifest_artifact_mb_est is not None:
        return manifest_artifact_mb_est <= artifact_limit_mb
    return None
```

- [ ] **Step 4: Wire it into build_forecast_row**

In `build_forecast_row()`, after computing `artifact_viable` from the forecast, add the fallback. Find the line that sets `artifact_viable=bool(artifact.get("has_viable_artifact_path"))` and replace with:

```python
raw_artifact_viable = artifact.get("has_viable_artifact_path")
artifact_viable = _resolve_artifact_viable(
    forecast_artifact_viable=raw_artifact_viable if raw_artifact_viable is not None else None,
    manifest_artifact_mb_est=_safe_float(manifest_row.get("artifact_mb_est")) if manifest_row else None,
    artifact_limit_mb=budget.artifact_limit_mb,
)
```

Note: `build_forecast_row` needs access to the manifest row. Check its current signature — if it doesn't take a manifest_row parameter, add one with default `None`.

- [ ] **Step 5: Run tests**

Run: `PYTHONPATH=python:../open-predictive-coder/src python3 -m pytest tests/test_artifact_fallback.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add python/chronohorn/fleet/forecast_results.py tests/test_artifact_fallback.py
git commit -m "feat: forecaster falls back to manifest artifact_mb_est for viability"
```

---

### Task 3: Pipeline checks local results before SSH

The pipeline's ResultStage should check `out/results/{name}.json` before SSHing to remote hosts.

**Files:**
- Modify: `chronohorn: python/chronohorn/pipeline.py`
- Create: `chronohorn: tests/test_result_cache.py`

- [ ] **Step 1: Write test**

```python
# tests/test_result_cache.py
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from chronohorn.pipeline import _local_result_path, _try_local_result


def test_local_result_path():
    path = _local_result_path("ex-a-s12-mlp")
    assert path.name == "ex-a-s12-mlp.json"
    assert "out/results" in str(path)


def test_try_local_result_returns_payload(tmp_path: Path):
    local = tmp_path / "test-job.json"
    local.write_text(json.dumps({"model": {"test_bpb": 2.05}}))
    payload = _try_local_result("test-job", result_dir=tmp_path)
    assert payload is not None
    assert payload["model"]["test_bpb"] == 2.05


def test_try_local_result_returns_none_when_missing(tmp_path: Path):
    payload = _try_local_result("nonexistent", result_dir=tmp_path)
    assert payload is None
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Add local result helpers to pipeline.py**

Add near the existing `_fetch_remote_result_payload` function:

```python
_DEFAULT_LOCAL_RESULT_DIR = Path("out/results")


def _local_result_path(name: str, *, result_dir: Path | None = None) -> Path:
    return (result_dir or _DEFAULT_LOCAL_RESULT_DIR) / f"{name}.json"


def _try_local_result(name: str, *, result_dir: Path | None = None) -> dict[str, Any] | None:
    path = _local_result_path(name, result_dir=result_dir)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except (json.JSONDecodeError, OSError):
        return None
```

- [ ] **Step 4: Wire into _collect_result_payload_rows**

In the remote result discovery loop (around line 286-309), before calling `_fetch_remote_result_payload`, try the local cache first:

```python
# Try local cache first (populated by fleet drain)
local_payload = _try_local_result(run.name)
if local_payload is not None:
    rows.append({
        "name": run.name,
        "source": str(_local_result_path(run.name)),
        "payload": local_payload,
        "run_id": run.run_id,
        "path": str(_local_result_path(run.name)),
    })
    continue
# Fall through to SSH fetch
```

- [ ] **Step 5: Run tests**

Run: `PYTHONPATH=python:../open-predictive-coder/src python3 -m pytest tests/test_result_cache.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add python/chronohorn/pipeline.py tests/test_result_cache.py
git commit -m "feat: pipeline checks local result cache before SSH fetch"
```

---

### Task 4: Add input_proj_scheme to export substrate

The export bundle's `deterministic_substrate` dict is missing `input_proj_scheme`. Without it, Rust replay would regenerate the wrong bank.

**Files:**
- Modify: `chronohorn: python/chronohorn/train/causal_bank_training_support.py`

- [ ] **Step 1: Add input_proj_scheme to the substrate dict**

In `build_causal_bank_deterministic_substrate()` (line 31-69 of `causal_bank_training_support.py`), add `input_proj_scheme` to the dict. Find the line with `"oscillatory_schedule"` and add after it:

```python
        "input_proj_scheme": config.input_proj_scheme,
```

Also add `oscillatory_period_min` and `oscillatory_period_max` which are also missing:

```python
        "oscillatory_period_min": float(config.oscillatory_period_min),
        "oscillatory_period_max": float(config.oscillatory_period_max),
```

- [ ] **Step 2: Verify the export still works**

Run: `PYTHONPATH=python:../open-predictive-coder/src python3 -c "
from open_predictive_coder.causal_bank import CausalBankConfig, scale_config
from chronohorn.train.causal_bank_training_support import build_causal_bank_deterministic_substrate
cfg = scale_config(CausalBankConfig(input_proj_scheme='split_banks', oscillatory_schedule='mincorr_greedy', oscillatory_frac=0.95), 14.0)
sub = build_causal_bank_deterministic_substrate(cfg)
assert sub['input_proj_scheme'] == 'split_banks'
assert sub['oscillatory_schedule'] == 'mincorr_greedy'
assert sub['oscillatory_period_min'] == 4.0
print('export substrate: OK')
"`
Expected: `export substrate: OK`

- [ ] **Step 3: Commit**

```bash
git add python/chronohorn/train/causal_bank_training_support.py
git commit -m "fix: add input_proj_scheme and period range to export deterministic_substrate"
```

---

### Task 5: OPC memory attachment protocol

Define a kernel-side protocol for optional memory attachment to any substrate family. This is the interface — Chronohorn implements the runtime.

**Files:**
- Create: `opc: src/open_predictive_coder/memory_protocol.py`
- Modify: `opc: src/open_predictive_coder/causal_bank.py`
- Modify: `opc: src/open_predictive_coder/__init__.py`

- [ ] **Step 1: Create memory_protocol.py in OPC**

```python
# src/open_predictive_coder/memory_protocol.py
"""Protocol for optional memory attachment to substrate families.

A memory attachment provides a residual probability correction to the
substrate's base prediction.  The kernel defines the protocol; descendants
own the runtime implementation.

Memory kinds:
  - "none"         no memory attachment (default)
  - "ngram"        uses NgramMemoryConfig from this package
  - "exact_context" uses ExactContextConfig from this package
  - "statistical_backoff" uses StatisticalBackoffConfig from this package
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MEMORY_KINDS = ("none", "ngram", "exact_context", "statistical_backoff")


@dataclass(frozen=True)
class MemoryAttachmentConfig:
    """Configuration for an optional memory attachment.

    The kernel provides the config shape.  Descendants own the implementation:
    how the memory is built, trained, mixed with the substrate prediction, and
    packed into an artifact.
    """

    kind: str = "none"
    vocabulary_size: int = 256
    max_order: int = 3
    alpha: float = 0.05
    trigram_bucket_count: int = 4096
    mix_mode: str = "learned"  # "learned" or "fixed"

    def __post_init__(self) -> None:
        if self.kind not in MEMORY_KINDS:
            raise ValueError(f"Unknown memory kind: {self.kind!r}; expected one of {MEMORY_KINDS}")
        if self.vocabulary_size < 1:
            raise ValueError("vocabulary_size must be positive")
        if self.max_order < 1:
            raise ValueError("max_order must be >= 1")
        if self.trigram_bucket_count < 1:
            raise ValueError("trigram_bucket_count must be positive")


def validate_memory_config(config: MemoryAttachmentConfig) -> None:
    """Validate a memory attachment config (called by __post_init__)."""
    config.__post_init__()
```

- [ ] **Step 2: Add memory_kind to CausalBankConfig**

In OPC's `causal_bank.py`, add to `CausalBankConfig` after `init_seed`:

```python
    memory_kind: str = "none"
```

And in `validate_config()`, add:

```python
    from open_predictive_coder.memory_protocol import MEMORY_KINDS
    if config.memory_kind not in MEMORY_KINDS:
        raise ValueError(f"Unknown causal-bank memory_kind: {config.memory_kind!r}; expected one of {MEMORY_KINDS}")
```

- [ ] **Step 3: Export from __init__.py**

Add to OPC's `__init__.py`:

```python
from open_predictive_coder.memory_protocol import (
    MEMORY_KINDS,
    MemoryAttachmentConfig,
)
```

- [ ] **Step 4: Verify**

Run: `PYTHONPATH=../open-predictive-coder/src python3 -c "
from open_predictive_coder.memory_protocol import MemoryAttachmentConfig, MEMORY_KINDS
from open_predictive_coder.causal_bank import CausalBankConfig, validate_config
print('MEMORY_KINDS:', MEMORY_KINDS)
cfg = CausalBankConfig(memory_kind='ngram')
validate_config(cfg)
print('ngram config: OK')
cfg = CausalBankConfig(memory_kind='none')
validate_config(cfg)
print('none config: OK')
"`
Expected: all OK

- [ ] **Step 5: Commit OPC changes**

```bash
cd /Users/asuramaya/Code/carving_machine_v3/open-predictive-coder
git add src/open_predictive_coder/memory_protocol.py src/open_predictive_coder/causal_bank.py src/open_predictive_coder/__init__.py
git commit -m "feat: add memory attachment protocol and memory_kind to CausalBankConfig"
```

---

### Task 6: Chronohorn memory_kind CLI and export wiring

Wire the OPC memory protocol through Chronohorn's training CLI and export substrate, without implementing the full training path yet.

**Files:**
- Modify: `chronohorn: python/chronohorn/train/causal_bank_training_primitives.py`
- Modify: `chronohorn: python/chronohorn/train/causal_bank_training_support.py`
- Modify: `chronohorn: python/chronohorn/families/causal_bank/scan.py`

- [ ] **Step 1: Add --memory-kind to CLI**

In `causal_bank_training_primitives.py`, add after the `--input-proj-scheme` arg:

```python
    parser.add_argument(
        "--memory-kind",
        choices=("none", "ngram", "exact_context", "statistical_backoff"),
        default="none",
    )
```

- [ ] **Step 2: Wire into config builder**

In `build_causal_bank_variant_config`, add after the `input_proj_scheme` line:

```python
    if hasattr(args, "memory_kind") and args.memory_kind != "none":
        variant_cfg = replace(variant_cfg, memory_kind=args.memory_kind)
```

- [ ] **Step 3: Add to export substrate**

In `build_causal_bank_deterministic_substrate()` in `causal_bank_training_support.py`, change the hardcoded `"memory_kind": "linear_bank+local_window"` to:

```python
        "memory_kind": f"linear_bank+local_window+{config.memory_kind}" if config.memory_kind != "none" else "linear_bank+local_window",
```

- [ ] **Step 4: Add to scan spec and command derivation**

In `_training_spec()`, add `memory_kind` parameter with default `"none"` and include it in the returned dict.

In `_SPEC_KEY_TO_FLAG`, add:

```python
    "memory_kind": "--memory-kind",
```

- [ ] **Step 5: Verify CLI**

Run: `PYTHONPATH=python:../open-predictive-coder/src python3 -m chronohorn train train-causal-bank-torch --help 2>&1 | grep memory-kind`
Expected: `--memory-kind {none,ngram,exact_context,statistical_backoff}`

- [ ] **Step 6: Run all tests**

Run: `PYTHONPATH=python:../open-predictive-coder/src python3 -m pytest tests/ -v`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add python/chronohorn/train/causal_bank_training_primitives.py python/chronohorn/train/causal_bank_training_support.py python/chronohorn/families/causal_bank/scan.py
git commit -m "feat: wire memory_kind through CLI, config builder, export, and scan"
```

---

## Execution Order

```
Task 1 (spec derivation) — independent
Task 2 (artifact fallback) — independent
Task 3 (result cache) — independent
Task 4 (export fix) — independent
Task 5 (OPC memory protocol) — independent
Task 6 (chronohorn memory wiring) — depends on Task 5
```

Parallelizable: Tasks 1-5 are all independent. Task 6 depends on Task 5.
