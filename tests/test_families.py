"""Tests for the family-agnostic registry system."""
import sys
from pathlib import Path

import pytest


def _has_decepticons() -> bool:
    try:
        import decepticons  # noqa: F401
        return True
    except ImportError:
        return False


_needs_decepticons = pytest.mark.skipif(
    not _has_decepticons(),
    reason="decepticons not installed — causal-bank adapter unavailable",
)


def test_family_auto_discovery():
    from chronohorn.families.registry import available_family_ids
    families = available_family_ids()
    assert "causal-bank" in families
    assert "polyhash" in families
    assert "transformer" in families


def test_resolve_causal_bank_adapter_without_decepticons_on_path(monkeypatch):
    import chronohorn.families.registry as registry

    decepticons_src = str(Path(__file__).resolve().parents[2] / "decepticons" / "src")
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != decepticons_src])

    registry._training_adapter_cache.pop("causal-bank", None)
    registry._adapter_load_failures.discard("causal-bank")
    registry._alias_map = None

    for module_name in list(sys.modules):
        if module_name == "chronohorn.families.causal_bank.adapter" or module_name.startswith(
            "chronohorn.families.causal_bank.training."
        ):
            sys.modules.pop(module_name, None)

    adapter = registry.resolve_training_adapter("causal-bank")

    assert adapter.family_id == "causal-bank"
    assert "chronohorn.families.causal_bank.training.causal_bank_training_primitives" not in sys.modules


@_needs_decepticons
def test_resolve_family_id():
    from chronohorn.families.registry import resolve_family_id
    assert resolve_family_id("polyhash_v6") == "polyhash"
    assert resolve_family_id("hash_embed") == "polyhash"
    assert resolve_family_id("opc") == "causal-bank"
    assert resolve_family_id("causal_bank") == "causal-bank"
    assert resolve_family_id("gpt2") == "transformer"
    assert resolve_family_id("llama") == "transformer"
    assert resolve_family_id("unknown_thing_xyz") is None


@_needs_decepticons
def test_detect_family_from_config():
    from chronohorn.families.registry import detect_family
    assert detect_family({"architecture": "polyhash_v6"}) == "polyhash"
    assert detect_family({"architecture": "gpt2"}) == "transformer"
    # OPC markers (no explicit architecture)
    assert detect_family({"linear_readout_kind": "mlp", "local_window": 4}) == "causal-bank"
    # Empty config
    assert detect_family({}) is None


@_needs_decepticons
def test_detect_illegal():
    from chronohorn.families.registry import detect_illegal
    # Polyhash: never illegal
    assert detect_illegal({"model": {"architecture": "polyhash"}, "config": {"train": {}}}) == False
    # Transformer: legal by default
    assert detect_illegal({"model": {"architecture": "gpt2"}, "config": {"train": {}}}) == False
    # Transformer: bidirectional is illegal
    assert detect_illegal({"model": {"architecture": "gpt2", "bidirectional": True}, "config": {"train": {}}}) == True
    # Causal-bank: patch leakage
    assert detect_illegal({
        "name": "sub1-patch4-test",
        "model": {"test_bpb": 0.55, "linear_readout_kind": "mlp", "local_window": 4},
        "config": {"train": {"steps": 2000}}
    }, family_id="causal-bank") == True


@_needs_decepticons
def test_adapter_config_summary():
    from chronohorn.families.registry import get_adapter
    # Polyhash
    ph = get_adapter("polyhash")
    summary = ph.config_summary({"config": {"train": {"num_tables": 8, "scan_dim": 256}}})
    assert "num_tables" in summary
    # Transformer
    tf = get_adapter("transformer")
    summary = tf.config_summary({"config": {"train": {"n_layers": 12, "n_heads": 8, "n_embd": 768}}})
    assert "n_layers" in summary


@_needs_decepticons
def test_adapter_training_entrypoints():
    from chronohorn.families.registry import get_adapter
    # Causal-bank has entrypoints
    cb = get_adapter("causal-bank")
    eps = cb.training_entrypoints()
    assert len(eps) > 0
    # Transformer has none (external training)
    tf = get_adapter("transformer")
    assert len(tf.training_entrypoints()) == 0


@_needs_decepticons
def test_causal_bank_torch_bridge_defaults_probe_diagnostics_off():
    import argparse

    from chronohorn.families.causal_bank.training.causal_bank_training_primitives import (
        add_torch_bridge_arguments,
    )

    parser = argparse.ArgumentParser()
    add_torch_bridge_arguments(parser)
    args = parser.parse_args([])

    assert args.probe_diagnostics is False


def test_concurrent_db_reads():
    """Verify DB doesn't segfault under concurrent reads."""
    import tempfile
    import threading

    from chronohorn.db import ChronohornDB

    with tempfile.NamedTemporaryFile(suffix=".db") as handle:
        db = ChronohornDB(handle.name)
    # Insert some data
        db.record_result("test-a", {
            "model": {"test_bpb": 1.5, "architecture": "test"},
            "config": {"train": {"steps": 1000}},
            "training": {"performance": {"tokens_per_second": 100000, "elapsed_sec": 60, "steps_completed": 1000}, "probes": [{"step": 500, "bpb": 2.0}, {"step": 1000, "bpb": 1.5}]},
        })

        errors = []
        def reader():
            try:
                for _ in range(50):
                    db.frontier(5)
                    db.summary()
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0, f"Concurrent read errors: {errors}"
        db.close()
