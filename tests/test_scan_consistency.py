from __future__ import annotations

from chronohorn.families.causal_bank.scan import (
    CAUSAL_BANK_FRONTIER_EMITTER,
    _torch_train_command,
    _training_spec,
    build_breakthrough_10k_scan,
    build_bottleneck_break_scan,
    build_exotic_16mb_scan,
    build_gated_retention_scan,
    build_toward_one_scan,
    build_toward_one_next_scan,
    default_frontier_topology,
)
from chronohorn.fleet.family_matrix import _topology_from_args, parse_args


def test_training_spec_keys_appear_in_command():
    """Every tuneable key in _training_spec should map to a CLI arg in _torch_train_command."""
    topology = default_frontier_topology()
    spec = _training_spec()
    command = _torch_train_command(row_name="test", topology=topology)

    assert "PYTHONPATH=python:../decepticons/src" in command
    assert "--table-path" not in command

    # Keys that map to positional/other args (not --key value)
    metadata_only = {"profile", "steps", "seq_len", "batch_size", "seed",
                     "linear_readout_num_experts", "scale", "variant",
                     "linear_readout_kind", "local_window"}

    key_to_arg = {
        "state_impl": "--state-impl",
        "oscillatory_frac": "--oscillatory-frac",
        "oscillatory_schedule": "--oscillatory-schedule",
        "oscillatory_period_min": "--oscillatory-period-min",
        "oscillatory_period_max": "--oscillatory-period-max",
        "input_proj_scheme": "--input-proj-scheme",
        "readout_bands": "--readout-bands",
        "linear_half_life_max": "--linear-half-life-max",
        "bank_gate_span": "--bank-gate-span",
        "learning_rate": "--learning-rate",
        "weight_decay": "--weight-decay",
    }

    for key in spec:
        if key in metadata_only:
            continue
        if key == "static_bank_gate":
            assert "--static-bank-gate" in command, "static_bank_gate not in command"
            continue
        if key == "local_scale_override":
            assert "--local-scale-override" in command, "local_scale_override not in command"
            continue
        if key in key_to_arg:
            assert key_to_arg[key] in command, (
                f"spec key {key!r} maps to {key_to_arg[key]!r} but that arg is missing from the command string"
            )


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


def test_exotic_16mb_scan_emits_rows():
    """Basic structural check on the exotic-16mb regime."""
    rows = build_exotic_16mb_scan()
    assert len(rows) > 0
    for row in rows:
        name = row["name"]
        assert "command" in row, f"{name}: missing command"
        assert "family" in row, f"{name}: missing family"
        assert row.get("gpu") is True, f"{name}: should request GPU"


def test_bottleneck_break_scan_targets_replication_and_crosses():
    rows = build_bottleneck_break_scan()
    names = {str(row["name"]) for row in rows}

    assert {
        "cb-break-s8-experts-seed43-50k",
        "cb-break-s8-experts-seed44-50k",
        "cb-break-s12-experts-50k",
        "cb-break-s8-experts-bands4-50k",
        "cb-break-s8-experts-mixing-50k",
        "cb-break-s8-experts-decays-50k",
        "cb-break-s8-experts-ssm16-50k",
        "cb-break-s8-experts-mincorr-split-50k",
    } <= names

    for row in rows:
        assert row["steps"] == 50_000
        assert row["linear_readout_kind"] == "routed_sqrelu_experts"
        assert row["linear_readout_num_experts"] == 4
        assert row["local_window"] == 8
        assert row["learning_rate"] == 0.0015
        assert row["static_bank_gate"] is False
        assert "--table-path" not in row["command"]

    scale8_rows = [row for row in rows if float(row["scale"]) == 8.0]
    assert scale8_rows
    for row in scale8_rows:
        assert row["min_gpu_mem_gb"] == 7.0
        assert row["gpu_placement_policy"] == "smallest_sufficient"

    scale12 = next(row for row in rows if float(row["scale"]) == 12.0)
    assert scale12["min_gpu_mem_gb"] == 12.0
    assert scale12.get("gpu_placement_policy") in {None, "fastest"}


def test_family_matrix_uses_family_default_topology():
    args = parse_args(["--family", "causal-bank", "--regime", "bottleneck-break"])
    topology = _topology_from_args(args, emitter=CAUSAL_BANK_FRONTIER_EMITTER)

    assert "slop-home" in topology.hosts
    assert topology.snapshot_paths
    assert topology.remote_data_root == "/data/chronohorn/fineweb10B_sp1024"


def test_breakthrough_10k_scan_is_dense_and_controlled():
    rows = build_breakthrough_10k_scan()
    names = {str(row["name"]) for row in rows}

    assert {
        "cb-rapid-s8-base-10k",
        "cb-rapid-s8-experts-10k",
        "cb-rapid-s12-base-10k",
        "cb-rapid-s8-bands4-10k",
        "cb-rapid-s8-mixing-10k",
        "cb-rapid-s8-decays-10k",
        "cb-rapid-s8-ssm16-10k",
        "cb-rapid-s8-mincorr-split-10k",
        "cb-rapid-s8-mixing-ssm16-10k",
        "cb-rapid-s8-decays-ssm16-10k",
        "cb-rapid-s8-mixing-mincorr-split-10k",
        "cb-rapid-s8-decays-mincorr-split-10k",
    } <= names

    assert len(rows) == 12
    for row in rows:
        assert row["steps"] == 10_000
        assert row["profile"] == "pilot"
        assert row["static_bank_gate"] is False
        assert row["local_window"] == 8
        assert row["learning_rate"] == 0.0015
        assert "--table-path" not in row["command"]

    scale8_rows = [row for row in rows if float(row["scale"]) == 8.0]
    assert scale8_rows
    for row in scale8_rows:
        assert row["min_gpu_mem_gb"] == 7.0
        assert row["gpu_placement_policy"] == "smallest_sufficient"

    scale12 = next(row for row in rows if float(row["scale"]) == 12.0)
    assert scale12["min_gpu_mem_gb"] == 12.0


def test_toward_one_scan_extends_breakthrough_with_scale_and_context_survival():
    rows = build_toward_one_scan()
    names = {str(row["name"]) for row in rows}

    assert {
        "cb-rapid-s8-base-10k",
        "cb-rapid-s8-experts-10k",
        "cb-frontier-s12-bands4-10k",
        "cb-frontier-s12-mixing-10k",
        "cb-frontier-s12-ssm16-10k",
        "cb-frontier-s12-scan-h4-10k",
        "cb-frontier-s12-retention-h4-10k",
        "cb-frontier-s12-mincorr-split-10k",
        "cb-frontier-s8-base-seq512-10k",
        "cb-frontier-s8-bands4-seq512-10k",
        "cb-frontier-s8-mixing-seq512-10k",
        "cb-frontier-s8-ssm16-seq512-10k",
        "cb-frontier-s8-hemi2-10k",
        "cb-frontier-s8-hemi2-mixing-10k",
        "cb-frontier-s8-block2-mixing-10k",
        "cb-frontier-s8-ssm32-10k",
        "cb-frontier-s8-scan-h4-10k",
        "cb-frontier-s8-retention-h4-10k",
        "cb-frontier-s8-mixing-ssm32-10k",
        "cb-frontier-s8-bands4-mixing-10k",
        "cb-frontier-s8-bands4-ssm16-10k",
        "cb-frontier-s8-bands4-mincorr-split-10k",
        "cb-frontier-s8-retention-h4-bands4-10k",
    } <= names

    assert len(rows) == 35
    for row in rows:
        assert row["steps"] == 10_000
        assert row["profile"] == "pilot"
        assert row["static_bank_gate"] is False
        assert row["learning_rate"] == 0.0015
        assert "--table-path" not in row["command"]

    seq512_rows = [row for row in rows if int(row["seq_len"]) == 512]
    assert seq512_rows
    for row in seq512_rows:
        assert row["batch_size"] == 8
        assert row["min_gpu_mem_gb"] == 7.0
        assert row["gpu_placement_policy"] == "smallest_sufficient"

    scale12_rows = [row for row in rows if float(row["scale"]) == 12.0]
    assert scale12_rows
    for row in scale12_rows:
        assert row["min_gpu_mem_gb"] == 12.0

    retention_rows = [row for row in rows if row.get("state_impl") == "retention"]
    assert retention_rows
    for row in retention_rows:
        assert row["state_dim"] == 16
        assert row["num_heads"] == 4


def test_toward_one_next_scan_stacks_winners_without_dropping_prior_frontier_rows():
    rows = build_toward_one_next_scan()
    names = {str(row["name"]) for row in rows}

    assert {
        "cb-frontier-s12-scan-h4-10k",
        "cb-frontier-s12-retention-h4-10k",
        "cb-next-s8-decays-ssm16-mincorr-split-10k",
        "cb-next-s8-bands4-decays-10k",
        "cb-next-s8-bands4-decays-ssm16-10k",
        "cb-next-s8-bands4-decays-ssm16-mincorr-split-10k",
        "cb-next-s8-decays-ssm16-seq512-10k",
        "cb-next-s8-bands4-decays-ssm16-seq512-10k",
        "cb-next-s8-scan-h4-bands4-10k",
        "cb-next-s8-retention-h4-decays-10k",
        "cb-next-s12-decays-ssm16-10k",
        "cb-next-s12-bands4-decays-ssm16-10k",
        "cb-next-s12-bands4-decays-ssm16-mincorr-split-10k",
        "cb-next-s12-scan-h4-bands4-10k",
        "cb-next-s12-retention-h4-decays-10k",
    } <= names

    assert len(rows) == 52

    seq512_rows = [row for row in rows if int(row["seq_len"]) == 512]
    assert seq512_rows
    for row in seq512_rows:
        assert row["batch_size"] == 8
        assert row["min_gpu_mem_gb"] == 7.0
        assert row["gpu_placement_policy"] == "smallest_sufficient"

    scale12_rows = [row for row in rows if float(row["scale"]) == 12.0]
    assert scale12_rows
    for row in scale12_rows:
        assert row["min_gpu_mem_gb"] == 12.0

    retention_rows = [row for row in rows if row.get("state_impl") == "retention"]
    assert retention_rows
    for row in retention_rows:
        assert row["state_dim"] == 16
        assert row["num_heads"] == 4

    scan_rows = [row for row in rows if row.get("state_impl") == "scan" and int(row.get("state_dim", 0)) == 16]
    assert scan_rows
    for row in scan_rows:
        if int(row.get("num_heads", 1)) > 1:
            assert row["num_heads"] == 4


def test_gated_retention_scan_appends_primary_learned_substrate_rows():
    rows = build_gated_retention_scan()
    names = {str(row["name"]) for row in rows}

    assert {
        "cb-frontier-s12-scan-h4-10k",
        "cb-next-s12-bands4-decays-ssm16-10k",
        "cb-substrate-s8-gret-h4-10k",
        "cb-substrate-s8-gret-h4-bands4-10k",
        "cb-substrate-s8-gret-h4-hemi2-10k",
        "cb-substrate-s8-gret-h4-seq512-10k",
        "cb-substrate-s12-gret-h4-10k",
        "cb-substrate-s12-gret-h4-bands4-10k",
        "cb-substrate-s12-gret-h4-hemi2-10k",
        "cb-substrate-s12-gret-h8-10k",
    } <= names

    assert len(rows) == 60

    gated_rows = [row for row in rows if "gret" in str(row["name"])]
    assert gated_rows
    for row in gated_rows:
        assert row["substrate_mode"] == "gated_retention"
        assert row["state_impl"] == "retention"
        assert int(row["state_dim"]) > 0
        assert int(row["num_heads"]) >= 4

    seq512_rows = [row for row in gated_rows if int(row["seq_len"]) == 512]
    assert seq512_rows
    for row in seq512_rows:
        assert row["batch_size"] == 8
        assert row["min_gpu_mem_gb"] == 7.0
        assert row["gpu_placement_policy"] == "smallest_sufficient"

    scale12_rows = [row for row in gated_rows if float(row["scale"]) == 12.0]
    assert scale12_rows
    for row in scale12_rows:
        assert row["min_gpu_mem_gb"] == 12.0

    scan_head_rows = [row for row in rows if row["name"] in {"cb-frontier-s8-scan-h4-10k", "cb-frontier-s12-scan-h4-10k"}]
    assert scan_head_rows
    for row in scan_head_rows:
        assert row["state_impl"] == "scan"
        assert row["num_heads"] == 4
