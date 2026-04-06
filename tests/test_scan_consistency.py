from __future__ import annotations

from chronohorn.families.causal_bank.scan import (
    _torch_train_command,
    _training_spec,
    build_exotic_16mb_scan,
    default_frontier_topology,
)


def test_training_spec_keys_appear_in_command():
    """Every tuneable key in _training_spec should map to a CLI arg in _torch_train_command."""
    topology = default_frontier_topology()
    spec = _training_spec()
    command = _torch_train_command(row_name="test", topology=topology)

    # Keys that map to positional/other args (not --key value)
    metadata_only = {"profile", "steps", "seq_len", "batch_size", "seed",
                     "linear_readout_num_experts", "scale", "variant",
                     "linear_readout_kind", "local_window"}

    key_to_arg = {
        "oscillatory_frac": "--oscillatory-frac",
        "oscillatory_schedule": "--oscillatory-schedule",
        "oscillatory_period_min": "--oscillatory-period-min",
        "oscillatory_period_max": "--oscillatory-period-max",
        "input_proj_scheme": "--input-proj-scheme",
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
