from __future__ import annotations

from chronohorn.families.causal_bank.scan import (
    _command_from_spec,
    _training_spec,
    default_frontier_topology,
)


def test_command_from_spec_includes_all_keys():
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


def test_static_bank_gate_conditional():
    spec_on = _training_spec(static_bank_gate=True)
    spec_off = _training_spec(static_bank_gate=False)
    topology = default_frontier_topology()

    cmd_on = _command_from_spec(spec_on, row_name="test", topology=topology)
    cmd_off = _command_from_spec(spec_off, row_name="test", topology=topology)

    assert "--static-bank-gate" in cmd_on
    assert "--static-bank-gate" not in cmd_off


def test_local_scale_override_conditional():
    spec_with = _training_spec(local_scale_override=0.5)
    spec_none = _training_spec(local_scale_override=None)
    topology = default_frontier_topology()

    cmd_with = _command_from_spec(spec_with, row_name="test", topology=topology)
    cmd_none = _command_from_spec(spec_none, row_name="test", topology=topology)

    assert "--local-scale-override 0.5" in cmd_with
    assert "--local-scale-override" not in cmd_none
