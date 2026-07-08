"""The checkpoint covenant gate: no training launch without a body.

Session 13 and the M4 micro-runs saved vitals only — the models are
gone. preflight_check now refuses training commands that provably save
no checkpoint, unless the manifest carries an explicit
``checkpoint_policy: exempt`` admission.
"""
from __future__ import annotations

import pytest

from chronohorn.fleet.preflight import PreflightError, preflight_check


def _job(command: str, **extra: object) -> dict[str, object]:
    return {"name": "covenant-test", "command": command, "host": "local", **extra}


def test_refuses_no_save_checkpoint_flag() -> None:
    job = _job("python -m chronohorn train --steps 1000 --no-save-checkpoint")
    with pytest.raises(PreflightError, match="checkpoint covenant"):
        preflight_check(job)


def test_no_save_checkpoint_allowed_when_exempt() -> None:
    job = _job(
        "python train_thing.py --steps 1000 --no-save-checkpoint",
        checkpoint_policy="exempt",
    )
    preflight_check(job)


def test_exempt_via_config_dict() -> None:
    job = _job(
        "python train_thing.py --steps 1000",
        config={"checkpoint_policy": "exempt"},
    )
    preflight_check(job)


def test_non_training_command_passes() -> None:
    preflight_check(_job("python -m chronohorn export --session 12"))


def test_registry_trainer_passes_without_explicit_flags() -> None:
    # train_causal_bank_torch saves by default; absence of flags is fine
    # for SHORT runs (long runs additionally need a periodic cadence).
    preflight_check(
        _job(
            "python -m chronohorn.families.causal_bank.training."
            "train_causal_bank_torch --steps 1000 --json out/run.json"
        )
    )


def test_long_run_without_cadence_refused() -> None:
    # The end-of-run body alone is a loophole: a crash at step 199,999
    # leaves nothing. Long runs must checkpoint periodically (#27).
    job = _job(
        "python -m chronohorn.families.causal_bank.training."
        "train_causal_bank_torch --steps 200000 --json out/run.json"
    )
    with pytest.raises(PreflightError, match="periodic cadence"):
        preflight_check(job)


def test_long_run_with_cadence_passes() -> None:
    preflight_check(
        _job(
            "python -m chronohorn.families.causal_bank.training."
            "train_causal_bank_torch --steps 200000 "
            "--save-checkpoint-every 10000 --json out/run.json"
        )
    )


def test_cadence_larger_than_steps_refused() -> None:
    job = _job(
        "python train.py --steps 50000 --save-checkpoint-every 60000 "
        "--checkpoint-path out/x.checkpoint.pt"
    )
    with pytest.raises(PreflightError, match="exceeds"):
        preflight_check(job)


def test_long_run_cadence_exempt_honored() -> None:
    preflight_check(
        _job(
            "python -m chronohorn.families.causal_bank.training."
            "train_causal_bank_torch --steps 200000 --json out/run.json",
            checkpoint_policy="exempt",
        )
    )


def test_standalone_trainer_without_evidence_refused() -> None:
    # The session-13 failure shape: an inline trainer with no save flags.
    job = _job("python /tmp/pilot_wave18.py --steps 200000 --json out/run.json")
    with pytest.raises(PreflightError, match="checkpoint covenant"):
        preflight_check(job)


def test_standalone_trainer_with_checkpoint_evidence_passes() -> None:
    preflight_check(
        _job(
            "python /tmp/pilot.py --steps 1000 "
            "--checkpoint-path out/pilot.checkpoint.pt"
        )
    )


def test_polyhash_requires_save_model() -> None:
    job = _job("python train_polyhash.py --steps 5000 --json out/ph.json")
    with pytest.raises(PreflightError, match="checkpoint covenant"):
        preflight_check(job)
    preflight_check(
        _job("python train_polyhash.py --steps 5000 --save-model out/ph.pt")
    )


def test_error_names_the_job() -> None:
    job = _job("python bespoke.py --steps 10")
    job["name"] = "wave19-mystery"
    with pytest.raises(PreflightError, match="wave19-mystery"):
        preflight_check(job)
