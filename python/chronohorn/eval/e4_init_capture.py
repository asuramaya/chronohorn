"""E4: capture the frozen init of the evict-first body.

The retrospective subspace is a property of the FROZEN bank — which state
directions linearly decode lagged bytes — so it can be exported from an
UNTRAINED body. That breaks the apparent chicken-and-egg in the E4 spec:

    directions  <- this capture (frozen init; no training needed)
    energy_frac <- measured retro R^2 of arm A's TRAINED readout

This tool writes the step-0 checkpoint pair that heinrich's mri + retro-subspace
export consume (profile-cb-retro-subspace --out directions.npz).

THE COVENANT THIS ENFORCES: a capture is only worth anything if it is the init
arm A actually trains from. So we build the body through the trainer's OWN config
path (same adapter, same seed_everything, same construction order) and stamp the
init signature — the identical hash the trainer logs at step 0. When arm A runs
with the same --seed/--variant/--scale/--linear-modes, its logged
init_signature_sha256 MUST equal the one printed here. If it does not, the
capture is unfaithful and arm C's directions describe a body that never existed.
Check it; do not assume it.

Run:  python -m chronohorn.eval.e4_init_capture --out out/results/e4-body-init
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from chronohorn.families.causal_bank.adapter import CAUSAL_BANK_TRAINING_ADAPTER
from chronohorn.families.causal_bank.training.causal_bank_training_stack import (
    load_training_backend_stack,
)
from decepticons.loader import save_checkpoint
from decepticons.models.causal_bank_torch import CausalBankModel

# The E4 body. Same architecture family as the production body (variant=base) so the
# retro-subspace notion transfers, but small enough that 1k/2k/5k-step arms are cheap.
#
# NOTE --scale MULTIPLIES --linear-modes (production: 256 x 8.0 = 2048 modes, 17.8M
# params). So this is 256 x 2.0 = 512 EFFECTIVE modes — a clean 4x shrink of the
# production body, not a differently-shaped one. Read the effective count off the
# built config, never off the flag.
#
# seq_len must exceed the largest retro lag (default lags 64,512), or the lag-512
# decode has nothing to decode from.
E4_BODY = dict(variant="base", scale=2.0, linear_modes=256, vocab_size=256,
               seq_len=1024, seed=42)


def init_signature(model: torch.nn.Module) -> str:
    """Byte-identical to the trainer's init-signature computation."""
    h = hashlib.sha256()
    sd = model.state_dict()
    for name in sorted(sd.keys()):
        h.update(name.encode("utf-8"))
        h.update(sd[name].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def build_e4_body(body: dict, device: str = "cpu") -> tuple[CausalBankModel, object]:
    """Build the E4 body through the TRAINER'S OWN parser and adapter.

    Not a hand-rolled Namespace: the trainer's parser carries dozens of defaults
    (local_window, share_embedding, state_dim, readout kind/depth/experts ...) that
    all shape the init. Reconstructing them by hand is how a capture silently comes
    to describe a different body than the one that trains. Parse the same argv the
    arms will, so every default is inherited rather than re-guessed.
    """
    from chronohorn.families.causal_bank.training.train_causal_bank_torch import (
        build_parser,
        seed_everything,
    )

    stack = load_training_backend_stack("torch")
    argv = [
        "--data-root", body.get("data_root", "unused-for-init-capture"),
        "--json", "unused-for-init-capture.json",   # required by the parser; never written
        "--seed", str(body["seed"]),
        "--variant", str(body["variant"]),
        "--scale", str(body["scale"]),
        "--linear-modes", str(body["linear_modes"]),
        "--vocab-size", str(body["vocab_size"]),
        "--seq-len", str(body["seq_len"]),
    ]
    args = build_parser().parse_args(argv)

    seed_everything(args.seed)          # the trainer's own RNG discipline
    config, _baseline_linear_hidden = CAUSAL_BANK_TRAINING_ADAPTER.build_variant_config(
        args,
        ConfigClass=stack.ConfigClass,
        scale_config=stack.scale_config,
        seq_len=body["seq_len"],
        vocab_size=body["vocab_size"],
    )
    model = CausalBankModel(vocab_size=body["vocab_size"], config=config).to(device)
    return model, config


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="out/results/e4-body-init")
    p.add_argument("--device", default="cpu",
                   help="cpu is correct here: nothing is trained, and the card is for science")
    for k, v in E4_BODY.items():
        p.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    a = p.parse_args(argv)

    body = {k: getattr(a, k) for k in E4_BODY}
    model, config = build_e4_body(body, device=a.device)
    sig = init_signature(model)
    n_params = int(sum(p_.numel() for p_ in model.parameters()))
    n_train = int(sum(p_.numel() for p_ in model.parameters() if p_.requires_grad))

    ckpt, js = save_checkpoint(model, a.out, extra={
        "title": "E4 evict-first body — FROZEN INIT (step 0, untrained)",
        "e4": {
            "role": "init_capture",
            "body": body,
            "init_signature_sha256": sig,
            "covenant": (
                "arm A trained with these body args MUST log this exact "
                "init_signature_sha256; if it differs, this capture describes a "
                "body that never trained and arm C's directions are void"),
        },
        "model": {"params": n_params, "trainable_params": n_train,
                  "linear_modes": config.linear_modes,
                  "init_signature_sha256": sig},
    })

    print(f"E4 body: {body}")
    print(f"  params           {n_params:,} ({n_train:,} trainable)")
    print(f"  linear_modes     {config.linear_modes}")
    print(f"  init signature   {sig}")
    print(f"  checkpoint       {ckpt}")
    print(f"  result json      {js}")
    print("\nNEXT (heinrich, Anubis): capture .seq.mri from this checkpoint, then")
    print(f"  profile-cb-retro-subspace --mri <body.seq.mri> --dims {config.linear_modes} "
          f"--lags 64,512 --out directions.npz")
    print("Then arm C: --init-hook project --init-directions directions.npz "
          "--init-energy-frac <measured retro R^2 of arm A>")
    print(json.dumps({"init_signature_sha256": sig, "params": n_params}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
