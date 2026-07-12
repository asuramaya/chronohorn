"""E3-micro: does the PORT beat the mixture? Three contenders, one set of tensors.

THE GATE (task #14). The λ-mixture is blind: it sees the organ only as a finished
distribution, so it cannot look at WHICH neighbours came back, how far they were, or
whether they agreed. The oracle purse says a lot of money sits above it. The question
is whether reading the retrieved SET directly collects more of that purse than tuning
a number in front of it.

Three contenders, fit on cal, scored on test, sharing ONE store / ONE query draw /
ONE frozen base pass (knn_prep) — so they differ only in how they USE the retrieval:

  scalar-λ    p = (1-λ)·p_base + λ·p_knn, λ swept on cal      [today's organ]
  soft-λ(x)   λ = f(retrieval features), learned              [THE BAR: 45.6% of purse]
  PORT        attention over the retrieved (key, successor)   [the contender]
              pairs, residual on the base logits

SCORED IN ONE CURRENCY: the fraction of the ORACLE PURSE each one collects. The oracle
(per-token min(base, knn)) is the ceiling any λ(x) could reach; "% of purse" makes a
port and a scalar directly comparable and makes the 45.6% bar meaningful.

WHY THE BAR IS RE-MEASURED HERE AND NOT QUOTED: 45.6% was measured on Anubis's
instrument. Comparing my port against their number would compare instruments, not
models. All three contenders are refit here, on the same tensors, and the port must
beat the soft-λ gate AS MEASURED IN THIS HARNESS. If the local bar comes out far from
45.6%, that discrepancy is itself a finding and must be reported, not smoothed over.

THE PORT IS SAFE BY CONSTRUCTION: its output gate is zero-init, so before a single
step it is an exact no-op and its bpb IS the base's. It can only earn its keep.

Run:  python -m chronohorn.eval.e3_micro --corpus pentad:world --store-bytes 64000000
"""
from __future__ import annotations

import argparse
import functools

import torch
from torch import nn

from decepticons.models.knn_port_torch import StateKNNPort

from .harness_util import bpb_from_logp
from .knn_datastore import KNNDatastoreConfig
from .knn_prep import prepare


def _oracle(base_logp: torch.Tensor, knn_logp: torch.Tensor, y: torch.Tensor) -> float:
    """Per-token min(base, knn) — the ceiling for ANY λ(x), and the purse's numerator."""
    nb = -base_logp.gather(-1, y[:, None]).squeeze(-1)
    nk = -knn_logp.gather(-1, y[:, None]).squeeze(-1)
    return float(torch.minimum(nb, nk).mean() / torch.log(torch.tensor(2.0)))


def _features(d: torch.Tensor, p_knn: torch.Tensor, base_logp: torch.Tensor) -> torch.Tensor:
    """What a soft gate is allowed to look at: retrieval quality + both distributions'
    confidence. Deliberately NOT the label — a gate that peeks is an oracle, not a gate."""
    top1 = d[:, :1]
    gap = d[:, 1:2] - d[:, :1] if d.shape[1] > 1 else torch.zeros_like(top1)
    mean_d = d.mean(-1, keepdim=True)
    h_knn = -(p_knn.clamp_min(1e-9) * p_knn.clamp_min(1e-9).log()).sum(-1, keepdim=True)
    p_base = base_logp.exp()
    h_base = -(p_base.clamp_min(1e-9) * base_logp).sum(-1, keepdim=True)
    return torch.cat([top1, gap, mean_d, h_knn, h_base], dim=-1)


class SoftLambdaGate(nn.Module):
    """λ(x) ∈ [0,1] from retrieval features. The bar the port must clear."""

    def __init__(self, n_feat: int = 5, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_feat, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.norm = nn.LayerNorm(n_feat)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(self.norm(feats)))


def _mix_logp(base_logp: torch.Tensor, p_knn: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """Mix in PROBABILITY space (conker's graveyard: prob-space mixing, not logit-space)."""
    return torch.log(((1 - lam) * base_logp.exp() + lam * p_knn).clamp_min(1e-12))


def _fit(module: nn.Module, fit_fn, val_fn, n_steps: int, lr: float, log,
         patience: int = 40, wd: float = 1e-2) -> nn.Module:
    """Fit on a slice of cal, EARLY-STOP on a held-out slice of cal.

    This is not a nicety. The port has ~99k parameters; the eval's default calibration
    set is ~9k positions. Trained to convergence on that, the port would memorise the
    cal set, lose on test, and I would have "proven" that reading the retrieved set
    does not work — when all I had proven is that I gave it 10x more parameters than
    examples. A rigged fight is worse than no fight: it retires a good idea.

    So: every contender gets the same protocol — same fit/val split, same early stop,
    same weight decay — and the winner wins on held-out data, not on capacity.
    """
    opt = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=wd)
    best, best_state, since = float("inf"), None, 0
    for step in range(n_steps):
        module.train()
        opt.zero_grad()
        loss = fit_fn()
        loss.backward()
        opt.step()

        module.eval()
        with torch.no_grad():
            v = float(val_fn())
        if v < best - 1e-6:
            best, since = v, 0
            best_state = {k: t.detach().clone() for k, t in module.state_dict().items()}
        else:
            since += 1
        if step % max(1, n_steps // 4) == 0:
            log(f"    step {step:4d}  fit {float(loss):.4f}  val {v:.4f}"
                f"{'  *' if since == 0 else ''}")
        if since >= patience:
            log(f"    early stop at step {step} (val has not improved in {patience})")
            break

    if best_state is not None:      # restore the best-on-val weights, not the last ones
        module.load_state_dict(best_state)
    log(f"    best val {best:.4f}")
    return module


def run_e3_micro(cfg: KNNDatastoreConfig, *, k: int = 64, vote_temp: float = 0.05,
                 eps: float = 0.1, steps: int = 400, lr: float = 3e-3,
                 n_heads: int = 4, head_dim: int = 32,
                 log=functools.partial(print, flush=True)) -> dict:
    T = prepare(cfg, log=log)
    dev = cfg.device
    V = T.vocab_size

    base_cal_logp = torch.log_softmax(T.base_cal.float(), -1)
    base_tst_logp = torch.log_softmax(T.base_tst.float(), -1)
    base_bpb = bpb_from_logp(base_tst_logp, T.y_tst)

    p_knn_cal = T.mem.vote(T.d_cal, T.i_cal, k=k, temperature=vote_temp, eps=eps)
    p_knn_tst = T.mem.vote(T.d_tst, T.i_tst, k=k, temperature=vote_temp, eps=eps)
    knn_bpb = bpb_from_logp(torch.log(p_knn_tst.clamp_min(1e-12)), T.y_tst)

    oracle = _oracle(base_tst_logp, torch.log(p_knn_tst.clamp_min(1e-12)), T.y_tst)
    purse = base_bpb - oracle
    log(f"\nbase {base_bpb:.4f} | kNN-alone {knn_bpb:.4f} | oracle {oracle:.4f} "
        f"| PURSE {purse:.4f}")
    if purse <= 0:
        log("purse is non-positive — the organ has nothing to give here; E3 is moot.")
        return {"base": base_bpb, "purse": purse}

    def take(bpb: float) -> float:
        return 100.0 * (base_bpb - bpb) / purse

    results: dict[str, float] = {}

    # Fit/val split WITHIN cal. Test is never touched until the verdict.
    n = len(T.y_cal)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)).to(dev)
    fit_ix, val_ix = perm[: int(0.8 * n)], perm[int(0.8 * n):]
    log(f"\ncal split: {len(fit_ix)} fit / {len(val_ix)} val positions "
        f"(test {len(T.y_tst)}, untouched until the verdict)")

    # ---- 1. scalar λ (today's organ) -------------------------------------------
    best = min(((bpb_from_logp(_mix_logp(base_cal_logp, p_knn_cal,
                                         torch.tensor(l, device=dev)), T.y_cal), l)
                for l in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)), key=lambda t: t[0])
    lam = torch.tensor(best[1], device=dev)
    results["scalar"] = bpb_from_logp(_mix_logp(base_tst_logp, p_knn_tst, lam), T.y_tst)
    log(f"\nscalar-lam   lam={best[1]:.2f}  TEST {results['scalar']:.4f}  "
        f"takes {take(results['scalar']):.1f}% of purse")

    # ---- 2. soft λ(x) — THE BAR -------------------------------------------------
    f_cal = _features(T.d_cal, p_knn_cal, base_cal_logp)
    f_tst = _features(T.d_tst, p_knn_tst, base_tst_logp)
    gate = SoftLambdaGate(f_cal.shape[-1]).to(dev)
    log(f"\nsoft-lam(x)  fitting the BAR ({sum(p.numel() for p in gate.parameters()):,} params)")

    def _gate_loss(ix):
        lp = _mix_logp(base_cal_logp[ix], p_knn_cal[ix], gate(f_cal[ix]))
        return -lp.gather(-1, T.y_cal[ix][:, None]).mean()

    _fit(gate, lambda: _gate_loss(fit_ix), lambda: _gate_loss(val_ix), steps, lr, log)
    with torch.no_grad():
        lam_x = gate(f_tst)
        results["soft_lambda"] = bpb_from_logp(_mix_logp(base_tst_logp, p_knn_tst, lam_x),
                                               T.y_tst)
    log(f"soft-lam(x)  TEST {results['soft_lambda']:.4f}  "
        f"takes {take(results['soft_lambda']):.1f}% of purse  "
        f"(mean lam {float(lam_x.mean()):.3f})")

    # ---- 3. THE PORT ------------------------------------------------------------
    nk_cal, nv_cal = T.neighbours(T.i_cal, k)
    nk_tst, nv_tst = T.neighbours(T.i_tst, k)
    h_cal, h_tst = T.encode(T.S_cal), T.encode(T.S_tst)
    dist_cal, dist_tst = T.d_cal[:, :k].float(), T.d_tst[:, :k].float()

    port = StateKNNPort(h_cal.shape[-1], V, n_heads=n_heads, head_dim=head_dim).to(dev)
    assert port.is_noop(), "port must be a no-op at init — the safety property"
    with torch.no_grad():   # the promise, checked rather than trusted:
        noop = bpb_from_logp(torch.log_softmax(
            T.base_tst.float() + port(h_tst, nk_tst, nv_tst, dist_tst), -1), T.y_tst)
    assert abs(noop - base_bpb) < 1e-9, f"port at init moved the base: {noop} vs {base_bpb}"
    n_port = sum(p.numel() for p in port.parameters())
    log(f"\nPORT         no-op check at init: {noop:.4f} == base {base_bpb:.4f} OK")
    log(f"PORT         fitting ({n_port:,} params on {len(fit_ix):,} fit positions "
        f"= {n_port / max(1, len(fit_ix)):.2f} params/example)")
    if n_port > len(fit_ix):
        log("  WARNING: more parameters than examples. A loss here is a verdict on the")
        log("  HARNESS, not on the port. Raise --n-cal before believing a negative result.")

    def _port_loss(ix):
        lp = torch.log_softmax(
            base_cal_logp[ix] + port(h_cal[ix], nk_cal[ix], nv_cal[ix], dist_cal[ix]), -1)
        return -lp.gather(-1, T.y_cal[ix][:, None]).mean()

    _fit(port, lambda: _port_loss(fit_ix), lambda: _port_loss(val_ix), steps, lr, log)
    with torch.no_grad():
        results["port"] = bpb_from_logp(torch.log_softmax(
            base_tst_logp + port(h_tst, nk_tst, nv_tst, dist_tst), -1), T.y_tst)
    log(f"PORT         TEST {results['port']:.4f}  takes {take(results['port']):.1f}% of purse")

    # ---- verdict ----------------------------------------------------------------
    bar, got = take(results["soft_lambda"]), take(results["port"])
    log("\n" + "=" * 68)
    log(f"{'contender':<14}{'TEST bpb':>10}{'vs base':>10}{'% of purse':>13}")
    log("-" * 68)
    log(f"{'base':<14}{base_bpb:>10.4f}{0.0:>10.4f}{0.0:>12.1f}%")
    for name in ("scalar", "soft_lambda", "port"):
        log(f"{name:<14}{results[name]:>10.4f}{results[name] - base_bpb:>10.4f}"
            f"{take(results[name]):>12.1f}%")
    log(f"{'oracle':<14}{oracle:>10.4f}{oracle - base_bpb:>10.4f}{100.0:>12.1f}%")
    log("=" * 68)
    verdict = ("PORT CLEARS THE BAR" if got > bar else "PORT DIES — it does not beat soft-λ(x)")
    log(f"VERDICT: {verdict}  (port {got:.1f}% vs bar {bar:.1f}%)")
    log("The bar is re-measured here, not quoted: comparing against another harness's")
    log("45.6% would compare instruments, not models.")

    return {"base": base_bpb, "knn_alone": knn_bpb, "oracle": oracle, "purse": purse,
            "bar_pct": bar, "port_pct": got, "port_clears": got > bar, **results}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", default="pentad:world")
    p.add_argument("--query-corpus", default=None)
    p.add_argument("--store-bytes", type=int, default=64_000_000)
    p.add_argument("--store-device", default="cpu")
    p.add_argument("--store-offset", type=int, default=0)
    p.add_argument("--basis-stride", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--k", type=int, default=64)
    # The eval's default n_cal=12 gives ~9k positions — 10x FEWER than the port's ~99k
    # parameters. That is not a test of the port, it is a test of overfitting. 256
    # windows (~196k positions) puts the port at ~0.5 params/example, and the soft-λ
    # gate (235 params) is unaffected either way, so nobody is advantaged by the change.
    p.add_argument("--n-cal", type=int, default=256,
                   help="calibration windows — MUST exceed the port's parameter count in positions")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=32)
    a = p.parse_args(argv)

    cfg = KNNDatastoreConfig(
        store_bytes=a.store_bytes, store_device=a.store_device, device=a.device,
        corpus=a.corpus, query_corpus=a.query_corpus, store_offset=a.store_offset,
        basis_stride=a.basis_stride, n_cal=a.n_cal)
    run_e3_micro(cfg, k=a.k, steps=a.steps, lr=a.lr,
                 n_heads=a.n_heads, head_dim=a.head_dim)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
