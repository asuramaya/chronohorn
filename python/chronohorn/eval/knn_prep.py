"""Shared prep for kNN experiments: store, query states, base logits, neighbours.

Extracted so the E3 port harness and the λ-mixture eval train and test on the SAME
tensors. That is not tidiness, it is the experiment's validity: the port's whole claim
is that it beats the mixture, and a comparison across two harnesses with two query
draws, two stores and two base passes measures the harnesses as much as the models.
One prep, one draw, one base — then the contenders differ only in how they USE it.

NOTE this file currently DUPLICATES the stream-mode prep inside knn_datastore, which
was deliberate: it was written while a chain was in flight and editing the eval module
mid-run would have meant earlier and later rows of one table running different code.
The dedupe (knn_datastore imports from here) lands once no chain is running, and is
verified by re-running the 8M regression rung to bit-identical numbers (2.9743 alone /
2.0223 mix). Until that check passes, treat any divergence as a bug in THIS file.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from decepticons.loader import load_checkpoint
from decepticons.models.state_knn_torch import StateKNNConfig, StateKNNMemory
from decepticons.models.state_stream_torch import LinearStateStreamer

from .knn_datastore import DEFAULT_CHECKPOINT, KNNDatastoreConfig, _corpus_arrays


@dataclass
class KNNTensors:
    """Everything a contender needs, prepared once and shared by all of them."""
    mem: StateKNNMemory
    S_cal: torch.Tensor      # [Nc, key_dim*] query states (pre-encode)
    S_tst: torch.Tensor
    y_cal: torch.Tensor      # [Nc] target bytes
    y_tst: torch.Tensor
    base_cal: torch.Tensor   # [Nc, V] FROZEN base logits
    base_tst: torch.Tensor
    d_cal: torch.Tensor      # [Nc, K] retrieval distances
    i_cal: torch.Tensor      # [Nc, K] retrieval indices into mem.keys/mem.values
    d_tst: torch.Tensor
    i_tst: torch.Tensor
    vocab_size: int

    def neighbours(self, idx: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(neighbour keys [N,k,D], their successor bytes [N,k]) for the port to attend over."""
        sel = idx[:, :k]
        keys = self.mem.keys[sel.to(self.mem.keys.device)].to(self.mem.device).float()
        vals = self.mem.values[sel.to(self.mem.values.device)].to(self.mem.device).long()
        return keys, vals

    def encode(self, states: torch.Tensor) -> torch.Tensor:
        """Query states in the SAME whitened space as the stored keys."""
        return self.mem._encode(states).float()


def prepare(cfg: KNNDatastoreConfig, log=print) -> KNNTensors:
    dev = cfg.device
    train, test = _corpus_arrays(cfg)
    inf = load_checkpoint(cfg.checkpoint or DEFAULT_CHECKPOINT, device=dev)
    W = inf.weights()

    emb = torch.tensor(W["linear_embedding.weight"], device=dev)
    in_proj = torch.tensor(W["linear_in_proj"], device=dev)
    decays = torch.tensor(W["linear_decays"], device=dev, dtype=torch.float64)
    n_modes = decays.shape[0]
    streamer = LinearStateStreamer.from_bank(emb, in_proj, decays, device=dev)

    mem = StateKNNMemory(n_modes, StateKNNConfig(
        key_dim=cfg.key_dim, k=max(cfg.k_grid), metric="cosine",
        key_transform=cfg.key_transform, store_dtype="float16",
        store_tile=getattr(cfg, "store_tile", 250_000)),
        device=dev, store_device=cfg.store_device)

    log(f"prep: store={cfg.store_bytes} corpus={cfg.corpus}@{cfg.store_offset} "
        f"queries={cfg.query_corpus or cfg.corpus}")

    store = train[cfg.store_offset: cfg.store_offset + cfg.store_bytes] \
        if cfg.store_offset else train[: cfg.store_bytes]

    if cfg.key_transform == "pca_whiten":
        for j, (st, _) in enumerate(streamer.stream(store)):
            if j % max(1, cfg.basis_stride) == 0:
                mem.observe(st)
        mem.finalize()
    n = len(store) - 1
    mem.reserve(n)
    for st, s in streamer.stream(store):
        hi = min(s + len(st), n)
        if hi <= s:
            break
        mem.add(st[: hi - s], torch.as_tensor(store[s + 1: hi + 1], device=dev))
    log(f"prep: store built ({len(mem.keys)} keys on {mem.store_device})")

    L, BURN = cfg.window, cfg.burn      # same window/burn-in the eval uses
    g = np.random.default_rng(cfg.seed)
    q_starts = g.integers(0, len(test) - L - 1, size=cfg.n_cal + cfg.n_test)

    def _positions(starts):
        return np.concatenate([np.arange(s + BURN, s + L - 1) for s in starts])

    needed = np.unique(_positions(q_starts))
    qbuf = torch.empty(len(needed), n_modes, device=dev)
    for st, s in streamer.stream(test):
        lo = int(np.searchsorted(needed, s))
        hi = int(np.searchsorted(needed, s + len(st)))
        if hi > lo:
            qbuf[lo:hi] = st[needed[lo:hi] - s]

    def _slots(starts):
        return torch.as_tensor(np.searchsorted(needed, _positions(starts)), device=dev)

    def _base(starts):
        Y, B = [], []
        for s in starts:
            seq = test[s: s + L]
            cap = inf.forward_captured(seq[None, :])
            B.append(torch.as_tensor(cap["logits"][0], device=dev)[BURN:-1])
            Y.append(torch.as_tensor(seq[BURN + 1:], device=dev))
        return torch.cat(Y), torch.cat(B)

    S_cal = qbuf[_slots(q_starts[: cfg.n_cal])]
    S_tst = qbuf[_slots(q_starts[cfg.n_cal:])]
    y_cal, base_cal = _base(q_starts[: cfg.n_cal])
    y_tst, base_tst = _base(q_starts[cfg.n_cal:])

    d_cal, i_cal = mem.search(S_cal)
    d_tst, i_tst = mem.search(S_tst)
    log(f"prep: searched (cal {len(y_cal)}, test {len(y_tst)} positions)")

    return KNNTensors(mem, S_cal, S_tst, y_cal, y_tst, base_cal, base_tst,
                      d_cal, i_cal, d_tst, i_tst, vocab_size=base_tst.shape[-1])
