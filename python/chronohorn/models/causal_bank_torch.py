from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from chronohorn._opc import ensure_open_predictive_coder_importable

ensure_open_predictive_coder_importable()

from open_predictive_coder.causal_bank import (  # noqa: E402
    CausalBankConfig,
    build_linear_bank,
    osc_pair_count,
    scale_config,
    validate_config,
)

from .common import _embedding_uniform, _rng_for, _xavier_uniform
from .readouts_torch import (
    MLP,
    RoutedSquaredReLUReadout,
    TiedRecursiveReadout,
    _copy_embedding_,
    _copy_linear_,
)

__all__ = [
    "CausalBankConfig",
    "CausalBankModel",
    "scale_config",
]

class CausalBankModel(nn.Module):
    def __init__(self, vocab_size: int, config: CausalBankConfig = CausalBankConfig()):
        super().__init__()
        validate_config(config)

        self.vocab_size = vocab_size
        self.config = config
        self.shared_embedding = None
        self.linear_embedding = None
        self.local_embedding = None
        self.linear_readout = None
        self.local_readout = None
        self.gate_proj = None
        self.bank_gate_logits = None
        self.non_osc_modes = 0
        self.osc_mode_count = 0

        if config.share_embedding and config.enable_linear and config.enable_local:
            self.shared_embedding = nn.Embedding(vocab_size, config.embedding_dim)

        if config.enable_linear:
            if self.shared_embedding is None:
                self.linear_embedding = nn.Embedding(vocab_size, config.embedding_dim)
            in_proj, decays, kernel = build_linear_bank(config)
            self.register_buffer("linear_in_proj", torch.from_numpy(in_proj))
            self.register_buffer("linear_decays", torch.from_numpy(decays.astype(np.float32)))
            if config.linear_impl == "kernel":
                self.register_buffer("linear_kernel", torch.from_numpy(kernel))
            else:
                self.linear_kernel = None
            linear_readout_in_dim = config.linear_modes + config.embedding_dim
            if config.linear_readout_kind == "mlp":
                self.linear_readout = MLP(
                    linear_readout_in_dim,
                    config.linear_hidden,
                    vocab_size,
                )
            elif config.linear_readout_kind == "tied_recursive":
                if len(config.linear_hidden) != 1:
                    raise ValueError(
                        "causal-bank tied_recursive linear readout currently expects exactly one hidden width."
                    )
                self.linear_readout = TiedRecursiveReadout(
                    linear_readout_in_dim,
                    config.linear_hidden[0],
                    vocab_size,
                    config.linear_readout_depth,
                )
            else:
                if len(config.linear_hidden) != 1:
                    raise ValueError(
                        "causal-bank routed_sqrelu_experts linear readout currently expects exactly one hidden width."
                    )
                self.linear_readout = RoutedSquaredReLUReadout(
                    linear_readout_in_dim,
                    config.linear_hidden[0],
                    vocab_size,
                    config.linear_readout_num_experts,
                )
            osc_pairs = osc_pair_count(config)
            self.non_osc_modes = config.linear_modes - 2 * osc_pairs
            self.osc_mode_count = 2 * osc_pairs
            if config.static_bank_gate and self.osc_mode_count > 0:
                self.bank_gate_logits = nn.Parameter(torch.zeros((2,), dtype=torch.float32))

        if config.enable_local:
            if self.shared_embedding is None:
                self.local_embedding = nn.Embedding(vocab_size, config.embedding_dim)
            self.local_readout = MLP(
                config.local_window * config.embedding_dim,
                config.local_hidden,
                vocab_size,
            )

        if config.enable_linear and config.enable_local and config.mix_mode == "gated":
            self.gate_proj = nn.Linear(6, 1)

        self._reset_trainable_parameters()

    def _reset_trainable_parameters(self) -> None:
        seed = int(self.config.init_seed)
        if self.shared_embedding is not None:
            weight = _embedding_uniform(tuple(self.shared_embedding.weight.shape), _rng_for(seed, "shared_embedding.weight"))
            _copy_embedding_(self.shared_embedding, weight)
        if self.linear_embedding is not None:
            weight = _embedding_uniform(tuple(self.linear_embedding.weight.shape), _rng_for(seed, "linear_embedding.weight"))
            _copy_embedding_(self.linear_embedding, weight)
        if self.local_embedding is not None:
            weight = _embedding_uniform(tuple(self.local_embedding.weight.shape), _rng_for(seed, "local_embedding.weight"))
            _copy_embedding_(self.local_embedding, weight)
        if self.linear_readout is not None:
            self.linear_readout.reset_parameters_with_seed(seed, "linear_readout")
        if self.local_readout is not None:
            self.local_readout.reset_parameters_with_seed(seed, "local_readout")
        if self.gate_proj is not None:
            gate_weight = _xavier_uniform(tuple(self.gate_proj.weight.shape), _rng_for(seed, "gate_proj.weight"))
            _copy_linear_(self.gate_proj, gate_weight)
        if self.bank_gate_logits is not None:
            with torch.no_grad():
                self.bank_gate_logits.zero_()

    @staticmethod
    def _logit_features(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        max_logit = torch.max(logits, dim=-1).values
        centered = logits - torch.mean(logits, dim=-1, keepdim=True)
        variance = torch.mean(centered * centered, dim=-1)
        return entropy, max_logit, variance

    def _embed_linear(self, chars: torch.Tensor) -> torch.Tensor:
        if self.shared_embedding is not None:
            return self.shared_embedding(chars)
        if self.linear_embedding is None:
            raise RuntimeError("causal-bank linear path has no embedding table.")
        return self.linear_embedding(chars)

    def _embed_local(self, chars: torch.Tensor) -> torch.Tensor:
        if self.shared_embedding is not None:
            return self.shared_embedding(chars)
        if self.local_embedding is None:
            raise RuntimeError("causal-bank local path has no embedding table.")
        return self.local_embedding(chars)

    def _linear_states_fft(self, drive: torch.Tensor, timesteps: int) -> torch.Tensor:
        drive_mb = drive.transpose(1, 2)
        n_fft = 1 << int(math.ceil(math.log2(max(2 * timesteps - 1, 1))))
        time = torch.arange(timesteps, dtype=drive.dtype, device=drive.device)
        decays = self.linear_decays.to(device=drive.device, dtype=drive.dtype)
        kernel = torch.pow(decays[:, None], time[None, :])
        drive_f = torch.fft.rfft(drive_mb, n=n_fft, dim=-1)
        kernel_f = torch.fft.rfft(kernel.unsqueeze(0), n=n_fft, dim=-1)
        states_mb = torch.fft.irfft(drive_f * kernel_f, n=n_fft, dim=-1)[..., :timesteps]
        return states_mb.transpose(1, 2)

    def _apply_mode_gate(self, states: torch.Tensor, mode_gate: torch.Tensor | None) -> torch.Tensor:
        if mode_gate is None:
            return states
        if mode_gate.ndim == 1:
            return states * mode_gate.view(1, 1, -1)
        if mode_gate.ndim == 2:
            return states * mode_gate[:, None, :]
        raise ValueError(f"causal-bank mode_gate must be rank-1 or rank-2, got shape {tuple(mode_gate.shape)}")

    def _static_bank_mode_gate(self) -> torch.Tensor | None:
        if self.bank_gate_logits is None or self.osc_mode_count <= 0:
            return None
        values = 1.0 + self.config.bank_gate_span * torch.tanh(self.bank_gate_logits)
        pieces = []
        if self.non_osc_modes > 0:
            pieces.append(values[0:1].expand(self.non_osc_modes))
        if self.osc_mode_count > 0:
            pieces.append(values[1:2].expand(self.osc_mode_count))
        return torch.cat(pieces, dim=0) if pieces else None

    def _linear_states(self, chars: torch.Tensor, mode_gate: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        _, timesteps = chars.shape
        if timesteps > self.config.max_seq_len:
            raise ValueError(
                f"causal-bank max_seq_len={self.config.max_seq_len} is smaller than input timesteps={timesteps}"
            )
        if self.linear_readout is None:
            raise RuntimeError("causal-bank linear path is disabled.")
        x = self._embed_linear(chars)
        linear_in_proj = self.linear_in_proj.to(device=x.device, dtype=x.dtype)
        drive = torch.matmul(x, linear_in_proj)
        if self.config.linear_impl == "kernel":
            if self.linear_kernel is None:
                raise RuntimeError("causal-bank kernel path is missing its materialized kernel.")
            kernels = self.linear_kernel[:, :timesteps, :timesteps].to(device=x.device, dtype=x.dtype)
            drive_mb = drive.permute(2, 0, 1)
            states_mb = torch.matmul(drive_mb, kernels.transpose(1, 2))
            states = states_mb.permute(1, 2, 0)
        else:
            states = self._linear_states_fft(drive, timesteps)
        states = self._apply_mode_gate(states, self._static_bank_mode_gate())
        return self._apply_mode_gate(states, mode_gate), x

    def _linear_logits(self, chars: torch.Tensor, mode_gate: torch.Tensor | None = None) -> torch.Tensor:
        states, x = self._linear_states(chars, mode_gate=mode_gate)
        return self.linear_readout(torch.cat([states, x], dim=-1))

    def _local_window_stack(self, x: torch.Tensor) -> torch.Tensor:
        batch, timesteps, dim = x.shape
        window = self.config.local_window
        if window == 1:
            return x
        pad = torch.zeros((batch, window - 1, dim), dtype=x.dtype, device=x.device)
        padded = torch.cat([pad, x], dim=1)
        views = []
        for offset in range(window):
            start = window - 1 - offset
            views.append(padded[:, start : start + timesteps, :])
        return torch.cat(views, dim=-1)

    def _local_logits(self, chars: torch.Tensor) -> torch.Tensor:
        if self.local_readout is None:
            raise RuntimeError("causal-bank local path is disabled.")
        x = self._embed_local(chars)
        stacked = self._local_window_stack(x)
        return self.local_readout(stacked)

    def forward(self, chars: torch.Tensor) -> torch.Tensor:
        logits_linear = self._linear_logits(chars) if self.config.enable_linear else None
        logits_local = self._local_logits(chars) if self.config.enable_local else None

        if logits_linear is None:
            return logits_local
        if logits_local is None:
            return logits_linear

        if self.gate_proj is None:
            gate = self.config.local_scale
        else:
            ent_l, max_l, var_l = self._logit_features(logits_linear)
            ent_r, max_r, var_r = self._logit_features(logits_local)
            features = torch.stack([ent_l, ent_r, max_l, max_r, var_l, var_r], dim=-1)
            gate = torch.sigmoid(self.gate_proj(features)) * self.config.local_scale

        return logits_linear + gate * logits_local

    def forward_with_mode_gate(self, chars: torch.Tensor, mode_gate: torch.Tensor | None) -> torch.Tensor:
        logits_linear = self._linear_logits(chars, mode_gate=mode_gate) if self.config.enable_linear else None
        logits_local = self._local_logits(chars) if self.config.enable_local else None

        if logits_linear is None:
            return logits_local
        if logits_local is None:
            return logits_linear

        if self.gate_proj is None:
            gate = self.config.local_scale
        else:
            ent_l, max_l, var_l = self._logit_features(logits_linear)
            ent_r, max_r, var_r = self._logit_features(logits_local)
            features = torch.stack([ent_l, ent_r, max_l, max_r, var_l, var_r], dim=-1)
            gate = torch.sigmoid(self.gate_proj(features)) * self.config.local_scale

        return logits_linear + gate * logits_local
