from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .common import _embedding_uniform, _rng_for, _xavier_uniform

_opc_cache: dict | None = None


def _get_opc():
    global _opc_cache
    if _opc_cache is not None:
        return _opc_cache
    try:
        from decepticons.causal_bank import (
            CausalBankConfig,
            build_linear_bank,
            learnable_substrate_keys,
            osc_pair_count,
            scale_config,
            validate_config,
        )
    except ImportError as exc:
        raise ImportError(
            "decepticons is required for causal-bank models. "
            "Install the decepticons package or make it importable."
        ) from exc
    _opc_cache = {
        "CausalBankConfig": CausalBankConfig,
        "build_linear_bank": build_linear_bank,
        "learnable_substrate_keys": learnable_substrate_keys,
        "osc_pair_count": osc_pair_count,
        "scale_config": scale_config,
        "validate_config": validate_config,
    }
    return _opc_cache
from .readouts_torch import (
    GRUReadout,
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

_OPC_REEXPORTS = ("CausalBankConfig", "scale_config", "build_linear_bank",
                   "learnable_substrate_keys", "osc_pair_count", "validate_config")


def __getattr__(name: str):
    if name in _OPC_REEXPORTS:
        return _get_opc()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

class CausalBankModel(nn.Module):
    def __init__(self, vocab_size: int, config=None):
        super().__init__()
        opc = _get_opc()
        CausalBankConfig = opc["CausalBankConfig"]
        validate_config = opc["validate_config"]
        build_linear_bank = opc["build_linear_bank"]
        learnable_substrate_keys = opc["learnable_substrate_keys"]
        osc_pair_count = opc["osc_pair_count"]

        if config is None:
            config = CausalBankConfig()
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
            learnable_keys = learnable_substrate_keys(config)

            if "linear_in_proj" in learnable_keys:
                self.linear_in_proj = nn.Parameter(torch.from_numpy(in_proj))
            else:
                self.register_buffer("linear_in_proj", torch.from_numpy(in_proj))

            if "linear_decays" in learnable_keys:
                self.linear_decays = nn.Parameter(torch.from_numpy(decays.astype(np.float32)))
            else:
                self.register_buffer("linear_decays", torch.from_numpy(decays.astype(np.float32)))

            self._learned_recurrence = "recurrence_gate" in learnable_keys
            if self._learned_recurrence:
                # Per-mode gate: input -> sigmoid gate value
                # W_gate: [embedding_dim, linear_modes], b_gate: [linear_modes]
                self._gate_weight = nn.Parameter(torch.zeros(config.embedding_dim, config.linear_modes))
                self._gate_bias = nn.Parameter(torch.zeros(config.linear_modes))
                # Initialize gate bias so initial gate ≈ decay values (smooth start)
                with torch.no_grad():
                    self._gate_bias.copy_(
                        torch.log(self.linear_decays / (1 - self.linear_decays + 1e-6)).clamp(-5, 5)
                    )
            else:
                self._learned_recurrence = False

            if "linear_decays" in learnable_keys:
                # Kernel will be recomputed in forward() from learnable decays
                self._recompute_kernel = True
                self.register_buffer("_kernel_time_idx", torch.arange(config.max_seq_len))
                self.linear_kernel = None
            elif config.linear_impl == "kernel":
                self._recompute_kernel = False
                self.register_buffer("linear_kernel", torch.from_numpy(kernel))
            else:
                self._recompute_kernel = False
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
            elif config.linear_readout_kind == "recurrent":
                self.linear_readout = GRUReadout(linear_readout_in_dim, vocab_size, config)
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
            # Substrate polynomial expansion
            self._substrate_poly = getattr(config, 'substrate_poly_order', 1)
            if self._substrate_poly >= 2 and config.enable_linear:
                # Project modes to smaller dim, then expand
                poly_dim = min(64, config.linear_modes // 4)
                self._substrate_poly_proj = nn.Linear(config.linear_modes, poly_dim)
                quad_dim = poly_dim * (poly_dim + 1) // 2
                extra_dim = quad_dim
                if self._substrate_poly >= 3:
                    extra_dim += poly_dim
                # Adjust the linear readout input dimension
                new_readout_dim = config.linear_modes + config.embedding_dim + extra_dim
                # Recreate linear readout with expanded input
                if config.linear_readout_kind == "mlp":
                    self.linear_readout = MLP(new_readout_dim, config.linear_hidden, vocab_size)
                elif config.linear_readout_kind == "tied_recursive":
                    self.linear_readout = TiedRecursiveReadout(
                        new_readout_dim, config.linear_hidden[0], vocab_size, config.linear_readout_depth,
                    )
                elif config.linear_readout_kind == "recurrent":
                    self.linear_readout = GRUReadout(new_readout_dim, vocab_size, config)
                else:
                    self.linear_readout = RoutedSquaredReLUReadout(
                        new_readout_dim, config.linear_hidden[0], vocab_size, config.linear_readout_num_experts,
                    )

            osc_pairs = osc_pair_count(config)
            self.non_osc_modes = config.linear_modes - 2 * osc_pairs
            self.osc_mode_count = 2 * osc_pairs
            if config.static_bank_gate and self.osc_mode_count > 0:
                self.bank_gate_logits = nn.Parameter(torch.zeros((2,), dtype=torch.float32))

        if config.enable_local:
            if self.shared_embedding is None:
                self.local_embedding = nn.Embedding(vocab_size, config.embedding_dim)
            base_local_dim = config.local_window * config.embedding_dim
            self._poly_order = getattr(config, 'local_poly_order', 1)
            if self._poly_order >= 2:
                poly_proj_dim = min(64, base_local_dim // 4)
                self._poly_proj = nn.Linear(base_local_dim, poly_proj_dim)
                # Quadratic features: poly_proj_dim * (poly_proj_dim + 1) / 2
                quad_dim = poly_proj_dim * (poly_proj_dim + 1) // 2
                expanded_dim = base_local_dim + quad_dim
                if self._poly_order >= 3:
                    # Cubic: element-wise cube of projection (cheap approximation)
                    expanded_dim += poly_proj_dim
                self.local_readout = MLP(expanded_dim, config.local_hidden, vocab_size)
            else:
                self.local_readout = MLP(base_local_dim, config.local_hidden, vocab_size)

        if config.enable_linear and config.enable_local and config.mix_mode == "gated":
            self.gate_proj = nn.Linear(6, 1)

        # Stacked substrate blocks
        self.num_blocks = getattr(config, 'num_blocks', 1)
        if self.num_blocks > 1 and config.enable_linear:
            mixing_dim = max(int(config.linear_modes * getattr(config, 'block_mixing_ratio', 0.25)), 1)
            self._block_layers = nn.ModuleList()
            for _ in range(self.num_blocks - 1):
                self._block_layers.append(nn.Sequential(
                    nn.Linear(config.linear_modes, mixing_dim),
                    nn.ReLU(),
                    nn.Linear(mixing_dim, config.linear_modes),
                ))

        # --- Selective scan (Mamba-style) ---
        self._use_selective_scan = getattr(config, 'state_dim', 0) > 0
        if self._use_selective_scan:
            sd = config.state_dim
            ed = config.embedding_dim
            # A: diagonal decay (learned, log-space)
            self._ssm_A = nn.Parameter(torch.zeros(sd))
            # B: input → state write
            self._ssm_B = nn.Linear(ed, sd, bias=False)
            # C: input → state read
            self._ssm_C = nn.Linear(ed, sd, bias=False)
            # D: skip connection
            self._ssm_D = nn.Parameter(torch.ones(1))
            # Output projection: state_dim → modes (to feed into readout)
            self._ssm_out = nn.Linear(sd, config.linear_modes, bias=False)

            # Initialize A from decay schedule
            with torch.no_grad():
                # Log-space: A_actual = exp(A_log), so state decays when A_log < 0
                init_decays = torch.linspace(-0.1, -5.0, sd)
                self._ssm_A.copy_(init_decays)

            self._num_hemispheres = getattr(config, 'num_hemispheres', 1)
            if self._num_hemispheres == 2:
                fast_dim = max(int(sd * getattr(config, 'fast_hemisphere_ratio', 0.25)), 1)
                slow_dim = sd - fast_dim
                self._fast_dim = fast_dim
                self._slow_dim = slow_dim
                # Re-initialize A with different decay ranges
                with torch.no_grad():
                    # Fast: short decay (forgets quickly, range -1 to -3)
                    self._ssm_A[:fast_dim] = torch.linspace(-1.0, -3.0, fast_dim)
                    # Slow: long decay (remembers, range -0.01 to -0.5)
                    self._ssm_A[fast_dim:] = torch.linspace(-0.01, -0.5, slow_dim)

        # --- Patch encoding ---
        self._patch_size = getattr(config, 'patch_size', 1)
        self._patch_causal_mode = getattr(config, 'patch_causal_decoder', 'none')
        if self._patch_size > 1:
            # Encoder: patch_size * embed_dim → embed_dim
            self._patch_encoder = nn.Linear(self._patch_size * config.embedding_dim, config.embedding_dim)
            # Decoder: readout features → patch_size * vocab_size
            readout_feat_dim = (
                config.linear_modes + config.embedding_dim
                if config.enable_linear
                else config.embedding_dim
            )
            if self._patch_causal_mode == 'autoregressive':
                # MEGABYTE-style autoregressive byte decoder
                ssm_out_dim = config.linear_modes + config.embedding_dim if config.enable_linear else config.embedding_dim
                decoder_input_dim = ssm_out_dim + config.embedding_dim  # SSM features + prev byte embedding
                self._patch_byte_embed = nn.Embedding(vocab_size, config.embedding_dim)
                self._patch_byte_decoder = nn.GRU(decoder_input_dim, config.embedding_dim, batch_first=True)
                self._patch_byte_output = nn.Linear(config.embedding_dim, vocab_size)
            elif self._patch_causal_mode == 'mlp_factored':
                # Factored MLP: each byte position gets its own output head
                ssm_out_dim = config.linear_modes + config.embedding_dim if config.enable_linear else config.embedding_dim
                self._patch_byte_heads = nn.ModuleList([
                    nn.Linear(ssm_out_dim, vocab_size)
                    for _ in range(self._patch_size)
                ])
            else:
                # Original (illegal) flat decoder
                self._patch_decoder = nn.Linear(readout_feat_dim, self._patch_size * vocab_size)

        # --- Trust-routing mode ---
        self._trust_routing = getattr(config, 'trust_routing', False)
        if self._trust_routing:
            trust_input_dim = config.linear_modes + config.embedding_dim if config.enable_linear else config.embedding_dim
            self._trust_gate = nn.Sequential(
                nn.Linear(trust_input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
            self._fallback_readout = nn.Linear(trust_input_dim, vocab_size)

            from chronohorn.families.polyhash.models.ngram_table import NgramTable
            table_path = getattr(config, 'table_path', '')
            if table_path and Path(table_path).exists():
                self._ngram_table = NgramTable.load(table_path)
            else:
                self._ngram_table = NgramTable(vocab_size=vocab_size)

        # Online causal memory (only useful with the linear path)
        self._use_online_memory = (
            getattr(config, "memory_kind", "none") != "none"
            and config.enable_linear
        )
        if self._use_online_memory:
            from decepticons.online_memory import OnlineCausalMemory, OnlineMemoryConfig
            mem_order = {"ngram": 3, "exact_context": 4, "statistical_backoff": 3}.get(config.memory_kind, 3)
            self._online_memory = OnlineCausalMemory(OnlineMemoryConfig(
                max_order=mem_order,
                bucket_count=8192,
                vocabulary_size=vocab_size,
            ))
            # Memory feature projection: 7 features -> readout input dim (additive residual)
            self._memory_proj = nn.Linear(7, linear_readout_in_dim)
            # Gate: scalar learned mix weight, initialised to 0 so memory has no effect at init
            self._memory_gate = nn.Parameter(torch.tensor(0.0))

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

    def param_groups(self, base_lr: float) -> list[dict]:
        """Return parameter groups with per-hemisphere learning rates."""
        if not getattr(self, '_num_hemispheres', 1) == 2 or not self._use_selective_scan:
            return [{"params": list(self.parameters()), "lr": base_lr}]

        fast_params = []
        other_params = []

        fast_mult = getattr(self.config, 'fast_lr_mult', 4.0)

        for name, param in self.named_parameters():
            if '_ssm_A' in name or '_ssm_B' in name or '_ssm_C' in name:
                # Split SSM params conceptually - but they're single tensors
                # Use the fast LR for all SSM params when hemispheres are active
                # (the decay initialization already handles the fast/slow split)
                fast_params.append(param)
            else:
                other_params.append(param)

        groups = []
        if fast_params:
            groups.append({"params": fast_params, "lr": base_lr * fast_mult})
        if other_params:
            groups.append({"params": other_params, "lr": base_lr})
        return groups

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

    def _linear_states_recurrent(self, drive: torch.Tensor, x_embed: torch.Tensor) -> torch.Tensor:
        """Learned recurrence: state[t] = gate * state[t-1] + (1-gate) * drive[t]

        gate = sigmoid(x_embed[t] @ W_gate + b_gate)  per-mode, input-dependent

        drive: [batch, seq, modes] (projected input)
        x_embed: [batch, seq, embed_dim] (raw embedding, for gate computation)
        Returns: [batch, seq, modes]
        """
        batch, seq_len, modes = drive.shape
        device = drive.device
        dtype = drive.dtype

        # Compute gates for all positions at once: [batch, seq, modes]
        gates = torch.sigmoid(
            torch.matmul(x_embed, self._gate_weight.to(dtype=dtype)) + self._gate_bias.to(dtype=dtype)
        )

        # Parallel scan using chunked sequential: process in chunks of K
        # positions. Within each chunk, use the parallel cumsum trick.
        # Between chunks, carry state forward. K=32 gives good GPU
        # utilisation while keeping the sequential loop short.
        b = (1.0 - gates) * drive               # [batch, seq, modes]

        K = min(32, seq_len)
        n_chunks = (seq_len + K - 1) // K
        states = torch.zeros(batch, seq_len, modes, device=device, dtype=dtype)
        h = torch.zeros(batch, modes, device=device, dtype=dtype)

        for c in range(n_chunks):
            start = c * K
            end = min(start + K, seq_len)
            a_chunk = gates[:, start:end, :]     # [batch, chunk, modes]
            b_chunk = b[:, start:end, :]

            # Within chunk: parallel cumsum in log-space
            log_a = torch.log(a_chunk.clamp(min=1e-6))
            log_cum_a = torch.cumsum(log_a, dim=1)
            cum_a = torch.exp(log_cum_a)

            # h[start+t] = cum_a[t] * h_prev + cum_a[t] * cumsum(b/cum_a)[t]
            inv_cum_a = 1.0 / cum_a.clamp(min=1e-8)
            weighted_b = b_chunk * inv_cum_a
            cum_wb = torch.cumsum(weighted_b, dim=1)

            chunk_states = cum_a * (h.unsqueeze(1) + cum_wb)
            states[:, start:end, :] = chunk_states
            h = chunk_states[:, -1, :]

        return states

    def _selective_scan(self, x_embed: torch.Tensor) -> torch.Tensor:
        """Mamba-style selective scan.

        x_embed: [batch, seq, embed_dim]
        Returns: [batch, seq, linear_modes]
        """
        batch, seq_len, _ = x_embed.shape
        sd = self._ssm_A.shape[0]
        device = x_embed.device
        dtype = x_embed.dtype

        # Compute input-dependent B and C for all positions
        B = self._ssm_B(x_embed)  # [batch, seq, state_dim]
        C = self._ssm_C(x_embed)  # [batch, seq, state_dim]

        # Discretize A: exp(A_log) where A_log < 0 gives decay in (0, 1)
        A = torch.exp(self._ssm_A.to(dtype=dtype))  # [state_dim]

        # Parallel chunked scan (same pattern as gated recurrence)
        K = min(32, seq_len)
        n_chunks = (seq_len + K - 1) // K

        y = torch.zeros(batch, seq_len, sd, device=device, dtype=dtype)
        h = torch.zeros(batch, sd, device=device, dtype=dtype)

        for c_idx in range(n_chunks):
            start = c_idx * K
            end = min(start + K, seq_len)

            B_chunk = B[:, start:end, :]  # [batch, chunk, sd]
            C_chunk = C[:, start:end, :]

            # drive = B * input (input projection into state)
            drive = B_chunk  # [batch, chunk, sd]

            log_a = self._ssm_A.to(dtype=dtype).unsqueeze(0).unsqueeze(0).expand(batch, end - start, -1)
            log_a = log_a.clamp(max=-1e-6)  # ensure decay
            log_cum_a = torch.cumsum(log_a, dim=1)
            cum_a = torch.exp(log_cum_a)

            inv_cum_a = 1.0 / cum_a.clamp(min=1e-8)
            weighted_drive = drive * inv_cum_a
            cum_wd = torch.cumsum(weighted_drive, dim=1)

            chunk_states = cum_a * (h.unsqueeze(1) + cum_wd)

            # Read with C
            chunk_y = chunk_states * C_chunk
            y[:, start:end, :] = chunk_y
            h = chunk_states[:, -1, :]

        # Skip connection + output projection
        linear_in_proj = self.linear_in_proj.to(dtype=dtype)
        out = self._ssm_out(y) + self._ssm_D * torch.matmul(x_embed, linear_in_proj)
        return out

    def _apply_substrate(
        self,
        drive: torch.Tensor,
        x: torch.Tensor,
        timesteps: int,
        kernels: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply the substrate (recurrence, kernel, or FFT) to *drive*.

        Returns states [batch, seq, modes].
        """
        if self._learned_recurrence:
            return self._linear_states_recurrent(drive, x)
        if self.config.linear_impl == "kernel":
            if kernels is None:
                raise RuntimeError("causal-bank kernel path called without kernels.")
            drive_mb = drive.permute(2, 0, 1)
            states_mb = torch.matmul(drive_mb, kernels.transpose(1, 2))
            return states_mb.permute(1, 2, 0)
        return self._linear_states_fft(drive, timesteps)

    def _linear_states(self, chars: torch.Tensor, mode_gate: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        _, timesteps = chars.shape
        if timesteps > self.config.max_seq_len:
            raise ValueError(
                f"causal-bank max_seq_len={self.config.max_seq_len} is smaller than input timesteps={timesteps}"
            )
        if self.linear_readout is None:
            raise RuntimeError("causal-bank linear path is disabled.")
        x = self._embed_linear(chars)

        # --- Selective scan replaces the entire substrate path ---
        if self._use_selective_scan:
            states = self._selective_scan(x)
            states = self._apply_mode_gate(states, self._static_bank_mode_gate())
            return self._apply_mode_gate(states, mode_gate), x

        linear_in_proj = self.linear_in_proj.to(device=x.device, dtype=x.dtype)
        drive = torch.matmul(x, linear_in_proj)

        # Prepare kernels once (used by kernel and learnable_decays paths)
        kernels = None
        if not self._learned_recurrence and self.config.linear_impl == "kernel":
            if self._recompute_kernel:
                # Recompute kernel from learnable decays
                time_idx = self._kernel_time_idx[:timesteps].to(device=x.device)
                delta = time_idx[:, None] - time_idx[None, :]
                mask = delta >= 0
                safe_delta = torch.where(mask, delta, torch.zeros_like(delta)).float()
                decays = self.linear_decays.to(device=x.device, dtype=x.dtype)
                kernel = decays[None, None, :] ** safe_delta[..., None]
                kernel = kernel * mask[..., None].float()
                # kernel shape: (T, T, modes) -> transpose to (modes, T, T)
                kernels = kernel.permute(2, 0, 1)
            elif self.linear_kernel is None:
                raise RuntimeError("causal-bank kernel path is missing its materialized kernel.")
            else:
                kernels = self.linear_kernel[:, :timesteps, :timesteps].to(device=x.device, dtype=x.dtype)

        # First substrate application
        states = self._apply_substrate(drive, x, timesteps, kernels)

        # Stacked blocks: mix then re-apply substrate
        block_stride = getattr(self.config, 'block_stride', 1)
        if self.num_blocks > 1 and hasattr(self, '_block_layers'):
            for idx, block_layer in enumerate(self._block_layers):
                if block_stride > 1 and idx > 0:
                    # Subsample: take every stride-th position
                    stride = block_stride ** idx  # exponential: stride 1, 4, 16...
                    if stride < states.shape[1]:
                        strided = states[:, ::stride, :]
                        mixed_strided = strided + block_layer(strided)
                        strided_states = self._apply_substrate(mixed_strided, x[:, ::stride, :], strided.shape[1], kernels)
                        # Upsample back: repeat
                        states_upsampled = strided_states.repeat_interleave(stride, dim=1)[:, :timesteps, :]
                        states = states + states_upsampled  # residual add
                    else:
                        mixed = states + block_layer(states)
                        states = self._apply_substrate(mixed, x, timesteps, kernels)
                else:
                    mixed = states + block_layer(states)
                    states = self._apply_substrate(mixed, x, timesteps, kernels)
                states = self._apply_mode_gate(states, self._static_bank_mode_gate())

        states = self._apply_mode_gate(states, self._static_bank_mode_gate())
        return self._apply_mode_gate(states, mode_gate), x

    def _compute_online_memory_features(self, chars: torch.Tensor) -> torch.Tensor:
        """Process sequence through the online causal memory, returning projected features.

        Returns tensor of shape [batch, seq_len, readout_in_dim].
        """
        device = chars.device
        batch_size, seq_len = chars.shape
        mem_features_list = []
        for b in range(batch_size):
            self._online_memory.reset()
            batch_features = []
            for t in range(seq_len):
                f = self._online_memory.query_features()
                batch_features.append(torch.from_numpy(f))
                self._online_memory.update(int(chars[b, t].item()))
            mem_features_list.append(torch.stack(batch_features))
        mem_features = torch.stack(mem_features_list).to(device=device)  # [batch, seq, 7]
        return torch.sigmoid(self._memory_gate) * self._memory_proj(mem_features)

    def _linear_logits(self, chars: torch.Tensor, mode_gate: torch.Tensor | None = None) -> torch.Tensor:
        states, x = self._linear_states(chars, mode_gate=mode_gate)
        noise_sigma = getattr(self.config, 'training_noise', 0.0)
        if noise_sigma > 0 and self.training:
            states = states + torch.randn_like(states) * noise_sigma
        features = torch.cat([states, x], dim=-1)
        if getattr(self, '_substrate_poly', 1) >= 2:
            proj = self._substrate_poly_proj(states)
            d = proj.shape[-1]
            i, j = torch.triu_indices(d, d, device=proj.device)
            quad = proj[..., i] * proj[..., j]
            poly_parts = [quad]
            if self._substrate_poly >= 3:
                poly_parts.append(proj ** 3)
            features = torch.cat([features] + poly_parts, dim=-1)
        if self._use_online_memory:
            features = features + self._compute_online_memory_features(chars)
        return self.linear_readout(features)

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

    def _expand_poly_features(self, local_features: torch.Tensor) -> torch.Tensor:
        """Expand local window features with polynomial terms (NVAR-style)."""
        if self._poly_order < 2:
            return local_features
        proj = self._poly_proj(local_features)  # [batch, seq, poly_proj_dim]
        # Quadratic: outer product (upper triangle)
        d = proj.shape[-1]
        i, j = torch.triu_indices(d, d, device=proj.device)
        quad = proj[..., i] * proj[..., j]  # [batch, seq, d*(d+1)/2]
        parts = [local_features, quad]
        if self._poly_order >= 3:
            # Cubic: element-wise cube of projection (cheap approximation)
            parts.append(proj ** 3)
        return torch.cat(parts, dim=-1)

    def _local_logits(self, chars: torch.Tensor) -> torch.Tensor:
        if self.local_readout is None:
            raise RuntimeError("causal-bank local path is disabled.")
        x = self._embed_local(chars)
        stacked = self._local_window_stack(x)
        if self._poly_order >= 2:
            stacked = self._expand_poly_features(stacked)
        return self.local_readout(stacked)

    def forward(self, chars: torch.Tensor) -> torch.Tensor:
        if self._trust_routing:
            return self._forward_trust_routing(chars)
        if self._patch_size > 1 and self._patch_causal_mode == "hybrid":
            return self._forward_hybrid(chars)
        if self._patch_size > 1:
            return self._forward_patched(chars)
        return self._forward_raw(chars)

    def _forward_trust_routing(self, chars: torch.Tensor) -> torch.Tensor:
        """Trust-routing: table predicts, neural calibrates."""
        batch, seq_len = chars.shape

        # Get substrate features (same as normal forward)
        if self.config.enable_linear:
            states, embed = self._linear_states(chars)
            features = torch.cat([states, embed], dim=-1)
        else:
            features = self._embed_local(chars)

        # Trust gate: [batch, seq, 1]
        trust = torch.sigmoid(self._trust_gate(features))

        # Fallback: cheap linear readout
        fallback_logits = self._fallback_readout(features)

        # Vectorized table lookup — no Python loop, pure numpy
        chars_np = chars.detach().cpu().numpy()
        table_log_probs_np = self._ngram_table.batch_lookup_log_probs(chars_np)
        table_logits = torch.from_numpy(table_log_probs_np).to(features.device)

        # Mix: trust * table + (1-trust) * fallback
        # Both in log-probability space
        fallback_log_probs = fallback_logits - torch.logsumexp(fallback_logits, dim=-1, keepdim=True)
        mixed_log_probs = trust * table_logits + (1 - trust) * fallback_log_probs

        return mixed_log_probs

    def _forward_hybrid(self, chars: torch.Tensor) -> torch.Tensor:
        """Hybrid patch: global SSM on patches, local window on raw bytes.

        The global/linear path encodes bytes into patches (reducing sequence
        length), runs the substrate, then upsamples back to per-byte features.
        The local path operates on raw bytes as usual.
        The readout merges both paths — the global context from patches plus
        the fine-grained byte context from the local window.
        """
        batch, seq_len = chars.shape
        P = self._patch_size
        n_patches = seq_len // P

        # === GLOBAL PATH: patch-level SSM ===
        if self.config.enable_linear:
            embed = self._embed_linear(chars)  # [batch, seq, embed_dim]
            # Group into patches and encode
            patch_embeds = embed.view(batch, n_patches, P * self.config.embedding_dim)
            encoded = self._patch_encoder(patch_embeds)  # [batch, n_patches, embed_dim]

            # Run substrate on patch sequence
            linear_in_proj = self.linear_in_proj.to(device=encoded.device, dtype=encoded.dtype)
            drive = torch.matmul(encoded, linear_in_proj)

            kernels = None
            if not self._learned_recurrence and self.config.linear_impl == "kernel":
                if self._recompute_kernel:
                    time_idx = self._kernel_time_idx[:n_patches].to(device=encoded.device)
                    delta = time_idx[:, None] - time_idx[None, :]
                    mask = delta >= 0
                    safe_delta = torch.where(mask, delta, torch.zeros_like(delta)).float()
                    decays = self.linear_decays.to(device=encoded.device, dtype=encoded.dtype)
                    kernel = decays[None, None, :] ** safe_delta[..., None]
                    kernel = kernel * mask[..., None].float()
                    kernels = kernel.permute(2, 0, 1)
                elif self.linear_kernel is not None:
                    kernels = self.linear_kernel[:, :n_patches, :n_patches].to(device=encoded.device, dtype=encoded.dtype)

            states = self._apply_substrate(drive, encoded, n_patches, kernels)

            if self.num_blocks > 1 and hasattr(self, '_block_layers'):
                for block_layer in self._block_layers:
                    mixed = states + block_layer(states)
                    states = self._apply_substrate(mixed, encoded, n_patches, kernels)
                    states = self._apply_mode_gate(states, self._static_bank_mode_gate())

            states = self._apply_mode_gate(states, self._static_bank_mode_gate())

            # Upsample: repeat each patch state for P byte positions
            # Shift by 1 patch: states[t] provides context for bytes in patch[t+1]
            # Pad with zeros at the start for the first patch
            shifted = torch.cat([
                torch.zeros(batch, 1, states.shape[-1], device=states.device, dtype=states.dtype),
                states[:, :-1, :],
            ], dim=1)  # [batch, n_patches, modes]
            upsampled_states = shifted.repeat_interleave(P, dim=1)  # [batch, seq, modes]
            upsampled_embed = encoded.repeat_interleave(P, dim=1)[:, :seq_len, :]

            # Patch-level features per byte position
            features = torch.cat([upsampled_states, embed], dim=-1)
            logits_linear = self.linear_readout(features)

            # Training noise
            noise_sigma = getattr(self.config, 'training_noise', 0.0)
            if noise_sigma > 0 and self.training:
                logits_linear = logits_linear + torch.randn_like(logits_linear) * noise_sigma
        else:
            logits_linear = None

        # === LOCAL PATH: raw byte-level (unchanged) ===
        logits_local = self._local_logits(chars) if self.config.enable_local else None

        # === MERGE ===
        if logits_linear is None:
            return logits_local
        if logits_local is None:
            return logits_linear

        if self.gate_proj is None:
            gate = self.config.local_scale
        else:
            ent_l, max_l, var_l = self._logit_features(logits_linear)
            ent_r, max_r, var_r = self._logit_features(logits_local)
            feat = torch.stack([ent_l, ent_r, max_l, max_r, var_l, var_r], dim=-1)
            gate = torch.sigmoid(self.gate_proj(feat)) * self.config.local_scale

        return logits_linear + gate * logits_local

    def _forward_raw(self, chars: torch.Tensor) -> torch.Tensor:
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

    def _forward_patched(self, chars: torch.Tensor) -> torch.Tensor:
        """Patch-encoded forward: group bytes into patches, run substrate, decode back to bytes."""
        batch, seq_len = chars.shape
        P = self._patch_size
        # Pad sequence length to be divisible by patch_size
        if seq_len % P != 0:
            pad_len = P - (seq_len % P)
            chars = torch.nn.functional.pad(chars, (0, pad_len), value=0)
            seq_len = chars.shape[1]
            padded = True
        else:
            pad_len = 0
            padded = False

        n_patches = seq_len // P

        # Embed all bytes
        x = self._embed_linear(chars)  # [batch, seq, embed_dim]

        # Reshape into patches: [batch, n_patches, P * embed_dim]
        x_patched = x.reshape(batch, n_patches, P * self.config.embedding_dim)

        # Encode patches: [batch, n_patches, embed_dim]
        x_enc = self._patch_encoder(x_patched)

        # Run the substrate on the patch sequence
        if self._use_selective_scan:
            states = self._selective_scan(x_enc)
            states = self._apply_mode_gate(states, self._static_bank_mode_gate())
        else:
            linear_in_proj = self.linear_in_proj.to(device=x_enc.device, dtype=x_enc.dtype)
            drive = torch.matmul(x_enc, linear_in_proj)
            # Use timesteps = n_patches for substrate
            kernels = None
            if not self._learned_recurrence and self.config.linear_impl == "kernel":
                if self._recompute_kernel:
                    time_idx = self._kernel_time_idx[:n_patches].to(device=x_enc.device)
                    delta = time_idx[:, None] - time_idx[None, :]
                    mask = delta >= 0
                    safe_delta = torch.where(mask, delta, torch.zeros_like(delta)).float()
                    decays = self.linear_decays.to(device=x_enc.device, dtype=x_enc.dtype)
                    kernel = decays[None, None, :] ** safe_delta[..., None]
                    kernel = kernel * mask[..., None].float()
                    kernels = kernel.permute(2, 0, 1)
                elif self.linear_kernel is None:
                    raise RuntimeError("causal-bank kernel path is missing its materialized kernel.")
                else:
                    kernels = self.linear_kernel[:, :n_patches, :n_patches].to(device=x_enc.device, dtype=x_enc.dtype)

            states = self._apply_substrate(drive, x_enc, n_patches, kernels)
            states = self._apply_mode_gate(states, self._static_bank_mode_gate())

        # Readout features: [batch, n_patches, modes + embed_dim]
        readout_input = torch.cat([states, x_enc], dim=-1)

        # === CAUSAL DECODER ===
        if self._patch_causal_mode == 'autoregressive':
            logits = self._decode_autoregressive(readout_input, chars, batch, seq_len, n_patches, P)
        elif self._patch_causal_mode == 'mlp_factored':
            logits = self._decode_factored(readout_input, batch, seq_len, n_patches, P)
        else:
            # Original (illegal) flat decoder
            patch_logits = self._patch_decoder(readout_input)
            logits = patch_logits.reshape(batch, seq_len, self.vocab_size)

        # Remove padding
        if padded:
            logits = logits[:, :seq_len - pad_len, :]

        return logits

    def _decode_autoregressive(self, ssm_features, chars, batch, seq_len, n_patches, P):
        """MEGABYTE-style: predict next patch's bytes autoregressively.

        For patch t, we predict the bytes of patch t+1:
          b0 from ssm_features[t]
          b1 from ssm_features[t] + embed(b0)
          b2 from ssm_features[t] + embed(b0) + embed(b1)  [via GRU hidden state]
          b3 from ssm_features[t] + embed(b2) + ... [via GRU hidden state]
        """
        vocab_size = self.vocab_size
        device = ssm_features.device
        dtype = ssm_features.dtype

        all_logits = torch.zeros(batch, seq_len, vocab_size, device=device, dtype=dtype)
        patches = chars.view(batch, n_patches, P)

        # For each patch position, generate P bytes autoregressively
        for t in range(n_patches):
            # Context: ssm_features for PREVIOUS patches (shifted by 1)
            if t == 0:
                ctx = torch.zeros(batch, ssm_features.shape[-1], device=device, dtype=dtype)
            else:
                ctx = ssm_features[:, t - 1, :]  # [batch, features]

            # GRU hidden state starts from zero
            h = torch.zeros(1, batch, self.config.embedding_dim, device=device, dtype=dtype)

            for b_idx in range(P):
                global_pos = t * P + b_idx
                if b_idx == 0:
                    prev_byte_embed = torch.zeros(batch, self.config.embedding_dim, device=device, dtype=dtype)
                else:
                    # Use the ACTUAL previous byte (teacher forcing during training)
                    prev_byte = patches[:, t, b_idx - 1]
                    prev_byte_embed = self._patch_byte_embed(prev_byte)

                # GRU input: [SSM context, previous byte embedding]
                gru_input = torch.cat([ctx, prev_byte_embed], dim=-1).unsqueeze(1)  # [batch, 1, dim]
                gru_out, h = self._patch_byte_decoder(gru_input, h)
                byte_logits = self._patch_byte_output(gru_out.squeeze(1))  # [batch, vocab]
                all_logits[:, global_pos, :] = byte_logits

        return all_logits

    def _decode_factored(self, ssm_features, batch, seq_len, n_patches, P):
        """Factored MLP: each byte position has its own head.

        No within-patch autoregression. Each byte is predicted independently
        from the SSM features of the PREVIOUS patch.
        """
        vocab_size = self.vocab_size
        device = ssm_features.device
        dtype = ssm_features.dtype

        all_logits = torch.zeros(batch, seq_len, vocab_size, device=device, dtype=dtype)

        for t in range(n_patches):
            if t == 0:
                ctx = torch.zeros(batch, ssm_features.shape[-1], device=device, dtype=dtype)
            else:
                ctx = ssm_features[:, t - 1, :]

            for b_idx in range(P):
                global_pos = t * P + b_idx
                byte_logits = self._patch_byte_heads[b_idx](ctx)
                all_logits[:, global_pos, :] = byte_logits

        return all_logits

    def substrate_regularization(self, step: int = 0) -> torch.Tensor:
        """Regularization for learnable substrate parameters.

        Returns a scalar loss term that should be added to the training loss.
        Returns 0 for frozen substrate.  When ``adaptive_reg=True`` on the
        config the strength grows with ``sqrt(step / 1000)``.
        """
        if not getattr(self, '_recompute_kernel', False) and not getattr(self, '_learned_recurrence', False):
            return torch.tensor(0.0, device=next(self.parameters()).device)

        dev = next(self.parameters()).device

        # --- Adaptive scale ---
        adaptive = getattr(self.config, 'adaptive_reg', False)
        scale = 1.0 + (step / 1000.0) ** 0.5 if adaptive else 1.0

        if not getattr(self, '_recompute_kernel', False):
            # Learned recurrence with no learnable decays: only adaptive scaling
            # applies to a small L2 penalty on gate weights if present.
            if self._learned_recurrence:
                reg = self._gate_weight.pow(2).sum() * 1e-4
                return reg * scale
            return torch.tensor(0.0, device=dev)

        decays = self.linear_decays
        reg = torch.tensor(0.0, device=decays.device)

        # 1. Keep decays in valid range (0, 1) via soft penalty
        # Decays outside (0.01, 0.9999) get penalized
        reg = reg + torch.relu(-decays + 0.01).sum() * 10.0  # penalty for < 0.01
        reg = reg + torch.relu(decays - 0.9999).sum() * 10.0  # penalty for > 0.9999

        # 2. Diversity penalty: penalize when modes become too similar
        # Use pairwise distance between decay values
        if decays.numel() > 1:
            flat = decays.flatten()
            diffs = flat.unsqueeze(0) - flat.unsqueeze(1)
            # Penalize small differences (modes collapsing together)
            similarity = torch.exp(-diffs.pow(2) / 0.01)
            # Zero out diagonal
            mask = 1.0 - torch.eye(flat.shape[0], device=decays.device)
            diversity_loss = (similarity * mask).sum() / max(mask.sum(), 1.0)
            reg = reg + diversity_loss * 0.1

        return reg * scale

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
