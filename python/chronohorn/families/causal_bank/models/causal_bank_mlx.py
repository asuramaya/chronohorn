from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .common import _embedding_uniform, _rng_for, _xavier_uniform
from .readouts_mlx import MLP, RoutedSquaredReLUReadout, TiedRecursiveReadout

_opc_cache: dict | None = None


def _get_opc():
    global _opc_cache
    if _opc_cache is not None:
        return _opc_cache
    try:
        from decepticons.causal_bank import (
            CausalBankConfig,
            build_linear_bank,
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
        "osc_pair_count": osc_pair_count,
        "scale_config": scale_config,
        "validate_config": validate_config,
    }
    return _opc_cache


_OPC_REEXPORTS = ("CausalBankConfig", "build_linear_bank", "osc_pair_count",
                   "scale_config", "validate_config")


def __getattr__(name: str):
    if name in _OPC_REEXPORTS:
        return _get_opc()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

_MISSING = object()


def _is_mx_array(value: object) -> bool:
    return mx is not None and isinstance(value, mx.array)


def _is_module_like(value: object) -> bool:
    return all(callable(getattr(value, attr, None)) for attr in ("parameters", "trainable_parameters", "update"))


class _ModuleLike(dict):
    def __init__(self):
        super().__init__()
        self._no_grad = set()
        self._training = True

    @property
    def training(self):
        return self._training

    @property
    def state(self):
        return self

    def __getattr__(self, key):
        if key in self:
            return self[key]
        return super().__getattribute__(key)

    def __setattr__(self, key, val):
        if key.startswith("_"):
            super().__setattr__(key, val)
            return
        if _is_mx_array(val) or isinstance(val, (dict, list, tuple)) or _is_module_like(val):
            if hasattr(self, key) and key not in self:
                try:
                    delattr(self, key)
                except AttributeError:
                    pass
            self[key] = val
        else:
            super().__setattr__(key, val)
            self.pop(key, None)

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            super().__delattr__(name)

    def _parameters_tree_value(self, value: object, trainable: bool) -> object:
        if _is_module_like(value):
            return value.trainable_parameters() if trainable else value.parameters()
        if isinstance(value, dict):
            out: dict[str, object] = {}
            for key, child in value.items():
                if isinstance(key, str) and key.startswith("_"):
                    continue
                subtree = self._parameters_tree_value(child, trainable=trainable)
                if subtree is not _MISSING:
                    out[key] = subtree
            return out
        if isinstance(value, list):
            out_list = []
            for child in value:
                subtree = self._parameters_tree_value(child, trainable=trainable)
                if subtree is not _MISSING:
                    out_list.append(subtree)
            return out_list
        if isinstance(value, tuple):
            out_tuple = []
            for child in value:
                subtree = self._parameters_tree_value(child, trainable=trainable)
                if subtree is not _MISSING:
                    out_tuple.append(subtree)
            return tuple(out_tuple)
        if _is_mx_array(value):
            return value
        return _MISSING

    def parameters(self):
        out: dict[str, object] = {}
        for key, value in self.items():
            if key.startswith("_"):
                continue
            subtree = self._parameters_tree_value(value, trainable=False)
            if subtree is not _MISSING:
                out[key] = subtree
        return out

    def trainable_parameters(self):
        out: dict[str, object] = {}
        for key, value in self.items():
            if key.startswith("_") or key in self._no_grad:
                continue
            subtree = self._parameters_tree_value(value, trainable=True)
            if subtree is not _MISSING:
                out[key] = subtree
        return out

    def update(self, parameters: dict, strict: bool = True):
        def apply(dst, params):
            if isinstance(dst, dict):
                if not isinstance(params, dict):
                    if strict:
                        raise ValueError(f"Received invalid type: {type(params).__name__}.")
                    return
                for key in params:
                    if key in dst:
                        current_value = dst[key]
                        new_value = params[key]
                        if _is_module_like(current_value):
                            current_value.update(new_value, strict=strict)
                        elif isinstance(current_value, dict):
                            apply(current_value, new_value)
                        elif isinstance(current_value, list):
                            apply(current_value, new_value)
                        elif isinstance(current_value, tuple):
                            apply(current_value, new_value)
                        else:
                            if strict and not _is_mx_array(new_value):
                                raise ValueError(
                                    f"Received invalid type: {type(new_value).__name__}."
                                )
                            dst[key] = new_value
                    elif strict:
                        raise ValueError(f'Module does not have parameter named "{key}".')
                return
            if isinstance(dst, list):
                if not isinstance(params, list):
                    if strict:
                        raise ValueError(f"Received invalid type: {type(params).__name__}.")
                    return
                if strict and len(params) > len(dst):
                    raise ValueError("Received too many list elements for module update.")
                for index in range(len(params)):
                    current_value = dst[index]
                    new_value = params[index]
                    if _is_module_like(current_value):
                        current_value.update(new_value, strict=strict)
                    elif isinstance(current_value, dict):
                        apply(current_value, new_value)
                    elif isinstance(current_value, list):
                        apply(current_value, new_value)
                    elif isinstance(current_value, tuple):
                        apply(current_value, new_value)
                    else:
                        if strict and not _is_mx_array(new_value):
                            raise ValueError(
                                f"Received invalid type: {type(new_value).__name__}."
                            )
                        dst[index] = new_value
                return
            if isinstance(dst, tuple):
                if not isinstance(params, tuple):
                    if strict:
                        raise ValueError(f"Received invalid type: {type(params).__name__}.")
                    return
                if strict and len(params) != len(dst):
                    raise ValueError("Received tuple with mismatched length for module update.")
                updated = list(dst)
                for index in range(len(params)):
                    current_value = dst[index]
                    new_value = params[index]
                    if _is_module_like(current_value):
                        current_value.update(new_value, strict=strict)
                    elif isinstance(current_value, dict):
                        apply(current_value, new_value)
                    elif isinstance(current_value, list):
                        apply(current_value, new_value)
                    elif isinstance(current_value, tuple):
                        apply(current_value, new_value)
                    else:
                        if strict and not _is_mx_array(new_value):
                            raise ValueError(
                                f"Received invalid type: {type(new_value).__name__}."
                            )
                        updated[index] = new_value
                return type(dst)(updated)
            if strict and not _is_mx_array(params):
                raise ValueError(f"Received invalid type: {type(params).__name__}.")
            return params

        apply(self, parameters)
        return self

    def freeze(self, *, recurse: bool = True, keys=None, strict: bool = False):
        del recurse
        local_keys = keys if isinstance(keys, list) else [keys] if keys is not None else None
        if local_keys is None:
            local_keys = [
                key
                for key, value in self.items()
                if not key.startswith("_") and _is_mx_array(value)
            ]
        if strict:
            for key in local_keys:
                if key not in self:
                    raise KeyError(f"Module doesn't contain member {key}.")
        self._no_grad.update(local_keys)
        return self

    def unfreeze(self, *, recurse: bool = True, keys=None, strict: bool = False):
        del recurse
        if keys is None:
            self._no_grad.clear()
            return self
        local_keys = keys if isinstance(keys, list) else [keys]
        if strict:
            for key in local_keys:
                if key not in self:
                    raise KeyError(f"Module doesn't contain member {key}.")
        self._no_grad.difference_update(local_keys)
        return self

    def train(self, mode: bool = True):
        self._training = mode
        return self

    def eval(self):
        return self.train(False)


class CausalBankModel(_ModuleLike):
    """Frozen linear substrate plus a parallel local residual coder."""

    def __init__(self, vocab_size: int, config=None):
        super().__init__()
        opc = _get_opc()
        CausalBankConfig = opc["CausalBankConfig"]
        validate_config = opc["validate_config"]
        build_linear_bank = opc["build_linear_bank"]
        osc_pair_count = opc["osc_pair_count"]

        if config is None:
            config = CausalBankConfig()
        validate_config(config)

        self.vocab_size = vocab_size
        self.config = config

        self.shared_embedding = None
        self.linear_embedding = None
        self.local_embedding = None
        self.linear_in_proj = None
        self.linear_decays = None
        self.linear_kernel = None
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
            self.linear_in_proj = mx.array(in_proj)
            self.linear_decays = mx.array(decays.astype(np.float32))

            if config.linear_impl == "kernel":
                self.linear_kernel = mx.array(kernel)

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
                self.bank_gate_logits = mx.zeros((2,), dtype=mx.float32)

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

        freeze_keys = [key for key in ("linear_in_proj", "linear_decays", "linear_kernel") if getattr(self, key) is not None]
        if freeze_keys:
            self.freeze(keys=freeze_keys, strict=False)

    def _reset_trainable_parameters(self) -> None:
        seed = int(self.config.init_seed)
        if self.shared_embedding is not None:
            self.shared_embedding.weight = mx.array(
                _embedding_uniform(tuple(self.shared_embedding.weight.shape), _rng_for(seed, "shared_embedding.weight"))
            )
        if self.linear_embedding is not None:
            self.linear_embedding.weight = mx.array(
                _embedding_uniform(tuple(self.linear_embedding.weight.shape), _rng_for(seed, "linear_embedding.weight"))
            )
        if self.local_embedding is not None:
            self.local_embedding.weight = mx.array(
                _embedding_uniform(tuple(self.local_embedding.weight.shape), _rng_for(seed, "local_embedding.weight"))
            )
        if self.linear_readout is not None:
            self.linear_readout.reset_parameters_with_seed(seed, "linear_readout")
        if self.local_readout is not None:
            self.local_readout.reset_parameters_with_seed(seed, "local_readout")
        if self.gate_proj is not None:
            self.gate_proj.weight = mx.array(
                _xavier_uniform(tuple(self.gate_proj.weight.shape), _rng_for(seed, "gate_proj.weight"))
            )
            if self.gate_proj.bias is not None:
                self.gate_proj.bias = mx.zeros(self.gate_proj.bias.shape, dtype=mx.float32)
        if self.bank_gate_logits is not None:
            self.bank_gate_logits = mx.zeros(self.bank_gate_logits.shape, dtype=mx.float32)

    def set_linear_decays(self, decays: np.ndarray) -> None:
        if not self.config.enable_linear:
            raise RuntimeError("causal-bank linear path is disabled.")
        decays = np.asarray(decays, dtype=np.float32)
        if decays.shape != (self.config.linear_modes,):
            raise ValueError(
                f"causal-bank expected {self.config.linear_modes} decays, got shape {decays.shape}"
            )
        self.linear_decays = mx.array(decays.astype(np.float32, copy=False))
        if self.config.linear_impl == "kernel":
            self.linear_kernel = mx.array(_kernel_from_decays(decays, self.config.max_seq_len))
        self.freeze(keys=["linear_decays", "linear_kernel"], strict=False)

    @staticmethod
    def _logit_features(logits: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        probs = mx.exp(log_probs)
        entropy = -mx.sum(probs * log_probs, axis=-1)
        max_logit = mx.max(logits, axis=-1)
        centered = logits - mx.mean(logits, axis=-1, keepdims=True)
        variance = mx.mean(centered * centered, axis=-1)
        return entropy, max_logit, variance

    def _embed_linear(self, chars: mx.array) -> mx.array:
        if self.shared_embedding is not None:
            return self.shared_embedding(chars)
        if self.linear_embedding is None:
            raise RuntimeError("causal-bank linear path has no embedding table.")
        return self.linear_embedding(chars)

    def _embed_local(self, chars: mx.array) -> mx.array:
        if self.shared_embedding is not None:
            return self.shared_embedding(chars)
        if self.local_embedding is None:
            raise RuntimeError("causal-bank local path has no embedding table.")
        return self.local_embedding(chars)

    def _linear_states_fft(self, drive: mx.array, timesteps: int) -> mx.array:
        if self.linear_decays is None:
            raise RuntimeError("causal-bank FFT path is missing linear decays.")
        drive_mb = mx.transpose(drive, (0, 2, 1))
        n_fft = 1 << int(math.ceil(math.log2(max(2 * timesteps - 1, 1))))
        time = mx.arange(timesteps, dtype=drive.dtype)
        kernel = mx.power(self.linear_decays[:, None], time[None, :])
        drive_f = mx.fft.rfft(drive_mb, n=n_fft, axis=-1)
        kernel_f = mx.fft.rfft(kernel[None, :, :], n=n_fft, axis=-1)
        states_mb = mx.fft.irfft(drive_f * kernel_f, n=n_fft, axis=-1)[..., :timesteps]
        return mx.transpose(states_mb, (0, 2, 1))

    def _apply_mode_gate(self, states: mx.array, mode_gate: mx.array | None) -> mx.array:
        if mode_gate is None:
            return states
        if mode_gate.ndim == 1:
            return states * mode_gate[None, None, :]
        if mode_gate.ndim == 2:
            return states * mode_gate[:, None, :]
        raise ValueError(f"causal-bank mode_gate must be rank-1 or rank-2, got shape {mode_gate.shape}")

    def _static_bank_mode_gate(self) -> mx.array | None:
        if self.bank_gate_logits is None or self.osc_mode_count <= 0:
            return None
        values = 1.0 + self.config.bank_gate_span * mx.tanh(self.bank_gate_logits)
        pieces = []
        if self.non_osc_modes > 0:
            pieces.append(mx.broadcast_to(values[0:1], (self.non_osc_modes,)))
        if self.osc_mode_count > 0:
            pieces.append(mx.broadcast_to(values[1:2], (self.osc_mode_count,)))
        return mx.concatenate(pieces, axis=0) if pieces else None

    def _linear_states(self, chars: mx.array, mode_gate: mx.array | None = None) -> tuple[mx.array, mx.array]:
        _, timesteps = chars.shape
        if timesteps > self.config.max_seq_len:
            raise ValueError(
                f"causal-bank max_seq_len={self.config.max_seq_len} is smaller than input timesteps={timesteps}"
            )
        if self.linear_in_proj is None or self.linear_readout is None:
            raise RuntimeError("causal-bank linear path is disabled.")
        x = self._embed_linear(chars)
        drive = mx.matmul(x, self.linear_in_proj)
        if self.config.linear_impl == "kernel":
            if self.linear_kernel is None:
                raise RuntimeError("causal-bank kernel path is missing its materialized kernel.")
            kernels = self.linear_kernel[:, :timesteps, :timesteps]
            drive_mb = mx.transpose(drive, (2, 0, 1))
            states_mb = mx.matmul(drive_mb, mx.transpose(kernels, (0, 2, 1)))
            states = mx.transpose(states_mb, (1, 2, 0))
        else:
            states = self._linear_states_fft(drive, timesteps)
        states = self._apply_mode_gate(states, self._static_bank_mode_gate())
        return self._apply_mode_gate(states, mode_gate), x

    def _linear_logits(self, chars: mx.array, mode_gate: mx.array | None = None) -> mx.array:
        states, x = self._linear_states(chars, mode_gate=mode_gate)
        return self.linear_readout(mx.concatenate([states, x], axis=-1))

    def _local_window_stack(self, x: mx.array) -> mx.array:
        batch, timesteps, dim = x.shape
        window = self.config.local_window
        if window == 1:
            return x
        pad = mx.zeros((batch, window - 1, dim), dtype=x.dtype)
        padded = mx.concatenate([pad, x], axis=1)
        views = []
        for offset in range(window):
            start = window - 1 - offset
            views.append(padded[:, start : start + timesteps, :])
        return mx.concatenate(views, axis=-1)

    def _local_logits(self, chars: mx.array) -> mx.array:
        if self.local_readout is None:
            raise RuntimeError("causal-bank local path is disabled.")
        x = self._embed_local(chars)
        stacked = self._local_window_stack(x)
        return self.local_readout(stacked)

    def __call__(self, chars: mx.array) -> mx.array:
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
            features = mx.stack([ent_l, ent_r, max_l, max_r, var_l, var_r], axis=-1)
            gate = mx.sigmoid(self.gate_proj(features)) * self.config.local_scale

        return logits_linear + gate * logits_local

    def forward_with_mode_gate(self, chars: mx.array, mode_gate: mx.array | None) -> mx.array:
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
            features = mx.stack([ent_l, ent_r, max_l, max_r, var_l, var_r], axis=-1)
            gate = mx.sigmoid(self.gate_proj(features)) * self.config.local_scale

        return logits_linear + gate * logits_local
