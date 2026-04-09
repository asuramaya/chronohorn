"""Tests for models/polyhash_v12.py — TTT, XSA, causality, scan modes."""
from __future__ import annotations

import pytest

try:
    import torch
except ImportError:
    pytest.skip("torch not installed", allow_module_level=True)

from chronohorn.families.polyhash.models.polyhash_v12 import PolyHashV12, TTTLayer, V12Config


def _tiny_config(**overrides):
    defaults = dict(
        pkm_enabled=True, pkm_sub_keys=16, pkm_top_k=4,
        pkm_key_dim=16, pkm_value_dim=16,
        isab_enabled=False, scan_dim=32, hidden_dim=64,
        num_layers=2, hash_buckets=256, num_hash_tables=2,
        hash_embed_dim=8, byte_embed_dim=16, conv_kernel=4,
        match_offsets=(1, 2, 3, 4),
    )
    defaults.update(overrides)
    return V12Config(**defaults)


def _rand_tokens(cfg: V12Config, shape: tuple[int, ...]):
    return torch.randint(0, cfg.vocab_size, shape)


class TestBasicForward:
    def test_forward_shape(self):
        cfg = _tiny_config()
        model = PolyHashV12(cfg)
        x = _rand_tokens(cfg, (2, 32))
        out = model(x)
        assert out.shape == (2, 32, cfg.vocab_size)

    def test_forward_no_nan(self):
        cfg = _tiny_config()
        model = PolyHashV12(cfg)
        x = _rand_tokens(cfg, (2, 32))
        out = model(x)
        assert not torch.isnan(out).any()

    def test_forward_shape_custom_vocab(self):
        cfg = _tiny_config(vocab_size=257)
        model = PolyHashV12(cfg)
        x = _rand_tokens(cfg, (2, 32))
        out = model(x)
        assert out.shape == (2, 32, cfg.vocab_size)


class TestXSA:
    def test_xsa_forward(self):
        cfg = _tiny_config(pkm_xsa=True)
        model = PolyHashV12(cfg)
        x = _rand_tokens(cfg, (2, 32))
        out = model(x)
        assert out.shape == (2, 32, cfg.vocab_size)
        assert not torch.isnan(out).any()

    def test_xsa_differs_from_standard(self):
        """XSA should produce different outputs than standard softmax."""
        torch.manual_seed(42)
        cfg_std = _tiny_config(pkm_xsa=False)
        cfg_xsa = _tiny_config(pkm_xsa=True)
        model_std = PolyHashV12(cfg_std)
        model_xsa = PolyHashV12(cfg_xsa)
        model_xsa.load_state_dict(model_std.state_dict())
        x = _rand_tokens(cfg_std, (1, 16))
        out_std = model_std(x)
        out_xsa = model_xsa(x)
        assert not torch.allclose(out_std, out_xsa), "XSA should differ from standard"


class TestTTT:
    def test_ttt_forward(self):
        cfg = _tiny_config(ttt_enabled=True, ttt_dim=16, ttt_lr=0.01, ttt_mini_batch=8)
        model = PolyHashV12(cfg)
        x = _rand_tokens(cfg, (2, 32))
        out = model(x)
        assert out.shape == (2, 32, cfg.vocab_size)
        assert not torch.isnan(out).any()

    def test_ttt_causality(self):
        """Changing future tokens must not affect past outputs."""
        ttt = TTTLayer(input_dim=32, ttt_dim=16, inner_lr=0.01, mini_batch=4)
        # set to inference mode
        ttt.train(False)
        x = torch.randn(1, 16, 32)
        x2 = x.clone()
        x2[0, 12:] = torch.randn(4, 32)

        with torch.no_grad():
            out1 = ttt(x)
            out2 = ttt(x2)

        diff = (out1[0, :12] - out2[0, :12]).abs().max().item()
        assert diff < 1e-6, f"Causality violation: past positions differ by {diff}"

    def test_ttt_stability(self):
        """TTT should not produce NaN during training."""
        cfg = _tiny_config(ttt_enabled=True, ttt_dim=16, ttt_lr=0.01, ttt_mini_batch=4)
        model = PolyHashV12(cfg)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        for _ in range(20):
            x = _rand_tokens(cfg, (2, 16))
            out = model(x)
            loss = torch.nn.functional.cross_entropy(
                out[:, :-1].reshape(-1, out.size(-1)), x[:, 1:].reshape(-1)
            )
            assert not torch.isnan(loss), "NaN loss during TTT training"
            opt.zero_grad()
            loss.backward()
            opt.step()

    def test_ttt_param_overhead(self):
        """TTT should add minimal parameters."""
        cfg_no_ttt = _tiny_config(ttt_enabled=False)
        cfg_ttt = _tiny_config(ttt_enabled=True, ttt_dim=16)
        p_base = sum(p.numel() for p in PolyHashV12(cfg_no_ttt).parameters())
        p_ttt = sum(p.numel() for p in PolyHashV12(cfg_ttt).parameters())
        overhead = p_ttt - p_base
        assert overhead > 0
        assert overhead < p_base * 0.1, f"TTT overhead {overhead} > 10% of base {p_base}"


class TestXSAPlusTTT:
    def test_combined_forward(self):
        cfg = _tiny_config(pkm_xsa=True, ttt_enabled=True, ttt_dim=16, ttt_lr=0.01)
        model = PolyHashV12(cfg)
        x = _rand_tokens(cfg, (2, 32))
        out = model(x)
        assert out.shape == (2, 32, cfg.vocab_size)
        assert not torch.isnan(out).any()


class TestScanModes:
    def test_gated_scan(self):
        cfg = _tiny_config(scan_dim=32, scan_rotation=False)
        model = PolyHashV12(cfg)
        x = _rand_tokens(cfg, (1, 16))
        out = model(x)
        assert out.shape == (1, 16, cfg.vocab_size)

    def test_rotation_scan(self):
        cfg = _tiny_config(scan_dim=32, scan_rotation=True)
        model = PolyHashV12(cfg)
        x = _rand_tokens(cfg, (1, 16))
        out = model(x)
        assert out.shape == (1, 16, cfg.vocab_size)
        assert not torch.isnan(out).any()

    def test_mamba_scan_fallback(self):
        """MambaScan falls back to GatedScan when mamba-ssm not installed."""
        from chronohorn.families.polyhash.models.polyhash_v12 import MambaScan
        cfg = _tiny_config(scan_dim=32, scan_mamba=True)
        model = PolyHashV12(cfg)
        assert isinstance(model.scan, MambaScan)
        x = _rand_tokens(cfg, (1, 16))
        out = model(x)
        assert out.shape == (1, 16, cfg.vocab_size)
        assert not torch.isnan(out).any()

    def test_mamba_scan_trains(self):
        """MambaScan (fallback mode) produces gradients and doesn't NaN."""
        cfg = _tiny_config(scan_dim=32, scan_mamba=True)
        model = PolyHashV12(cfg)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        for _ in range(10):
            x = _rand_tokens(cfg, (2, 16))
            out = model(x)
            loss = torch.nn.functional.cross_entropy(
                out[:, :-1].reshape(-1, out.size(-1)), x[:, 1:].reshape(-1)
            )
            assert not torch.isnan(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()

    def test_no_scan(self):
        cfg = _tiny_config(scan_dim=0)
        model = PolyHashV12(cfg)
        x = _rand_tokens(cfg, (1, 16))
        out = model(x)
        assert out.shape == (1, 16, cfg.vocab_size)
