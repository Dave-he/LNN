"""Round 16 (N20) — tests for int8 quantization of LNN distillation students.

Validates per-tensor and per-channel quantization, model-level in-place
quantization, and size accounting.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.quantization import (
    dequantize_int8,
    quantize_int8_per_channel,
    quantize_int8_per_tensor,
    quantize_model_inplace,
    total_compressed_size_bytes,
    total_fp32_size_bytes,
)


def _seed(seed: int = 42):
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Per-tensor quantization
# ---------------------------------------------------------------------------


def test_quantize_int8_per_tensor_shape():
    _seed()
    w = torch.randn(8, 4)
    wq, scale, zp = quantize_int8_per_tensor(w)
    assert wq.shape == w.shape
    assert wq.dtype == torch.int8
    assert scale.dim() == 0
    assert zp.dim() == 0
    assert zp.item() == 0  # symmetric


def test_quantize_int8_per_tensor_range():
    _seed()
    w = torch.randn(8, 4) * 5.0
    wq, scale, _ = quantize_int8_per_tensor(w)
    assert wq.min() >= -128
    assert wq.max() <= 127


def test_quantize_int8_per_tensor_recovery_error():
    """Round-trip error should be bounded by scale/2."""
    _seed()
    w = torch.randn(8, 4)
    wq, scale, _ = quantize_int8_per_tensor(w)
    w_deq = dequantize_int8(wq, scale)
    err = (w - w_deq).abs().max().item()
    # Each int8 bucket spans (scale * 2) values; quantization error <= scale
    assert err <= scale.item() * 1.01, f"recovery error {err} > scale {scale.item()}"


def test_quantize_int8_per_tensor_zero_weight():
    """All-zero weight should quantize to all-zero int8 without division-by-zero."""
    _seed()
    w = torch.zeros(3, 3)
    wq, scale, _ = quantize_int8_per_tensor(w)
    assert (wq == 0).all()
    assert scale.item() == 1.0  # fallback when wmax==0


# ---------------------------------------------------------------------------
# Per-channel quantization
# ---------------------------------------------------------------------------


def test_quantize_int8_per_channel_shape():
    _seed()
    w = torch.randn(8, 4)
    wq, scale = quantize_int8_per_channel(w, channel_dim=0)
    assert wq.shape == w.shape
    assert wq.dtype == torch.int8
    assert scale.shape == (8,)  # one scale per output channel


# ---------------------------------------------------------------------------
# Module-level in-place quantization
# ---------------------------------------------------------------------------


def test_quantize_model_inplace_returns_meta():
    _seed()
    model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 1))
    meta = quantize_model_inplace(model, per_channel=True)
    assert len(meta) >= 2  # at least 2 Linear weights (and optionally biases)
    for k, v in meta.items():
        assert "scale" in v
        assert "int8_size_bytes" in v
        assert "fp32_size_bytes" in v


def test_quantize_model_inplace_keeps_forward_runnable():
    """After quantization, the model should still produce valid output."""
    _seed()
    model = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 1))
    x = torch.randn(3, 4)
    y_pre = model(x).detach().clone()
    meta = quantize_model_inplace(model, per_channel=True)
    y_post = model(x)
    assert y_post.shape == y_pre.shape
    assert torch.isfinite(y_post).all()


def test_quantization_introduces_bounded_error():
    """Quantizing a Linear then re-running should keep MSE bounded."""
    _seed()
    torch.manual_seed(0)
    x = torch.randn(64, 4)
    y = torch.randn(64, 1)
    model = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 1))
    y_pre = model(x).detach()
    meta = quantize_model_inplace(model, per_channel=True)
    y_post = model(x).detach()
    mse = torch.nn.functional.mse_loss(y_post, y_pre).item()
    # Per-channel int8 with 256 buckets gives roughly scale/2 error.
    # For random N(0,1) weights, scales are small (<0.1), so MSE < 1e-3.
    assert mse < 1e-3, f"Quantization MSE {mse} too large"


def test_total_sizes_account_correctly():
    """int8 size should be ~1/4 of fp32 size (1 byte vs 4 bytes per weight)."""
    _seed()
    model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 1))
    meta = quantize_model_inplace(model, per_channel=True)
    int8_size = total_compressed_size_bytes(meta)
    fp32_size = total_fp32_size_bytes(meta)
    # All params in this model are 1D/2D matrices — int8 should be 1/4 of fp32
    assert int8_size == fp32_size // 4, f"int8={int8_size}, fp32={fp32_size}"
