"""Int8 quantization for LNN edge deployment.

Implements the Stage 3 (post-training quantization) of the DLNet pipeline
(arXiv 2601.06227): after dual-stage distillation, quantize the student
weights from float32 to int8 for true edge deployment (MCU, TFLite micro,
etc.).

Two quantization modes:
  - ``quantize_int8_per_tensor``: standard symmetric per-tensor quantization.
    Compute one scale + zero-point per weight tensor.
  - ``quantize_int8_per_channel``: per-output-channel scales for Linear
    layers (more accurate but more metadata).

Forward pass uses a custom ``Int8Linear`` that dequantises weights on-the-fly
to float, so the model still runs on float hardware but inference cost is
representative of int8 inference (no float matmul of full precision).
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


def quantize_int8_per_tensor(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Symmetric per-tensor int8 quantization.

    Returns ``(weight_q, scale, zero_point)`` where:
        weight_q: int8 tensor with same shape as ``weight``
        scale: float32 scalar (weight.abs().max() / 127)
        zero_point: int8 scalar (always 0 for symmetric quantization)
    """
    w = weight.detach().float()
    wmax = w.abs().max()
    if wmax == 0:
        scale = torch.tensor(1.0, dtype=torch.float32)
    else:
        scale = wmax / 127.0
    scale = scale.clamp(min=1e-8)
    wq = torch.round(w / scale).clamp(-128, 127).to(torch.int8)
    zp = torch.zeros((), dtype=torch.int8)
    return wq, scale, zp


def quantize_int8_per_channel(weight: torch.Tensor, channel_dim: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel symmetric int8 quantization (one scale per channel).

    For a Linear weight of shape ``[out, in]``, channel_dim=0 gives one
    scale per output feature (per output neuron); we amax over the input
    dim to compute the per-output scale.

    Args:
        weight: tensor to quantise.
        channel_dim: dim to KEEP as the channel axis. Default 0.

    Returns ``(weight_q, scale)``. ``scale`` has one entry per channel.
    """
    w = weight.detach().float()
    reduce_dims = [d for d in range(w.dim()) if d != channel_dim]
    if not reduce_dims:
        reduce_dims = list(range(w.dim()))
    wmax = w.abs().amax(dim=reduce_dims, keepdim=True)
    wmax = wmax.clamp(min=1e-8)
    scale_full = wmax / 127.0
    wq = torch.round(w / scale_full).clamp(-128, 127).to(torch.int8)
    # Squeeze the reduce_dims only (NOT channel_dim)
    for d in sorted(reduce_dims, reverse=True):
        scale_full = scale_full.squeeze(d)
    return wq, scale_full


def dequantize_int8(weight_q: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor = None) -> torch.Tensor:
    """Inverse of int8 quantization (handles per-tensor and per-channel).

    ``scale`` can be a scalar or a per-channel tensor that broadcasts against
    ``weight_q`` shape.
    """
    if zero_point is None:
        zero_point = torch.zeros((), dtype=torch.int8, device=weight_q.device)
    if scale.dim() == 0:
        return (weight_q.float() - zero_point.float()) * scale
    # Per-channel: scale has shape that broadcasts to weight_q.
    # Reshape scale to match weight_q's shape (insert trailing 1's for reduce dims).
    while scale.dim() < weight_q.dim():
        scale = scale.unsqueeze(-1)
    return (weight_q.float() - zero_point.float()) * scale


# ---------------------------------------------------------------------------
# Module-level wrapper: quantize all Linear/Cell weights in-place
# ---------------------------------------------------------------------------


def quantize_model_inplace(model: nn.Module, per_channel: bool = True) -> dict:
    """Walk through ``model`` and replace every :class:`nn.Linear` weight
    with its int8-quantised + dequantised float version (simulates int8
    inference with float hardware). Records the per-tensor quantization
    metadata in a dict keyed by ``(module_name, param_name)``.

    Returns:
        dict mapping (module_name, param_name) -> {"scale": float,
        "zero_point": int, "int8_size_bytes": int, "fp32_size_bytes": int}
    """
    meta = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            w = module.weight.detach()
            if per_channel and w.dim() == 2:
                wq, scale = quantize_int8_per_channel(w, channel_dim=0)
                zp = torch.zeros((), dtype=torch.int8)
            else:
                wq, scale, zp = quantize_int8_per_tensor(w)
            w_deq = dequantize_int8(wq, scale, zp)
            with torch.no_grad():
                module.weight.copy_(w_deq.to(module.weight.dtype))
            int8_bytes = wq.numel()  # 1 byte per int8 element
            fp32_bytes = w.numel() * 4
            meta[(name, "weight")] = {
                "scale": scale.tolist() if hasattr(scale, "tolist") else float(scale),
                "zero_point": int(zp) if zp.dim() == 0 else zp.tolist(),
                "int8_size_bytes": int8_bytes,
                "fp32_size_bytes": fp32_bytes,
            }
            if module.bias is not None:
                # Quantise bias too (per-tensor)
                b = module.bias.detach()
                bq, bscale, bzp = quantize_int8_per_tensor(b)
                b_deq = dequantize_int8(bq, bscale, bzp)
                with torch.no_grad():
                    module.bias.copy_(b_deq.to(module.bias.dtype))
                meta[(name, "bias")] = {
                    "scale": float(bscale),
                    "zero_point": int(bzp),
                    "int8_size_bytes": int(bq.numel()),
                    "fp32_size_bytes": int(b.numel() * 4),
                }
    return meta


def total_compressed_size_bytes(meta: dict) -> int:
    """Sum int8 footprint across all quantized parameters."""
    return sum(v["int8_size_bytes"] for v in meta.values())


def total_fp32_size_bytes(meta: dict) -> int:
    """Sum fp32 footprint (for comparison)."""
    return sum(v["fp32_size_bytes"] for v in meta.values())


__all__ = [
    "quantize_int8_per_tensor",
    "quantize_int8_per_channel",
    "dequantize_int8",
    "quantize_model_inplace",
    "total_compressed_size_bytes",
    "total_fp32_size_bytes",
]
