"""Unit tests for LiquidTAD-style hierarchical decay-rate sharing.

These tests exercise the new HierarchicalDecayLiquidBlock and
HierarchicalDecayLiquidTADHead added per the LiquidTAD paper (arXiv:2604.18274)
research report at docs/reports/LiquidTAD_Efficient_Temporal_Action_Detection_研读报告.md
(PRD §8 task #2, stage A).
"""
from __future__ import annotations

import pytest
import torch

from lnn.core.long_sequence import (
    HierarchicalDecayLiquidBlock,
    HierarchicalDecayLiquidTADHead,
    parallel_liquid_relaxation,
)


def test_hierarchical_decay_block_shapes_and_grad():
    torch.manual_seed(0)
    block = HierarchicalDecayLiquidBlock(input_size=8, hidden_size=16, kernel_size=5)
    x = torch.randn(2, 24, 8, requires_grad=True)
    y = block(x)
    assert y.shape == (2, 24, 16)
    assert torch.isfinite(y).all()
    # Backward must reach the shared decay parameter (single learnable scalar per channel).
    y.sum().backward()
    assert block.retain_logits.grad is not None
    assert block.retain_logits.grad.abs().sum().item() > 0.0
    assert block.retain_logits.grad.shape == (16,)


def test_hierarchical_decay_block_retain_in_unit_interval():
    block = HierarchicalDecayLiquidBlock(input_size=4, hidden_size=8, init_decay=0.7)
    retain = block.effective_retain()
    assert retain.shape == (8,)
    assert (retain > 0.0).all() and (retain < 1.0).all()
    # init_decay should round-trip through the logit transform.
    assert torch.allclose(retain, torch.full((8,), 0.7), atol=1e-5)


def test_hierarchical_decay_block_mask_zeroes_padding():
    block = HierarchicalDecayLiquidBlock(input_size=4, hidden_size=8)
    x = torch.randn(1, 10, 4)
    mask = torch.ones(1, 10)
    mask[0, 7:] = 0.0
    y = block(x, mask=mask)
    assert torch.allclose(y[0, 7:], torch.zeros_like(y[0, 7:]))
    assert not torch.allclose(y[0, :7], torch.zeros_like(y[0, :7]))


def test_hierarchical_decay_tad_head_outputs_and_decay_growth():
    head = HierarchicalDecayLiquidTADHead(
        input_size=6,
        num_classes=4,
        hidden_size=12,
        num_blocks=3,
        init_decay=0.80,
        decay_growth=1.05,
    )
    # Deeper blocks should start with strictly larger decay coefficients.
    schedule = head.decay_schedule
    assert len(schedule) == 3
    assert all(s2 > s1 for s1, s2 in zip(schedule, schedule[1:]))
    assert max(schedule) <= 0.99

    x = torch.randn(2, 20, 6)
    out = head(x)
    assert out["frame_logits"].shape == (2, 20, 4)
    assert out["boundaries"].shape == (2, 20, 2)
    assert ((out["boundaries"] >= 0.0) & (out["boundaries"] <= 1.0)).all()


def test_hierarchical_decay_tad_head_share_decay_ties_params():
    head = HierarchicalDecayLiquidTADHead(
        input_size=4,
        num_classes=2,
        hidden_size=8,
        num_blocks=3,
        share_decay=True,
    )
    # When share_decay=True the same nn.Parameter object is reused across blocks.
    first = head.blocks[0].retain_logits
    for block in head.blocks[1:]:
        assert block.retain_logits is first


def test_parallel_liquid_relaxation_matches_recurrent_reference():
    """The cumsum-log parallel form must equal the explicit recurrent update."""
    torch.manual_seed(1)
    batch, time_steps, channels = 2, 16, 4
    retain = torch.rand(batch, time_steps, channels) * 0.9 + 0.05
    value = torch.randn(batch, time_steps, channels)

    parallel = parallel_liquid_relaxation(retain, value)
    retain_clamped = retain.clamp(0.02, 0.98)

    recurrent = torch.zeros(batch, time_steps, channels)
    h = torch.zeros(batch, channels)
    for t in range(time_steps):
        h = retain_clamped[:, t] * h + (1.0 - retain_clamped[:, t]) * value[:, t]
        recurrent[:, t] = h

    # cumsum-log path drifts slightly from the strict recurrence for small retain
    # values because of the prefix.clamp_min(1e-8) guard.  1e-3 tolerance covers
    # both forms; tighten if you remove the guard.
    assert torch.allclose(parallel, recurrent, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
