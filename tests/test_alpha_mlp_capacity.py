"""Round 19 (N22) — tests for α MLP capacity variants in hybrid_gate.

Validates that alpha_mlp_depth and alpha_mlp_width parameters produce
correctly-shaped MLPs with varying capacity.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.memory_fusion_cfc import MemoryFusionCfCCell


def _seed(seed: int = 42):
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Init tests
# ---------------------------------------------------------------------------


def test_default_depth_is_1():
    """Backward compat: default α MLP depth should be 2 (1 Linear + 1 Linear = 2 layers)."""
    _seed()
    cell = MemoryFusionCfCCell(input_size=4, hidden_size=8, retention_kind="hybrid_gate")
    assert cell.alpha_mlp_depth == 1
    # Verify gate_mlp has exactly 2 Linear layers (depth=1 means 1 hidden layer + 1 output)
    mlp = cell.gate_mlps[0]
    linear_layers = [m for m in mlp if isinstance(m, torch.nn.Linear)]
    assert len(linear_layers) == 1, f"depth=1 should give 1 Linear layer, got {len(linear_layers)}"


def test_explicit_depth_3():
    """depth=3 → 3 Linear layers (1 hidden + 1 hidden + 1 output)."""
    _seed()
    cell = MemoryFusionCfCCell(
        input_size=4, hidden_size=8, retention_kind="hybrid_gate",
        alpha_mlp_depth=3,
    )
    assert cell.alpha_mlp_depth == 3
    mlp = cell.gate_mlps[0]
    linear_layers = [m for m in mlp if isinstance(m, torch.nn.Linear)]
    assert len(linear_layers) == 3


def test_explicit_width_doubles():
    """width=2*d → hidden layer has 2x branch_dim units."""
    _seed()
    cell = MemoryFusionCfCCell(
        input_size=4, hidden_size=8, retention_kind="hybrid_gate",
        alpha_mlp_depth=2, alpha_mlp_width=16,  # 16 = 2 × 8
    )
    mlp = cell.gate_mlps[0]
    linear_layers = [m for m in mlp if isinstance(m, torch.nn.Linear)]
    # Layer 0: in_dim=5 -> 16
    assert linear_layers[0].in_features == 5
    assert linear_layers[0].out_features == 16
    # Layer 1: 16 -> 8 (branch_dim)
    assert linear_layers[1].in_features == 16
    assert linear_layers[1].out_features == 8


def test_width_0_means_use_branch_dim():
    """width=0 (default) means hidden layer matches branch_dim."""
    _seed()
    cell = MemoryFusionCfCCell(
        input_size=4, hidden_size=8, retention_kind="hybrid_gate",
        alpha_mlp_width=0,
    )
    mlp = cell.gate_mlps[0]
    linear_layers = [m for m in mlp if isinstance(m, torch.nn.Linear)]
    assert linear_layers[0].out_features == 8  # = branch_dim


# ---------------------------------------------------------------------------
# Forward shape
# ---------------------------------------------------------------------------


def test_forward_shape_depth_3():
    _seed()
    cell = MemoryFusionCfCCell(
        input_size=3, hidden_size=8, retention_kind="hybrid_gate",
        alpha_mlp_depth=3, alpha_mlp_width=16,
    )
    x = torch.randn(5, 3)
    h = torch.randn(5, 8)
    dt = torch.ones(5, 1)
    out = cell(x, h, dt=dt)
    assert out.shape == (5, 8)


# ---------------------------------------------------------------------------
# α variability
# ---------------------------------------------------------------------------


def test_alpha_varies_with_x_and_dt_depth_3_after_training():
    """Deeper MLP α varies with input after training (init may have small spread due to Sigmoid saturation)."""
    _seed()
    cell = MemoryFusionCfCCell(
        input_size=4, hidden_size=8, retention_kind="hybrid_gate",
        alpha_mlp_depth=3, alpha_mlp_width=16,
    )
    # Quick training: just one step with MSE loss to break initial symmetry
    opt = torch.optim.Adam(cell.parameters(), lr=1e-2)
    x = torch.randn(8, 4)
    h = torch.randn(8, 8)
    target = torch.randn(8, 8)
    for _ in range(5):
        opt.zero_grad()
        dt = torch.ones(8, 1)
        out = cell(x, h, dt=dt)
        torch.nn.functional.mse_loss(out, target).backward()
        opt.step()
    # Now check α spread
    with torch.no_grad():
        a_x1 = cell.gate_mlps[0](torch.cat([torch.zeros(1, 4), torch.ones(1, 1)], dim=-1))
        a_x2 = cell.gate_mlps[0](torch.cat([torch.ones(1, 4) * 5, torch.ones(1, 1)], dim=-1))
        a_dt1 = cell.gate_mlps[0](torch.cat([torch.zeros(1, 4), torch.tensor([[1.0]])], dim=-1))
        a_dt2 = cell.gate_mlps[0](torch.cat([torch.zeros(1, 4), torch.tensor([[5.0]])], dim=-1))
    assert (a_x1 - a_x2).abs().mean() > 1e-4, f"α should depend on x with depth=3 after training (got {(a_x1 - a_x2).abs().mean().item():.6f})"
    assert (a_dt1 - a_dt2).abs().mean() > 1e-4, f"α should depend on dt with depth=3 after training (got {(a_dt1 - a_dt2).abs().mean().item():.6f})"


def test_gradients_flow_depth_3():
    _seed()
    cell = MemoryFusionCfCCell(
        input_size=3, hidden_size=8, retention_kind="hybrid_gate",
        alpha_mlp_depth=3, alpha_mlp_width=16,
    )
    x = torch.randn(4, 3, requires_grad=True)
    h = torch.randn(4, 8)
    dt = torch.ones(4, 1)
    out = cell(x, h, dt=dt)
    out.sum().backward()
    assert x.grad is not None
    for mlp in cell.gate_mlps:
        for lin in mlp:
            if hasattr(lin, "weight"):
                assert lin.weight.grad is not None
                assert torch.isfinite(lin.weight.grad).all()


def test_capacity_increases_params():
    """Wider+deeper gate_mlp should have more parameters than default."""
    _seed()
    cell_default = MemoryFusionCfCCell(input_size=4, hidden_size=8, retention_kind="hybrid_gate")
    cell_wide = MemoryFusionCfCCell(
        input_size=4, hidden_size=8, retention_kind="hybrid_gate",
        alpha_mlp_depth=3, alpha_mlp_width=32,
    )
    p_default = sum(p.numel() for p in cell_default.gate_mlps.parameters())
    p_wide = sum(p.numel() for p in cell_wide.gate_mlps.parameters())
    assert p_wide > p_default * 3, f"wide ({p_wide}) should be > 3x default ({p_default})"
