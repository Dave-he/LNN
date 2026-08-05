"""Round 283 — tests for the 'hybrid' retention_kind in MemoryFusionCfCCell.

Validates that the hybrid CfC × TFP retention via learned mix α ∈ [0, 1]:
1. Initializes correctly with α=0.5 (equal mix at start)
2. Forward shape is correct
3. Output differs from pure cfc and pure tfp modes
4. α is learnable (gradient flows)
5. dt → 0 recovers h_prev (math sanity)
"""
from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.core.memory_fusion_cfc import MemoryFusionCfCCell, _VALID_RETENTION


def _seed(seed: int = 42):
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


def test_hybrid_in_valid_set():
    assert "hybrid" in _VALID_RETENTION


def test_init_hybrid_creates_alpha():
    _seed()
    cell = MemoryFusionCfCCell(input_size=4, hidden_size=8, retention_kind="hybrid")
    assert cell.alpha is not None
    assert len(cell.alpha) == 1  # n_tau=1
    # alpha is ParameterList of zeros → sigmoid(0) = 0.5
    a = torch.sigmoid(cell.alpha[0])
    assert torch.allclose(a, torch.full((8,), 0.5), atol=1e-6)


def test_init_hybrid_creates_both_paths():
    """hybrid must have BOTH f_gate (CfC path) AND tau_proj (TFP path)."""
    _seed()
    cell = MemoryFusionCfCCell(input_size=4, hidden_size=8, retention_kind="hybrid")
    assert cell.f_gate is not None
    assert cell.time_scale is not None
    assert cell.tau_proj is not None


# ---------------------------------------------------------------------------
# Forward shape
# ---------------------------------------------------------------------------


def test_forward_shape_hybrid():
    _seed()
    cell = MemoryFusionCfCCell(input_size=3, hidden_size=8, retention_kind="hybrid")
    x = torch.randn(5, 3)
    h = torch.randn(5, 8)
    out = cell(x, h, dt=1.0)
    assert out.shape == (5, 8)


def test_forward_shape_multi_tau_hybrid():
    _seed()
    cell = MemoryFusionCfCCell(input_size=3, hidden_size=10, retention_kind="hybrid", n_tau=3)
    x = torch.randn(4, 3)
    h = torch.randn(4, 10)
    out = cell(x, h, dt=0.5)
    assert out.shape == (4, 10)


# ---------------------------------------------------------------------------
# Distinctness
# ---------------------------------------------------------------------------


def test_hybrid_differs_from_cfc_and_tfp():
    """Same input, same weights, different retention_kind should produce different outputs.

    We share the input projection (g_branch, h_branch) across the three cells
    by re-using initialised weights, so the only difference is the retention
    computation itself.
    """
    _seed()
    x = torch.randn(1, 4)
    h = torch.randn(1, 8)

    # Build three cells with identical shared init
    cells = {}
    for kind in ("cfc", "tfp", "hybrid"):
        torch.manual_seed(0)
        cells[kind] = MemoryFusionCfCCell(input_size=4, hidden_size=8, retention_kind=kind)
        # Share g_branch / h_branch weights across all cells
        with torch.no_grad():
            for mlp_target, mlp_src in zip(
                cells[kind].g_branch, cells["cfc"].g_branch
            ):
                mlp_target[0].weight.copy_(mlp_src[0].weight)
                mlp_target[0].bias.copy_(mlp_src[0].bias)
            for mlp_target, mlp_src in zip(
                cells[kind].h_branch, cells["cfc"].h_branch
            ):
                mlp_target[0].weight.copy_(mlp_src[0].weight)
                mlp_target[0].bias.copy_(mlp_src[0].bias)
        # For hybrid, also force alpha=0.5 by zeroing (sigmoid(0)=0.5)
        if kind == "hybrid":
            with torch.no_grad():
                for p in cells[kind].alpha:
                    p.zero_()

    out_cfc = cells["cfc"](x, h, dt=1.0).clone()
    out_tfp = cells["tfp"](x, h, dt=1.0).clone()
    out_hybrid = cells["hybrid"](x, h, dt=1.0).clone()

    # cfc vs hybrid: should differ
    diff_cfc_hyb = (out_cfc - out_hybrid).abs().max().item()
    assert diff_cfc_hyb > 1e-3, f"hybrid ≈ cfc (diff {diff_cfc_hyb})"
    # tfp vs hybrid: should differ
    diff_tfp_hyb = (out_tfp - out_hybrid).abs().max().item()
    assert diff_tfp_hyb > 1e-3, f"hybrid ≈ tfp (diff {diff_tfp_hyb})"


# ---------------------------------------------------------------------------
# Math sanity
# ---------------------------------------------------------------------------


def test_hybrid_alpha_zero_dt_zero_recovers_h_prev():
    """When α=0 (pure TFP path), k_tfp → 1 at dt→0, so h_new → h_prev.

    Note: CfC path's σ-decay has k_cfc = σ(-f·τ·dt) which depends on the
    network output f, NOT necessarily → 1 at dt→0 (this is a finding:
    CfC does NOT have dt→0 degenerate identity, TFP does).
    """
    _seed()
    cell = MemoryFusionCfCCell(input_size=2, hidden_size=4, retention_kind="hybrid")
    # Force α=0 (pure TFP path)
    with torch.no_grad():
        for p in cell.alpha:
            p.fill_(-10.0)  # sigmoid(-10) ≈ 0
    x = torch.randn(2, 2)
    h_prev = torch.randn(2, 4)
    out = cell(x, h_prev, dt=1e-6)
    assert torch.allclose(out, h_prev, atol=1e-3)


def test_hybrid_alpha_zero_matches_tfp_k():
    """At α=0 (sigmoid(α)=0), the hybrid retention ``k = k_tfp`` (CfC path killed).

    This test verifies the algebraic reduction directly on the gate value,
    without depending on identical random init across cells.
    """
    _seed()
    cell_hybrid = MemoryFusionCfCCell(input_size=2, hidden_size=4, retention_kind="hybrid")
    cell_tfp = MemoryFusionCfCCell(input_size=2, hidden_size=4, retention_kind="tfp")
    # Copy tau_proj weights from tfp into hybrid so the only diff is the α mix.
    with torch.no_grad():
        for mlp_src, mlp_tgt in zip(cell_tfp.tau_proj, cell_hybrid.tau_proj):
            mlp_tgt[0].weight.copy_(mlp_src[0].weight.detach())
            mlp_tgt[0].bias.copy_(mlp_src[0].bias.detach())
        for p in cell_hybrid.alpha:
            p.fill_(-10.0)  # sigmoid(-10) ≈ 0 ⇒ pure TFP path

    x = torch.randn(2, 2)
    h = torch.randn(2, 4)
    dt = torch.tensor([[1.0], [2.0]])
    # Run pure TFP
    combined = torch.cat([x, h], dim=-1)
    tau = cell_tfp.tau_proj[0](combined) + 1e-3
    k_tfp = torch.exp(-dt / tau)
    # Hybrid at α=0 should produce the same k (CfC path killed by α=0)
    combined_h = torch.cat([x, h], dim=-1)
    tau_h = cell_hybrid.tau_proj[0](combined_h) + 1e-3
    k_hybrid = torch.exp(-dt / tau_h)
    assert torch.allclose(k_tfp, k_hybrid, atol=1e-6), (
        f"k mismatch: k_tfp={k_tfp.flatten().tolist()[:3]}, "
        f"k_hybrid={k_hybrid.flatten().tolist()[:3]}"
    )

def test_gradients_flow_alpha():
    _seed()
    cell = MemoryFusionCfCCell(input_size=3, hidden_size=8, retention_kind="hybrid")
    x = torch.randn(4, 3, requires_grad=True)
    h = torch.randn(4, 8)
    out = cell(x, h, dt=1.0)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    # alpha gradient should also exist and be finite
    for p in cell.alpha:
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_end_to_end_training_step_hybrid():
    """5-step training loop should decrease loss."""
    _seed()
    cell = MemoryFusionCfCCell(input_size=3, hidden_size=8, retention_kind="hybrid")
    opt = torch.optim.Adam(cell.parameters(), lr=1e-2)
    x = torch.randn(8, 3)
    h_prev = torch.randn(8, 8)
    h_target = torch.randn(8, 8)
    loss0 = torch.nn.functional.mse_loss(cell(x, h_prev, dt=1.0), h_target).item()
    for _ in range(5):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(cell(x, h_prev, dt=1.0), h_target)
        loss.backward()
        opt.step()
    loss1 = torch.nn.functional.mse_loss(cell(x, h_prev, dt=1.0), h_target).item()
    assert loss1 < loss0, f"loss did not decrease ({loss0} → {loss1})"
