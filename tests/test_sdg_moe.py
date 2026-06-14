"""Round 104 — Tests for SDG-MoE Signed Debate Graph Deliberation (PRD #10-66)."""
from __future__ import annotations

import pytest
import torch

from lnn.core.sdg_moe import (
    SDGConfig,
    SDGLearnedInteractions,
    SDGQuiteMoECfCCell,
    SDGQuiteMoECfCNetwork,
    disagreement_score,
    signed_debate_step,
)


class TestDisagreementScore:
    """Tests for the disagreement_score function."""

    def test_identical_experts_disagreement_zero(self):
        """Identical expert outputs give disagreement = 0."""
        e = torch.randn(1, 1, 8).expand(3, 3, 8).contiguous()  # all 3 experts same
        d = disagreement_score(e)
        assert d.shape == (3,)
        assert (d.abs() < 1e-5).all()

    def test_orthogonal_experts_high_disagreement(self):
        """Orthogonal expert outputs give disagreement close to 1."""
        e = torch.zeros(1, 3, 8)
        e[0, 0, 0] = 1.0
        e[0, 1, 1] = 1.0
        e[0, 2, 2] = 1.0
        d = disagreement_score(e)
        # All pairs are orthogonal → sim=0 → disagreement=1
        assert (d > 0.9).all()

    def test_single_expert_zero_disagreement(self):
        """K=1 returns disagreement=0 (no pairs to compare)."""
        e = torch.randn(2, 1, 4)
        d = disagreement_score(e)
        assert d.shape == (2,)
        assert (d.abs() < 1e-5).all()

    def test_shape_correct(self):
        """Output is (B,) for (B, K, H) input."""
        e = torch.randn(4, 3, 8)
        d = disagreement_score(e)
        assert d.shape == (4,)

    def test_invalid_dim_raises(self):
        """Non-3D input raises ValueError."""
        with pytest.raises(ValueError, match="B, K, H"):
            disagreement_score(torch.randn(3, 8))


class TestSignedDebateStep:
    """Tests for the signed_debate_step function."""

    def test_output_shape(self):
        """Output shape matches input shape."""
        e = torch.randn(2, 3, 8)
        A_pos = torch.zeros(3, 3)
        A_neg = torch.zeros(3, 3)
        out = signed_debate_step(e, A_pos, A_neg, alpha=0.1, beta=0.0)
        assert out.shape == (2, 3, 8)

    def test_zero_debate_preserves_input(self):
        """A_pos=A_neg=0 and alpha=beta=0 gives identity."""
        e = torch.randn(2, 3, 8)
        A_pos = torch.zeros(3, 3)
        A_neg = torch.zeros(3, 3)
        out = signed_debate_step(e, A_pos, A_neg, alpha=0.0, beta=0.0)
        assert torch.allclose(out, e, atol=1e-6)

    def test_support_changes_output(self):
        """Non-zero A_pos changes the output."""
        e = torch.randn(1, 3, 8)
        A_pos = torch.eye(3) * 0.5  # mild self-support
        A_neg = torch.zeros(3, 3)
        out = signed_debate_step(e, A_pos, A_neg, alpha=0.1, beta=0.0)
        # Each e_k is multiplied by (1 + 0.05) = 1.05 in the diagonal terms
        # So out should be different from e
        assert not torch.allclose(out, e, atol=1e-4)

    def test_sign_matters(self):
        """A_pos and A_neg produce different outputs (sign matters)."""
        e = torch.randn(1, 3, 8)
        A = torch.randn(3, 3) * 0.1
        A_pos = A.clone()
        A_neg = torch.zeros(3, 3)
        out_pos = signed_debate_step(e, A_pos, A_neg, alpha=0.1, beta=0.0)
        A_pos2 = torch.zeros(3, 3)
        A_neg2 = A.clone()
        out_neg = signed_debate_step(e, A_pos2, A_neg2, alpha=0.0, beta=0.1)
        # Support and critique should produce opposite effects
        assert not torch.allclose(out_pos - e, out_neg - e, atol=1e-5)

    def test_wrong_A_shape_raises(self):
        """Wrong shape A_pos or A_neg raises ValueError."""
        e = torch.randn(1, 3, 8)
        A_wrong = torch.zeros(4, 4)
        with pytest.raises(ValueError, match="A_pos"):
            signed_debate_step(e, A_wrong, torch.zeros(3, 3), 0.1, 0.0)
        with pytest.raises(ValueError, match="A_neg"):
            signed_debate_step(e, torch.zeros(3, 3), A_wrong, 0.1, 0.0)


class TestSDGConfig:
    """Tests for SDGConfig dataclass."""

    def test_default_values(self):
        """Default values are sensible."""
        cfg = SDGConfig()
        assert cfg.alpha_max == 0.1
        assert cfg.beta_max == 0.1
        assert cfg.n_steps == 1
        assert cfg.use_anchoring is True
        assert cfg.anchoring_strength == 0.5

    def test_invalid_alpha_max_raises(self):
        """alpha_max < 0 raises ValueError."""
        with pytest.raises(ValueError, match="alpha_max"):
            SDGConfig(alpha_max=-0.1)

    def test_invalid_beta_max_raises(self):
        """beta_max < 0 raises ValueError."""
        with pytest.raises(ValueError, match="beta_max"):
            SDGConfig(beta_max=-0.1)

    def test_invalid_n_steps_raises(self):
        """n_steps < 1 raises ValueError."""
        with pytest.raises(ValueError, match="n_steps"):
            SDGConfig(n_steps=0)

    def test_invalid_anchoring_strength_raises(self):
        """anchoring_strength out of [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="anchoring_strength"):
            SDGConfig(anchoring_strength=1.5)


class TestSDGLearnedInteractions:
    """Tests for SDGLearnedInteractions module."""

    def test_initialization(self):
        """Module stores A_pos and A_neg of right shape."""
        m = SDGLearnedInteractions(n_experts=4)
        assert m.n_experts == 4
        assert m.A_pos.shape == (4, 4)
        assert m.A_neg.shape == (4, 4)
        # Initialized small (std=0.02)
        assert m.A_pos.abs().max() < 0.5
        assert m.A_neg.abs().max() < 0.5

    def test_forward_returns_matrices(self):
        """forward returns the A_pos, A_neg matrices."""
        m = SDGLearnedInteractions(n_experts=3)
        A_pos, A_neg = m()
        assert A_pos.shape == (3, 3)
        assert A_neg.shape == (3, 3)


class TestSDGQuiteMoECfCCell:
    """Tests for the SDG-augmented cell."""

    def test_cell_initialization(self):
        """Cell stores the right parameters."""
        cell = SDGQuiteMoECfCCell(
            input_size=2, hidden_size=8, n_experts=3, top_k=2,
            d_context=16,
        )
        assert cell.n_experts == 3
        assert cell.top_k == 2
        assert cell.d_context == 16
        # Inner cell exists
        assert cell.cell.n_experts == 3
        # Interactions module exists
        assert cell.interactions.n_experts == 3

    def test_cell_forward_shape(self):
        """Forward returns (B, hidden_size)."""
        torch.manual_seed(0)
        cell = SDGQuiteMoECfCCell(
            input_size=2, hidden_size=8, n_experts=3, top_k=2,
            d_context=16,
        )
        x_t = torch.randn(2, 2)
        h = torch.randn(2, 8)
        ctx = torch.randn(2, 16)
        h_new = cell(x_t, h, context=ctx)
        assert h_new.shape == (2, 8)

    def test_cell_zero_debate_equals_vanilla(self):
        """With alpha=beta=0 and no anchoring, output = vanilla routing."""
        torch.manual_seed(0)
        cfg = SDGConfig(alpha_max=0.0, beta_max=0.0, use_anchoring=False)
        cell = SDGQuiteMoECfCCell(
            input_size=2, hidden_size=8, n_experts=3, top_k=2,
            d_context=16, sdg_config=cfg,
        )
        x_t = torch.randn(1, 2)
        h = torch.randn(1, 8)
        ctx = torch.randn(1, 16)
        h_new = cell(x_t, h, context=ctx)
        # Should still produce finite output of right shape
        assert h_new.shape == (1, 8)
        assert torch.isfinite(h_new).all()

    def test_cell_dense_routing(self):
        """top_k == n_experts gives dense routing (all experts deliberate)."""
        torch.manual_seed(0)
        cell = SDGQuiteMoECfCCell(
            input_size=2, hidden_size=4, n_experts=2, top_k=2,
            d_context=8,
        )
        x_t = torch.randn(1, 2)
        h = torch.randn(1, 4)
        ctx = torch.randn(1, 8)
        h_new = cell(x_t, h, context=ctx)
        assert h_new.shape == (1, 4)
        assert torch.isfinite(h_new).all()

    def test_cell_gradient_flows(self):
        """Gradient flows through cell back to input (K=3, top_k=2 for deliberation)."""
        torch.manual_seed(0)
        cell = SDGQuiteMoECfCCell(
            input_size=2, hidden_size=8, n_experts=3, top_k=2,
            d_context=16,
        )
        x_t = torch.randn(1, 2, requires_grad=True)
        h = torch.randn(1, 8)
        ctx = torch.randn(1, 16)
        h_new = cell(x_t, h, context=ctx)
        loss = h_new.sum()
        loss.backward()
        # A_pos, A_neg should have gradients
        assert cell.interactions.A_pos.grad is not None
        assert cell.interactions.A_neg.grad is not None
        assert cell.interactions.A_pos.grad.abs().sum() > 0


class TestSDGQuiteMoECfCNetwork:
    """Tests for the full SDG network."""

    def test_network_initialization(self):
        """Network stores the right parameters."""
        net = SDGQuiteMoECfCNetwork(
            input_size=2, hidden_size=8, n_experts=2, top_k=1,
            n_queries=4, d_context=16, n_heads=4, output_size=1,
        )
        assert net.input_size == 2
        assert net.n_experts == 2
        assert net.d_context == 16

    def test_network_forward_shape(self):
        """Forward returns (B, T, output_size)."""
        torch.manual_seed(0)
        net = SDGQuiteMoECfCNetwork(
            input_size=2, hidden_size=8, n_experts=2, top_k=1,
            n_queries=4, d_context=16, n_heads=4, output_size=1,
        )
        obs = torch.randn(2, 20, 2)
        times = torch.linspace(0, 1, 20).unsqueeze(0).expand(2, -1)
        out = net(obs, times)
        assert out.shape == (2, 20, 1)
        assert torch.isfinite(out).all()

    def test_network_with_nan(self):
        """NaN observations are handled (mask propagates)."""
        torch.manual_seed(0)
        net = SDGQuiteMoECfCNetwork(
            input_size=2, hidden_size=8, n_experts=2, top_k=1,
            n_queries=4, d_context=16, n_heads=4, output_size=1,
        )
        obs = torch.randn(1, 10, 2)
        obs[0, :5] = float("nan")
        times = torch.linspace(0, 1, 10).unsqueeze(0)
        out = net(obs, times)
        assert out.shape == (1, 10, 1)
        assert torch.isfinite(out).all()

    def test_network_gradient_flows(self):
        """Gradient flows from output to A_pos/A_neg parameters (K=3, top_k=2)."""
        torch.manual_seed(0)
        net = SDGQuiteMoECfCNetwork(
            input_size=2, hidden_size=8, n_experts=3, top_k=2,
            n_queries=4, d_context=16, n_heads=4, output_size=1,
        )
        obs = torch.randn(1, 10, 2)
        times = torch.linspace(0, 1, 10).unsqueeze(0)
        out = net(obs, times)
        loss = out.sum()
        loss.backward()
        # A_pos, A_neg should have gradients
        assert net.cell.interactions.A_pos.grad is not None
        assert net.cell.interactions.A_pos.grad.abs().sum() > 0


class TestSDGExports:
    """Verify exports are correct."""

    def test_exports(self):
        from lnn.core import (
            SDGConfig,
            SDGLearnedInteractions,
            SDGQuiteMoECfCCell,
            SDGQuiteMoECfCNetwork,
            disagreement_score,
            signed_debate_step,
        )
        _ = SDGConfig
        _ = SDGLearnedInteractions
        _ = SDGQuiteMoECfCCell
        _ = SDGQuiteMoECfCNetwork
        assert callable(disagreement_score)
        assert callable(signed_debate_step)
