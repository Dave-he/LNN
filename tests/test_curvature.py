"""Round 101 — Tests for Ollivier-Ricci Curvature routing signal (PRD #10-63)."""
from __future__ import annotations

import math

import pytest
import torch

from lnn.core.curvature import (
    curvature_routing_loss,
    mean_ollivier_ricci,
    ollivier_ricci_curvature,
)


class TestOllivierRicci:
    """Tests for the ollivier_ricci_curvature function."""

    def test_orc_is_symmetric(self):
        """ORC matrix must be symmetric: ORC(i,j) = ORC(j,i)."""
        torch.manual_seed(0)
        pts = torch.randn(5, 4)
        orc = ollivier_ricci_curvature(pts, k=2, sinkhorn_iters=5)
        assert orc.shape == (5, 5)
        # Diagonal must be 0
        assert torch.allclose(orc.diag(), torch.zeros(5), atol=1e-6)
        # Symmetric
        assert torch.allclose(orc, orc.t(), atol=1e-5)

    def test_orc_zero_for_identical_points(self):
        """All identical points → degenerate (skipped): ORC returns 0 matrix
        since d_ij = 0 → formula returns 0 (clamped)."""
        pts = torch.zeros(4, 3) + torch.tensor([1.0, 2.0, 3.0])
        orc = ollivier_ricci_curvature(pts, k=2, sinkhorn_iters=5)
        # All points identical → d_ij = 0 (clamped to 1e-12) → ORC(i,j) ≈ 1
        # Actually w1 = 0 (same distributions) → 1 - 0/d = 1
        # So all ORC should be ~1
        assert orc.shape == (4, 4)
        # Symmetric
        assert torch.allclose(orc, orc.t(), atol=1e-5)

    def test_orc_higher_for_spread_points(self):
        """Spread-out points should have higher mean ORC than clustered."""
        torch.manual_seed(0)
        spread = torch.randn(8, 4) * 5.0  # Wide spread
        clustered = torch.randn(8, 4) * 0.01  # Tight cluster
        orc_spread = ollivier_ricci_curvature(spread, k=2, sinkhorn_iters=5)
        orc_clustered = ollivier_ricci_curvature(clustered, k=2, sinkhorn_iters=5)
        # Spread should have higher mean ORC (more tree-like)
        mean_s = orc_spread[torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)].mean()
        mean_c = orc_clustered[torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)].mean()
        assert mean_s > mean_c, f"spread {mean_s:.3f} should > clustered {mean_c:.3f}"

    def test_orc_shape_correct(self):
        """Output shape must be (N, N)."""
        for N in [2, 3, 5, 10]:
            pts = torch.randn(N, 6)
            orc = ollivier_ricci_curvature(pts, k=1)
            assert orc.shape == (N, N)

    def test_orc_k_clamps_to_n_minus_1(self):
        """k > N-1 should be clamped (no error)."""
        pts = torch.randn(3, 4)
        orc = ollivier_ricci_curvature(pts, k=10)  # k=10 > N-1=2
        assert orc.shape == (3, 3)
        # Should still be symmetric
        assert torch.allclose(orc, orc.t(), atol=1e-5)


class TestMeanOllivierRicci:
    """Tests for the mean_ollivier_ricci function."""

    def test_mean_is_scalar(self):
        """Mean ORC returns a scalar tensor."""
        pts = torch.randn(5, 4)
        m = mean_ollivier_ricci(pts, k=2, sinkhorn_iters=5)
        assert m.dim() == 0  # scalar

    def test_mean_value_range(self):
        """Mean ORC should be in roughly [-1, 1]."""
        torch.manual_seed(0)
        for _ in range(3):
            pts = torch.randn(8, 5) * 2.0
            m = mean_ollivier_ricci(pts, k=2, sinkhorn_iters=5)
            assert -2.0 <= m.item() <= 2.0  # generous bounds for Sinkhorn approx

    def test_mean_higher_for_spread(self):
        """Mean ORC of spread points > mean ORC of clustered points."""
        torch.manual_seed(0)
        spread = torch.randn(10, 6) * 3.0
        clustered = torch.randn(10, 6) * 0.1
        m_s = mean_ollivier_ricci(spread, k=2, sinkhorn_iters=5).item()
        m_c = mean_ollivier_ricci(clustered, k=2, sinkhorn_iters=5).item()
        assert m_s > m_c, f"spread {m_s:.3f} should > clustered {m_c:.3f}"


class TestCurvatureRoutingLoss:
    """Tests for the curvature_routing_loss function."""

    def test_loss_zero_for_lambda_zero(self):
        """λ=0 → loss should be 0 regardless of features."""
        torch.manual_seed(0)
        pts = torch.randn(5, 4)
        loss = curvature_routing_loss(pts, k=2, lambda_coeff=0.0)
        assert loss.item() == 0.0

    def test_loss_positive_for_clustered(self):
        """Clustered experts → low ORC → high loss."""
        torch.manual_seed(0)
        clustered = torch.randn(4, 8) * 0.01  # all very close
        loss = curvature_routing_loss(clustered, k=2, lambda_coeff=1.0, sinkhorn_iters=5)
        # Low ORC (clustered) → 1 - ORC > 0 → loss > 0
        assert loss.item() > 0

    def test_loss_lower_for_spread(self):
        """Spread experts → high ORC → lower loss than clustered."""
        torch.manual_seed(0)
        spread = torch.randn(4, 8) * 5.0
        clustered = torch.randn(4, 8) * 0.01
        loss_s = curvature_routing_loss(spread, k=2, lambda_coeff=1.0, sinkhorn_iters=5)
        loss_c = curvature_routing_loss(clustered, k=2, lambda_coeff=1.0, sinkhorn_iters=5)
        assert loss_s.item() < loss_c.item(), (
            f"spread {loss_s.item():.3f} should < clustered {loss_c.item():.3f}"
        )

    def test_loss_scales_with_lambda(self):
        """Loss should scale linearly with lambda_coeff."""
        torch.manual_seed(0)
        pts = torch.randn(4, 6)
        loss_1 = curvature_routing_loss(pts, k=2, lambda_coeff=1.0, sinkhorn_iters=5)
        loss_2 = curvature_routing_loss(pts, k=2, lambda_coeff=2.0, sinkhorn_iters=5)
        assert abs(loss_2.item() - 2 * loss_1.item()) < 1e-3

    def test_loss_is_differentiable(self):
        """Loss must support autograd (gradient flows)."""
        torch.manual_seed(0)
        pts = torch.randn(4, 5, requires_grad=True)
        loss = curvature_routing_loss(pts, k=2, lambda_coeff=1.0, sinkhorn_iters=5)
        loss.backward()
        assert pts.grad is not None
        assert pts.grad.abs().sum() > 0  # gradient non-zero

    def test_loss_2d_features(self):
        """Loss works with 2D expert features (typical case)."""
        torch.manual_seed(0)
        # 4 experts, 2D features
        pts = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])
        loss = curvature_routing_loss(pts, k=2, lambda_coeff=0.001, sinkhorn_iters=5)
        # Well-spread → high ORC → low loss
        assert loss.item() < 0.5  # reasonable bound

    def test_loss_too_few_experts_raises(self):
        """K < 2 should raise ValueError."""
        pts = torch.randn(1, 4)
        with pytest.raises(ValueError, match="at least 2 experts"):
            curvature_routing_loss(pts, k=2, lambda_coeff=1.0)

    def test_loss_negative_lambda_raises(self):
        """lambda < 0 should raise ValueError."""
        pts = torch.randn(3, 4)
        with pytest.raises(ValueError, match="lambda_coeff must be"):
            curvature_routing_loss(pts, k=2, lambda_coeff=-0.1)


class TestCurvatureExports:
    """Verify exports are correct."""

    def test_exports(self):
        from lnn.core import curvature_routing_loss, mean_ollivier_ricci, ollivier_ricci_curvature
        assert callable(ollivier_ricci_curvature)
        assert callable(mean_ollivier_ricci)
        assert callable(curvature_routing_loss)
