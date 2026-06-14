"""Tests for round 100 soft nearest neighbor loss (PRD #10-62)."""
from __future__ import annotations

import pytest
import torch

from lnn.core.snnl import (
    expert_snnl_loss,
    soft_nearest_neighbor_loss,
)


# ---------------------------------------------------------------------------
# soft_nearest_neighbor_loss
# ---------------------------------------------------------------------------

class TestSNNL:
    def test_zero_for_perfectly_clustered(self) -> None:
        """Features cluster perfectly by label → loss ≈ 0."""
        # Two clusters, perfectly separated
        f1 = torch.tensor([[0.0, 0.0], [0.1, 0.1], [-0.1, -0.1]])  # class 0
        f2 = torch.tensor([[10.0, 10.0], [10.1, 10.1], [10.0, 9.9]])  # class 1
        features = torch.cat([f1, f2], dim=0)
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        loss = soft_nearest_neighbor_loss(features, labels, temperature=0.5)
        # Same-class pairs are MUCH closer than different-class pairs
        # → probability ≈ 1 → log ≈ 0 → loss ≈ 0
        assert loss.item() < 0.1

    def test_high_for_randomly_mixed(self) -> None:
        """Random features with no clustering → loss > 0."""
        torch.manual_seed(0)
        features = torch.randn(8, 4)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        loss = soft_nearest_neighbor_loss(features, labels, temperature=1.0)
        # Some same-class pairs will be far apart → loss > 0
        assert loss.item() > 0.0

    def test_temperature_scaling(self) -> None:
        """Lower T → sharper distribution → lower loss for clustered data."""
        f1 = torch.tensor([[0.0, 0.0], [0.1, 0.1]])
        f2 = torch.tensor([[5.0, 5.0], [5.1, 5.1]])
        features = torch.cat([f1, f2], dim=0)
        labels = torch.tensor([0, 0, 1, 1])
        loss_low_T = soft_nearest_neighbor_loss(features, labels, temperature=0.1)
        loss_high_T = soft_nearest_neighbor_loss(features, labels, temperature=10.0)
        # Lower T → tighter distribution → better clustering → lower loss
        assert loss_low_T.item() < loss_high_T.item()

    def test_single_class_returns_zero(self) -> None:
        """All-same-class batch → no positive pairs → return 0."""
        features = torch.randn(4, 3)
        labels = torch.tensor([0, 0, 0, 0])
        loss = soft_nearest_neighbor_loss(features, labels)
        assert loss.item() == 0.0

    def test_gradient_flows(self) -> None:
        """SNNL is differentiable w.r.t. features."""
        features = torch.randn(4, 3, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1])
        loss = soft_nearest_neighbor_loss(features, labels)
        loss.backward()
        assert features.grad is not None
        assert features.grad.abs().sum().item() > 0

    def test_rejects_zero_temperature(self) -> None:
        """temperature <= 0 is rejected."""
        features = torch.randn(4, 3)
        labels = torch.tensor([0, 0, 1, 1])
        with pytest.raises(ValueError):
            soft_nearest_neighbor_loss(features, labels, temperature=0.0)
        with pytest.raises(ValueError):
            soft_nearest_neighbor_loss(features, labels, temperature=-0.5)

    def test_handles_1d_features(self) -> None:
        """1D features are reshaped internally."""
        features = torch.tensor([0.0, 0.1, 5.0, 5.1])
        labels = torch.tensor([0, 0, 1, 1])
        loss = soft_nearest_neighbor_loss(features, labels, temperature=0.5)
        assert loss.item() < 0.1

    def test_handles_empty_batch(self) -> None:
        """B < 2 → return 0."""
        features = torch.randn(1, 3)
        labels = torch.tensor([0])
        loss = soft_nearest_neighbor_loss(features, labels)
        assert loss.item() == 0.0

    def test_handles_perfectly_separated(self) -> None:
        """Perfectly separated features → loss is small."""
        features = torch.tensor([[0.0, 0.0], [0.0, 0.0], [100.0, 100.0], [100.0, 100.0]])
        labels = torch.tensor([0, 0, 1, 1])
        loss = soft_nearest_neighbor_loss(features, labels, temperature=1.0)
        assert loss.item() < 0.01

    def test_no_positive_pairs(self) -> None:
        """All different labels → no positive pairs → return 0."""
        features = torch.randn(4, 3)
        labels = torch.tensor([0, 1, 2, 3])
        loss = soft_nearest_neighbor_loss(features, labels)
        assert loss.item() == 0.0


# ---------------------------------------------------------------------------
# expert_snnl_loss
# ---------------------------------------------------------------------------

class TestExpertSNNL:
    def test_basic(self) -> None:
        """Expert SNNL works on (K, d) expert features."""
        K, d = 4, 8
        torch.manual_seed(0)
        # 4 experts, each with a feature vector
        features = torch.randn(K, d)
        routing = torch.arange(K)  # each expert handles its own class
        loss = expert_snnl_loss(features, routing, temperature=1.0)
        # No same-class pairs (all labels different) → 0
        assert loss.item() == 0.0

    def test_with_clusters(self) -> None:
        """Expert SNNL with multiple experts per class → meaningful loss."""
        K, d = 6, 4
        features = torch.tensor([
            [0.0, 0.0], [0.1, 0.1],  # class 0
            [5.0, 5.0], [5.1, 5.1],  # class 0
            [10.0, 10.0], [10.1, 10.1],  # class 1
        ])
        routing = torch.tensor([0, 0, 0, 0, 1, 1])
        loss = expert_snnl_loss(features, routing, temperature=0.5)
        # Class 0 is well-clustered, class 1 is well-clustered → loss ≈ 0
        assert loss.item() < 0.1


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

class TestSNNLExports:
    def test_soft_nearest_neighbor_loss_exported(self) -> None:
        from lnn.core import soft_nearest_neighbor_loss as fn
        assert fn is soft_nearest_neighbor_loss

    def test_expert_snnl_loss_exported(self) -> None:
        from lnn.core import expert_snnl_loss as fn
        assert fn is expert_snnl_loss

    def test_in_all_list(self) -> None:
        import lnn.core as core
        assert "soft_nearest_neighbor_loss" in core.__all__
        assert "expert_snnl_loss" in core.__all__
