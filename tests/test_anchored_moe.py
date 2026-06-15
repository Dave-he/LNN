"""Tests for round 108 Anchored MoE (PRD #10-70)."""
from __future__ import annotations

import pytest
import torch

from lnn.core.anchored_moe import (
    AnchoredMoEConfig,
    AnchoredMoECfCCell,
    AnchoredMoECfCNetwork,
    AnchoredRouter,
    RegimePredictor,
    StructuralPrior,
)


# ----------------------------------------------------------------------
# TestAnchoredMoEConfig
# ----------------------------------------------------------------------
class TestAnchoredMoEConfig:
    def test_defaults(self):
        cfg = AnchoredMoEConfig()
        assert cfg.n_experts == 4
        assert cfg.top_k == 2
        assert cfg.anchor_mode == "logit"
        assert cfg.anchor_alpha == 0.5
        assert cfg.anchor_lambda == 0.1

    def test_custom(self):
        cfg = AnchoredMoEConfig(
            n_experts=8, top_k=4, anchor_mode="mix", anchor_alpha=0.7,
        )
        assert cfg.n_experts == 8
        assert cfg.top_k == 4
        assert cfg.anchor_mode == "mix"
        assert cfg.anchor_alpha == 0.7


# ----------------------------------------------------------------------
# TestRegimePredictor
# ----------------------------------------------------------------------
class TestRegimePredictor:
    def test_shape(self):
        reg = RegimePredictor(input_size=2, d_hidden=8)
        x = torch.randn(2, 5, 2)
        d = reg(x)
        assert d.shape == (2, 4)

    def test_descriptors_in_unit_range(self):
        reg = RegimePredictor(input_size=2, d_hidden=8)
        x = torch.randn(4, 8, 2) * 5  # wide range
        d = reg(x)
        assert (d >= 0).all() and (d <= 1).all()

    def test_nan_aware(self):
        reg = RegimePredictor(input_size=2, d_hidden=8)
        x = torch.randn(2, 5, 2)
        x[0, 2, 0] = float("nan")
        d = reg(x)
        assert not torch.isnan(d).any()

    def test_gradient_flows(self):
        reg = RegimePredictor(input_size=2, d_hidden=8)
        x = torch.randn(2, 5, 2, requires_grad=True)
        d = reg(x)
        d.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


# ----------------------------------------------------------------------
# TestStructuralPrior
# ----------------------------------------------------------------------
class TestStructuralPrior:
    def test_shape(self):
        sp = StructuralPrior(descriptor_dim=4, n_experts=3)
        d = torch.randn(2, 4)
        p = sp(d)
        assert p.shape == (2, 3)

    def test_valid_probability(self):
        sp = StructuralPrior(descriptor_dim=4, n_experts=5)
        d = torch.randn(3, 4)
        p = sp(d)
        assert (p >= 0).all()
        assert torch.allclose(p.sum(dim=-1), torch.ones(3), atol=1e-5)

    def test_descriptors_change_prior(self):
        sp = StructuralPrior(descriptor_dim=4, n_experts=3, d_hidden=8)
        d1 = torch.zeros(1, 4)
        d2 = torch.ones(1, 4)
        p1 = sp(d1)
        p2 = sp(d2)
        # Different descriptors → different priors
        assert not torch.allclose(p1, p2, atol=1e-4)


# ----------------------------------------------------------------------
# TestAnchoredRouter
# ----------------------------------------------------------------------
class TestAnchoredRouter:
    def test_shape(self):
        r = AnchoredRouter(input_size=2, hidden_size=8, n_experts=3, top_k=2)
        x_t = torch.randn(2, 2)
        h = torch.randn(2, 8)
        w, idx = r(x_t, h)
        assert w.shape == (2, 3)
        assert idx.shape == (2, 2)

    def test_topk_indices_in_range(self):
        r = AnchoredRouter(input_size=2, hidden_size=8, n_experts=4, top_k=2)
        x_t = torch.randn(3, 2)
        h = torch.randn(3, 8)
        w, idx = r(x_t, h)
        assert (idx >= 0).all() and (idx < 4).all()

    def test_logit_anchoring(self):
        r = AnchoredRouter(input_size=2, hidden_size=8, n_experts=3, top_k=2, anchor_mode="logit")
        x_t = torch.randn(2, 2)
        h = torch.randn(2, 8)
        prior = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])
        w_anchored, _ = r(x_t, h, prior=prior)
        # The two priors have very different shapes; routing should
        # respond to the prior (one or two top-k slots should align)
        w_unanchored, _ = r(x_t, h, prior=None)
        # At least the *weight magnitude* should differ (anchored
        # adds log-prior which can change the softmax)
        assert not torch.allclose(w_anchored, w_unanchored, atol=1e-4)

    def test_mix_anchoring(self):
        r = AnchoredRouter(input_size=2, hidden_size=8, n_experts=3, top_k=2, anchor_mode="mix", anchor_alpha=0.5)
        x_t = torch.randn(2, 2)
        h = torch.randn(2, 8)
        prior = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]])
        w, _ = r(x_t, h, prior=prior)
        # alpha=0.5 → mixture; routing should be a blend
        assert w.shape == (2, 3)
        # The high-prior expert should get higher weight (but not 1.0)
        assert w[0, 0] > w[0, 1]
        assert w[1, 2] > w[1, 0]

    def test_kl_mode_loss_nonzero(self):
        r = AnchoredRouter(input_size=2, hidden_size=8, n_experts=3, top_k=2, anchor_mode="kl")
        x_t = torch.randn(2, 2)
        h = torch.randn(2, 8)
        prior = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])
        r(x_t, h, prior=prior)
        kl = r.get_kl_regularization(prior)
        assert kl.item() > 0


# ----------------------------------------------------------------------
# TestAnchoredMoECfCCell
# ----------------------------------------------------------------------
class TestAnchoredMoECfCCell:
    def test_forward_shape(self):
        cell = AnchoredMoECfCCell(input_size=2, hidden_size=8, n_experts=3, top_k=2)
        x = torch.randn(2, 5, 2)
        h = torch.randn(2, 8)
        out = cell(x, h)
        assert out.shape == (2, 8)

    def test_nan_aware(self):
        cell = AnchoredMoECfCCell(input_size=2, hidden_size=8, n_experts=3, top_k=2)
        x = torch.randn(2, 5, 2)
        x[0, 2, 0] = float("nan")
        h = torch.randn(2, 8)
        out = cell(x, h)
        assert not torch.isnan(out).any()

    def test_utilization_recorded(self):
        cell = AnchoredMoECfCCell(input_size=2, hidden_size=8, n_experts=3, top_k=2)
        x = torch.randn(2, 5, 2)
        h = torch.randn(2, 8)
        cell(x, h)
        util = cell.get_utilization()
        assert "routing_entropy" in util
        assert "routing_max_min_ratio" in util
        assert "descriptors_mean" in util
        assert "prior_entropy" in util
        assert len(util["expert_avg_weights"]) == 3

    def test_kl_regularization(self):
        cell = AnchoredMoECfCCell(
            input_size=2, hidden_size=8, n_experts=3, top_k=2,
            anchor_mode="kl", anchor_lambda=0.1,
        )
        x = torch.randn(2, 5, 2)
        h = torch.randn(2, 8)
        cell(x, h)
        reg = cell.get_regularization_loss()
        assert reg.item() > 0

    def test_logit_mode_zero_regularization(self):
        cell = AnchoredMoECfCCell(
            input_size=2, hidden_size=8, n_experts=3, top_k=2,
            anchor_mode="logit",
        )
        x = torch.randn(2, 5, 2)
        h = torch.randn(2, 8)
        cell(x, h)
        reg = cell.get_regularization_loss()
        assert reg.item() == 0.0

    def test_anchored_routing_differs_from_unanchored(self):
        torch.manual_seed(0)
        cell_a = AnchoredMoECfCCell(
            input_size=2, hidden_size=8, n_experts=3, top_k=2,
            anchor_mode="logit",
        )
        torch.manual_seed(0)
        cell_u = AnchoredMoECfCCell(
            input_size=2, hidden_size=8, n_experts=3, top_k=2,
            anchor_mode="none",  # no anchoring
        )
        x = torch.randn(4, 5, 2)
        h = torch.randn(4, 8)
        out_a = cell_a(x, h)
        out_u = cell_u(x, h)
        # Different initialization regimes — outputs should differ
        # (the seed=0 only gives same init, but anchor_mode affects forward differently)
        assert not torch.allclose(out_a, out_u, atol=1e-4)


# ----------------------------------------------------------------------
# TestAnchoredMoECfCNetwork
# ----------------------------------------------------------------------
class TestAnchoredMoECfCNetwork:
    def test_forward_shape(self):
        net = AnchoredMoECfCNetwork(input_size=2, hidden_size=8, n_experts=3, top_k=2, output_size=2)
        x = torch.randn(2, 5, 2)
        out = net(x)
        assert out.shape == (2, 5, 2)

    def test_nan_aware(self):
        net = AnchoredMoECfCNetwork(input_size=2, hidden_size=8, n_experts=3, top_k=2, output_size=2)
        x = torch.randn(2, 5, 2)
        x[0, 2, 0] = float("nan")
        out = net(x)
        assert not torch.isnan(out).any()

    def test_get_utilization(self):
        net = AnchoredMoECfCNetwork(input_size=2, hidden_size=8, n_experts=3, top_k=2, output_size=2)
        x = torch.randn(2, 5, 2)
        net(x)
        util = net.get_utilization()
        assert "routing_entropy" in util
        assert "expert_avg_weights" in util

    def test_gradient_flows(self):
        net = AnchoredMoECfCNetwork(input_size=2, hidden_size=8, n_experts=3, top_k=2, output_size=2)
        x = torch.randn(2, 5, 2, requires_grad=True)
        out = net(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


# ----------------------------------------------------------------------
# TestAnchoredMoEExports
# ----------------------------------------------------------------------
class TestAnchoredMoEExports:
    def test_all_exports(self):
        from lnn.core import (
            AnchoredMoEConfig,
            AnchoredMoECfCCell,
            AnchoredMoECfCNetwork,
            RegimePredictor,
            StructuralPrior,
            AnchoredRouter,
        )
        assert AnchoredMoEConfig is not None
        assert AnchoredMoECfCCell is not None
        assert AnchoredMoECfCNetwork is not None
        assert RegimePredictor is not None
        assert StructuralPrior is not None
        assert AnchoredRouter is not None
