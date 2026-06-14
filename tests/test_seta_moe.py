"""Tests for round 105 SETA sparse shared + unique experts (PRD #10-67)."""
from __future__ import annotations

import math

import pytest
import torch

from lnn.core.seta_moe import (
    SETAConfig,
    SETAMoECfCCell,
    SETAMoECfCNetwork,
    SETARouter,
    elastic_anchoring_loss,
    routing_regularization,
    snapshot_expert_weights,
    update_ema_anchors,
)
from lnn.core.cfc import CfCCell


# ----------------------------------------------------------------------
# TestSETAConfig
# ----------------------------------------------------------------------
class TestSETAConfig:
    def test_defaults(self):
        cfg = SETAConfig()
        assert cfg.n_shared == 2
        assert cfg.n_unique == 3
        assert cfg.top_k == 2

    def test_total_experts(self):
        cfg = SETAConfig(n_shared=2, n_unique=3, top_k=2)
        assert cfg.n_shared + cfg.n_unique == 5  # K=5 total

    def test_custom_target_entropy(self):
        cfg = SETAConfig(target_routing_entropy=0.5)
        assert cfg.target_routing_entropy == 0.5

    def test_ema_disabled(self):
        cfg = SETAConfig(use_ema_anchor=False)
        assert cfg.use_ema_anchor is False

    def test_invalid_top_k(self):
        with pytest.raises(ValueError):
            SETAConfig(n_shared=1, n_unique=2, top_k=3)  # top_k > n_unique


# ----------------------------------------------------------------------
# TestElasticAnchoringLoss
# ----------------------------------------------------------------------
class TestElasticAnchoringLoss:
    def test_zero_at_anchor(self):
        cfcs = [CfCCell(input_size=2, hidden_size=4) for _ in range(2)]
        shared = torch.nn.ModuleList(cfcs)
        snap = snapshot_expert_weights(shared)
        loss = elastic_anchoring_loss(shared, snap, lambda_val=1.0)
        assert float(loss.item()) < 1e-6

    def test_positive_when_diverged(self):
        shared = torch.nn.ModuleList(
            [CfCCell(input_size=2, hidden_size=4) for _ in range(2)],
        )
        snap = snapshot_expert_weights(shared)
        # Perturb
        for p in shared.parameters():
            p.data.add_(0.5)
        loss = elastic_anchoring_loss(shared, snap, lambda_val=1.0)
        assert float(loss.item()) > 0.0

    def test_lambda_scaling(self):
        shared = torch.nn.ModuleList(
            [CfCCell(input_size=2, hidden_size=4) for _ in range(2)],
        )
        snap = snapshot_expert_weights(shared)
        for p in shared.parameters():
            p.data.add_(0.1)
        l1 = elastic_anchoring_loss(shared, snap, lambda_val=1.0)
        l2 = elastic_anchoring_loss(shared, snap, lambda_val=2.0)
        assert abs(float(l2.item()) - 2.0 * float(l1.item())) < 1e-4

    def test_missing_key_uses_zero_anchor(self):
        shared = torch.nn.ModuleList(
            [CfCCell(input_size=2, hidden_size=4) for _ in range(1)],
        )
        # Empty anchor — should still produce a valid (large) loss
        loss = elastic_anchoring_loss(shared, {}, lambda_val=1.0)
        assert float(loss.item()) > 0.0


# ----------------------------------------------------------------------
# TestRoutingRegularization
# ----------------------------------------------------------------------
class TestRoutingRegularization:
    def test_zero_at_target(self):
        router = SETARouter(input_size=2, hidden_size=4, d_context=0, n_unique=4, top_k=2)
        # Force uniform top-k weights
        B = 3
        g = torch.full((B, 2), 0.5)
        router.last_g = g
        # target = log(2) — uniform over 2 is exactly log(2)
        loss = routing_regularization(router, math.log(2), lambda_val=1.0)
        assert abs(float(loss.item())) < 1e-4

    def test_positive_when_collapsed(self):
        router = SETARouter(input_size=2, hidden_size=4, d_context=0, n_unique=4, top_k=2)
        # 1-hot → entropy = 0
        router.last_g = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        loss = routing_regularization(router, math.log(2), lambda_val=1.0)
        # 1-hot has H=0, target=log(2)=0.693, so squared dev = 0.48
        assert float(loss.item()) > 0.3

    def test_lambda_scaling(self):
        router = SETARouter(input_size=2, hidden_size=4, d_context=0, n_unique=4, top_k=2)
        router.last_g = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        l1 = routing_regularization(router, math.log(2), lambda_val=1.0)
        l2 = routing_regularization(router, math.log(2), lambda_val=2.0)
        assert abs(float(l2.item()) - 2.0 * float(l1.item())) < 1e-4

    def test_no_last_g_returns_zero(self):
        router = SETARouter(input_size=2, hidden_size=4, d_context=0, n_unique=4, top_k=2)
        # No last_g yet
        loss = routing_regularization(router, 0.5, lambda_val=1.0)
        assert float(loss.item()) == 0.0


# ----------------------------------------------------------------------
# TestUpdateEMAAnchors
# ----------------------------------------------------------------------
class TestUpdateEMAAnchors:
    def test_initializes_with_current_weights(self):
        shared = torch.nn.ModuleList(
            [CfCCell(input_size=2, hidden_size=4) for _ in range(2)],
        )
        anchors = update_ema_anchors({}, shared, decay=0.99)
        assert len(anchors) > 0

    def test_ema_moves_slowly(self):
        shared = torch.nn.ModuleList(
            [CfCCell(input_size=2, hidden_size=4) for _ in range(1)],
        )
        snap = snapshot_expert_weights(shared)
        # Move weights by 0.1
        for p in shared.parameters():
            p.data.add_(0.1)
        anchors = update_ema_anchors(snap, shared, decay=0.99)
        # First key check
        for k in anchors:
            delta = (anchors[k] - snap[k]).abs().max().item()
            assert delta < 0.1  # decayed move, not full


# ----------------------------------------------------------------------
# TestSETARouter
# ----------------------------------------------------------------------
class TestSETARouter:
    def test_output_shape(self):
        router = SETARouter(input_size=3, hidden_size=8, d_context=4, n_unique=4, top_k=2)
        x = torch.randn(5, 3)
        h = torch.randn(5, 8)
        ctx = torch.randn(5, 4)
        g = router(x, h, context=ctx)
        assert g.shape == (5, 4)
        # Sparsity: exactly top_k non-zero per row
        n_active = (g > 1e-6).sum(dim=-1)
        assert (n_active == 2).all()

    def test_topk_indices_match_active(self):
        router = SETARouter(input_size=3, hidden_size=8, d_context=4, n_unique=4, top_k=2)
        g = router(torch.randn(2, 3), torch.randn(2, 8), context=torch.randn(2, 4))
        idx = router.last_top_idx
        assert idx.shape == (2, 2)
        # Active columns should match top_idx
        active = (g > 1e-6).nonzero(as_tuple=False)
        active_set = set(map(tuple, active.tolist()))
        idx_set = set()
        for b in range(2):
            for k in range(2):
                idx_set.add((b, int(idx[b, k].item())))
        assert active_set == idx_set

    def test_no_context(self):
        router = SETARouter(input_size=3, hidden_size=8, d_context=4, n_unique=4, top_k=2)
        g = router(torch.randn(2, 3), torch.randn(2, 8), context=None)
        assert g.shape == (2, 4)


# ----------------------------------------------------------------------
# TestSETAMoECfCCell
# ----------------------------------------------------------------------
class TestSETAMoECfCCell:
    def test_output_shape(self):
        cfg = SETAConfig(n_shared=2, n_unique=3, top_k=2)
        cell = SETAMoECfCCell(input_size=2, hidden_size=8, sdta_config=cfg, d_context=4)
        x = torch.randn(3, 2)
        h = torch.randn(3, 8)
        out = cell(x, h, context=torch.randn(3, 4))
        assert out.shape == (3, 8)

    def test_shared_always_active(self):
        cfg = SETAConfig(n_shared=2, n_unique=3, top_k=1)
        cell = SETAMoECfCCell(input_size=2, hidden_size=8, sdta_config=cfg, d_context=4)
        x = torch.randn(3, 2)
        h = torch.randn(3, 8)
        cell(x, h, context=None)
        # Shared stack should be (B, S, H) = (3, 2, 8)
        assert cell.last_shared_stack.shape == (3, 2, 8)

    def test_unique_sparse(self):
        cfg = SETAConfig(n_shared=1, n_unique=4, top_k=1)
        cell = SETAMoECfCCell(input_size=2, hidden_size=8, sdta_config=cfg, d_context=4)
        cell(torch.randn(3, 2), torch.randn(3, 8), context=None)
        # unique g should have exactly 1 non-zero per row
        n_active = (cell.last_g_unique > 1e-6).sum(dim=-1)
        assert (n_active == 1).all()

    def test_gradient_flows(self):
        cfg = SETAConfig(n_shared=2, n_unique=3, top_k=2, elastic_lambda=0.0, routing_lambda=0.0)
        cell = SETAMoECfCCell(input_size=2, hidden_size=8, sdta_config=cfg, d_context=4)
        x = torch.randn(3, 2)
        h = torch.randn(3, 8)
        out = cell(x, h, context=torch.randn(3, 4))
        loss = out.sum()
        loss.backward()
        # Shared experts should have non-zero grad
        for expert in cell.shared_experts:
            has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in expert.parameters())
            assert has_grad
        # Unique experts should have non-zero grad
        for expert in cell.unique_experts:
            has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in expert.parameters())
            assert has_grad

    def test_regularization_loss_lazy_init(self):
        cfg = SETAConfig(n_shared=2, n_unique=3, top_k=2, elastic_lambda=1e-3, routing_lambda=1e-2)
        cell = SETAMoECfCCell(input_size=2, hidden_size=8, sdta_config=cfg, d_context=4)
        # First call: should initialize anchors
        cell(torch.randn(3, 2), torch.randn(3, 8), context=None)
        loss1 = cell.regularization_loss()
        # Second call: should still work and EMA-update
        cell(torch.randn(3, 2), torch.randn(3, 8), context=None)
        loss2 = cell.regularization_loss()
        assert float(loss1.item()) >= 0.0
        assert float(loss2.item()) >= 0.0

    def test_utilization_shared_always_active(self):
        cfg = SETAConfig(n_shared=2, n_unique=3, top_k=1)
        cell = SETAMoECfCCell(input_size=2, hidden_size=8, sdta_config=cfg, d_context=4)
        cell(torch.randn(3, 2), torch.randn(3, 8), context=None)
        util = cell.collect_expert_utilization()
        assert util["shared_n_active"] == 2  # always active
        assert util["shared_entropy"] == pytest.approx(math.log(2), abs=1e-4)


# ----------------------------------------------------------------------
# TestSETAMoECfCNetwork
# ----------------------------------------------------------------------
class TestSETAMoECfCNetwork:
    def test_forward(self):
        cfg = SETAConfig(n_shared=2, n_unique=3, top_k=2)
        net = SETAMoECfCNetwork(
            input_size=2, hidden_size=8, sdta_config=cfg,
            n_queries=2, d_context=4, n_heads=2, output_size=2,
        )
        obs = torch.randn(2, 5, 2)
        times = torch.linspace(0, 1, 5).unsqueeze(0).expand(2, -1)
        out = net(obs, times)
        assert out.shape == (2, 5, 2)

    def test_nan_aware_mask(self):
        cfg = SETAConfig(n_shared=2, n_unique=3, top_k=2)
        net = SETAMoECfCNetwork(
            input_size=2, hidden_size=8, sdta_config=cfg,
            n_queries=2, d_context=4, n_heads=2, output_size=2,
        )
        obs = torch.randn(2, 5, 2)
        obs[0, 2, 0] = float("nan")
        times = torch.linspace(0, 1, 5).unsqueeze(0).expand(2, -1)
        out = net(obs, times)
        assert not torch.isnan(out).any()

    def test_gradient_flows(self):
        cfg = SETAConfig(n_shared=2, n_unique=3, top_k=2, elastic_lambda=0.0, routing_lambda=0.0)
        net = SETAMoECfCNetwork(
            input_size=2, hidden_size=8, sdta_config=cfg,
            n_queries=2, d_context=4, n_heads=2, output_size=2,
        )
        obs = torch.randn(2, 5, 2)
        times = torch.linspace(0, 1, 5).unsqueeze(0).expand(2, -1)
        out = net(obs, times)
        loss = out.sum()
        loss.backward()
        # QuITE params should have grad
        for p in net.quite.parameters():
            assert p.grad is not None
            assert p.grad.abs().sum() > 0

    def test_regularization_loss(self):
        cfg = SETAConfig(n_shared=2, n_unique=3, top_k=2, elastic_lambda=1e-3, routing_lambda=1e-2)
        net = SETAMoECfCNetwork(
            input_size=2, hidden_size=8, sdta_config=cfg,
            n_queries=2, d_context=4, n_heads=2, output_size=2,
        )
        obs = torch.randn(2, 5, 2)
        times = torch.linspace(0, 1, 5).unsqueeze(0).expand(2, -1)
        net(obs, times)
        reg = net.regularization_loss()
        assert float(reg.item()) >= 0.0


# ----------------------------------------------------------------------
# TestExports
# ----------------------------------------------------------------------
class TestSETAExports:
    def test_all_exports(self):
        from lnn.core import (
            SETAConfig,
            SETAMoECfCCell,
            SETAMoECfCNetwork,
            SETARouter,
            elastic_anchoring_loss,
            routing_regularization,
        )
        assert SETAConfig is not None
        assert SETAMoECfCCell is not None
        assert SETAMoECfCNetwork is not None
        assert SETARouter is not None
        assert elastic_anchoring_loss is not None
        assert routing_regularization is not None
