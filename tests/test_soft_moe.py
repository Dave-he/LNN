"""Tests for round 107 Soft MoE routing (PRD #10-69)."""
from __future__ import annotations

import math

import pytest
import torch

from lnn.core.soft_moe import (
    SoftMoEConfig,
    SoftMoERouter,
    SoftMoECfCCell,
    SoftMoESETAMoECfCCell,
    SoftMoESETAMoECfCNetwork,
)
from lnn.core.seta_moe import SETAConfig


# ----------------------------------------------------------------------
# TestSoftMoEConfig
# ----------------------------------------------------------------------
class TestSoftMoEConfig:
    def test_defaults(self):
        cfg = SoftMoEConfig()
        assert cfg.n_experts == 4
        assert cfg.d_slot == 16
        assert cfg.normalize is False

    def test_custom(self):
        cfg = SoftMoEConfig(n_experts=8, d_slot=32, normalize=True)
        assert cfg.n_experts == 8
        assert cfg.d_slot == 32
        assert cfg.normalize is True


# ----------------------------------------------------------------------
# TestSoftMoERouter
# ----------------------------------------------------------------------
class TestSoftMoERouter:
    def test_dispatch_shape(self):
        router = SoftMoERouter(input_size=4, hidden_size=8, n_experts=3, d_slot=8)
        x = torch.randn(2, 5, 4)
        out = router(x)
        assert out.shape == (2, 5, 8)

    def test_dispatch_weights_sum_to_one(self):
        router = SoftMoERouter(input_size=4, hidden_size=8, n_experts=3, d_slot=8)
        x = torch.randn(2, 5, 4)
        router(x)
        # last_combine_weights are the per-token, per-expert soft scores
        scores = router.last_combine_weights
        # Sum over experts (last dim) should be 1 for each token
        assert torch.allclose(
            scores.sum(dim=-1), torch.ones(2, 5), atol=1e-5,
        )

    def test_all_experts_receive_signal(self):
        """H=0 lock-in is structurally impossible — every expert sees
        a weighted combination of all tokens."""
        router = SoftMoERouter(input_size=4, hidden_size=8, n_experts=3, d_slot=8)
        x = torch.randn(2, 5, 4)
        router(x)
        # last_dispatch_weights is (B, K, D), each row is the weighted
        # average of all tokens for that expert. None should be all-zero
        # unless ALL inputs are zero.
        for k in range(3):
            assert router.last_dispatch_weights[:, k, :].abs().sum() > 0

    def test_permutation_invariance(self):
        """Swapping the order of slots should swap the order of
        expert outputs but not change the final combined output —
        provided the shared ``phi`` projection is also copied."""
        torch.manual_seed(0)
        router1 = SoftMoERouter(input_size=4, hidden_size=8, n_experts=3, d_slot=8)
        # Build router2 with same phi weights, but permuted slots+experts
        router2 = SoftMoERouter(input_size=4, hidden_size=8, n_experts=3, d_slot=8)
        with torch.no_grad():
            perm = [2, 0, 1]
            router2.phi.weight.data = router1.phi.weight.data
            router2.slots.data = router1.slots.data[perm]
            for new_k, old_k in enumerate(perm):
                router2.experts[new_k].weight.data = router1.experts[old_k].weight.data
                router2.experts[new_k].bias.data = router1.experts[old_k].bias.data
        x = torch.randn(2, 5, 4)
        out1 = router1(x)
        out2 = router2(x)
        # The combined output should be identical (just permuted inside)
        assert torch.allclose(out1, out2, atol=1e-5)

    def test_gradient_flows(self):
        router = SoftMoERouter(input_size=4, hidden_size=8, n_experts=3, d_slot=8)
        x = torch.randn(2, 5, 4)
        out = router(x)
        loss = out.sum()
        loss.backward()
        # Gradients flow to slots and expert params
        assert router.slots.grad is not None
        assert router.slots.grad.abs().sum() > 0
        for expert in router.experts:
            assert expert.weight.grad is not None
            assert expert.weight.grad.abs().sum() > 0
        # And to phi
        assert router.phi.weight.grad is not None

    def test_nan_aware(self):
        router = SoftMoERouter(input_size=4, hidden_size=8, n_experts=3, d_slot=8)
        x = torch.randn(2, 5, 4)
        x[0, 2, 0] = float("nan")
        # Forward should not produce NaN
        out = router(x)
        assert not torch.isnan(out).any()

    def test_normalize_routing(self):
        """Cosine-similarity routing should also work."""
        router = SoftMoERouter(
            input_size=4, hidden_size=8, n_experts=3, d_slot=8, normalize=True,
        )
        x = torch.randn(2, 5, 4)
        out = router(x)
        assert out.shape == (2, 5, 8)
        assert not torch.isnan(out).any()

    def test_get_utilization(self):
        router = SoftMoERouter(input_size=4, hidden_size=8, n_experts=3, d_slot=8)
        x = torch.randn(2, 5, 4)
        router(x)
        util = router.get_utilization()
        assert "expert_norms" in util
        assert "expert_norm_std" in util
        assert "expert_norm_max_min_ratio" in util
        assert len(util["expert_norms"]) == 3


# ----------------------------------------------------------------------
# TestSoftMoECfCCell
# ----------------------------------------------------------------------
class TestSoftMoECfCCell:
    def test_forward_shape(self):
        cell = SoftMoECfCCell(
            input_size=2, hidden_size=8, n_experts=3, d_slot=8,
        )
        x = torch.randn(2, 5, 2)
        h = torch.randn(2, 8)
        out = cell(x, h, dt=1.0)
        assert out.shape == (2, 5, 8)

    def test_nan_aware(self):
        cell = SoftMoECfCCell(
            input_size=2, hidden_size=8, n_experts=3, d_slot=8,
        )
        x = torch.randn(2, 5, 2)
        x[0, 2, 0] = float("nan")
        h = torch.randn(2, 8)
        out = cell(x, h, dt=1.0)
        assert not torch.isnan(out).any()

    def test_router_utilization_recorded(self):
        cell = SoftMoECfCCell(
            input_size=2, hidden_size=8, n_experts=3, d_slot=8,
        )
        x = torch.randn(2, 5, 2)
        h = torch.randn(2, 8)
        cell(x, h, dt=1.0)
        assert "expert_norms" in cell.last_router_util


# ----------------------------------------------------------------------
# TestSoftMoESETAMoECfCCell
# ----------------------------------------------------------------------
class TestSoftMoESETAMoECfCCell:
    def test_forward_shape(self):
        cfg = SETAConfig(n_shared=2, n_unique=3, top_k=2)
        cell = SoftMoESETAMoECfCCell(
            input_size=2, hidden_size=8, sdta_config=cfg, d_context=4,
            d_slot=8,
        )
        x = torch.randn(3, 2)
        h = torch.randn(3, 8)
        out = cell(x, h, context=torch.randn(3, 4))
        assert out.shape == (3, 8)

    def test_router_is_softmoe(self):
        cfg = SETAConfig(n_shared=2, n_unique=3, top_k=2)
        cell = SoftMoESETAMoECfCCell(
            input_size=2, hidden_size=8, sdta_config=cfg, d_context=4,
            d_slot=8,
        )
        from lnn.core.soft_moe import SoftMoESETARouter
        assert isinstance(cell.router, SoftMoESETARouter)

    def test_utilization_includes_softmoe(self):
        cfg = SETAConfig(n_shared=2, n_unique=3, top_k=2)
        cell = SoftMoESETAMoECfCCell(
            input_size=2, hidden_size=8, sdta_config=cfg, d_context=4,
            d_slot=8,
        )
        cell(torch.randn(3, 2), torch.randn(3, 8), context=None)
        util = cell.collect_expert_utilization()
        assert "softmoe_expert_norm_std" in util
        assert "softmoe_expert_norm_max_min_ratio" in util

    def test_shared_always_active(self):
        cfg = SETAConfig(n_shared=2, n_unique=3, top_k=2)
        cell = SoftMoESETAMoECfCCell(
            input_size=2, hidden_size=8, sdta_config=cfg, d_context=4,
            d_slot=8,
        )
        cell(torch.randn(3, 2), torch.randn(3, 8), context=None)
        # SETA's structural fix: shared experts produce a (B, S, H) stack
        assert cell.last_shared_stack.shape == (3, 2, 8)


# ----------------------------------------------------------------------
# TestSoftMoESETAMoECfCNetwork
# ----------------------------------------------------------------------
class TestSoftMoESETAMoECfCNetwork:
    def test_forward(self):
        cfg = SETAConfig(n_shared=2, n_unique=3, top_k=2)
        net = SoftMoESETAMoECfCNetwork(
            input_size=2, hidden_size=8, sdta_config=cfg,
            n_queries=2, d_context=4, n_heads=2, output_size=2,
            d_slot=8,
        )
        obs = torch.randn(2, 5, 2)
        times = torch.linspace(0, 1, 5).unsqueeze(0).expand(2, -1)
        out = net(obs, times)
        assert out.shape == (2, 5, 2)

    def test_nan_aware_mask(self):
        cfg = SETAConfig(n_shared=2, n_unique=3, top_k=2)
        net = SoftMoESETAMoECfCNetwork(
            input_size=2, hidden_size=8, sdta_config=cfg,
            n_queries=2, d_context=4, n_heads=2, output_size=2,
            d_slot=8,
        )
        obs = torch.randn(2, 5, 2)
        obs[0, 2, 0] = float("nan")
        times = torch.linspace(0, 1, 5).unsqueeze(0).expand(2, -1)
        out = net(obs, times)
        assert not torch.isnan(out).any()

    def test_get_utilization(self):
        cfg = SETAConfig(n_shared=2, n_unique=3, top_k=2)
        net = SoftMoESETAMoECfCNetwork(
            input_size=2, hidden_size=8, sdta_config=cfg,
            n_queries=2, d_context=4, n_heads=2, output_size=2,
            d_slot=8,
        )
        obs = torch.randn(2, 5, 2)
        times = torch.linspace(0, 1, 5).unsqueeze(0).expand(2, -1)
        net(obs, times)
        util = net.get_utilization()
        assert "softmoe_expert_norm_std" in util
        assert "shared_entropy" in util


# ----------------------------------------------------------------------
# TestExports
# ----------------------------------------------------------------------
class TestSoftMoEExports:
    def test_all_exports(self):
        from lnn.core import (
            SoftMoEConfig,
            SoftMoERouter,
            SoftMoECfCCell,
            SoftMoESETAMoECfCCell,
            SoftMoESETAMoECfCNetwork,
        )
        assert SoftMoEConfig is not None
        assert SoftMoERouter is not None
        assert SoftMoECfCCell is not None
        assert SoftMoESETAMoECfCCell is not None
        assert SoftMoESETAMoECfCNetwork is not None
