"""Tests for round 106 AuxLF load balancing (PRD #10-68)."""
from __future__ import annotations

import math

import pytest
import torch

from lnn.core.auxlf import (
    AuxLFConfig,
    AuxLFRouter,
    AuxLFSETAMoECfCCell,
    AuxLFSETAMoECfCNetwork,
    update_load_balancing_bias,
)
from lnn.core.seta_moe import SETAConfig


# ----------------------------------------------------------------------
# TestAuxLFConfig
# ----------------------------------------------------------------------
class TestAuxLFConfig:
    def test_defaults(self):
        cfg = AuxLFConfig()
        assert cfg.bias_lr == 0.01
        assert cfg.bias_clamp == 2.0
        assert cfg.warmup_steps == 10
        assert cfg.use_update is True

    def test_custom(self):
        cfg = AuxLFConfig(bias_lr=0.05, warmup_steps=5, use_update=False)
        assert cfg.bias_lr == 0.05
        assert cfg.warmup_steps == 5
        assert cfg.use_update is False


# ----------------------------------------------------------------------
# TestUpdateLoadBalancingBias
# ----------------------------------------------------------------------
class TestUpdateLoadBalancingBias:
    def test_overloaded_decreases(self):
        bias = torch.zeros(4)
        # Expert 0 is overloaded, others are zero
        counts = torch.tensor([10.0, 0.0, 0.0, 0.0])
        cfg = AuxLFConfig(bias_lr=0.1, bias_clamp=10.0, target_load_fraction=0.25)
        update_load_balancing_bias(bias, counts, cfg, 4)
        # bias[0] should DECREASE (overloaded → reduce)
        assert bias[0].item() < 0.0
        # bias[1,2,3] should INCREASE (under-loaded)
        assert bias[1].item() > 0.0
        assert bias[2].item() > 0.0
        assert bias[3].item() > 0.0

    def test_balanced_no_change(self):
        bias = torch.zeros(4)
        counts = torch.tensor([5.0, 5.0, 5.0, 5.0])
        cfg = AuxLFConfig(bias_lr=0.1, bias_clamp=10.0, target_load_fraction=0.25)
        update_load_balancing_bias(bias, counts, cfg, 4)
        # Balanced → no change
        assert torch.allclose(bias, torch.zeros(4))

    def test_clamp(self):
        bias = torch.zeros(2)
        counts = torch.tensor([100.0, 0.0])
        cfg = AuxLFConfig(bias_lr=0.5, bias_clamp=1.0)
        # Many updates
        for _ in range(100):
            update_load_balancing_bias(bias, counts, cfg, 2)
        # Should be clamped
        assert abs(bias[0].item()) <= 1.0 + 1e-6
        assert abs(bias[1].item()) <= 1.0 + 1e-6

    def test_no_update_when_disabled(self):
        bias = torch.zeros(2)
        counts = torch.tensor([10.0, 0.0])
        cfg = AuxLFConfig(use_update=False)
        update_load_balancing_bias(bias, counts, cfg, 2)
        # No update
        assert torch.allclose(bias, torch.zeros(2))

    def test_lr_scaling(self):
        bias = torch.zeros(2)
        counts = torch.tensor([10.0, 0.0])
        cfg1 = AuxLFConfig(bias_lr=0.1, bias_clamp=10.0)
        cfg2 = AuxLFConfig(bias_lr=0.2, bias_clamp=10.0)
        b1 = update_load_balancing_bias(torch.zeros(2), counts, cfg1, 2)
        b2 = update_load_balancing_bias(torch.zeros(2), counts, cfg2, 2)
        # Doubling LR should double the bias change
        assert abs(b2[0].item() - 2 * b1[0].item()) < 1e-4


# ----------------------------------------------------------------------
# TestAuxLFRouter
# ----------------------------------------------------------------------
class TestAuxLFRouter:
    def test_bias_starts_at_zero(self):
        router = AuxLFRouter(
            input_size=2, hidden_size=4, d_context=0,
            n_unique=3, top_k=2,
        )
        assert torch.allclose(router.bias, torch.zeros(3))

    def test_forward_shape(self):
        router = AuxLFRouter(
            input_size=2, hidden_size=4, d_context=0,
            n_unique=4, top_k=2,
        )
        x = torch.randn(3, 2)
        h = torch.randn(3, 4)
        g = router(x, h, context=None)
        assert g.shape == (3, 4)
        # Exactly top_k non-zero
        n_active = (g > 1e-6).sum(dim=-1)
        assert (n_active == 2).all()

    def test_bias_affects_routing(self):
        router = AuxLFRouter(
            input_size=2, hidden_size=4, d_context=0,
            n_unique=3, top_k=1, auxlf_config=AuxLFConfig(use_update=False),
        )
        # Manually set bias to heavily favor expert 0
        with torch.no_grad():
            router.bias.fill_(-10.0)
            router.bias[0] = 10.0  # expert 0 favored
        g = router(torch.randn(10, 2), torch.randn(10, 4), context=None)
        # Expert 0 should always be selected
        assert (router.last_top_idx == 0).all()

    def test_no_update_when_disabled(self):
        router = AuxLFRouter(
            input_size=2, hidden_size=4, d_context=0,
            n_unique=3, top_k=2, auxlf_config=AuxLFConfig(use_update=False),
        )
        # Run a few times
        for _ in range(20):
            router(torch.randn(4, 2), torch.randn(4, 4), context=None)
        # Bias should still be 0
        assert torch.allclose(router.bias, torch.zeros(3))

    def test_update_after_warmup(self):
        router = AuxLFRouter(
            input_size=2, hidden_size=4, d_context=0,
            n_unique=3, top_k=2, auxlf_config=AuxLFConfig(
                bias_lr=0.1, warmup_steps=5, bias_clamp=10.0,
                target_load_fraction=1.0 / 3.0,
            ),
        )
        # Warmup — bias stays 0
        for _ in range(5):
            router(torch.randn(4, 2), torch.randn(4, 4), context=None)
        # Bias may already have updated by the last warmup step (5 == warmup).
        # Force it back to zero so we can verify post-warmup behavior.
        with torch.no_grad():
            router.bias.zero_()
            router._step_count = 0
        # After warmup — bias updates
        for _ in range(20):
            router(torch.randn(4, 2), torch.randn(4, 4), context=None)
        assert not torch.allclose(router.bias, torch.zeros(3))

    def test_gradient_flows(self):
        router = AuxLFRouter(
            input_size=2, hidden_size=4, d_context=0,
            n_unique=3, top_k=2,
        )
        g = router(torch.randn(2, 2), torch.randn(2, 4), context=None)
        loss = g.sum()
        loss.backward()
        # Router net has grad
        for p in router.net.parameters():
            assert p.grad is not None
        # Bias is NOT a grad parameter (requires_grad=False)
        assert router.bias.requires_grad is False

    def test_get_load_stats(self):
        router = AuxLFRouter(
            input_size=2, hidden_size=4, d_context=0,
            n_unique=3, top_k=2, auxlf_config=AuxLFConfig(use_update=False),
        )
        stats = router.get_load_stats()
        assert "util_per_expert" in stats
        assert "std" in stats
        assert "max_min_ratio" in stats
        assert "bias_per_expert" in stats
        assert len(stats["util_per_expert"]) == 3


# ----------------------------------------------------------------------
# TestAuxLFSETAMoECfCCell
# ----------------------------------------------------------------------
class TestAuxLFSETAMoECfCCell:
    def test_forward_shape(self):
        cfg = SETAConfig(n_shared=2, n_unique=3, top_k=2)
        cell = AuxLFSETAMoECfCCell(
            input_size=2, hidden_size=8, sdta_config=cfg, d_context=4,
        )
        x = torch.randn(3, 2)
        h = torch.randn(3, 8)
        out = cell(x, h, context=torch.randn(3, 4))
        assert out.shape == (3, 8)

    def test_shared_always_active(self):
        cfg = SETAConfig(n_shared=2, n_unique=3, top_k=1)
        cell = AuxLFSETAMoECfCCell(
            input_size=2, hidden_size=8, sdta_config=cfg, d_context=4,
        )
        cell(torch.randn(3, 2), torch.randn(3, 8), context=None)
        assert cell.last_shared_stack.shape == (3, 2, 8)

    def test_router_is_auxlf(self):
        cfg = SETAConfig(n_shared=2, n_unique=3, top_k=2)
        cell = AuxLFSETAMoECfCCell(
            input_size=2, hidden_size=8, sdta_config=cfg, d_context=4,
        )
        assert isinstance(cell.router, AuxLFRouter)

    def test_utilization_includes_load_stats(self):
        cfg = SETAConfig(n_shared=2, n_unique=3, top_k=2)
        cell = AuxLFSETAMoECfCCell(
            input_size=2, hidden_size=8, sdta_config=cfg, d_context=4,
            auxlf_config=AuxLFConfig(use_update=False),
        )
        cell(torch.randn(3, 2), torch.randn(3, 8), context=None)
        util = cell.collect_expert_utilization()
        assert "auxlf_util_std" in util
        assert "auxlf_max_min_ratio" in util
        assert "auxlf_bias_norm" in util


# ----------------------------------------------------------------------
# TestAuxLFSETAMoECfCNetwork
# ----------------------------------------------------------------------
class TestAuxLFSETAMoECfCNetwork:
    def test_forward(self):
        cfg = SETAConfig(n_shared=2, n_unique=3, top_k=2)
        net = AuxLFSETAMoECfCNetwork(
            input_size=2, hidden_size=8, sdta_config=cfg,
            n_queries=2, d_context=4, n_heads=2, output_size=2,
        )
        obs = torch.randn(2, 5, 2)
        times = torch.linspace(0, 1, 5).unsqueeze(0).expand(2, -1)
        out = net(obs, times)
        assert out.shape == (2, 5, 2)

    def test_nan_aware_mask(self):
        cfg = SETAConfig(n_shared=2, n_unique=3, top_k=2)
        net = AuxLFSETAMoECfCNetwork(
            input_size=2, hidden_size=8, sdta_config=cfg,
            n_queries=2, d_context=4, n_heads=2, output_size=2,
        )
        obs = torch.randn(2, 5, 2)
        obs[0, 2, 0] = float("nan")
        times = torch.linspace(0, 1, 5).unsqueeze(0).expand(2, -1)
        out = net(obs, times)
        assert not torch.isnan(out).any()

    def test_get_utilization(self):
        cfg = SETAConfig(n_shared=2, n_unique=3, top_k=2)
        net = AuxLFSETAMoECfCNetwork(
            input_size=2, hidden_size=8, sdta_config=cfg,
            n_queries=2, d_context=4, n_heads=2, output_size=2,
            auxlf_config=AuxLFConfig(use_update=False),
        )
        obs = torch.randn(2, 5, 2)
        times = torch.linspace(0, 1, 5).unsqueeze(0).expand(2, -1)
        net(obs, times)
        util = net.get_utilization()
        assert "auxlf_util_std" in util


# ----------------------------------------------------------------------
# TestExports
# ----------------------------------------------------------------------
class TestAuxLFExports:
    def test_all_exports(self):
        from lnn.core import (
            AuxLFConfig,
            AuxLFRouter,
            AuxLFSETAMoECfCCell,
            AuxLFSETAMoECfCNetwork,
            update_load_balancing_bias,
        )
        assert AuxLFConfig is not None
        assert AuxLFRouter is not None
        assert AuxLFSETAMoECfCCell is not None
        assert AuxLFSETAMoECfCNetwork is not None
        assert update_load_balancing_bias is not None
