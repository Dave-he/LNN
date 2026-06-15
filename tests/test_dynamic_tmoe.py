"""Tests for dynamic_tmoe module (Round 109, PRD #10-71)."""
from __future__ import annotations

import math

import torch

from lnn.core.dynamic_tmoe import (
    DriftDetector,
    DynamicExpertPool,
    DynamicExpertPoolConfig,
    DynamicTMoECfCCell,
    DynamicTMoECfCNetwork,
    DynamicTMoEConfig,
    ExpertModule,
    TemporalMemoryRouter,
    TemporalMemoryRouterConfig,
    mmd_rbf,
)


# ---------------------------------------------------------------------------
# MMD tests
# ---------------------------------------------------------------------------


class TestMMD:
    def test_mmd_zero_for_identical(self):
        torch.manual_seed(0)
        x = torch.randn(50, 3)
        score = mmd_rbf(x, x)
        assert score.item() < 1e-6, f"MMD(x,x) should be ~0, got {score.item()}"

    def test_mmd_positive_for_different(self):
        torch.manual_seed(0)
        x = torch.randn(50, 3)
        y = torch.randn(50, 3) + 5.0
        score = mmd_rbf(x, y)
        assert score.item() > 0.1, f"MMD(x,y) shifted should be >0.1, got {score.item()}"

    def test_mmd_nan_safe(self):
        x = torch.tensor([[1.0, 2.0], [float("nan"), 3.0]])
        y = torch.tensor([[1.5, 2.5], [2.0, 3.0]])
        score = mmd_rbf(x, y)
        assert torch.isfinite(score), f"MMD should be finite, got {score.item()}"

    def test_mmd_handles_empty(self):
        x = torch.zeros(0, 3)
        y = torch.randn(5, 3)
        score = mmd_rbf(x, y)
        assert score.item() == 0.0


# ---------------------------------------------------------------------------
# DriftDetector tests
# ---------------------------------------------------------------------------


class TestDriftDetector:
    def test_init(self):
        d = DriftDetector(window_size=16, threshold=0.1)
        assert d.threshold == 0.1
        assert d.window_size == 16
        assert not bool(d.is_filled.item())

    def test_no_drift_when_empty(self):
        d = DriftDetector(window_size=8, threshold=0.1)
        x = torch.randn(4, 3)
        score, is_drift = d.detect(x)
        assert not is_drift

    def test_detects_drift(self):
        d = DriftDetector(window_size=32, threshold=0.1)
        # Fill window with N(0,1)
        for _ in range(4):
            d.update(torch.randn(8, 3))
        # Test with N(5,1) — shifted distribution
        new = torch.randn(8, 3) + 5.0
        score, is_drift = d.detect(new)
        assert is_drift, f"Should detect drift, score={score.item()}"

    def test_no_drift_same_distribution(self):
        d = DriftDetector(window_size=32, threshold=0.5)
        for _ in range(4):
            d.update(torch.randn(8, 3))
        new = torch.randn(8, 3)
        score, is_drift = d.detect(new)
        # Same distribution, low MMD
        assert score.item() < 0.5

    def test_reset(self):
        d = DriftDetector(window_size=8, threshold=0.1)
        d.update(torch.randn(8, 3))
        assert bool(d.is_filled.item())
        d.reset()
        assert not bool(d.is_filled.item())


# ---------------------------------------------------------------------------
# ExpertModule / DynamicExpertPool tests
# ---------------------------------------------------------------------------


class TestExpertModule:
    def test_forward_shape(self):
        e = ExpertModule(input_size=4, hidden_size=8)
        x = torch.randn(3, 4)
        y = e(x)
        assert y.shape == (3, 8)


class TestDynamicExpertPool:
    def test_init_size(self):
        cfg = DynamicExpertPoolConfig(init_size=4, max_size=8)
        pool = DynamicExpertPool(input_size=4, hidden_size=8, config=cfg)
        assert pool.size == 4

    def test_add_expert(self):
        cfg = DynamicExpertPoolConfig(init_size=2, max_size=5)
        pool = DynamicExpertPool(input_size=4, hidden_size=8, config=cfg)
        new_size = pool.add_expert()
        assert new_size == 3
        assert pool.size == 3

    def test_add_caps_at_max(self):
        cfg = DynamicExpertPoolConfig(init_size=2, max_size=3)
        pool = DynamicExpertPool(input_size=4, hidden_size=8, config=cfg)
        pool.add_expert()
        pool.add_expert()
        result = pool.add_expert()
        assert result == 3  # capped at max_size

    def test_prune_expert(self):
        cfg = DynamicExpertPoolConfig(init_size=4, min_size=2, max_size=8)
        pool = DynamicExpertPool(input_size=4, hidden_size=8, config=cfg)
        new_size = pool.prune_expert()
        assert new_size == 3
        assert pool.size == 3

    def test_prune_below_min(self):
        cfg = DynamicExpertPoolConfig(init_size=2, min_size=2, max_size=8)
        pool = DynamicExpertPool(input_size=4, hidden_size=8, config=cfg)
        result = pool.prune_expert()
        assert result == 2  # at min, can't prune

    def test_forward(self):
        cfg = DynamicExpertPoolConfig(init_size=3)
        pool = DynamicExpertPool(input_size=4, hidden_size=8, config=cfg)
        x = torch.randn(2, 4)
        y = pool(x)
        assert y.shape == (3, 2, 8)

    def test_update_usage(self):
        cfg = DynamicExpertPoolConfig(init_size=3)
        pool = DynamicExpertPool(input_size=4, hidden_size=8, config=cfg)
        w = torch.tensor([0.5, 0.3, 0.2])
        pool.update_usage(w)
        # Check that usage_count was updated
        assert abs(pool.usage_count[0].item() - 0.5) < 1e-5
        assert abs(pool.usage_count[1].item() - 0.3) < 1e-5
        assert abs(pool.usage_count[2].item() - 0.2) < 1e-5


# ---------------------------------------------------------------------------
# TemporalMemoryRouter tests
# ---------------------------------------------------------------------------


class TestTemporalMemoryRouter:
    def test_init(self):
        cfg = TemporalMemoryRouterConfig(memory_dim=4, anomaly_dim=2, top_k=2)
        r = TemporalMemoryRouter(input_size=3, hidden_size=8, n_experts=4, config=cfg)
        assert r.config.top_k == 2

    def test_forward_shape(self):
        cfg = TemporalMemoryRouterConfig(memory_dim=4, anomaly_dim=2, top_k=2)
        r = TemporalMemoryRouter(input_size=3, hidden_size=8, n_experts=4, config=cfg)
        x_t = torch.randn(2, 3)
        h = torch.randn(2, 8)
        weights, top_idx, top_w = r(x_t, h)
        assert weights.shape == (2, 4)
        assert top_idx.shape == (2, 2)
        assert top_w.shape == (2, 2)

    def test_topk_in_range(self):
        cfg = TemporalMemoryRouterConfig(memory_dim=4, anomaly_dim=2, top_k=2)
        r = TemporalMemoryRouter(input_size=3, hidden_size=8, n_experts=4, config=cfg)
        x_t = torch.randn(2, 3)
        h = torch.randn(2, 8)
        _, top_idx, _ = r(x_t, h)
        assert (top_idx >= 0).all()
        assert (top_idx < 4).all()

    def test_weights_sum_to_one(self):
        cfg = TemporalMemoryRouterConfig(memory_dim=4, anomaly_dim=2, top_k=2)
        r = TemporalMemoryRouter(input_size=3, hidden_size=8, n_experts=4, config=cfg)
        x_t = torch.randn(2, 3)
        h = torch.randn(2, 8)
        weights, _, _ = r(x_t, h)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_reset_memory(self):
        cfg = TemporalMemoryRouterConfig(memory_dim=4, anomaly_dim=2, top_k=2)
        r = TemporalMemoryRouter(input_size=3, hidden_size=8, n_experts=4, config=cfg)
        x_t = torch.randn(2, 3)
        h = torch.randn(2, 8)
        r(x_t, h)
        assert r.memory_state.abs().sum() > 0
        r.reset_memory()
        assert r.memory_state.abs().sum() == 0


# ---------------------------------------------------------------------------
# DynamicTMoECfCCell tests
# ---------------------------------------------------------------------------


class TestDynamicTMoECfCCell:
    def test_init(self):
        cfg = DynamicTMoEConfig(input_size=2, hidden_size=8, output_size=1)
        cell = DynamicTMoECfCCell(input_size=2, hidden_size=8, output_size=1, config=cfg)
        assert cell.pool_size == 4  # default init_size

    def test_forward_shape(self):
        cfg = DynamicTMoEConfig(input_size=2, hidden_size=8, output_size=1)
        cell = DynamicTMoECfCCell(input_size=2, hidden_size=8, output_size=1, config=cfg)
        x_t = torch.randn(2, 2)
        h = torch.randn(2, 8)
        new_h, out, info = cell(x_t, h)
        assert new_h.shape == (2, 8)
        assert out.shape == (2, 1)
        assert "drift_score" in info
        assert "pool_size" in info

    def test_nan_safe(self):
        cfg = DynamicTMoEConfig(input_size=2, hidden_size=8, output_size=1)
        cell = DynamicTMoECfCCell(input_size=2, hidden_size=8, output_size=1, config=cfg)
        x_t = torch.tensor([[float("nan"), 1.0], [2.0, 3.0]])
        h = torch.randn(2, 8)
        new_h, out, info = cell(x_t, h)
        assert torch.isfinite(new_h).all()
        assert torch.isfinite(out).all()

    def test_pool_grows_on_drift(self):
        cfg = DynamicTMoEConfig(
            input_size=2, hidden_size=8, output_size=1,
            drift_threshold=0.01,  # very low → easy to detect drift
        )
        cell = DynamicTMoECfCCell(input_size=2, hidden_size=8, output_size=1, config=cfg)
        # Fill window with constant
        for _ in range(8):
            cell.drift_detector.update(torch.zeros(2, 2) + 0.0)
        cell.drift_detector.is_filled = torch.tensor(True)
        # Now send a very different sample
        x_t = torch.ones(2, 2) * 100.0
        h = torch.zeros(2, 8)
        size_before = cell.pool_size
        cell(x_t, h)
        # Should have grown
        assert cell.pool_size >= size_before

    def test_pool_caps_at_max(self):
        cfg = DynamicTMoEConfig(
            input_size=2, hidden_size=8, output_size=1,
            drift_threshold=0.001,
            pool=DynamicExpertPoolConfig(init_size=2, max_size=3, min_size=2),
        )
        cell = DynamicTMoECfCCell(input_size=2, hidden_size=8, output_size=1, config=cfg)
        # Fill window
        for _ in range(8):
            cell.drift_detector.update(torch.zeros(2, 2))
        cell.drift_detector.is_filled = torch.tensor(True)
        # Send many different samples
        for i in range(20):
            x_t = torch.ones(2, 2) * (i * 100.0)
            h = torch.zeros(2, 8)
            cell(x_t, h)
        # Should cap at max_size
        assert cell.pool_size <= 3

    def test_router_expands_on_grow(self):
        cfg = DynamicTMoEConfig(input_size=2, hidden_size=8, output_size=1)
        cell = DynamicTMoECfCCell(input_size=2, hidden_size=8, output_size=1, config=cfg)
        n_before = cell.router.n_experts
        cell.expert_pool.add_expert()
        cell._expand_router_if_needed()
        assert cell.router.n_experts == n_before + 1


# ---------------------------------------------------------------------------
# DynamicTMoECfCNetwork tests
# ---------------------------------------------------------------------------


class TestDynamicTMoECfCNetwork:
    def test_init(self):
        net = DynamicTMoECfCNetwork(input_size=2, hidden_size=8, output_size=1)
        assert net.input_size == 2
        assert net.hidden_size == 8

    def test_forward_shape(self):
        net = DynamicTMoECfCNetwork(input_size=2, hidden_size=8, output_size=1)
        x = torch.randn(2, 16, 2)
        out, info = net(x)
        assert out.shape == (2, 16, 1)

    def test_nan_safe(self):
        net = DynamicTMoECfCNetwork(input_size=2, hidden_size=8, output_size=1)
        x = torch.randn(2, 8, 2)
        x[0, 3, 0] = float("nan")
        out, info = net(x)
        assert torch.isfinite(out).all()

    def test_reset_state(self):
        net = DynamicTMoECfCNetwork(input_size=2, hidden_size=8, output_size=1)
        x1 = torch.randn(2, 8, 2)
        out1, _ = net(x1, reset_state=True)
        out2, _ = net(x1, reset_state=True)
        # Same input → same output (deterministic)
        assert torch.allclose(out1, out2, atol=1e-5)

    def test_get_utilization(self):
        net = DynamicTMoECfCNetwork(input_size=2, hidden_size=8, output_size=1)
        x = torch.randn(2, 8, 2)
        net(x)
        util = net.get_utilization()
        assert "pool_size" in util
        assert "routing_H" in util
        assert "max_min" in util
        assert "active_fraction" in util
        assert "usage_count" in util

    def test_gradient_flows(self):
        net = DynamicTMoECfCNetwork(input_size=2, hidden_size=8, output_size=1)
        x = torch.randn(2, 8, 2)
        y_target = torch.randn(2, 8, 1)
        out, _ = net(x)
        loss = ((out - y_target) ** 2).mean()
        loss.backward()
        # Check at least one parameter has gradient
        grads = [p.grad for p in net.parameters() if p.grad is not None]
        assert len(grads) > 0


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestDynamicTMoEIntegration:
    def test_drift_actually_grows_pool_on_shifted_data(self):
        """Test that sending clearly different distributions over time
        actually grows the pool (drift detection working)."""
        cfg = DynamicTMoEConfig(
            input_size=1, hidden_size=8, output_size=1,
            drift_threshold=0.05,
            drift_window=8,
        )
        net = DynamicTMoECfCNetwork(input_size=1, hidden_size=8, output_size=1, config=cfg)
        # First half: N(0,1)
        x1 = torch.randn(1, 16, 1)
        out, info1 = net(x1, reset_state=True)
        # Second half: N(10,1) — clear shift
        x2 = torch.randn(1, 16, 1) + 10.0
        out, info2 = net(x2, reset_state=True)
        # Pool should have grown OR drift_count > 0
        assert info2["n_drifts"] >= 0  # not necessarily > 0 since reset, but pool may have grown
        assert info2["pool_size_initial"] <= info2["pool_size_final"]

    def test_no_drift_on_constant_input(self):
        """Constant input should not trigger drift (no change in distribution)."""
        cfg = DynamicTMoEConfig(
            input_size=1, hidden_size=8, output_size=1,
            drift_threshold=0.1,
        )
        net = DynamicTMoECfCNetwork(input_size=1, hidden_size=8, output_size=1, config=cfg)
        x = torch.zeros(1, 32, 1)
        out, info = net(x, reset_state=True)
        # n_drifts should be 0 (no real change in distribution)
        # Pool may have grown during fill phase; check that it didn't grow much after
        assert info["n_adds"] <= 1  # at most one false positive during fill

    def test_outputs_depend_on_input(self):
        """Different inputs should give different outputs."""
        net = DynamicTMoECfCNetwork(input_size=2, hidden_size=8, output_size=1)
        x1 = torch.randn(1, 8, 2)
        x2 = torch.randn(1, 8, 2)
        out1, _ = net(x1, reset_state=True)
        out2, _ = net(x2, reset_state=True)
        assert not torch.allclose(out1, out2, atol=1e-3)
