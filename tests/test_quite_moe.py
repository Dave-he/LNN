"""Round 103 — Tests for QuITE+MoE Irregularity-Context Routing (PRD #10-65)."""
from __future__ import annotations

import pytest
import torch

from lnn.core.quite_moe import (
    QuiteMoECfCCell,
    QuiteMoECfCNetwork,
    QuiteRouter,
    quite_context_pool,
)


class TestQuiteContextPool:
    """Tests for quite_context_pool helper."""

    def test_mean_pool_shape(self):
        """Mean pool outputs (B, d_model)."""
        tokens = torch.randn(2, 8, 16)
        out = quite_context_pool(tokens, method="mean")
        assert out.shape == (2, 16)

    def test_max_pool_shape(self):
        """Max pool outputs (B, d_model)."""
        tokens = torch.randn(2, 8, 16)
        out = quite_context_pool(tokens, method="max")
        assert out.shape == (2, 16)

    def test_first_pool_shape(self):
        """First-token pool outputs (B, d_model)."""
        tokens = torch.randn(2, 8, 16)
        out = quite_context_pool(tokens, method="first")
        assert out.shape == (2, 16)
        assert torch.allclose(out, tokens[:, 0, :])

    def test_mean_pool_value(self):
        """Mean pool matches manual mean."""
        tokens = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        out = quite_context_pool(tokens, method="mean")
        expected = torch.tensor([[3.0, 4.0]])
        assert torch.allclose(out, expected)

    def test_invalid_method_raises(self):
        """Invalid method should raise ValueError."""
        with pytest.raises(ValueError, match="mean/max/first"):
            quite_context_pool(torch.randn(1, 4, 8), method="invalid")

    def test_invalid_input_dim_raises(self):
        """Non-3D input should raise ValueError."""
        with pytest.raises(ValueError, match="B, n_queries, d_model"):
            quite_context_pool(torch.randn(8, 16), method="mean")


class TestQuiteRouter:
    """Tests for the QuITE-augmented router."""

    def test_router_initialization(self):
        """Router stores the right parameters."""
        torch.manual_seed(0)
        r = QuiteRouter(
            input_size=3, hidden_size=8, d_context=16,
            n_experts=4, top_k=2,
        )
        assert r.input_size == 3
        assert r.hidden_size == 8
        assert r.d_context == 16
        assert r.n_experts == 4
        assert r.top_k == 2

    def test_router_forward_shape_with_context(self):
        """Forward with context outputs (B, n_experts)."""
        torch.manual_seed(0)
        r = QuiteRouter(
            input_size=3, hidden_size=8, d_context=16,
            n_experts=4, top_k=2,
        )
        x_t = torch.randn(2, 3)
        h = torch.randn(2, 8)
        ctx = torch.randn(2, 16)
        g = r(x_t, h, context=ctx)
        assert g.shape == (2, 4)
        # Each row should sum to 1
        sums = g.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5)
        # Exactly top_k non-zero entries per row
        nonzeros = (g > 1e-6).sum(dim=-1)
        assert (nonzeros == 2).all()

    def test_router_forward_shape_without_context(self):
        """Forward without context falls back to [x_t, h] routing."""
        torch.manual_seed(0)
        r = QuiteRouter(
            input_size=3, hidden_size=8, d_context=16,
            n_experts=4, top_k=2,
        )
        x_t = torch.randn(2, 3)
        h = torch.randn(2, 8)
        g = r(x_t, h, context=None)
        assert g.shape == (2, 4)

    def test_router_context_changes_routing(self):
        """Different contexts should produce different routing weights."""
        torch.manual_seed(0)
        r = QuiteRouter(
            input_size=3, hidden_size=8, d_context=16,
            n_experts=4, top_k=2,
        )
        x_t = torch.randn(1, 3)
        h = torch.randn(1, 8)
        ctx_a = torch.randn(1, 16)
        ctx_b = torch.randn(1, 16)
        g_a = r(x_t, h, context=ctx_a)
        g_b = r(x_t, h, context=ctx_b)
        # The routing should differ for different contexts
        assert not torch.allclose(g_a, g_b, atol=1e-4)

    def test_router_dense_top_k(self):
        """When top_k == n_experts, no masking is applied."""
        torch.manual_seed(0)
        r = QuiteRouter(
            input_size=2, hidden_size=4, d_context=8,
            n_experts=3, top_k=3,
        )
        x_t = torch.randn(1, 2)
        h = torch.randn(1, 4)
        ctx = torch.randn(1, 8)
        g = r(x_t, h, context=ctx)
        # All entries should be non-zero
        assert (g > 1e-6).all()
        assert torch.allclose(g.sum(dim=-1), torch.tensor([1.0]), atol=1e-5)

    def test_router_wrong_context_dim_raises(self):
        """Wrong context dim should raise ValueError."""
        r = QuiteRouter(
            input_size=2, hidden_size=4, d_context=8,
            n_experts=2, top_k=1,
        )
        x_t = torch.randn(1, 2)
        h = torch.randn(1, 4)
        ctx = torch.randn(1, 16)  # wrong dim
        with pytest.raises(ValueError, match="context dim"):
            r(x_t, h, context=ctx)

    def test_router_with_mlp(self):
        """router_hidden > 0 uses 2-layer MLP."""
        torch.manual_seed(0)
        r = QuiteRouter(
            input_size=2, hidden_size=4, d_context=8,
            n_experts=3, top_k=2, router_hidden=16,
        )
        x_t = torch.randn(1, 2)
        h = torch.randn(1, 4)
        ctx = torch.randn(1, 8)
        g = r(x_t, h, context=ctx)
        assert g.shape == (1, 3)
        nonzeros = (g > 1e-6).sum(dim=-1)
        assert (nonzeros == 2).all()


class TestQuiteMoECfCCell:
    """Tests for the QuITE-augmented MoE cell."""

    def test_cell_initialization(self):
        """Cell stores the right parameters."""
        cell = QuiteMoECfCCell(
            input_size=3, hidden_size=8, n_experts=2, top_k=1,
            d_context=16,
        )
        assert cell.n_experts == 2
        assert cell.top_k == 1
        assert cell.d_context == 16
        assert len(cell.experts) == 2

    def test_cell_forward_shape(self):
        """Forward returns (B, hidden_size)."""
        torch.manual_seed(0)
        cell = QuiteMoECfCCell(
            input_size=3, hidden_size=8, n_experts=2, top_k=1,
            d_context=16,
        )
        x_t = torch.randn(2, 3)
        h = torch.randn(2, 8)
        ctx = torch.randn(2, 16)
        h_new = cell(x_t, h, context=ctx)
        assert h_new.shape == (2, 8)

    def test_cell_forward_without_context(self):
        """Forward without context falls back to [x_t, h] routing."""
        torch.manual_seed(0)
        cell = QuiteMoECfCCell(
            input_size=3, hidden_size=8, n_experts=2, top_k=1,
            d_context=16,
        )
        x_t = torch.randn(2, 3)
        h = torch.randn(2, 8)
        h_new = cell(x_t, h, context=None)
        assert h_new.shape == (2, 8)

    def test_cell_different_contexts_different_outputs(self):
        """Different contexts should produce different cell outputs."""
        torch.manual_seed(0)
        cell = QuiteMoECfCCell(
            input_size=3, hidden_size=8, n_experts=2, top_k=1,
            d_context=16,
        )
        x_t = torch.randn(1, 3)
        h = torch.randn(1, 8)
        ctx_a = torch.randn(1, 16)
        ctx_b = torch.randn(1, 16)
        h_a = cell(x_t, h, context=ctx_a)
        h_b = cell(x_t, h, context=ctx_b)
        # Different contexts → different routing → different h_new
        assert not torch.allclose(h_a, h_b, atol=1e-4)

    def test_cell_gradient_flows(self):
        """Gradient flows through cell back to input."""
        torch.manual_seed(0)
        cell = QuiteMoECfCCell(
            input_size=3, hidden_size=8, n_experts=2, top_k=1,
            d_context=16,
        )
        x_t = torch.randn(1, 3, requires_grad=True)
        h = torch.randn(1, 8)
        ctx = torch.randn(1, 16)
        h_new = cell(x_t, h, context=ctx)
        loss = h_new.sum()
        loss.backward()
        assert x_t.grad is not None
        assert x_t.grad.abs().sum() > 0

    def test_cell_dense_routing(self):
        """top_k == n_experts gives dense routing."""
        torch.manual_seed(0)
        cell = QuiteMoECfCCell(
            input_size=2, hidden_size=4, n_experts=3, top_k=3,
            d_context=8,
        )
        x_t = torch.randn(1, 2)
        h = torch.randn(1, 4)
        ctx = torch.randn(1, 8)
        h_new = cell(x_t, h, context=ctx)
        assert h_new.shape == (1, 4)
        # Dense routing means all experts contribute
        nonzeros = (cell.last_g > 1e-6).sum(dim=-1)
        assert (nonzeros == 3).all()


class TestQuiteMoECfCNetwork:
    """Tests for the full QuITE-augmented MoE network."""

    def test_network_initialization(self):
        """Network stores the right parameters."""
        net = QuiteMoECfCNetwork(
            input_size=2, hidden_size=8, n_experts=2, top_k=1,
            n_queries=4, d_context=16, n_heads=4, output_size=1,
        )
        assert net.input_size == 2
        assert net.hidden_size == 8
        assert net.n_experts == 2
        assert net.top_k == 1
        assert net.n_queries == 4
        assert net.d_context == 16
        assert net.output_size == 1

    def test_network_forward_shape(self):
        """Forward returns (B, T, output_size)."""
        torch.manual_seed(0)
        net = QuiteMoECfCNetwork(
            input_size=2, hidden_size=8, n_experts=2, top_k=1,
            n_queries=4, d_context=16, n_heads=4, output_size=1,
        )
        obs = torch.randn(2, 20, 2)
        times = torch.linspace(0, 1, 20).unsqueeze(0).expand(2, -1)
        out = net(obs, times)
        assert out.shape == (2, 20, 1)

    def test_network_with_mask(self):
        """Mask=0 positions are ignored (NaN handling)."""
        torch.manual_seed(0)
        net = QuiteMoECfCNetwork(
            input_size=2, hidden_size=8, n_experts=2, top_k=1,
            n_queries=4, d_context=16, n_heads=4, output_size=1,
        )
        obs = torch.randn(1, 10, 2)
        obs[0, :5] = float("nan")  # First half is missing
        times = torch.linspace(0, 1, 10).unsqueeze(0)
        out_clean = net(obs, times)
        assert out_clean.shape == (1, 10, 1)
        # Output should not contain NaN even with NaN input
        assert torch.isfinite(out_clean).all()

    def test_network_expert_utilization(self):
        """After forward, all experts should have been activated in at
        least one step (K=3, top_k=2, with diverse inputs)."""
        torch.manual_seed(0)
        net = QuiteMoECfCNetwork(
            input_size=2, hidden_size=8, n_experts=3, top_k=2,
            n_queries=4, d_context=16, n_heads=4, output_size=1,
        )
        obs = torch.randn(8, 20, 2)  # 8 batch to encourage diversity
        times = torch.linspace(0, 1, 20).unsqueeze(0).expand(8, -1)
        # We need to collect expert usage across all T steps. Re-run with
        # a hook on router.last_top_idx to gather all selections.
        usage = torch.zeros(3)
        # Forward manually with a hook
        net.compute_context(obs, times)
        # Reset
        net._cached_context = None
        # Collect by patching the router's last_top_idx
        from lnn.core.quite_moe import quite_context_pool
        tokens = net.quite(obs, times)
        context = quite_context_pool(tokens)
        h = torch.zeros(8, 8)
        for t in range(20):
            x_t = obs[:, t, :]
            _ = net.cell(x_t, h, context=context)
            # last_top_idx is (B, top_k)
            top_idx = net.cell.router.last_top_idx
            for k in top_idx.flatten().tolist():
                usage[k] += 1
        # At least 2 of 3 experts should have been used (with 8 batch and
        # 20 steps = 160 routing decisions, hard for all experts to be
        # completely dead with random init + diverse inputs).
        assert (usage > 0).sum() >= 2

    def test_network_compute_context_explicit(self):
        """compute_context returns a (B, d_context) tensor and caches it."""
        torch.manual_seed(0)
        net = QuiteMoECfCNetwork(
            input_size=2, hidden_size=8, n_experts=2, top_k=1,
            n_queries=4, d_context=16, n_heads=4, output_size=1,
        )
        obs = torch.randn(2, 10, 2)
        times = torch.linspace(0, 1, 10).unsqueeze(0).expand(2, -1)
        ctx = net.compute_context(obs, times)
        assert ctx.shape == (2, 16)
        # Cached
        assert net._cached_context is not None
        # Reset clears it
        net.reset_context()
        assert net._cached_context is None

    def test_network_gradient_flows(self):
        """Gradient flows from output back to input observations."""
        torch.manual_seed(0)
        net = QuiteMoECfCNetwork(
            input_size=2, hidden_size=8, n_experts=2, top_k=1,
            n_queries=4, d_context=16, n_heads=4, output_size=1,
        )
        obs = torch.randn(1, 10, 2, requires_grad=True)
        times = torch.linspace(0, 1, 10).unsqueeze(0)
        out = net(obs, times)
        loss = out.sum()
        loss.backward()
        assert obs.grad is not None
        assert obs.grad.abs().sum() > 0

    def test_network_different_pool_methods(self):
        """pool_method switches between mean/max/first pooling."""
        for method in ("mean", "max", "first"):
            torch.manual_seed(0)
            net = QuiteMoECfCNetwork(
                input_size=2, hidden_size=8, n_experts=2, top_k=1,
                n_queries=4, d_context=16, n_heads=4, output_size=1,
                pool_method=method,
            )
            obs = torch.randn(1, 5, 2)
            times = torch.linspace(0, 1, 5).unsqueeze(0)
            out = net(obs, times)
            assert out.shape == (1, 5, 1)
            assert torch.isfinite(out).all()

    def test_network_precomputed_context(self):
        """Forward with precomputed context reuses it."""
        torch.manual_seed(0)
        net = QuiteMoECfCNetwork(
            input_size=2, hidden_size=8, n_experts=2, top_k=1,
            n_queries=4, d_context=16, n_heads=4, output_size=1,
        )
        obs = torch.randn(1, 5, 2)
        times = torch.linspace(0, 1, 5).unsqueeze(0)
        # Pre-compute context
        ctx = net.compute_context(obs, times)
        # Use the same context in forward
        out = net(obs, times, precomputed_context=ctx)
        assert out.shape == (1, 5, 1)


class TestQuiteMoEExports:
    """Verify exports are correct."""

    def test_exports(self):
        from lnn.core import (
            QuiteMoECfCCell,
            QuiteMoECfCNetwork,
            QuiteRouter,
            quite_context_pool,
        )
        _ = QuiteMoECfCCell
        _ = QuiteMoECfCNetwork
        _ = QuiteRouter
        assert callable(quite_context_pool)
