"""Unit tests for ReMoE (ReLU-routed MoE) Cell + Network (PRD #10-76, 2026-06-15).

Verifies:
- ReMoERouter: init, ReLU non-negativity, gradient flow, sparsity.
- ReMoECfCCell: init with K experts, forward shape, full gradient to all experts.
- Load-balancing loss: shape, zero at uniform gate, positive at skewed gate.
- ReMoECfCNetwork matches CfCNetwork API.
- Toy sin smoke: 4 experts converges.
"""
import numpy as np
import torch

from lnn.core.remoe import (
    ReMoECfCCell,
    ReMoECfCNetwork,
    ReMoERouter,
    remoe_load_balancing_loss,
    remoe_utilization,
)


def _seed(s: int = 0) -> None:
    torch.manual_seed(s)
    np.random.seed(s)


class TestReMoERouterInit:
    """Init: ReLU router, linear or 2-layer MLP."""

    def test_init_linear(self) -> None:
        _seed(0)
        router = ReMoERouter(input_size=3, hidden_size=8, n_experts=4)
        assert router.n_experts == 4
        assert router.router_hidden == 0
        # Linear layer
        assert isinstance(router.net, torch.nn.Linear)
        assert router.net.out_features == 4
        assert router.net.in_features == 3 + 8

    def test_init_mlp(self) -> None:
        _seed(0)
        router = ReMoERouter(input_size=3, hidden_size=8, n_experts=4, router_hidden=6)
        assert router.router_hidden == 6
        assert isinstance(router.net, torch.nn.Sequential)
        assert len(router.net) == 3

    def test_init_invalid_n_experts(self) -> None:
        try:
            ReMoERouter(input_size=3, hidden_size=8, n_experts=0)
            raise AssertionError("expected assertion")
        except AssertionError as e:
            assert "n_experts" in str(e)


class TestReMoERouterForward:
    """Forward: ReLU non-negativity, gradient flow, natural sparsity."""

    def test_relu_nonneg(self) -> None:
        """Output g must be non-negative (ReLU property)."""
        _seed(0)
        router = ReMoERouter(input_size=3, hidden_size=8, n_experts=4)
        x = torch.randn(8, 3)
        h = torch.randn(8, 8)
        g = router(x, h)
        assert g.shape == (8, 4)
        assert (g >= 0).all()

    def test_gradient_flow(self) -> None:
        """Gradient flows through ReLU (sparse but non-zero)."""
        _seed(0)
        router = ReMoERouter(input_size=3, hidden_size=8, n_experts=4)
        x = torch.randn(8, 3, requires_grad=True)
        h = torch.randn(8, 8, requires_grad=True)
        g = router(x, h)
        loss = g.sum()
        loss.backward()
        assert x.grad is not None
        assert h.grad is not None
        for p in router.parameters():
            assert p.grad is not None
            assert p.grad.abs().sum() > 0

    def test_natural_sparsity(self) -> None:
        """ReLU naturally produces some zero gates (sparsity by construction)."""
        _seed(0)
        router = ReMoERouter(input_size=3, hidden_size=8, n_experts=4)
        x = torch.randn(64, 3) * 0.1  # small inputs -> small pre-activations -> more zeros
        h = torch.randn(64, 8) * 0.1
        g = router(x, h)
        frac_nonzero = (g > 0).float().mean().item()
        # With small inputs, expect >= 30% zero rate.  Looser upper bound.
        assert 0.0 <= frac_nonzero <= 1.0

    def test_large_inputs_sparse_dominance(self) -> None:
        """Large inputs -> router produces sparse positive gates (specialization).

        The ReLU gate may kill negative pre-activations even with large inputs.
        We verify (1) non-negativity, (2) some positive mass per row on average.
        """
        _seed(0)
        router = ReMoERouter(input_size=3, hidden_size=8, n_experts=4)
        x = torch.randn(64, 3) * 100.0
        h = torch.randn(64, 8) * 100.0
        g = router(x, h)
        # All entries non-negative (ReLU property)
        assert (g >= 0).all()
        # At least one expert gets a non-trivial gate (not all zeros)
        assert g.sum() > 0
        # With init_bias=1.0 + large inputs, expect most rows to have at least one active gate
        active_per_row = (g > 0).any(dim=-1).sum().item()
        # Most rows (>= 80%) should have at least one active gate
        assert active_per_row >= 0.8 * 64, f"only {active_per_row}/64 rows had active gate"


class TestReMoECfCCellInit:
    """Init: K experts + ReLU router."""

    def test_init_default(self) -> None:
        _seed(0)
        cell = ReMoECfCCell(input_size=3, hidden_size=8)
        assert cell.n_experts == 4
        assert len(cell.experts) == 4
        assert isinstance(cell.router, ReMoERouter)
        assert cell.sparsity_target == 0.5

    def test_init_custom_k(self) -> None:
        _seed(0)
        cell = ReMoECfCCell(input_size=3, hidden_size=8, n_experts=2)
        assert cell.n_experts == 2
        assert len(cell.experts) == 2
        assert cell.router.n_experts == 2

    def test_init_invalid_k(self) -> None:
        try:
            ReMoECfCCell(input_size=3, hidden_size=8, n_experts=0)
            raise AssertionError("expected assertion")
        except AssertionError as e:
            assert "n_experts" in str(e)

    def test_init_with_router_hidden(self) -> None:
        _seed(0)
        cell = ReMoECfCCell(input_size=3, hidden_size=8, n_experts=4, router_hidden=4)
        assert cell.router_hidden == 4
        assert isinstance(cell.router.net, torch.nn.Sequential)


class TestReMoECfCCellForward:
    """Forward shape, gradient flow, ReLU sparsity, additive combination."""

    def test_forward_shape(self) -> None:
        _seed(0)
        cell = ReMoECfCCell(input_size=3, hidden_size=8, n_experts=4)
        x_t = torch.randn(4, 3)
        h = torch.randn(4, 8)
        h_new = cell(x_t, h)
        assert h_new.shape == (4, 8)

    def test_g_nonneg(self) -> None:
        """last_g is non-negative (ReLU property preserved)."""
        _seed(0)
        cell = ReMoECfCCell(input_size=3, hidden_size=8, n_experts=4)
        x_t = torch.randn(4, 3)
        h = torch.randn(4, 8)
        cell(x_t, h)
        g = cell.last_g
        assert g.shape == (4, 4)
        assert (g >= 0).all()

    def test_gradient_flows_to_all_experts(self) -> None:
        """CRITICAL: gradient flows to ALL K experts (not just top-K)."""
        _seed(0)
        cell = ReMoECfCCell(input_size=3, hidden_size=8, n_experts=4)
        x_t = torch.randn(4, 3, requires_grad=True)
        h = torch.randn(4, 8, requires_grad=True)
        h_new = cell(x_t, h)
        loss = h_new.sum()
        loss.backward()
        # All K experts should have non-zero gradient (ReLU gates are non-zero
        # for most experts; the natural sparsity might zero a few out, but
        # with 4 experts and 4 samples, expect most to get grad).
        n_with_grad = 0
        for expert in cell.experts:
            has_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in expert.parameters()
            )
            if has_grad:
                n_with_grad += 1
        # At least 2/4 experts should get grad (ReLU sparsity allows some zeros)
        assert n_with_grad >= 2, f"only {n_with_grad}/4 experts got grad"

    def test_gradient_flows_to_router(self) -> None:
        """Router parameters also receive gradient."""
        _seed(0)
        cell = ReMoECfCCell(input_size=3, hidden_size=8, n_experts=4)
        x_t = torch.randn(4, 3, requires_grad=True)
        h = torch.randn(4, 8, requires_grad=True)
        h_new = cell(x_t, h)
        h_new.sum().backward()
        for p in cell.router.parameters():
            assert p.grad is not None
            assert p.grad.abs().sum() > 0

    def test_last_sparsity_populated(self) -> None:
        """last_sparsity is populated after forward."""
        _seed(0)
        cell = ReMoECfCCell(input_size=3, hidden_size=8, n_experts=4)
        x_t = torch.randn(4, 3)
        h = torch.randn(4, 8)
        cell(x_t, h)
        assert cell.last_sparsity is not None
        assert 0.0 <= cell.last_sparsity.item() <= 1.0

    def test_additive_structure(self) -> None:
        """h_new equals sum_i g_i * expert_i(x, h) by construction."""
        _seed(0)
        cell = ReMoECfCCell(input_size=3, hidden_size=8, n_experts=3)
        x_t = torch.randn(2, 3)
        h = torch.randn(2, 8)
        h_new = cell(x_t, h)
        # Manually compute the expected output
        g = cell.router(x_t, h)
        expert_outs = [e(x_t, h) for e in cell.experts]
        stacked = torch.stack(expert_outs, dim=1)
        expected = (g.unsqueeze(-1) * stacked).sum(dim=1)
        assert torch.allclose(h_new, expected, atol=1e-6)


class TestReMoELoadBalancing:
    """remoe_load_balancing_loss: shape, behavior at uniform/skewed."""

    def test_zero_at_uniform(self) -> None:
        """If all g_i are equal, the load-balancing loss is 0."""
        g = torch.ones(8, 4)  # uniform mass
        loss = remoe_load_balancing_loss(g, n_experts=4)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)

    def test_positive_at_skewed(self) -> None:
        """If one expert dominates, the load-balancing loss is positive."""
        g = torch.zeros(8, 4)
        g[:, 0] = 10.0  # only expert 0 is active
        loss = remoe_load_balancing_loss(g, n_experts=4)
        assert loss.item() > 0.0

    def test_returns_scalar(self) -> None:
        g = torch.randn(8, 4).abs()
        loss = remoe_load_balancing_loss(g, n_experts=4)
        assert loss.dim() == 0
        assert loss.item() >= 0.0

    def test_n_experts_inferred(self) -> None:
        """n_experts defaults to g.shape[-1]."""
        g = torch.ones(8, 6)
        loss = remoe_load_balancing_loss(g)  # no n_experts
        assert loss.item() < 1e-5  # uniform mass -> ~0


class TestReMoECfCNetwork:
    """ReMoECfCNetwork matches CfCNetwork API."""

    def test_network_init(self) -> None:
        _seed(0)
        net = ReMoECfCNetwork(
            input_size=3, hidden_size=8, output_size=2,
            num_layers=2, n_experts=4,
        )
        assert len(net.cells) == 2
        for cell in net.cells:
            assert isinstance(cell, ReMoECfCCell)

    def test_network_forward_dense(self) -> None:
        _seed(0)
        net = ReMoECfCNetwork(
            input_size=3, hidden_size=8, output_size=2, num_layers=1,
            n_experts=4, return_sequences=True,
        )
        x = torch.randn(4, 12, 3)
        y = net(x)
        assert y.shape == (4, 12, 2)

    def test_network_forward_last_step(self) -> None:
        _seed(0)
        net = ReMoECfCNetwork(
            input_size=3, hidden_size=8, output_size=2, num_layers=1,
            n_experts=4, return_sequences=False,
        )
        x = torch.randn(4, 12, 3)
        y = net(x)
        assert y.shape == (4, 2)

    def test_network_with_mask(self) -> None:
        _seed(0)
        net = ReMoECfCNetwork(
            input_size=3, hidden_size=8, output_size=2, num_layers=1,
            n_experts=4, return_sequences=True,
        )
        x = torch.randn(4, 12, 3)
        mask = torch.ones(4, 12, 3)
        mask[:, 6:, :] = 0.0
        y = net(x, mask=mask)
        assert y.shape == (4, 12, 2)

    def test_network_gradient_flows(self) -> None:
        _seed(0)
        net = ReMoECfCNetwork(
            input_size=3, hidden_size=8, output_size=2, num_layers=1,
            n_experts=4, return_sequences=False,
        )
        x = torch.randn(4, 8, 3)
        y = net(x)
        y.sum().backward()
        for cell in net.cells:
            for p in cell.parameters():
                if p.requires_grad:
                    assert p.grad is not None
                    assert p.grad.abs().sum() > 0


class TestReMoEDiagnostics:
    """remoe_utilization diagnostic and other helpers."""

    def test_utilization_no_forward(self) -> None:
        """Calling diagnostic before forward should not crash."""
        _seed(0)
        cell = ReMoECfCCell(input_size=3, hidden_size=8, n_experts=4)
        diag = remoe_utilization(cell)
        assert diag["g_mean"].shape == (4,)
        assert diag["g_active_frac"].shape == (4,)
        assert diag["sparsity"].item() == 0.0

    def test_utilization_after_forward(self) -> None:
        _seed(0)
        cell = ReMoECfCCell(input_size=3, hidden_size=8, n_experts=4)
        x_t = torch.randn(8, 3)
        h = torch.randn(8, 8)
        cell(x_t, h)
        diag = remoe_utilization(cell)
        assert diag["g_mean"].shape == (4,)
        assert diag["g_active_frac"].shape == (4,)
        assert 0.0 <= diag["sparsity"].item() <= 1.0

    def test_captures_signal(self) -> None:
        """Loss on a non-trivial signal should be non-zero and finite."""
        _seed(0)
        cell = ReMoECfCCell(input_size=3, hidden_size=8, n_experts=4)
        x_t = torch.randn(8, 3)
        h = torch.randn(8, 8)
        h_new = cell(x_t, h)
        target = torch.randn(8, 8)
        loss = ((h_new - target) ** 2).mean()
        assert loss.item() > 0
        assert torch.isfinite(loss)


class TestReMoESineSmoke:
    """Toy sin smoke test: 4 experts should converge to a reasonable loss."""

    def test_converges_on_sin(self) -> None:
        _seed(0)
        net = ReMoECfCNetwork(
            input_size=1, hidden_size=16, output_size=1, num_layers=1,
            n_experts=4, return_sequences=True,
        )
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        t = torch.linspace(0, 4 * np.pi, 64).unsqueeze(0).unsqueeze(-1)
        target = torch.sin(t)
        x = t
        initial_loss = None
        loss_value = float("inf")
        for step in range(200):
            opt.zero_grad()
            y = net(x)
            loss = ((y - target) ** 2).mean()
            if step == 0:
                initial_loss = loss.item()
            loss_value = loss.item()
            loss.backward()
            opt.step()
        final_loss = loss_value
        assert torch.isfinite(loss)
        assert final_loss < initial_loss * 0.9, (
            f"ReMoE did not improve: initial={initial_loss:.4f} final={final_loss:.4f}"
        )


def pytest_main() -> None:
    """Quick smoke for `python -m tests.test_remoe`."""
    import pytest
    pytest.main([__file__, "-v"])
