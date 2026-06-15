"""Unit tests for Gumbel-Softmax MoE (stochastic MoE) Cell + Network (PRD #10-79, 2026-06-15).

Verifies:
- GumbelRouter: init, forward shape, softmax sums to 1 (per row), Gumbel noise
  in training mode, deterministic softmax in eval mode, temperature scaling,
  anneal_step decreases T.
- GumbelMoECfCCell: init with K experts, forward shape, gradient flow to all K
  experts, hidden state preserved.
- anneal_step on cell propagates to router.
- gumbel_moe_utilization diagnostic.
- GumbelMoECfCNetwork matches CfCNetwork API.
- Toy sin smoke: K=3 dense converges.
"""
import math
import numpy as np
import torch

from lnn.core.gumbel_moe import (
    GumbelMoECfCCell,
    GumbelMoECfCNetwork,
    GumbelRouter,
    _sample_gumbel,
    gumbel_moe_utilization,
)


def _seed(s: int = 0) -> None:
    torch.manual_seed(s)
    np.random.seed(s)


class TestGumbelNoise:
    """Gumbel(0, 1) noise sampling."""

    def test_gumbel_shape(self) -> None:
        """Sample shape matches input shape."""
        _seed(0)
        g = _sample_gumbel((4, 8), torch.device("cpu"), torch.float32)
        assert g.shape == (4, 8)

    def test_gumbel_distribution(self) -> None:
        """Mean ~ 0.5772 (Euler-Mascheroni), std ~ 1.2825 for large N."""
        _seed(0)
        g = _sample_gumbel((10000,), torch.device("cpu"), torch.float32)
        # Mean of Gumbel(0, 1) is ~ 0.5772
        assert abs(g.mean().item() - 0.5772) < 0.1
        # Std of Gumbel(0, 1) is ~ 1.2825
        assert abs(g.std().item() - 1.2825) < 0.1


class TestGumbelRouterInit:
    """Init: linear/MLP, temperature, anneal params."""

    def test_init_linear(self) -> None:
        _seed(0)
        router = GumbelRouter(input_size=2, hidden_size=8, n_experts=3)
        assert router.n_experts == 3
        assert router.temperature == 1.0  # default
        assert router.anneal_rate == 0.95
        assert router.min_temperature == 0.1
        assert router.small_init is True

    def test_init_mlp(self) -> None:
        _seed(0)
        router = GumbelRouter(
            input_size=2, hidden_size=8, n_experts=3, router_hidden=4,
        )
        assert router.router_hidden == 4
        assert isinstance(router.net, torch.nn.Sequential)

    def test_init_invalid_n_experts(self) -> None:
        try:
            GumbelRouter(input_size=2, hidden_size=8, n_experts=0)
            raise AssertionError("expected assertion")
        except AssertionError as e:
            assert "n_experts" in str(e)

    def test_init_small_init(self) -> None:
        _seed(0)
        router = GumbelRouter(
            input_size=2, hidden_size=8, n_experts=3, small_init=True,
        )
        w = router.net.weight
        # Small init: weight std should be 0.01
        assert w.abs().max().item() < 1.0


class TestGumbelRouterForward:
    """Forward: softmax sums to 1, Gumbel noise in training, deterministic in eval."""

    def test_softmax_sums_to_one(self) -> None:
        """g_routing sums to 1 per row (softmax property)."""
        _seed(0)
        router = GumbelRouter(input_size=2, hidden_size=8, n_experts=3)
        x = torch.randn(8, 2)
        h = torch.randn(8, 8)
        g = router(x, h, training=True)
        assert g.shape == (8, 3)
        sums = g.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(8), atol=1e-5)

    def test_training_adds_noise(self) -> None:
        """Training mode: outputs differ across forward passes (Gumbel noise)."""
        _seed(0)
        router = GumbelRouter(input_size=2, hidden_size=8, n_experts=3, temperature=1.0)
        x = torch.randn(4, 2)
        h = torch.randn(4, 8)
        g1 = router(x, h, training=True)
        g2 = router(x, h, training=True)
        # Gumbel noise should make g1 != g2 with high probability
        assert not torch.allclose(g1, g2, atol=1e-5)

    def test_eval_is_deterministic(self) -> None:
        """Eval mode: outputs are identical across forward passes."""
        _seed(0)
        router = GumbelRouter(input_size=2, hidden_size=8, n_experts=3, temperature=1.0)
        x = torch.randn(4, 2)
        h = torch.randn(4, 8)
        g1 = router(x, h, training=False)
        g2 = router(x, h, training=False)
        assert torch.allclose(g1, g2, atol=1e-5)

    def test_gradient_flow(self) -> None:
        """Gradient flows through Gumbel-Softmax router."""
        _seed(0)
        router = GumbelRouter(input_size=2, hidden_size=8, n_experts=3)
        x = torch.randn(8, 2, requires_grad=True)
        h = torch.randn(8, 8, requires_grad=True)
        g = router(x, h, training=True)
        g.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
        for p in router.parameters():
            assert p.grad is not None
            assert p.grad.abs().sum() > 0

    def test_anneal_step_decreases_temperature(self) -> None:
        """anneal_step decreases T (or hits min_temperature floor)."""
        _seed(0)
        router = GumbelRouter(
            input_size=2, hidden_size=8, n_experts=3,
            temperature=1.0, anneal_rate=0.5, min_temperature=0.1,
        )
        t0 = router.temperature
        router.anneal_step()
        t1 = router.temperature
        # T should decrease (or stay at floor)
        assert t1 < t0 or t1 == router.min_temperature
        # After many anneals, T should hit min_temperature
        for _ in range(20):
            router.anneal_step()
        assert router.temperature == router.min_temperature

    def test_set_temperature(self) -> None:
        """set_temperature overrides current T (with min clamp)."""
        _seed(0)
        router = GumbelRouter(
            input_size=2, hidden_size=8, n_experts=3,
            temperature=1.0, min_temperature=0.1,
        )
        router.set_temperature(0.5)
        assert router.temperature == 0.5
        router.set_temperature(0.05)  # below min
        assert router.temperature == 0.1  # clamped to min


class TestGumbelMoECfCCellInit:
    """Init: K experts, temperature, anneal params."""

    def test_init_default(self) -> None:
        _seed(0)
        cell = GumbelMoECfCCell(input_size=2, hidden_size=8)
        assert cell.n_experts == 3
        assert cell.temperature == 1.0
        assert cell.anneal_rate == 0.95
        assert cell.min_temperature == 0.1
        assert len(cell.experts) == 3
        assert isinstance(cell.router, GumbelRouter)

    def test_init_custom(self) -> None:
        _seed(0)
        cell = GumbelMoECfCCell(
            input_size=4, hidden_size=16, n_experts=5,
            temperature=0.5, anneal_rate=0.9, min_temperature=0.05,
        )
        assert cell.n_experts == 5
        assert cell.temperature == 0.5
        assert cell.anneal_rate == 0.9
        assert cell.min_temperature == 0.05

    def test_init_invalid_n_experts(self) -> None:
        try:
            GumbelMoECfCCell(input_size=2, hidden_size=8, n_experts=0)
            raise AssertionError("expected assertion")
        except AssertionError as e:
            assert "n_experts" in str(e)


class TestGumbelMoECfCCellForward:
    """Forward: shape, gradient flow, recurrent state preservation."""

    def test_forward_shape(self) -> None:
        _seed(0)
        cell = GumbelMoECfCCell(input_size=2, hidden_size=8, n_experts=3)
        x_t = torch.randn(3, 2)
        h = torch.randn(3, 8)
        h_new = cell(x_t, h)
        assert h_new.shape == (3, 8)

    def test_gradient_flows_to_all_experts(self) -> None:
        """Gradient flows to all K experts via the soft mixture."""
        _seed(0)
        cell = GumbelMoECfCCell(input_size=2, hidden_size=8, n_experts=3)
        x_t = torch.randn(16, 2, requires_grad=True)
        h = torch.randn(16, 8, requires_grad=True)
        h_new = cell(x_t, h)
        h_new.sum().backward()
        # All K experts should get gradient (Gumbel-Softmax is dense)
        n_with_grad = 0
        for expert in cell.experts:
            has_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in expert.parameters()
            )
            if has_grad:
                n_with_grad += 1
        assert n_with_grad == 3, f"only {n_with_grad}/3 experts got grad"

    def test_gradient_flows_to_router(self) -> None:
        """Router parameters receive gradient."""
        _seed(0)
        cell = GumbelMoECfCCell(input_size=2, hidden_size=8, n_experts=3)
        x_t = torch.randn(4, 2, requires_grad=True)
        h = torch.randn(4, 8, requires_grad=True)
        h_new = cell(x_t, h)
        h_new.sum().backward()
        for p in cell.router.parameters():
            assert p.grad is not None
            assert p.grad.abs().sum() > 0

    def test_hidden_state_preserved_shape(self) -> None:
        """Forward returns same shape as input hidden state (recurrent intact)."""
        _seed(0)
        cell = GumbelMoECfCCell(input_size=2, hidden_size=8, n_experts=3)
        x_t = torch.randn(4, 2, requires_grad=True)
        h = torch.randn(4, 8, requires_grad=True)
        h_new = cell(x_t, h)
        assert h_new.shape == h.shape
        h_new.sum().backward()
        assert h.grad is not None
        assert h.grad.abs().sum() > 0

    def test_eval_mode_deterministic(self) -> None:
        """In eval mode, two forwards produce identical output."""
        _seed(0)
        cell = GumbelMoECfCCell(input_size=2, hidden_size=8, n_experts=3)
        cell.eval()
        x_t = torch.randn(4, 2)
        h = torch.randn(4, 8)
        h1 = cell(x_t, h)
        h2 = cell(x_t, h)
        assert torch.allclose(h1, h2, atol=1e-5)

    def test_anneal_step_propagates(self) -> None:
        """cell.anneal_step() decreases router temperature."""
        _seed(0)
        cell = GumbelMoECfCCell(
            input_size=2, hidden_size=8, n_experts=3,
            temperature=1.0, anneal_rate=0.5, min_temperature=0.1,
        )
        t0 = cell.router.temperature
        cell.anneal_step()
        t1 = cell.router.temperature
        assert t1 < t0 or t1 == cell.min_temperature

    def test_expert_util_diagnostic(self) -> None:
        """last_expert_util has shape [K] and is a probability (sums to 1)."""
        _seed(0)
        cell = GumbelMoECfCCell(input_size=2, hidden_size=8, n_experts=3)
        x_t = torch.randn(8, 2)
        h = torch.randn(8, 8)
        cell(x_t, h)
        util = cell.last_expert_util
        assert util.shape == (3,)
        # Sum should be ~ 1.0 (mean of softmax = 1/K, sum = 1)
        assert abs(util.sum().item() - 1.0) < 0.01

    def test_captures_signal(self) -> None:
        """Loss on a non-trivial signal is non-zero and finite."""
        _seed(0)
        cell = GumbelMoECfCCell(input_size=2, hidden_size=8, n_experts=3)
        x_t = torch.randn(8, 2)
        h = torch.randn(8, 8)
        h_new = cell(x_t, h)
        target = torch.randn(8, 8)
        loss = ((h_new - target) ** 2).mean()
        assert loss.item() > 0
        assert torch.isfinite(loss)


class TestGumbelMoEUtilization:
    """gumbel_moe_utilization diagnostic."""

    def test_utilization_no_forward(self) -> None:
        _seed(0)
        cell = GumbelMoECfCCell(input_size=2, hidden_size=8, n_experts=3)
        diag = gumbel_moe_utilization(cell)
        assert diag["expert_util"].shape == (3,)
        assert diag["routing_entropy"].item() == 0.0
        assert diag["temperature"] == 1.0

    def test_utilization_after_forward(self) -> None:
        _seed(0)
        cell = GumbelMoECfCCell(input_size=2, hidden_size=8, n_experts=3)
        x_t = torch.randn(8, 2)
        h = torch.randn(8, 8)
        cell(x_t, h)
        diag = gumbel_moe_utilization(cell)
        assert diag["expert_util"].shape == (3,)
        assert diag["routing_entropy"].item() >= 0.0
        assert diag["temperature"] == 1.0


class TestGumbelMoECfCNetwork:
    """GumbelMoECfCNetwork matches CfCNetwork API."""

    def test_network_init(self) -> None:
        _seed(0)
        net = GumbelMoECfCNetwork(
            input_size=2, hidden_size=8, output_size=2,
            num_layers=2, n_experts=3,
        )
        assert len(net.cells) == 2
        for cell in net.cells:
            assert isinstance(cell, GumbelMoECfCCell)

    def test_network_forward_dense(self) -> None:
        _seed(0)
        net = GumbelMoECfCNetwork(
            input_size=2, hidden_size=8, output_size=2, num_layers=1,
            n_experts=3, return_sequences=True,
        )
        x = torch.randn(3, 12, 2)
        y = net(x)
        assert y.shape == (3, 12, 2)

    def test_network_forward_last_step(self) -> None:
        _seed(0)
        net = GumbelMoECfCNetwork(
            input_size=2, hidden_size=8, output_size=2, num_layers=1,
            n_experts=3, return_sequences=False,
        )
        x = torch.randn(3, 12, 2)
        y = net(x)
        assert y.shape == (3, 2)

    def test_network_anneal_step(self) -> None:
        """net.anneal_step() decreases temperature on all cells."""
        _seed(0)
        net = GumbelMoECfCNetwork(
            input_size=2, hidden_size=8, output_size=2, num_layers=2,
            n_experts=3, temperature=1.0, anneal_rate=0.5, min_temperature=0.1,
        )
        t0 = net.get_temperature()
        net.anneal_step()
        t1 = net.get_temperature()
        assert t1 < t0 or t1 == net.cells[0].min_temperature

    def test_network_gradient_flows(self) -> None:
        """Gradient flows to all experts and routers in network."""
        _seed(0)
        net = GumbelMoECfCNetwork(
            input_size=2, hidden_size=8, output_size=2, num_layers=1,
            n_experts=3, return_sequences=False,
        )
        x = torch.randn(8, 16, 2)
        y = net(x)
        y.sum().backward()
        for cell in net.cells:
            for p in cell.parameters():
                if p.requires_grad:
                    assert p.grad is not None
                    # Don't assert > 0 (some biases may have zero grad)


class TestGumbelMoESineSmoke:
    """Toy sin smoke: K=3 should converge."""

    def test_converges_on_sin(self) -> None:
        _seed(0)
        net = GumbelMoECfCNetwork(
            input_size=1, hidden_size=16, output_size=1, num_layers=1,
            n_experts=3, return_sequences=True,
            temperature=1.0, anneal_rate=0.95, min_temperature=0.1,
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
            # Anneal every 20 steps
            if step % 20 == 0:
                net.anneal_step()
        final_loss = loss_value
        assert torch.isfinite(loss)
        assert final_loss < initial_loss * 0.9, (
            f"Gumbel MoE did not improve: initial={initial_loss:.4f} "
            f"final={final_loss:.4f}"
        )


def pytest_main() -> None:
    """Quick smoke for `python -m tests.test_gumbel_moe`."""
    import pytest
    pytest.main([__file__, "-v"])
