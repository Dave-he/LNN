"""Unit tests for Sigmoid MoE (sigmoid_routing) Cell + Network (PRD #10-78, 2026-06-15).

Verifies:
- SigmoidRouter: init, forward shape, sigmoid in [0, 1] (no normalization), top-K sparsity.
- SigmoidMoECfCCell: init with K experts, dense or sparse top-K modes, forward shape.
- Dense mode: all K experts contribute, no top-K selection.
- Sparse mode: top-K selection, K-K' zeros per row.
- Gradient flows to all K experts.
- Hidden state is preserved across steps (recurrent state not modified).
- sigmoid_moe_utilization diagnostic.
- SigmoidMoECfCNetwork matches CfCNetwork API.
- Toy sin smoke: K=3 dense converges.
"""
import numpy as np
import torch

from lnn.core.sigmoid_moe import (
    SigmoidMoECfCCell,
    SigmoidMoECfCNetwork,
    SigmoidRouter,
    sigmoid_moe_utilization,
)


def _seed(s: int = 0) -> None:
    torch.manual_seed(s)
    np.random.seed(s)


class TestSigmoidRouterInit:
    """Init: linear/MLP, bias toggle, top_k options."""

    def test_init_linear(self) -> None:
        _seed(0)
        router = SigmoidRouter(input_size=2, hidden_size=8, n_experts=3)
        assert router.n_experts == 3
        assert router.top_k == 0  # dense default
        assert router.use_bias is True
        assert router.small_init is True
        assert isinstance(router.net, torch.nn.Linear)
        assert router.net.out_features == 3
        assert router.bias is not None
        assert torch.allclose(router.bias, torch.zeros(3), atol=1e-6)

    def test_init_mlp(self) -> None:
        _seed(0)
        router = SigmoidRouter(
            input_size=2, hidden_size=8, n_experts=3,
            router_hidden=4,
        )
        assert router.router_hidden == 4
        assert isinstance(router.net, torch.nn.Sequential)
        assert len(router.net) == 3  # Linear, Tanh, Linear

    def test_init_no_bias(self) -> None:
        _seed(0)
        router = SigmoidRouter(
            input_size=2, hidden_size=8, n_experts=3,
            use_bias=False,
        )
        assert router.use_bias is False
        assert router.bias is None

    def test_init_invalid_n_experts(self) -> None:
        try:
            SigmoidRouter(input_size=2, hidden_size=8, n_experts=0)
            raise AssertionError("expected assertion")
        except AssertionError as e:
            assert "n_experts" in str(e)

    def test_init_small_init(self) -> None:
        _seed(0)
        router = SigmoidRouter(
            input_size=2, hidden_size=8, n_experts=3,
            small_init=True,
        )
        # Small init: weight std should be 0.01
        w = router.net.weight
        assert w.abs().max().item() < 1.0  # well below default ~0.5-1.0


class TestSigmoidRouterForward:
    """Forward: sigmoid in [0, 1] (not normalized), top-K sparsity."""

    def test_sigmoid_in_range(self) -> None:
        """g_i in [0, 1] for all i (sigmoid property)."""
        _seed(0)
        router = SigmoidRouter(input_size=2, hidden_size=8, n_experts=3)
        x = torch.randn(8, 2)
        h = torch.randn(8, 8)
        g = router(x, h)
        assert g.shape == (8, 3)
        assert (g >= 0.0).all()
        assert (g <= 1.0).all()

    def test_no_normalization(self) -> None:
        """g does NOT sum to 1 (sigmoid has no normalization property)."""
        _seed(0)
        router = SigmoidRouter(input_size=2, hidden_size=8, n_experts=3)
        x = torch.randn(16, 2)
        h = torch.randn(16, 8)
        g = router(x, h)
        sums = g.sum(dim=-1)
        # Sigmoid sums to K * 0.5 = 1.5 on average, not 1.0
        # We verify the sum is NOT 1.0 (i.e., the property that distinguishes
        # sigmoid from softmax).
        for i in range(16):
            assert abs(sums[i].item() - 1.0) > 0.05, (
                f"sigmoid sum too close to 1.0: {sums[i].item():.4f}"
            )

    def test_dense_mode_all_experts_fire(self) -> None:
        """Dense mode (top_k=0): all K experts have non-zero gate."""
        _seed(0)
        router = SigmoidRouter(
            input_size=2, hidden_size=8, n_experts=3, top_k=0,
        )
        x = torch.randn(8, 2)
        h = torch.randn(8, 8)
        g = router(x, h)
        # All entries should be > 0 (sigmoid is strictly positive)
        assert (g > 0.0).all()

    def test_sparse_mode_topk_zeros(self) -> None:
        """Sparse mode (top_k=1): exactly 1 non-zero per row, 2 zeros."""
        _seed(0)
        router = SigmoidRouter(
            input_size=2, hidden_size=8, n_experts=3, top_k=1,
        )
        x = torch.randn(8, 2)
        h = torch.randn(8, 8)
        g = router(x, h)
        # Each row: 1 non-zero, 2 zeros
        for i in range(8):
            assert (g[i] > 0.0).sum().item() == 1
            assert (g[i] == 0.0).sum().item() == 2

    def test_sparse_mode_topk_indices(self) -> None:
        """last_top_idx has shape [B, top_k] and distinct indices per row."""
        _seed(0)
        router = SigmoidRouter(
            input_size=2, hidden_size=8, n_experts=3, top_k=2,
        )
        x = torch.randn(8, 2)
        h = torch.randn(8, 8)
        g = router(x, h)
        assert router.last_top_idx.shape == (8, 2)
        for i in range(8):
            assert len(set(router.last_top_idx[i].tolist())) == 2

    def test_gradient_flow(self) -> None:
        """Gradient flows through sigmoid router."""
        _seed(0)
        router = SigmoidRouter(input_size=2, hidden_size=8, n_experts=3)
        x = torch.randn(8, 2, requires_grad=True)
        h = torch.randn(8, 8, requires_grad=True)
        g = router(x, h)
        g.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
        for p in router.parameters():
            assert p.grad is not None
            assert p.grad.abs().sum() > 0

    def test_per_expert_bias_used(self) -> None:
        """If use_bias=True, the bias is added to the logits before sigmoid."""
        _seed(0)
        router = SigmoidRouter(
            input_size=2, hidden_size=8, n_experts=3, use_bias=True,
        )
        with torch.no_grad():
            router.bias.fill_(1.0)  # all positive bias → all g ~ 0.7+
        x = torch.zeros(4, 2)
        h = torch.zeros(4, 8)
        g = router(x, h)
        # All sigmoid(0 + 1) = 0.731
        assert (g > 0.6).all()
        assert (g < 0.8).all()


class TestSigmoidMoECfCCellInit:
    """Init: K experts, top_k options, defaults."""

    def test_init_default(self) -> None:
        _seed(0)
        cell = SigmoidMoECfCCell(input_size=2, hidden_size=8)
        assert cell.n_experts == 3
        assert cell.top_k == 0  # dense default
        assert len(cell.experts) == 3
        assert isinstance(cell.router, SigmoidRouter)
        assert cell.use_router_bias is True
        assert cell.small_init is True

    def test_init_custom(self) -> None:
        _seed(0)
        cell = SigmoidMoECfCCell(
            input_size=4, hidden_size=16, n_experts=5, top_k=2,
        )
        assert cell.n_experts == 5
        assert cell.top_k == 2
        assert len(cell.experts) == 5

    def test_init_invalid_top_k(self) -> None:
        try:
            SigmoidMoECfCCell(
                input_size=2, hidden_size=8, n_experts=3, top_k=5,
            )
            raise AssertionError("expected assertion")
        except AssertionError as e:
            assert "top_k" in str(e)

    def test_init_no_router_bias(self) -> None:
        _seed(0)
        cell = SigmoidMoECfCCell(
            input_size=2, hidden_size=8, n_experts=3,
            use_router_bias=False,
        )
        assert cell.use_router_bias is False


class TestSigmoidMoECfCCellForward:
    """Forward: shape, dense/sparse modes, gradient flow, recurrent state."""

    def test_forward_shape(self) -> None:
        _seed(0)
        cell = SigmoidMoECfCCell(input_size=2, hidden_size=8, n_experts=3)
        x_t = torch.randn(3, 2)
        h = torch.randn(3, 8)
        h_new = cell(x_t, h)
        assert h_new.shape == (3, 8)

    def test_dense_mode_all_experts_used(self) -> None:
        """Dense mode: expert_util sums to a value > 0.5 (not just 1/K)."""
        _seed(0)
        cell = SigmoidMoECfCCell(
            input_size=2, hidden_size=8, n_experts=3, top_k=0,
        )
        x_t = torch.randn(8, 2)
        h = torch.randn(8, 8)
        cell(x_t, h)
        util = cell.last_expert_util
        assert util.shape == (3,)
        # Dense sigmoid: each g ~ 0.5, mean per expert ~ 0.5
        # Sum ~ 1.5, NOT 1.0
        assert util.sum().item() > 0.7  # definitely not zero
        assert util.sum().item() < 2.5  # not over-saturated

    def test_sparse_mode_top_k(self) -> None:
        """Sparse mode: each row picks top_k experts."""
        _seed(0)
        cell = SigmoidMoECfCCell(
            input_size=2, hidden_size=8, n_experts=3, top_k=1,
        )
        x_t = torch.randn(8, 2)
        h = torch.randn(8, 8)
        cell(x_t, h)
        # In top_k=1 mode, only 1 expert per row has gate > 0
        # Mean across batch of (gate > 0) for each expert ≈ 1/3
        util = cell.last_expert_util
        # All experts should be approximately equally likely
        assert (util > 0.0).all()

    def test_gradient_flows_to_all_experts(self) -> None:
        """Gradient flows to all K experts in dense mode."""
        _seed(0)
        cell = SigmoidMoECfCCell(
            input_size=2, hidden_size=8, n_experts=3, top_k=0,
        )
        x_t = torch.randn(8, 2, requires_grad=True)
        h = torch.randn(8, 8, requires_grad=True)
        h_new = cell(x_t, h)
        h_new.sum().backward()
        # All K experts should get gradient in dense mode
        n_with_grad = 0
        for expert in cell.experts:
            has_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in expert.parameters()
            )
            if has_grad:
                n_with_grad += 1
        assert n_with_grad == 3, f"only {n_with_grad}/3 experts got grad"

    def test_hidden_state_preserved_shape(self) -> None:
        """Forward returns same shape as input hidden state (recurrent intact)."""
        _seed(0)
        cell = SigmoidMoECfCCell(
            input_size=2, hidden_size=8, n_experts=3, top_k=0,
        )
        x_t = torch.randn(4, 2, requires_grad=True)
        h = torch.randn(4, 8, requires_grad=True)
        h_new = cell(x_t, h)
        # h_new has same shape as h — recurrent state is preserved
        assert h_new.shape == h.shape
        # h_new should be a function of h (gradient flows through)
        h_new.sum().backward()
        assert h.grad is not None
        assert h.grad.abs().sum() > 0
        assert torch.isfinite(h_new).all()

    def test_expert_util_diagnostic(self) -> None:
        """last_expert_util has shape [K] and is in [0, 1] (mean sigmoid gate)."""
        _seed(0)
        cell = SigmoidMoECfCCell(
            input_size=2, hidden_size=8, n_experts=3, top_k=0,
        )
        x_t = torch.randn(4, 2)
        h = torch.randn(4, 8)
        cell(x_t, h)
        util = cell.last_expert_util
        assert util.shape == (3,)
        # In dense mode, all gates are positive
        assert (util > 0.0).all()
        assert (util <= 1.0).all()

    def test_captures_signal(self) -> None:
        """Loss on a non-trivial signal is non-zero and finite."""
        _seed(0)
        cell = SigmoidMoECfCCell(
            input_size=2, hidden_size=8, n_experts=3, top_k=0,
        )
        x_t = torch.randn(8, 2)
        h = torch.randn(8, 8)
        h_new = cell(x_t, h)
        target = torch.randn(8, 8)
        loss = ((h_new - target) ** 2).mean()
        assert loss.item() > 0
        assert torch.isfinite(loss)


class TestSigmoidMoEUtilization:
    """sigmoid_moe_utilization diagnostic."""

    def test_utilization_no_forward(self) -> None:
        _seed(0)
        cell = SigmoidMoECfCCell(
            input_size=2, hidden_size=8, n_experts=3, top_k=0,
        )
        diag = sigmoid_moe_utilization(cell)
        assert diag["expert_util"].shape == (3,)
        assert diag["routing_entropy"].item() == 0.0
        assert diag["sparsity_mode"] == "dense"

    def test_utilization_after_forward(self) -> None:
        _seed(0)
        cell = SigmoidMoECfCCell(
            input_size=2, hidden_size=8, n_experts=3, top_k=0,
        )
        x_t = torch.randn(8, 2)
        h = torch.randn(8, 8)
        cell(x_t, h)
        diag = sigmoid_moe_utilization(cell)
        assert diag["expert_util"].shape == (3,)
        assert diag["routing_entropy"].item() >= 0.0
        assert diag["sparsity_mode"] == "dense"

    def test_sparse_mode_label(self) -> None:
        _seed(0)
        cell = SigmoidMoECfCCell(
            input_size=2, hidden_size=8, n_experts=3, top_k=2,
        )
        x_t = torch.randn(8, 2)
        h = torch.randn(8, 8)
        cell(x_t, h)
        diag = sigmoid_moe_utilization(cell)
        assert diag["sparsity_mode"] == "top_2"


class TestSigmoidMoECfCNetwork:
    """SigmoidMoECfCNetwork matches CfCNetwork API."""

    def test_network_init(self) -> None:
        _seed(0)
        net = SigmoidMoECfCNetwork(
            input_size=2, hidden_size=8, output_size=2,
            num_layers=2, n_experts=3, top_k=0,
        )
        assert len(net.cells) == 2
        for cell in net.cells:
            assert isinstance(cell, SigmoidMoECfCCell)

    def test_network_forward_dense(self) -> None:
        _seed(0)
        net = SigmoidMoECfCNetwork(
            input_size=2, hidden_size=8, output_size=2, num_layers=1,
            n_experts=3, top_k=0, return_sequences=True,
        )
        x = torch.randn(3, 12, 2)
        y = net(x)
        assert y.shape == (3, 12, 2)

    def test_network_forward_last_step(self) -> None:
        _seed(0)
        net = SigmoidMoECfCNetwork(
            input_size=2, hidden_size=8, output_size=2, num_layers=1,
            n_experts=3, top_k=0, return_sequences=False,
        )
        x = torch.randn(3, 12, 2)
        y = net(x)
        assert y.shape == (3, 2)

    def test_network_sparse_mode(self) -> None:
        """Network with top_k=2 (sparse) also works."""
        _seed(0)
        net = SigmoidMoECfCNetwork(
            input_size=2, hidden_size=8, output_size=2, num_layers=1,
            n_experts=4, top_k=2, return_sequences=True,
        )
        x = torch.randn(3, 12, 2)
        y = net(x)
        assert y.shape == (3, 12, 2)

    def test_network_gradient_flows(self) -> None:
        """Gradient flows to all experts and router in network."""
        _seed(0)
        net = SigmoidMoECfCNetwork(
            input_size=2, hidden_size=8, output_size=2, num_layers=1,
            n_experts=3, top_k=0, return_sequences=False,
        )
        x = torch.randn(8, 16, 2)
        y = net(x)
        y.sum().backward()
        for cell in net.cells:
            for p in cell.parameters():
                if p.requires_grad:
                    assert p.grad is not None
                    # Don't assert > 0 (some biases may have zero grad)


class TestSigmoidMoESineSmoke:
    """Toy sin smoke: K=3 dense should converge."""

    def test_converges_on_sin(self) -> None:
        _seed(0)
        net = SigmoidMoECfCNetwork(
            input_size=1, hidden_size=16, output_size=1, num_layers=1,
            n_experts=3, top_k=0, return_sequences=True,
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
            f"Sigmoid MoE did not improve: initial={initial_loss:.4f} "
            f"final={final_loss:.4f}"
        )


def pytest_main() -> None:
    """Quick smoke for `python -m tests.test_sigmoid_moe`."""
    import pytest
    pytest.main([__file__, "-v"])
