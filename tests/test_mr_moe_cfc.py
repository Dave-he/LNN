"""Unit tests for MRMoECfCCell and MRMoECfCNetwork (PRD #10-24, 2026-06-14).

Verifies:
- ``n_experts=1`` reduces to a single ``CfCCell``-style forward.
- K-expert router outputs are valid probability distributions (sum=1).
- Gradients flow to every expert (no expert collapse at init).
- Combined with round 76 ``n_tau`` (n_tau_per_expert>1) the K×K' cell
  is forward-stable.
- ``MRMoECfCNetwork`` matches ``CfCNetwork`` API: return_sequences,
  mask, dt.
- Toy sin smoke: K=3 does not catastrophically regress vs K=1.
"""
import numpy as np
import torch

from lnn.core.mr_moe_cfc import MRMoECfCCell, MRMoECfCNetwork


def _seed(s: int = 0) -> None:
    torch.manual_seed(s)
    np.random.seed(s)


class TestMRMoEKOneEquivalence:
    """n_experts=1 should reduce to a single CfCCell with a no-op router."""

    def test_k_1_forward_shape(self) -> None:
        _seed(0)
        cell = MRMoECfCCell(input_size=3, hidden_size=8, n_experts=1)
        x_t = torch.randn(2, 3)
        h = torch.randn(2, 8)
        out = cell(x_t, h, dt=1.0)
        assert out.shape == (2, 8)
        assert torch.isfinite(out).all()

    def test_k_1_router_softmax_is_one(self) -> None:
        """With K=1, softmax of any single logit is 1.0 → pure pass-through."""
        _seed(1)
        cell = MRMoECfCCell(input_size=3, hidden_size=8, n_experts=1)
        x_t = torch.randn(4, 3)
        h = torch.randn(4, 8)
        cell(x_t, h, dt=1.0)
        g = cell.last_g
        assert g.shape == (4, 1)
        assert torch.allclose(g, torch.ones_like(g), atol=1e-6)


class TestMRMoEKThree:
    """K=3 default path: router, expert mix, gradients."""

    def test_k_3_forward_shape(self) -> None:
        _seed(2)
        cell = MRMoECfCCell(input_size=3, hidden_size=12, n_experts=3)
        x_t = torch.randn(4, 3)
        h = torch.randn(4, 12)
        out = cell(x_t, h, dt=1.0)
        assert out.shape == (4, 12)
        assert torch.isfinite(out).all()

    def test_k_3_router_sums_to_1(self) -> None:
        _seed(3)
        cell = MRMoECfCCell(input_size=3, hidden_size=12, n_experts=3)
        x_t = torch.randn(8, 3)
        h = torch.randn(8, 12)
        cell(x_t, h, dt=1.0)
        g = cell.last_g
        assert g.shape == (8, 3)
        assert torch.allclose(g.sum(dim=-1), torch.ones(8), atol=1e-6)
        # All weights in (0, 1).
        assert (g > 0).all() and (g < 1).all()

    def test_k_3_gradient_flows_to_all_experts(self) -> None:
        _seed(4)
        cell = MRMoECfCCell(input_size=3, hidden_size=12, n_experts=3)
        x_t = torch.randn(4, 3)
        h = torch.randn(4, 12)
        out = cell(x_t, h, dt=1.0)
        out.sum().backward()
        for i, expert in enumerate(cell.experts):
            grad_sum = sum(p.grad.abs().sum().item() for p in expert.parameters() if p.grad is not None)
            assert grad_sum > 0, f"expert[{i}] has zero grad"
        # Router also has grad.
        for p in cell.router.parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0

    def test_k_3_router_entropy_positive(self) -> None:
        """At init the router should not be a one-hot collapse (entropy > 0)."""
        _seed(5)
        cell = MRMoECfCCell(input_size=4, hidden_size=8, n_experts=3)
        x_t = torch.randn(16, 4)
        h = torch.randn(16, 8)
        cell(x_t, h, dt=1.0)
        g = cell.last_g
        entropy = -(g * g.clamp_min(1e-12).log()).sum(dim=-1)  # [16]
        # Max entropy for K=3 is log(3)≈1.0986; init entropy should be in (0.1, 1.10).
        assert (entropy > 0.1).all(), f"init router collapsed: entropy min {entropy.min().item():.4f}"
        assert (entropy < 1.10).all(), f"init router suspiciously uniform: entropy max {entropy.max().item():.4f}"


class TestMRMoEWithNtau:
    """Combined MR-MoE K-experts with round 76 n_tau per expert."""

    def test_k_3_with_n_tau_3(self) -> None:
        """K=3 experts × n_tau=3 (3 τ groups per expert) = 9 effective τ groups."""
        _seed(6)
        cell = MRMoECfCCell(
            input_size=3, hidden_size=12, n_experts=3, n_tau_per_expert=3,
            tau_scales=(0.1, 1.0, 10.0),
        )
        x_t = torch.randn(2, 3)
        h = torch.randn(2, 12)
        out = cell(x_t, h, dt=1.0)
        assert out.shape == (2, 12)
        assert torch.isfinite(out).all()
        # All 3 experts should have multi-τ paths active.
        for expert in cell.experts:
            assert expert._multi_tau is True
            assert expert.n_tau == 3

    def test_router_mlp_variant(self) -> None:
        """router_hidden > 0 should use the 2-layer MLP variant."""
        _seed(7)
        cell = MRMoECfCCell(input_size=3, hidden_size=8, n_experts=3, router_hidden=16)
        x_t = torch.randn(2, 3)
        h = torch.randn(2, 8)
        out = cell(x_t, h, dt=1.0)
        assert out.shape == (2, 8)
        # ModuleList of Linear + Tanh + Linear → 3 modules in router Sequential.
        assert len(list(cell.router)) == 3


class TestMRMoEInvalid:
    def test_n_experts_0_raises(self) -> None:
        try:
            MRMoECfCCell(input_size=2, hidden_size=4, n_experts=0)
        except AssertionError:
            return
        raise AssertionError("Expected AssertionError for n_experts=0")


class TestMRMoENetwork:
    """MRMoECfCNetwork should mirror CfCNetwork API."""

    def test_network_k_1_smoke(self) -> None:
        _seed(8)
        net = MRMoECfCNetwork(input_size=3, hidden_size=12, output_size=2, num_layers=2, n_experts=1)
        x = torch.randn(2, 16, 3)
        out = net(x)
        assert out.shape == (2, 16, 2)
        assert torch.isfinite(out).all()

    def test_network_k_3_smoke(self) -> None:
        _seed(9)
        net = MRMoECfCNetwork(input_size=3, hidden_size=12, output_size=2, num_layers=2, n_experts=3)
        x = torch.randn(2, 16, 3)
        out = net(x)
        assert out.shape == (2, 16, 2)

    def test_network_k_3_with_mask(self) -> None:
        _seed(10)
        net = MRMoECfCNetwork(input_size=3, hidden_size=12, output_size=2, num_layers=1, n_experts=3)
        x = torch.randn(2, 10, 3)
        mask = torch.ones(2, 10, 3)
        mask[:, 4:6, :] = 0.0
        out = net(x, mask=mask)
        assert out.shape == (2, 10, 2)
        assert torch.isfinite(out).all()

    def test_network_return_sequences_false(self) -> None:
        _seed(11)
        net = MRMoECfCNetwork(
            input_size=3, hidden_size=12, output_size=2,
            num_layers=1, n_experts=3, return_sequences=False,
        )
        x = torch.randn(2, 16, 3)
        out = net(x)
        assert out.shape == (2, 2)


class TestMRMoESineSmoke:
    """Tiny training smoke test: K=3 should at least match K=1 on a simple sin curve.

    Per the iter#24/35/37 honest-negative pattern, LNNs do not dominate
    LSTM/MLP on toy noise-free sin datasets.  We therefore assert only
    that K=3 is within 2x of K=1 (no catastrophic regression from
    over-parameterisation) and that all seeds converge.
    """

    def test_k_3_converges_on_sin(self) -> None:
        torch.manual_seed(7)
        T = 32
        N = 64
        t = torch.linspace(0, 2 * np.pi, T).unsqueeze(0).expand(N, -1)
        x = torch.sin(t).unsqueeze(-1)
        y = torch.cos(t).unsqueeze(-1)

        def _train(n_experts: int) -> float:
            torch.manual_seed(42)
            net = MRMoECfCNetwork(
                input_size=1, hidden_size=16, output_size=1,
                num_layers=1, n_experts=n_experts,
            )
            opt = torch.optim.Adam(net.parameters(), lr=0.01)
            loss_fn = torch.nn.MSELoss()
            final = 0.0
            for _ in range(30):
                opt.zero_grad()
                pred = net(x)
                loss = loss_fn(pred, y)
                loss.backward()
                opt.step()
                final = float(loss.item())
            return final

        m1 = _train(n_experts=1)
        m3 = _train(n_experts=3)
        # Both should converge to < 0.5 on a simple sin/cos task.
        assert m1 < 0.5, f"K=1 failed to converge: {m1}"
        assert m3 < 0.5, f"K=3 failed to converge: {m3}"
        # K=3 should be within 2x of K=1 (no catastrophic regression from
        # 3-expert over-parameterisation on toy data).
        assert m3 < 2.0 * m1 + 1e-3, f"K=3 ({m3}) is >2x worse than K=1 ({m1})"
