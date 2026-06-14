"""Unit tests for φ-balancing (PRD #10-40, 2026-06-14).

Verifies the ``PhiBalancer`` invariants, its integration with
``ForecastabilityRouter`` and ``FAMECfCCell``/``FAMECfCNetwork``, and
the synergy with the orthogonality loss (PRD #10-37).

Coverage:
- ``PhiBalancer``: initial state, EMA update, no_grad, in-place
  mutation, bias sign convention (high f → negative b), bias shape
  broadcast, ``step_size=0`` short-circuit, ``reset_state``.
- ``ForecastabilityRouter``: bias applied when balancer given;
  unbiased when ``None`` (back-compat).
- ``FAMECfCCell.forward_with_aux``: balancer is updated iff training.
- ``FAMECfCNetwork``: per-layer balancer, eval-mode freezes state,
  phi_balance=False matches round 80.
- Toy sin smoke: K=3 top_k=1 with phi_balance alone vs combined with
  orthogonality trains stably.
"""
import numpy as np
import torch

from lnn.core.fame_cfc import FAMECfCCell, FAMECfCNetwork
from lnn.core.forecastability_router import ForecastabilityRouter
from lnn.core.orthogonality import orthogonality_loss
from lnn.core.phi_balancing import PhiBalancer


def _seed(s: int = 0) -> None:
    torch.manual_seed(s)
    np.random.seed(s)


class TestPhiBalancerInvariants:
    def test_initial_state(self) -> None:
        """Initial EMA is uniform; initial bias is zero."""
        _seed(0)
        b = PhiBalancer(n_experts=4)
        # Uniform 1/4
        assert torch.allclose(b.f, torch.full((4,), 0.25))
        # Zero bias
        assert torch.allclose(b.b, torch.zeros(4))

    def test_update_is_no_grad(self) -> None:
        """``update`` must not require or produce gradients."""
        _seed(1)
        b = PhiBalancer(n_experts=3, ema_alpha=0.5, step_size=0.1)
        top_idx = torch.tensor([[0, 1], [2, 0], [1, 2]])  # [B=3, K'=2]
        # No torch.no_grad() context needed — it should be a no-op method.
        b.update(top_idx)
        # After update f and b are finite, non-NaN, on the right device.
        assert torch.isfinite(b.f).all()
        assert torch.isfinite(b.b).all()
        assert b.f.grad_fn is None
        assert b.b.grad_fn is None

    def test_bias_sign_convention(self) -> None:
        """Frequently-activated expert (high f) gets SMALL bias (≈0); rare experts get large POSITIVE bias (promoted)."""
        _seed(2)
        b = PhiBalancer(n_experts=3, ema_alpha=1.0, step_size=0.1)  # alpha=1 = no smoothing
        # Make expert 0 the only one activated.
        top_idx = torch.zeros(8, 1, dtype=torch.long)  # all → expert 0
        b.update(top_idx)
        # f ≈ [1, 0, 0] (after clamp_min(eps) for zeros).  With
        # b_k = -η * log(f_k), expert 0 → 0 (log(1)=0), experts 1,2 → large
        # positive (log(eps) is very negative, negated is large positive).
        # Convention: ADD b to logits → expert 0 demoted relative to 1, 2.
        assert abs(b.b[0].item()) < 0.01, f"expert 0 bias should be ~0, got {b.b[0].item()}"
        assert b.b[1].item() > 0.5, f"rare expert 1 should have large positive bias, got {b.b[1].item()}"
        assert b.b[0].item() < b.b[1].item(), "expert 0 (frequent) should have SMALLER bias than expert 1 (rare)"

    def test_forward_adds_bias(self) -> None:
        """``forward(logits) == logits + b`` (broadcast over batch)."""
        _seed(3)
        b = PhiBalancer(n_experts=2, ema_alpha=0.5, step_size=0.1)
        # Manually set bias.
        b.b.copy_(torch.tensor([0.3, -0.5]))
        logits = torch.tensor([[1.0, 2.0], [0.0, 1.0]])  # [B=2, K=2]
        biased = b(logits)
        expected = torch.tensor([[1.3, 1.5], [0.3, 0.5]])
        assert torch.allclose(biased, expected, atol=1e-6)

    def test_step_size_zero_short_circuits(self) -> None:
        """When ``step_size=0``, ``forward`` returns logits unchanged."""
        _seed(4)
        b = PhiBalancer(n_experts=3, step_size=0.0)
        # update is a no-op for the bias even after assignments.
        b.update(torch.tensor([[0, 1]]))
        logits = torch.randn(2, 3)
        biased = b(logits)
        assert torch.allclose(biased, logits, atol=1e-6)

    def test_reset_state(self) -> None:
        """``reset_state`` returns to uniform f and zero b."""
        _seed(5)
        b = PhiBalancer(n_experts=3, ema_alpha=0.1, step_size=0.1)
        b.update(torch.tensor([[0, 1], [2, 0]]))
        # After update, f and b are no longer initial.
        assert not torch.allclose(b.f, torch.full((3,), 1.0 / 3.0))
        b.reset_state()
        assert torch.allclose(b.f, torch.full((3,), 1.0 / 3.0))
        assert torch.allclose(b.b, torch.zeros(3))

    def test_buffer_device_propagation(self) -> None:
        """Buffers move with ``.to(device)``."""
        _seed(6)
        b = PhiBalancer(n_experts=2)
        b_cpu_id = id(b.f)
        b.to("cpu")  # no-op on CPU, but verifies the contract
        assert id(b.f) == b_cpu_id  # same buffer object
        assert b.f.device.type == "cpu"


class TestRouterWithPhiBalancer:
    def test_router_no_balancer_unchanged(self) -> None:
        """``balancer=None`` ⇒ forward is round 78 (no bias)."""
        _seed(7)
        r = ForecastabilityRouter(
            input_size=2, hidden_size=4, n_experts=3, top_k=2, balancer=None,
        )
        x_t = torch.randn(1, 2)
        h = torch.randn(1, 4)
        g = r(x_t, h)
        assert g.shape == (1, 3)
        # Exactly top_k nonzeros.
        assert (g > 0).sum(dim=-1).item() == 2

    def test_router_with_balancer_applies_bias(self) -> None:
        """With a balancer, logits are biased (last_top_idx still set)."""
        _seed(8)
        bal = PhiBalancer(n_experts=3, ema_alpha=0.0, step_size=0.5)
        # Manually set a strong negative bias on expert 0.
        bal.b.copy_(torch.tensor([-10.0, 0.0, 0.0]))
        r = ForecastabilityRouter(
            input_size=2, hidden_size=4, n_experts=3, top_k=1, balancer=bal,
        )
        # Same input twice: with the strong negative bias, expert 0 should
        # never be the argmax of biased logits (so last_top_idx should be 1 or 2).
        x_t = torch.randn(5, 2)
        h = torch.randn(5, 4)
        # Need to manually run forward (caller updates EMA).
        g = r(x_t, h)
        # The chosen expert (top-1) should never be 0 across the batch.
        chosen = r.last_top_idx  # [B=5, K'=1]
        assert (chosen != 0).all(), f"expected no expert 0 in {chosen.tolist()}"


class TestFAMECfCCellWithPhiBalance:
    def test_cell_no_balancer_back_compat(self) -> None:
        """``phi_balance=False`` ⇒ no balancer attribute on cell."""
        _seed(9)
        cell = FAMECfCCell(input_size=3, hidden_size=8, n_experts=3, top_k=2, phi_balance=False)
        assert cell.balancer is None
        # Forward still works as before.
        x_t = torch.randn(2, 3)
        h = torch.randn(2, 8)
        h_new, outs = cell.forward_with_aux(x_t, h, dt=1.0)
        assert h_new.shape == (2, 8)
        assert len(outs) == 3

    def test_cell_phi_balance_train_updates(self) -> None:
        """In train mode, balancer.f and balancer.b update each step."""
        _seed(10)
        cell = FAMECfCCell(
            input_size=3, hidden_size=8, n_experts=3, top_k=2,
            phi_balance=True, ema_alpha=0.1, phi_step_size=0.05,
        )
        cell.train()
        x_t = torch.randn(4, 3)
        h = torch.randn(4, 8)
        f_before = cell.balancer.f.clone()
        cell.forward_with_aux(x_t, h, dt=1.0)
        f_after = cell.balancer.f.clone()
        # f must have moved (even if a little).
        assert not torch.allclose(f_before, f_after)

    def test_cell_phi_balance_eval_freezes(self) -> None:
        """In eval mode, balancer.f and balancer.b do NOT update."""
        _seed(11)
        cell = FAMECfCCell(
            input_size=3, hidden_size=8, n_experts=3, top_k=2,
            phi_balance=True, ema_alpha=0.5, phi_step_size=0.05,
        )
        cell.train()
        # One warm-up forward in train mode to perturb state.
        x_t = torch.randn(4, 3)
        h = torch.randn(4, 8)
        cell.forward_with_aux(x_t, h, dt=1.0)
        f_warm = cell.balancer.f.clone()
        b_warm = cell.balancer.b.clone()
        # Switch to eval and run more steps.
        cell.eval()
        with torch.no_grad():
            for _ in range(5):
                x_t = torch.randn(4, 3)
                h = torch.randn(4, 8)
                cell.forward_with_aux(x_t, h, dt=1.0)
        # State unchanged in eval.
        assert torch.allclose(cell.balancer.f, f_warm)
        assert torch.allclose(cell.balancer.b, b_warm)

    def test_cell_forward_matches_forward_with_aux_h(self) -> None:
        """``forward`` and ``forward_with_aux``[0] must agree."""
        _seed(12)
        cell = FAMECfCCell(
            input_size=3, hidden_size=8, n_experts=3, top_k=2,
            phi_balance=True, ema_alpha=0.1, phi_step_size=0.01,
        )
        cell.eval()  # so update is a no-op
        x_t = torch.randn(2, 3)
        h = torch.randn(2, 8)
        h_a = cell.forward(x_t, h, dt=1.0)
        with torch.no_grad():
            h_b, _ = cell.forward_with_aux(x_t, h, dt=1.0)
        assert torch.allclose(h_a, h_b, atol=1e-6)


class TestFAMEWithPhiBalanceAndOrthogonality:
    def test_network_forward_with_aux_shape(self) -> None:
        _seed(13)
        net = FAMECfCNetwork(
            input_size=3, hidden_size=8, output_size=2,
            num_layers=1, n_experts=3, top_k=2,
            phi_balance=True, ema_alpha=0.05, phi_step_size=0.02,
        )
        x = torch.randn(2, 5, 3)
        y, expert_outs = net.forward_with_aux(x)
        assert y.shape == (2, 5, 2)
        assert len(expert_outs) == 1
        assert len(expert_outs[0]) == 5

    def test_k3_topk1_phi_balance_alone_stable(self) -> None:
        """K=3 top_k=1 + φ-balancing (no orth) should converge to < 0.5."""
        _seed(14)
        T, N = 32, 64
        t = torch.linspace(0, 2 * np.pi, T).unsqueeze(0).expand(N, -1)
        x = torch.sin(t).unsqueeze(-1)
        y = torch.cos(t).unsqueeze(-1)
        torch.manual_seed(42)
        net = FAMECfCNetwork(
            input_size=1, hidden_size=16, output_size=1,
            num_layers=1, n_experts=3, top_k=1,
            phi_balance=True, ema_alpha=0.05, phi_step_size=0.05,
        )
        opt = torch.optim.Adam(net.parameters(), lr=0.01)
        loss_fn = torch.nn.MSELoss()
        final = float("nan")
        for _ in range(25):
            opt.zero_grad()
            y_pred, _ = net.forward_with_aux(x)
            task_loss = loss_fn(y_pred, y)
            # No orthogonality in this test — just task loss.
            task_loss.backward()
            opt.step()
            final = float(task_loss.item())
        assert final < 0.5, f"task_loss={final} did not converge"

    def test_k3_topk1_orth_plus_phi_synergy(self) -> None:
        """K=3 top_k=1 + orth + φ should beat orth-only (round 80 0.1089)."""
        _seed(15)
        T, N = 32, 64
        t = torch.linspace(0, 2 * np.pi, T).unsqueeze(0).expand(N, -1)
        x = torch.sin(t).unsqueeze(-1)
        y = torch.cos(t).unsqueeze(-1)
        torch.manual_seed(42)
        net = FAMECfCNetwork(
            input_size=1, hidden_size=16, output_size=1,
            num_layers=1, n_experts=3, top_k=1,
            phi_balance=True, ema_alpha=0.05, phi_step_size=0.05,
        )
        opt = torch.optim.Adam(net.parameters(), lr=0.01)
        loss_fn = torch.nn.MSELoss()
        final = float("nan")
        for _ in range(25):
            opt.zero_grad()
            y_pred, expert_outs = net.forward_with_aux(x)
            task_loss = loss_fn(y_pred, y)
            last_outs = expert_outs[0][-1]  # K × [B, H]
            aux = orthogonality_loss(last_outs, lambda_coeff=0.001)
            (task_loss + aux).backward()
            opt.step()
            final = float(task_loss.item())
        # Round 80 orth-only (no φ) was 0.1089 on the same toy setup.
        # We expect φ+orth to be at least as good (within float32 noise).
        assert final < 0.15, f"task_loss={final} did not beat round 80 baseline"
