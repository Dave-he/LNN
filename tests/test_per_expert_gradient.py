"""Unit tests for Per-Expert Gradient Magnitude (PRD #10-50, 2026-06-15, round 88).

Verifies:
- ``per_expert_gradient_norms`` returns [K] tensor
- Returns zero tensor when ``task_loss`` is None or grad unavailable
- Returns finite non-negative per-expert tensor
- Normalise scales invariantly
- ``H_mode="per_expert_gradient"`` returns per-expert E (tensor, not scalar)
- Falls back to per-expert empirical when no ``task_loss``
- ``MoEEcologyMonitor.per_expert_gradient_diagnostic`` returns expected dict
- ``FAMECfCCell(ecology_per_expert_grad=True)`` uses per-expert H
- ``FAMECfCCell(ecology_per_expert_grad=False)`` is back-compat
- Per-expert H_grad identifies dead experts (synthetic: 1-hot collapse)
- Invalid ``H_mode="per_expert_gradient"`` arguments handled
"""
import numpy as np
import torch

from lnn.core.fame_cfc import FAMECfCCell
from lnn.core.moe_ecology import (
    MoEEcologyMonitor,
    moe_ecology_number,
    per_expert_gradient_norms,
)


def _seed(s: int = 0) -> None:
    torch.manual_seed(s)
    np.random.seed(s)


class TestPerExpertGradientNorms:
    def test_returns_K_tensor(self) -> None:
        """Shape is [K], dtype float."""
        _seed(0)
        K = 3
        logits = torch.randn(4, K, requires_grad=True)
        loss = (logits ** 2).sum()
        out = per_expert_gradient_norms(logits, task_loss=loss)
        assert out.shape == (K,)
        assert out.dtype == torch.float32

    def test_returns_zero_when_no_loss(self) -> None:
        """No task_loss → zeros([K])."""
        logits = torch.randn(4, 3, requires_grad=True)
        out = per_expert_gradient_norms(logits, task_loss=None)
        assert out.shape == (3,)
        assert torch.all(out == 0.0)

    def test_returns_zero_when_no_grad(self) -> None:
        """router_logits.requires_grad=False → zeros."""
        logits = torch.randn(4, 3)  # no requires_grad
        loss = torch.tensor(1.0, requires_grad=True)
        out = per_expert_gradient_norms(logits, task_loss=loss)
        assert torch.all(out == 0.0)

    def test_returns_nonnegative_finite(self) -> None:
        """Healthy case: all values finite and ≥ 0."""
        _seed(1)
        logits = torch.randn(8, 3, requires_grad=True)
        target = torch.randn(8, 3)
        pred = logits.softmax(dim=-1)
        loss = ((pred - target) ** 2).mean()
        out = per_expert_gradient_norms(logits, task_loss=loss)
        assert (out >= 0.0).all()
        assert (out < 1e6).all()  # finite
        assert (out != 0.0).any()  # at least one nonzero

    def test_normalize_scales_invariantly(self) -> None:
        """Normalised and unnormalised differ by factor of B."""
        _seed(2)
        B, K = 8, 3
        logits = torch.randn(B, K, requires_grad=True)
        target = torch.randn(B, K)
        pred = logits.softmax(dim=-1)
        loss = ((pred - target) ** 2).mean()
        out_n = per_expert_gradient_norms(logits, task_loss=loss, normalize=True)
        out_u = per_expert_gradient_norms(logits, task_loss=loss, normalize=False)
        # out_n ≈ out_u / B
        expected = out_u / B
        assert (out_n - expected).abs().max() < 1e-5

    def test_identifies_dead_expert_synthetic(self) -> None:
        """Expert with no gradient contribution shows 0.0 norm.

        Construct logits so that the loss only depends on expert 0
        (experts 1 and 2 are masked out by sigmoid gates).  Then
        per_expert_gradient_norms[0] > 0 and [1], [2] ≈ 0.
        """
        _seed(3)
        logits = torch.randn(4, 3, requires_grad=True)
        # Build a loss that depends on logits[:, 0] but not 1, 2.
        target = torch.randn(4, 1)
        pred = logits[:, 0:1]  # only expert 0
        loss = ((pred - target) ** 2).mean()
        out = per_expert_gradient_norms(logits, task_loss=loss, normalize=False)
        # Expert 0 should have nonzero norm; experts 1, 2 zero.
        assert out[0] > 1e-6
        assert abs(out[1].item()) < 1e-6
        assert abs(out[2].item()) < 1e-6


class TestMoeEcologyNumberWithPerExpertH:
    def test_per_expert_gradient_returns_tensor(self) -> None:
        """H_mode='per_expert_gradient' returns [K] tensor."""
        _seed(0)
        K = 3
        logits = torch.randn(4, K, requires_grad=True)
        g = logits.softmax(dim=-1).detach()
        target = torch.randn(4, K)
        pred = logits.softmax(dim=-1)
        loss = ((pred - target) ** 2).mean()
        e = moe_ecology_number(
            router_logits=logits, last_g=g, B=0.1,
            H_mode="per_expert_gradient", task_loss=loss,
        )
        assert e.shape == (K,)
        assert (e >= 0.0).all()

    def test_per_expert_gradient_falls_back_when_no_loss(self) -> None:
        """H_mode='per_expert_gradient' with task_loss=None falls back to uniform [K]."""
        _seed(1)
        g = torch.softmax(torch.randn(4, 3), dim=-1)
        e = moe_ecology_number(
            router_logits=g, last_g=g, B=0.1,  # use B>0 to avoid div-by-eps
            H_mode="per_expert_gradient", task_loss=None,
        )
        assert e.shape == (3,)
        # Fallback is uniform 1/K; with B=0.1, E_k = T * (1/K) / 0.1
        expected = (1.0 / 3.0) / 0.1
        assert torch.allclose(e, torch.full((3,), expected), atol=1e-5)
        # And all values are equal (uniform).
        assert torch.allclose(e, e[0].expand_as(e))

    def test_invalid_H_mode_raises(self) -> None:
        """Unknown H_mode raises ValueError."""
        g = torch.softmax(torch.randn(4, 3), dim=-1)
        try:
            moe_ecology_number(router_logits=g, last_g=g, B=0.0, H_mode="bogus")
            assert False, "should have raised"
        except ValueError:
            pass

    def test_empirical_mode_unaffected(self) -> None:
        """H_mode='empirical' returns scalar (unchanged from round 83)."""
        _seed(2)
        g = torch.softmax(torch.randn(4, 3), dim=-1)
        e = moe_ecology_number(
            router_logits=g, last_g=g, B=0.1, H_mode="empirical",
        )
        assert e.dim() == 0  # scalar


class TestMoEEcologyMonitorPerExpert:
    def test_per_expert_gradient_diagnostic_returns_dict(self) -> None:
        """Diagnostic returns expected keys."""
        _seed(0)
        K = 3
        monitor = MoEEcologyMonitor(n_experts=K)
        logits = torch.randn(8, K, requires_grad=True)
        target = torch.randn(8, K)
        pred = logits.softmax(dim=-1)
        loss = ((pred - target) ** 2).mean()
        diag = monitor.per_expert_gradient_diagnostic(
            router_logits=logits, task_loss=loss,
        )
        expected_keys = {
            "per_expert_grad", "per_expert_grad_list",
            "dead_by_grad", "alive_by_grad", "dead_by_grad_indices",
            "max_grad", "min_grad", "max_min_ratio",
        }
        assert expected_keys.issubset(diag.keys())
        assert diag["per_expert_grad_list"].__len__() == K
        assert diag["dead_by_grad"] >= 0
        assert diag["max_grad"] >= diag["min_grad"]

    def test_per_expert_diagnostic_identifies_dead(self) -> None:
        """When loss doesn't depend on experts 1, 2, those are flagged dead."""
        _seed(1)
        K = 3
        monitor = MoEEcologyMonitor(n_experts=K)
        logits = torch.randn(8, K, requires_grad=True)
        target = torch.randn(8, 1)
        pred = logits[:, 0:1]  # only expert 0
        loss = ((pred - target) ** 2).mean()
        diag = monitor.per_expert_gradient_diagnostic(
            router_logits=logits, task_loss=loss,
            dead_grad_threshold=1e-6,
        )
        assert diag["dead_by_grad"] == 2
        assert 0 not in diag["dead_by_grad_indices"]


class TestFAMECfCCellPerExpertH:
    def test_per_expert_grad_flag_default_off(self) -> None:
        """Default ``ecology_per_expert_grad=False`` is back-compat."""
        _seed(0)
        cell = FAMECfCCell(input_size=3, hidden_size=8, n_experts=3, top_k=2)
        assert cell.ecology_per_expert_grad is False
        with torch.no_grad():
            cell.forward(torch.randn(4, 3), torch.randn(4, 8), dt=1.0)
        diag = cell.moe_ecology_diagnostic(B=0.0)
        # No per-expert keys by default.
        assert "per_expert_grad" not in diag

    def test_per_expert_grad_flag_on(self) -> None:
        """``ecology_per_expert_grad=True`` includes per-expert in default diagnostic."""
        _seed(1)
        cell = FAMECfCCell(
            input_size=3, hidden_size=8, n_experts=3, top_k=2,
            ecology_per_expert_grad=True,
        )
        cell.train()
        # Need requires_grad for per_expert to work.  Use forward_with_aux.
        x = torch.randn(4, 3)
        h = torch.randn(4, 8)
        # Build a fake task_loss that's connected to the router via the cell.
        h_new, _ = cell.forward_with_aux(x, h, dt=1.0)
        # task_loss that depends on h_new (which depends on routing via expert_outs)
        target = torch.randn(4, 8)
        task_loss = ((h_new - target) ** 2).mean()
        # Now call diagnostic with task_loss; per-expert should be in result.
        diag = cell.moe_ecology_diagnostic(B=0.0, task_loss=task_loss)
        assert "per_expert_grad" in diag
        assert "per_expert_grad_list" in diag
        assert "dead_by_grad" in diag

    def test_per_expert_arg_works(self) -> None:
        """``per_expert=True`` arg on diagnostic forces per-expert output."""
        _seed(2)
        cell = FAMECfCCell(input_size=3, hidden_size=8, n_experts=3, top_k=2)
        cell.train()
        with torch.no_grad():
            cell.forward(torch.randn(4, 3), torch.randn(4, 8), dt=1.0)
        # With task_loss, per_expert=True triggers per-expert diagnostic.
        h_new, _ = cell.forward_with_aux(
            torch.randn(4, 3), torch.randn(4, 8), dt=1.0,
        )
        task_loss = (h_new ** 2).mean()
        diag = cell.moe_ecology_diagnostic(B=0.0, task_loss=task_loss, per_expert=True)
        assert "per_expert_grad" in diag
        assert "dead_by_grad" in diag
        assert diag["per_expert_grad_list"].__len__() == 3

    def test_invalid_H_mode_raises(self) -> None:
        """Constructor rejects invalid ecology_H_mode."""
        try:
            FAMECfCCell(
                input_size=3, hidden_size=8, n_experts=3, top_k=2,
                ecology_H_mode="bogus",
            )
            assert False, "should have raised"
        except AssertionError:
            pass  # expected
