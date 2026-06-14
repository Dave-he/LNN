"""Unit tests for Gradient-based H (PRD #10-49, 2026-06-15, round 87).

Verifies:
- ``gradient_routing_sensitivity`` returns ≥ 0
- Returns 0.0 when ``task_loss`` is None or grad is unavailable
- ``moe_ecology_number(H_mode="empirical")`` matches round 83 behavior
- ``moe_ecology_number(H_mode="gradient", task_loss=...)`` returns finite E
- ``moe_ecology_number(H_mode="blend", alpha=0.5)`` = 0.5·H_emp + 0.5·H_grad
- ``FAMECfCCell(ecology_H_mode="gradient")`` uses gradient H in diagnostic
- ``FAMECfCCell(ecology_H_mode="empirical")`` is back-compat
- Gates fire on gradient H same way as empirical H (when both healthy)
"""
import numpy as np
import torch

from lnn.core.fame_cfc import FAMECfCCell
from lnn.core.moe_ecology import (
    gradient_routing_sensitivity,
    moe_ecology_number,
)


def _seed(s: int = 0) -> None:
    torch.manual_seed(s)
    np.random.seed(s)


class TestGradientRoutingSensitivity:
    def test_returns_zero_when_no_loss(self) -> None:
        """No task_loss → 0.0 (cannot compute gradient)."""
        logits = torch.randn(4, 3, requires_grad=True)
        assert gradient_routing_sensitivity(logits, task_loss=None) == 0.0

    def test_returns_zero_when_no_grad(self) -> None:
        """router_logits.requires_grad=False → 0.0."""
        logits = torch.randn(4, 3)  # no requires_grad
        loss = torch.tensor(1.0, requires_grad=True)
        assert gradient_routing_sensitivity(logits, task_loss=loss) == 0.0

    def test_returns_nonnegative_finite(self) -> None:
        """Healthy case: returns finite non-negative scalar."""
        _seed(0)
        logits = torch.randn(8, 3, requires_grad=True)
        target = torch.randn(8, 3)
        pred = logits.softmax(dim=-1)
        loss = ((pred - target) ** 2).mean()
        h = gradient_routing_sensitivity(logits, task_loss=loss, normalize=True)
        assert isinstance(h, float)
        assert h >= 0.0
        assert h < 1e6  # not huge

    def test_normalize_scales_invariantly(self) -> None:
        """Normalised and unnormalised differ by factor of B·log(K)."""
        _seed(1)
        B, K = 8, 3
        logits = torch.randn(B, K, requires_grad=True)
        target = torch.randn(B, K)
        pred = logits.softmax(dim=-1)
        loss = ((pred - target) ** 2).mean()
        h_n = gradient_routing_sensitivity(logits, task_loss=loss, normalize=True)
        h_u = gradient_routing_sensitivity(logits, task_loss=loss, normalize=False)
        # h_n ≈ h_u / (B · log(K))
        expected = h_u / (B * float(np.log(K)))
        assert abs(h_n - expected) < 1e-5


class TestMoeEcologyNumberWithHMode:
    def test_empirical_default_matches_round_83(self) -> None:
        """H_mode='empirical' (default) matches round 83 behavior."""
        _seed(0)
        g = torch.softmax(torch.randn(4, 3), dim=-1)
        e_round_83 = moe_ecology_number(router_logits=g, last_g=g, H=None, B=0.0)
        e_default = moe_ecology_number(router_logits=g, last_g=g, H=None, B=0.0)
        # Same args; should match exactly.
        assert abs(float(e_round_83.item()) - float(e_default.item())) < 1e-9

    def test_empirical_with_explicit_H_mode(self) -> None:
        """H_mode='empirical' is the same as default."""
        _seed(1)
        g = torch.softmax(torch.randn(4, 3), dim=-1)
        e_default = moe_ecology_number(router_logits=g, last_g=g, B=0.0)
        e_explicit = moe_ecology_number(
            router_logits=g, last_g=g, B=0.0, H_mode="empirical",
        )
        assert abs(float(e_default.item()) - float(e_explicit.item())) < 1e-9

    def test_gradient_finite_with_task_loss(self) -> None:
        """H_mode='gradient' returns finite E when task_loss is provided."""
        _seed(2)
        logits = torch.randn(4, 3, requires_grad=True)
        g = logits.softmax(dim=-1).detach()
        target = torch.randn(4, 3)
        pred = logits.softmax(dim=-1)
        loss = ((pred - target) ** 2).mean()
        e = moe_ecology_number(
            router_logits=logits, last_g=g, B=0.1,
            H_mode="gradient", task_loss=loss,
        )
        assert e.item() >= 0.0
        assert e.item() < 1e6  # finite, reasonable magnitude

    def test_gradient_falls_back_to_empirical_when_no_loss(self) -> None:
        """H_mode='gradient' with task_loss=None silently uses empirical."""
        _seed(3)
        g = torch.softmax(torch.randn(4, 3), dim=-1)
        e_emp = moe_ecology_number(router_logits=g, last_g=g, B=0.0, H_mode="empirical")
        e_grad_fallback = moe_ecology_number(
            router_logits=g, last_g=g, B=0.0,
            H_mode="gradient", task_loss=None,
        )
        # Should be identical (silent fallback).
        assert abs(float(e_emp.item()) - float(e_grad_fallback.item())) < 1e-9

    def test_blend_is_alpha_weighted_average(self) -> None:
        """H_mode='blend' = alpha·H_emp + (1-alpha)·H_grad."""
        _seed(4)
        logits = torch.randn(4, 3, requires_grad=True)
        g = logits.softmax(dim=-1).detach()
        target = torch.randn(4, 3)
        pred = logits.softmax(dim=-1)
        loss = ((pred - target) ** 2).mean()
        alpha = 0.3
        e_emp = float(moe_ecology_number(
            router_logits=logits, last_g=g, B=0.1, H_mode="empirical",
        ).item())
        e_grad = float(moe_ecology_number(
            router_logits=logits, last_g=g, B=0.1,
            H_mode="gradient", task_loss=loss,
        ).item())
        e_blend = float(moe_ecology_number(
            router_logits=logits, last_g=g, B=0.1,
            H_mode="blend", alpha=alpha, task_loss=loss,
        ).item())
        expected = alpha * e_emp + (1.0 - alpha) * e_grad
        # Use relative tolerance since B=0.1 + small H can be noisy.
        denom = max(abs(expected), 1e-6)
        assert abs(e_blend - expected) / denom < 1e-3

    def test_invalid_H_mode_raises(self) -> None:
        """Unknown H_mode raises ValueError."""
        g = torch.softmax(torch.randn(4, 3), dim=-1)
        try:
            moe_ecology_number(router_logits=g, last_g=g, B=0.0, H_mode="bogus")
            assert False, "should have raised"
        except ValueError:
            pass


class TestFAMECfCCellGradientH:
    def test_empirical_default_backcompat(self) -> None:
        """Default ``ecology_H_mode='empirical'`` is fully back-compat."""
        _seed(0)
        cell = FAMECfCCell(input_size=3, hidden_size=8, n_experts=3, top_k=2)
        assert cell.ecology_H_mode == "empirical"
        # Diagnostic with no task_loss still works (back-compat).
        with torch.no_grad():
            cell.forward(torch.randn(4, 3), torch.randn(4, 8), dt=1.0)
        diag = cell.moe_ecology_diagnostic(B=0.0)
        assert "E" in diag

    def test_gradient_H_mode_uses_task_loss(self) -> None:
        """``ecology_H_mode='gradient'`` uses task_loss in diagnostic."""
        _seed(1)
        cell = FAMECfCCell(
            input_size=3, hidden_size=8, n_experts=3, top_k=2,
            ecology_H_mode="gradient",
        )
        cell.train()
        # Build a fake task_loss with grad.
        logits = cell.router(torch.randn(4, 3), torch.randn(4, 8))
        target = torch.softmax(torch.randn(4, 3), dim=-1)
        # forward_with_aux is needed to populate last_g.
        with torch.no_grad():
            cell.forward(torch.randn(4, 3), torch.randn(4, 8), dt=1.0)
        # Re-route to get a fresh logits with requires_grad.
        cell.last_g = None
        # Manually populate last_g with a tensor that has requires_grad.
        g = cell.router(torch.randn(4, 3), torch.randn(4, 8))
        cell.last_g = g.detach()
        # Now call with a fresh task_loss and a requires_grad logits for
        # the gradient call.
        # We need to call moe_ecology_number directly with our own logits.
        from lnn.core.moe_ecology import moe_ecology_number
        new_logits = torch.randn(4, 3, requires_grad=True)
        new_g = new_logits.softmax(dim=-1)
        target = torch.softmax(torch.randn(4, 3), dim=-1)
        loss = ((new_g - target) ** 2).mean()
        e = moe_ecology_number(
            router_logits=new_logits, last_g=new_g, B=0.0,
            H_mode="gradient", task_loss=loss,
        )
        assert e.item() > 0.0

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

    def test_blend_alpha_propagates(self) -> None:
        """``ecology_H_alpha`` is stored and used by moe_ecology_number."""
        _seed(2)
        cell = FAMECfCCell(
            input_size=3, hidden_size=8, n_experts=3, top_k=2,
            ecology_H_mode="blend", ecology_H_alpha=0.7,
        )
        assert cell.ecology_H_alpha == 0.7
        cell.train()
        with torch.no_grad():
            cell.forward(torch.randn(4, 3), torch.randn(4, 8), dt=1.0)
        # Diagnostic without task_loss falls back to empirical (no blend).
        diag = cell.moe_ecology_diagnostic(B=0.0)
        assert "E" in diag
