"""Unit tests for round 91 smoothness metrics (PRD #10-53, response
to arXiv:2606.07670, Li/Pal/Tan June 2026).

Verifies:
- total_variation: 0 for constant, increases with oscillation
- l2_derivative: 0 for constant, RMS grows with amplitude
- max_gradient: 0 for constant, = max finite-diff for varying
- smoothness_summary: returns all 3 + n
- Edge cases: 0-length, 1-length tensors return 0
- Linear signal: TV = constant value
"""
import torch

from lnn.core.smoothness_metrics import (
    l2_derivative,
    max_gradient,
    smoothness_summary,
    total_variation,
)


class TestTotalVariation:
    def test_constant_signal_tv_zero(self) -> None:
        """Constant signal has TV=0."""
        y = torch.ones(10) * 3.14
        assert total_variation(y) == 0.0

    def test_oscillating_signal_tv_positive(self) -> None:
        """Sine wave has positive TV."""
        t = torch.linspace(0, 1, 100)
        y = torch.sin(2 * 3.14159 * t)
        tv = total_variation(y)
        assert tv > 0.0
        # TV of sin over 1 period with N=100 is roughly 4/N * 2 = 0.08
        # (4 crossings of zero derivative, ~2 * amplitude / N).
        assert 0.01 < tv < 1.0

    def test_linear_signal_tv_constant_step(self) -> None:
        """Linear signal y=t has TV = mean step = 1/N."""
        N = 100
        y = torch.linspace(0, 1, N)
        tv = total_variation(y)
        expected = 1.0 / (N - 1)  # mean |y[i+1] - y[i]|
        assert abs(tv - expected) < 1e-5

    def test_zero_length_returns_zero(self) -> None:
        assert total_variation(torch.tensor([])) == 0.0

    def test_one_element_returns_zero(self) -> None:
        assert total_variation(torch.tensor([1.0])) == 0.0


class TestL2Derivative:
    def test_constant_signal_zero(self) -> None:
        y = torch.ones(10) * 2.0
        assert l2_derivative(y) == 0.0

    def test_linear_signal_rms(self) -> None:
        """Linear signal y=t with dt=1: deriv=1, RMS=1."""
        y = torch.linspace(0, 9, 10)  # step = 1
        assert abs(l2_derivative(y) - 1.0) < 1e-5

    def test_scaling(self) -> None:
        """Doubling y doubles l2_deriv (homogeneity of degree 1)."""
        y1 = torch.linspace(0, 1, 50)
        y2 = y1 * 2
        assert abs(l2_derivative(y2) - 2 * l2_derivative(y1)) < 1e-5


class TestMaxGradient:
    def test_constant_signal_zero(self) -> None:
        y = torch.ones(10)
        assert max_gradient(y) == 0.0

    def test_step_function(self) -> None:
        """Step function: max gradient is the step size."""
        y = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        # Max step is 1.0 at the transition.
        assert abs(max_gradient(y) - 1.0) < 1e-5

    def test_dt_scaling(self) -> None:
        """Smaller dt → larger gradient (scaled by 1/dt)."""
        y = torch.tensor([0.0, 1.0, 2.0, 3.0])  # step=1
        assert abs(max_gradient(y, dt=0.1) - 10.0) < 1e-5


class TestSmoothnessSummary:
    def test_returns_all_keys(self) -> None:
        y = torch.linspace(0, 1, 50)
        s = smoothness_summary(y)
        assert "tv" in s
        assert "l2_deriv" in s
        assert "max_grad" in s
        assert "n" in s
        assert s["n"] == 50

    def test_constant_signal_all_zero(self) -> None:
        y = torch.ones(20) * 5.0
        s = smoothness_summary(y)
        assert s["tv"] == 0.0
        assert s["l2_deriv"] == 0.0
        assert s["max_grad"] == 0.0


class TestExports:
    def test_metrics_exported_from_lnn_core(self) -> None:
        from lnn.core import (  # noqa: F401
            l2_derivative as ld,
            max_gradient as mg,
            smoothness_summary as ss,
            total_variation as tv,
        )
        assert callable(ld)
        assert callable(mg)
        assert callable(ss)
        assert callable(tv)
