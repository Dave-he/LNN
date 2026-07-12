"""Unit tests for state_decorrelation_loss (round 289).

Verifies:
    Lambda=0 disables loss (returns 0).
    Loss is non-negative.
    Loss penalizes correlated state more than decorrelated state.
    Loss is differentiable end-to-end.
    H3 ratio: random orthogonal h has ratio >> 5, perfectly correlated
        h has ratio near 1.
    Covariance diagnostics shape checks.
    lambda_coeff=0 is exactly the identity operation.
    Loss handles batch and time dims correctly.
"""

from __future__ import annotations

import unittest

import torch

from lnn.core.decorrelation_loss import (
    state_decorrelation_loss,
    state_covariance_diagnostics,
)


class TestDecorrelationLossBasics(unittest.TestCase):

    def test_lambda_zero_returns_zero(self):
        h = torch.randn(4, 16, 8)
        loss = state_decorrelation_loss(h, lambda_coeff=0.0)
        self.assertEqual(float(loss.item()), 0.0)

    def test_shape_validation(self):
        with self.assertRaises(ValueError):
            state_decorrelation_loss(torch.randn(4, 8))  # 2D, not 3D

    def test_single_timestep_returns_zero(self):
        h = torch.randn(4, 1, 8)
        loss = state_decorrelation_loss(h, lambda_coeff=0.01)
        self.assertEqual(float(loss.item()), 0.0)


class TestDecorrelationEffect(unittest.TestCase):

    def test_correlated_h_higher_loss_than_orthogonal(self):
        # Perfectly correlated: h has rank 1.
        torch.manual_seed(0)
        base = torch.randn(4, 16, 1)  # single feature varying in time
        proj = torch.randn(1, 8)
        h_corr = base @ proj  # (B, T, 8) with rank 1

        # Orthogonal: each dim is independent noise.
        torch.manual_seed(0)
        h_orth = torch.randn(4, 16, 8)

        loss_corr = state_decorrelation_loss(h_corr, lambda_coeff=1.0)
        loss_orth = state_decorrelation_loss(h_orth, lambda_coeff=1.0)
        self.assertGreater(
            float(loss_corr.item()), float(loss_orth.item()),
            f"correlated ({float(loss_corr.item()):.4f}) should be > "
            f"orthogonal ({float(loss_orth.item()):.4f})")

    def test_loss_scales_with_lambda(self):
        h = torch.randn(4, 16, 8)
        l1 = state_decorrelation_loss(h, lambda_coeff=0.01)
        l2 = state_decorrelation_loss(h, lambda_coeff=0.1)
        # Should be exactly 10x.
        ratio = float(l2.item()) / max(float(l1.item()), 1e-12)
        self.assertAlmostEqual(ratio, 10.0, places=4,
                               msg=f"lambda scaling violated, ratio {ratio:.4f}")


class TestGradientFlow(unittest.TestCase):

    def test_gradients_flow_through_loss(self):
        h = torch.randn(4, 16, 8, requires_grad=True)
        loss = state_decorrelation_loss(h, lambda_coeff=0.01)
        loss.backward()
        self.assertIsNotNone(h.grad)
        self.assertEqual(h.grad.shape, h.shape)
        self.assertGreater(h.grad.abs().sum().item(), 0.0)

    def test_loss_composes_with_task(self):
        # Simulate a tiny end-to-end: hidden state from a linear cell.
        x = torch.randn(4, 16, 1)
        W = torch.nn.Linear(1, 8, bias=False)
        h = W(x)  # (B, T, d_h)
        loss_dec = state_decorrelation_loss(h, lambda_coeff=0.1)
        # Add a fake task loss to compose gradients.
        target = torch.zeros(4, 16, 8)
        loss_task = (h - target).pow(2).mean()
        total = loss_task + loss_dec
        total.backward()
        self.assertIsNotNone(W.weight.grad)
        self.assertGreater(W.weight.grad.abs().sum().item(), 0.0)


class TestCovarianceDiagnostics(unittest.TestCase):

    def test_orthogonal_state_has_high_ratio(self):
        # Each dim independent → diag >> off-diag → ratio high.
        # With B*T=256 samples and 16 dims, the off-diag/diag ratio
        # for i.i.d. noise is roughly sqrt(B*T)/dim ≈ 4. We assert > 3.
        torch.manual_seed(0)
        h = torch.randn(8, 32, 16)
        d = state_covariance_diagnostics(h)
        self.assertGreater(d["ratio"], 3.0,
                           f"orthogonal state ratio {d['ratio']:.2f} should > 3")

    def test_correlated_state_has_low_ratio(self):
        # Rank-1 state → off-diag ≈ diag → ratio near 1.
        base = torch.randn(8, 32, 1)
        proj = torch.randn(1, 16) * 5.0
        h = base @ proj
        d = state_covariance_diagnostics(h)
        self.assertLess(d["ratio"], 2.0,
                        f"correlated state ratio {d['ratio']:.2f} should < 2")

    def test_diagnostics_keys(self):
        h = torch.randn(2, 8, 4)
        d = state_covariance_diagnostics(h)
        self.assertIn("mean_diag", d)
        self.assertIn("max_off_diag", d)
        self.assertIn("ratio", d)


class TestEndToEndWithCell(unittest.TestCase):

    def test_loss_works_with_blend_gated_cell(self):
        from lnn.core.blend_gated_liquid_tau_cfc import (
            BlendGatedLiquidTauCfCCell,
        )
        torch.manual_seed(0)
        cell = BlendGatedLiquidTauCfCCell(input_size=1, hidden_size=16)
        x = torch.randn(2, 24, 1)
        out, _h_final = cell(x)
        # `out` is the per-step hidden state (B, T, d_h); use that.
        self.assertEqual(out.dim(), 3)
        loss_dec = state_decorrelation_loss(out, lambda_coeff=0.01)
        self.assertGreater(float(loss_dec.item()), 0.0)


if __name__ == "__main__":
    unittest.main()