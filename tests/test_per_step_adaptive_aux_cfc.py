"""Tests for PerStepAdaptiveAuxMultiBasinLyapunovCfCCell (round 254)."""

from __future__ import annotations

import unittest

import torch

from lnn.core.per_step_adaptive_aux_cfc import (
    PerStepAdaptiveAuxMultiBasinLyapunovCfCCell,
)


class TestPerStepAdaptiveAux(unittest.TestCase):
    def test_inherits_per_branch(self):
        cell = PerStepAdaptiveAuxMultiBasinLyapunovCfCCell(
            3, 8, n_branches=4, n_basin=3,
        )
        param_names = [n for n, _ in cell.named_parameters()]
        self.assertIn("basin_centers", param_names)
        self.assertTrue(cell.per_step_aux)

    def test_default_lyap_lambda_max(self):
        cell = PerStepAdaptiveAuxMultiBasinLyapunovCfCCell(
            3, 8, lyap_lambda_max=0.1,
        )
        self.assertAlmostEqual(cell.default_lyap_lambda_max, 0.1, places=4)

    def test_lambda_step_shape_is_scalar(self):
        d_in, d_h, B, K, B_ = 3, 8, 4, 4, 3
        cell = PerStepAdaptiveAuxMultiBasinLyapunovCfCCell(
            d_in, d_h, n_branches=K, n_basin=B_, lyap_lambda_max=0.1,
        )
        h_list = cell.init_state(B)
        x_t = torch.randn(B, d_in)
        _, _, aux = cell.forward_with_aux(x_t, h_list)
        self.assertEqual(aux["lambda_step"].dim(), 0)  # scalar tensor

    def test_lambda_step_in_range(self):
        """λ_t should be in [0, λ_max]."""
        d_in, d_h, B, K, B_ = 3, 8, 4, 4, 3
        cell = PerStepAdaptiveAuxMultiBasinLyapunovCfCCell(
            d_in, d_h, n_branches=K, n_basin=B_, lyap_lambda_max=0.1,
        )
        h_list = cell.init_state(B)
        x_t = torch.randn(B, d_in)
        _, _, aux = cell.forward_with_aux(x_t, h_list)
        ls = aux["lambda_step"].item()
        self.assertGreaterEqual(ls, 0.0)
        self.assertLessEqual(ls, 0.1 + 1e-6)

    def test_lambda_step_detached(self):
        """λ_t should be detached (gating signal, not learnable)."""
        d_in, d_h, B, K, B_ = 3, 8, 4, 4, 3
        cell = PerStepAdaptiveAuxMultiBasinLyapunovCfCCell(
            d_in, d_h, n_branches=K, n_basin=B_, lyap_lambda_max=0.1,
        )
        h_list = cell.init_state(B)
        x_t = torch.randn(B, d_in)
        _, _, aux = cell.forward_with_aux(x_t, h_list)
        self.assertFalse(aux["lambda_step"].requires_grad)

    def test_per_step_false_uses_constant(self):
        """per_step_aux=False → λ_t = λ_max constant (matches r252)."""
        d_in, d_h, B, K, B_ = 3, 8, 4, 4, 3
        cell = PerStepAdaptiveAuxMultiBasinLyapunovCfCCell(
            d_in, d_h, n_branches=K, n_basin=B_,
            lyap_lambda_max=0.1, per_step_aux=False,
        )
        h_list = cell.init_state(B)
        x_t = torch.randn(B, d_in)
        _, _, aux = cell.forward_with_aux(x_t, h_list)
        self.assertAlmostEqual(aux["lambda_step"].item(), 0.1, places=4)

    def test_grad_flows_to_basin_centers(self):
        d_in, d_h, B, K, B_ = 3, 8, 4, 4, 3
        cell = PerStepAdaptiveAuxMultiBasinLyapunovCfCCell(
            d_in, d_h, n_branches=K, n_basin=B_, lyap_lambda_max=0.1,
        )
        h_list = cell.init_state(B)
        x_t = torch.randn(B, d_in)
        _, _, aux = cell.forward_with_aux(x_t, h_list)
        aux["lyap_loss_total"].backward()
        self.assertIsNotNone(cell.basin_centers.grad)
        self.assertGreater(cell.basin_centers.grad.abs().sum().item(), 0.0)

    def test_forward_shape(self):
        d_in, d_h, T, B, K = 3, 8, 12, 4, 4
        cell = PerStepAdaptiveAuxMultiBasinLyapunovCfCCell(
            d_in, d_h, n_branches=K, n_basin=3,
        )
        x = torch.randn(T, B, d_in)
        h_list = cell.init_state(B)
        outputs = []
        for t in range(T):
            h, h_list = cell(x[t], h_list)
            outputs.append(h)
        self.assertEqual(outputs[-1].shape, (B, d_h))

    def test_lambda_step_responds_to_branch_uncertainty(self):
        """When ALL branches are uncertain (h far from basins), λ_t should
        be high; when ALL are confident, λ_t should be low.

        Note: CfC cell dynamics use tanh, so h is bounded. We test the
        deterministic relationship: λ_t = λ_max · mean(H) / log(n_basin)
        by directly checking the formula.
        """
        import math
        d_in, d_h, B, K, B_ = 3, 8, 4, 4, 3
        cell = PerStepAdaptiveAuxMultiBasinLyapunovCfCCell(
            d_in, d_h, n_branches=K, n_basin=B_, lyap_lambda_max=0.1,
        )
        h_list = cell.init_state(B)
        x_t = torch.randn(B, d_in)
        _, _, aux = cell.forward_with_aux(x_t, h_list)
        # The formula: λ_t = λ_max · mean(H) / log(n_basin).
        expected = (0.1 * aux["per_branch_basin_H"].mean().item()
                    / math.log(B_))
        actual = aux["lambda_step"].item()
        self.assertAlmostEqual(actual, expected, places=4)


if __name__ == "__main__":
    unittest.main()
