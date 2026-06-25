"""Tests for CombinedPerBranchPerStepAuxMultiBasinLyapunovCfCCell (round 255)."""

from __future__ import annotations

import math
import unittest

import torch

from lnn.core.combined_per_branch_per_step_aux_cfc import (
    CombinedPerBranchPerStepAuxMultiBasinLyapunovCfCCell,
)


class TestCombinedAux(unittest.TestCase):
    def test_inherits_per_branch(self):
        cell = CombinedPerBranchPerStepAuxMultiBasinLyapunovCfCCell(
            3, 8, n_branches=4, n_basin=3,
        )
        param_names = [n for n, _ in cell.named_parameters()]
        self.assertIn("basin_centers", param_names)
        self.assertEqual(cell.combination, "product")

    def test_default_lyap_lambda_max(self):
        cell = CombinedPerBranchPerStepAuxMultiBasinLyapunovCfCCell(
            3, 8, lyap_lambda_max=0.1,
        )
        self.assertAlmostEqual(cell.default_lyap_lambda_max, 0.1, places=4)

    def test_combined_lambda_le_per_branch(self):
        """Product ≤ min(per_branch, per_step) so combined ≤ per_branch."""
        d_in, d_h, B, K, B_ = 3, 8, 4, 4, 3
        cell = CombinedPerBranchPerStepAuxMultiBasinLyapunovCfCCell(
            d_in, d_h, n_branches=K, n_basin=B_,
            lyap_lambda_max=0.1, combination="product",
        )
        h_list = cell.init_state(B)
        x_t = torch.randn(B, d_in)
        _, _, aux = cell.forward_with_aux(x_t, h_list)
        combined = aux["lambda_combined"]
        per_branch = aux["lambda_per_branch"]
        # All combined should be <= per_branch (product with [0, 1]).
        self.assertTrue((combined <= per_branch + 1e-6).all().item())

    def test_combined_lambda_in_range(self):
        """Combined λ should be in [0, λ_max]."""
        d_in, d_h, B, K, B_ = 3, 8, 4, 4, 3
        cell = CombinedPerBranchPerStepAuxMultiBasinLyapunovCfCCell(
            d_in, d_h, n_branches=K, n_basin=B_, lyap_lambda_max=0.1,
        )
        h_list = cell.init_state(B)
        x_t = torch.randn(B, d_in)
        _, _, aux = cell.forward_with_aux(x_t, h_list)
        lam = aux["lambda_combined"]
        self.assertTrue((lam >= 0).all().item())
        self.assertTrue((lam <= 0.1 + 1e-6).all().item())

    def test_combined_lambda_detached(self):
        """Combined λ should be detached (gating signal)."""
        d_in, d_h, B, K, B_ = 3, 8, 4, 4, 3
        cell = CombinedPerBranchPerStepAuxMultiBasinLyapunovCfCCell(
            d_in, d_h, n_branches=K, n_basin=B_, lyap_lambda_max=0.1,
        )
        h_list = cell.init_state(B)
        x_t = torch.randn(B, d_in)
        _, _, aux = cell.forward_with_aux(x_t, h_list)
        self.assertFalse(aux["lambda_combined"].requires_grad)

    def test_max_combination_uses_max(self):
        """combination='max' → λ_k = λ_max · max(per_branch_k, per_step)."""
        d_in, d_h, B, K, B_ = 3, 8, 4, 4, 3
        cell = CombinedPerBranchPerStepAuxMultiBasinLyapunovCfCCell(
            d_in, d_h, n_branches=K, n_basin=B_,
            lyap_lambda_max=0.1, combination="max",
        )
        h_list = cell.init_state(B)
        x_t = torch.randn(B, d_in)
        _, _, aux = cell.forward_with_aux(x_t, h_list)
        combined = aux["lambda_combined"]
        per_branch = aux["lambda_per_branch"]
        step = aux["lambda_step"]
        expected = 0.1 * torch.max(per_branch / 0.1, step / 0.1)
        self.assertTrue(torch.allclose(combined, expected, atol=1e-5))

    def test_grad_flows_to_basin_centers(self):
        d_in, d_h, B, K, B_ = 3, 8, 4, 4, 3
        cell = CombinedPerBranchPerStepAuxMultiBasinLyapunovCfCCell(
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
        cell = CombinedPerBranchPerStepAuxMultiBasinLyapunovCfCCell(
            d_in, d_h, n_branches=K, n_basin=3,
        )
        x = torch.randn(T, B, d_in)
        h_list = cell.init_state(B)
        outputs = []
        for t in range(T):
            h, h_list = cell(x[t], h_list)
            outputs.append(h)
        self.assertEqual(outputs[-1].shape, (B, d_h))

    def test_combined_stricter_than_alone(self):
        """Product combination is more conservative than either alone."""
        d_in, d_h, B, K, B_ = 3, 8, 4, 4, 3
        cell = CombinedPerBranchPerStepAuxMultiBasinLyapunovCfCCell(
            d_in, d_h, n_branches=K, n_basin=B_,
            lyap_lambda_max=0.1, combination="product",
        )
        h_list = cell.init_state(B)
        x_t = torch.randn(B, d_in)
        _, _, aux = cell.forward_with_aux(x_t, h_list)
        # Combined should be smaller than per_step (which equals max
        # when all per_branch are equal, but should be <= per_step in general).
        self.assertLessEqual(
            aux["lambda_combined"].max().item(),
            aux["lambda_step"].item() + 1e-6,
        )


if __name__ == "__main__":
    unittest.main()
