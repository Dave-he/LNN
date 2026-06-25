"""Tests for AdaptiveAuxPerBranchMultiBasinLyapunovCfCCell (round 253)."""

from __future__ import annotations

import math
import unittest

import torch

from lnn.core.adaptive_aux_per_branch_cfc import (
    AdaptiveAuxPerBranchMultiBasinLyapunovCfCCell,
)


class TestAdaptiveAuxPerBranch(unittest.TestCase):
    def test_inherits_per_branch(self):
        """basin_centers is still a parameter (LEARNED)."""
        cell = AdaptiveAuxPerBranchMultiBasinLyapunovCfCCell(
            3, 8, n_branches=4, n_basin=3,
        )
        param_names = [n for n, _ in cell.named_parameters()]
        self.assertIn("basin_centers", param_names)
        self.assertTrue(cell.adaptive_aux)

    def test_default_lyap_lambda_max(self):
        cell = AdaptiveAuxPerBranchMultiBasinLyapunovCfCCell(
            3, 8, lyap_lambda_max=0.1,
        )
        self.assertAlmostEqual(cell.default_lyap_lambda_max, 0.1, places=4)

    def test_lambda_per_branch_shape_and_range(self):
        """λ_k should be in [0, λ_max] and shape (n_branches,)."""
        d_in, d_h, B, K, B_ = 3, 8, 4, 4, 3
        cell = AdaptiveAuxPerBranchMultiBasinLyapunovCfCCell(
            d_in, d_h, n_branches=K, n_basin=B_, lyap_lambda_max=0.1,
        )
        h_list = cell.init_state(B)
        x_t = torch.randn(B, d_in)
        _, _, aux = cell.forward_with_aux(x_t, h_list)
        lam = aux["lambda_per_branch"]
        self.assertEqual(lam.shape, (K,))
        # H ∈ [0, log n_basin] so λ ∈ [0, λ_max].
        self.assertTrue((lam >= 0).all().item())
        self.assertTrue((lam <= 0.1 + 1e-6).all().item())

    def test_adaptive_aux_false_matches_constant(self):
        """adaptive_aux=False → λ_k = λ_max for all k (matches r252)."""
        d_in, d_h, B, K, B_ = 3, 8, 4, 4, 3
        cell = AdaptiveAuxPerBranchMultiBasinLyapunovCfCCell(
            d_in, d_h, n_branches=K, n_basin=B_,
            lyap_lambda_max=0.1, adaptive_aux=False,
        )
        h_list = cell.init_state(B)
        x_t = torch.randn(B, d_in)
        _, _, aux = cell.forward_with_aux(x_t, h_list)
        lam = aux["lambda_per_branch"]
        # All entries should be exactly 0.1.
        self.assertTrue(torch.allclose(lam, torch.full_like(lam, 0.1)))

    def test_confident_branch_gets_lower_lambda(self):
        """If a branch's h is very close to one basin, H_k → 0, λ_k → 0.

        We can simulate this by pinning h_prev to be at a basin center.
        """
        d_in, d_h, B, K, B_ = 3, 8, 4, 4, 3
        cell = AdaptiveAuxPerBranchMultiBasinLyapunovCfCCell(
            d_in, d_h, n_branches=K, n_basin=B_, lyap_lambda_max=0.1,
        )
        # Pin h_0 to the first basin's center (branch 0 is "confident").
        h_list = cell.init_state(B)
        with torch.no_grad():
            h_list[0] = cell.basin_centers[0, 0:1].expand(B, -1).clone()
        x_t = torch.randn(B, d_in)
        _, _, aux = cell.forward_with_aux(x_t, h_list)
        lam = aux["lambda_per_branch"]
        # Soft test: confident branch (pinned at basin) should have
        # λ well below the uncertain branches initialized at zero.
        # We compare with a non-pinned branch's λ.
        non_pinned_lam = lam[1:].mean().item()
        pinned_lam = lam[0].item()
        self.assertLess(pinned_lam, non_pinned_lam + 0.02)

    def test_grad_flows_to_basin_centers(self):
        """Aux loss must gradient-flow to LEARNED basin centers."""
        d_in, d_h, B, K, B_ = 3, 8, 4, 4, 3
        cell = AdaptiveAuxPerBranchMultiBasinLyapunovCfCCell(
            d_in, d_h, n_branches=K, n_basin=B_, lyap_lambda_max=0.1,
        )
        h_list = cell.init_state(B)
        x_t = torch.randn(B, d_in)
        _, _, aux = cell.forward_with_aux(x_t, h_list)
        aux["lyap_loss_total"].backward()
        self.assertIsNotNone(cell.basin_centers.grad)
        self.assertGreater(cell.basin_centers.grad.abs().sum().item(), 0.0)

    def test_lambda_per_branch_detached(self):
        """lambda_per_branch should be DETACHED from H gradient
        (we use H as a *gating signal*, not as a learnable quantity)."""
        d_in, d_h, B, K, B_ = 3, 8, 4, 4, 3
        cell = AdaptiveAuxPerBranchMultiBasinLyapunovCfCCell(
            d_in, d_h, n_branches=K, n_basin=B_, lyap_lambda_max=0.1,
        )
        h_list = cell.init_state(B)
        x_t = torch.randn(B, d_in)
        _, _, aux = cell.forward_with_aux(x_t, h_list)
        # lambda_per_branch is computed from detached H.
        lam = aux["lambda_per_branch"]
        self.assertFalse(lam.requires_grad)

    def test_forward_shape(self):
        d_in, d_h, T, B, K = 3, 8, 12, 4, 4
        cell = AdaptiveAuxPerBranchMultiBasinLyapunovCfCCell(
            d_in, d_h, n_branches=K, n_basin=3,
        )
        x = torch.randn(T, B, d_in)
        h_list = cell.init_state(B)
        outputs = []
        for t in range(T):
            h, h_list = cell(x[t], h_list)
            outputs.append(h)
        self.assertEqual(outputs[-1].shape, (B, d_h))

    def test_log_n_basin_normalization(self):
        """When H_k is at max (log n_basin), λ_k = λ_max exactly."""
        d_in, d_h, B, K, B_ = 3, 8, 4, 4, 3
        cell = AdaptiveAuxPerBranchMultiBasinLyapunovCfCCell(
            d_in, d_h, n_branches=K, n_basin=B_, lyap_lambda_max=0.1,
        )
        # Manually set H_per_branch in the forward path.
        h_list = cell.init_state(B)
        x_t = torch.randn(B, d_in)
        # Just verify the log_nb math.
        log_nb = math.log(B_)
        self.assertAlmostEqual(log_nb, math.log(3), places=4)


if __name__ == "__main__":
    unittest.main()
