"""Tests for LyapAuxPerBranchMultiBasinLyapunovCfCCell (round 252)."""

from __future__ import annotations

import unittest

import torch

from lnn.core.lyap_aux_per_branch_multibasin_cfc import (
    LyapAuxPerBranchMultiBasinLyapunovCfCCell,
)


class TestLyapAuxPerBranch(unittest.TestCase):
    def test_inherits_per_branch(self):
        """basin_centers should still be a parameter (LEARNED)."""
        cell = LyapAuxPerBranchMultiBasinLyapunovCfCCell(3, 8, n_branches=4,
                                                         n_basin=3)
        param_names = [n for n, _ in cell.named_parameters()]
        self.assertIn("basin_centers", param_names)
        buffer_names = [n for n, _ in cell.named_buffers()]
        self.assertNotIn("basin_centers", buffer_names)

    def test_default_lyap_lambda(self):
        cell = LyapAuxPerBranchMultiBasinLyapunovCfCCell(3, 8, lyap_lambda=0.1)
        self.assertAlmostEqual(cell.default_lyap_lambda, 0.1, places=4)

    def test_forward_with_aux_applies_default(self):
        d_in, d_h, B, K = 3, 8, 4, 4
        cell = LyapAuxPerBranchMultiBasinLyapunovCfCCell(
            d_in, d_h, n_branches=K, n_basin=3, lyap_lambda=0.1,
        )
        h_list = cell.init_state(B)
        x_t = torch.randn(B, d_in)
        h, _, aux = cell.forward_with_aux(x_t, h_list)
        self.assertIn("lyap_loss_total", aux)
        expected = 0.1 * aux["lyap_loss"]
        self.assertAlmostEqual(aux["lyap_loss_total"].item(),
                                expected.item(), places=4)

    def test_forward_with_aux_explicit_zero(self):
        d_in, d_h, B, K = 3, 8, 4, 4
        cell = LyapAuxPerBranchMultiBasinLyapunovCfCCell(
            d_in, d_h, n_branches=K, n_basin=3, lyap_lambda=0.1,
        )
        h_list = cell.init_state(B)
        x_t = torch.randn(B, d_in)
        h, _, aux = cell.forward_with_aux(x_t, h_list, lyap_lambda=0.0)
        self.assertNotIn("lyap_loss_total", aux)

    def test_grad_flows_to_basin_centers(self):
        """Aux loss must gradient-flow to LEARNED basin centers (unlike r251)."""
        d_in, d_h, B, K = 3, 8, 4, 4
        cell = LyapAuxPerBranchMultiBasinLyapunovCfCCell(
            d_in, d_h, n_branches=K, n_basin=3, lyap_lambda=0.1,
        )
        h_list = cell.init_state(B)
        x_t = torch.randn(B, d_in)
        h, _, aux = cell.forward_with_aux(x_t, h_list)
        aux["lyap_loss_total"].backward()
        # basin_centers is a parameter — must have non-zero grad.
        self.assertIsNotNone(cell.basin_centers.grad)
        self.assertGreater(cell.basin_centers.grad.abs().sum().item(), 0.0)

    def test_forward_shape(self):
        d_in, d_h, T, B, K = 3, 8, 12, 4, 4
        cell = LyapAuxPerBranchMultiBasinLyapunovCfCCell(d_in, d_h, n_branches=K,
                                                         n_basin=3)
        x = torch.randn(T, B, d_in)
        h_list = cell.init_state(B)
        outputs = []
        for t in range(T):
            h, h_list = cell(x[t], h_list)
            outputs.append(h)
        self.assertEqual(outputs[-1].shape, (B, d_h))


if __name__ == "__main__":
    unittest.main()