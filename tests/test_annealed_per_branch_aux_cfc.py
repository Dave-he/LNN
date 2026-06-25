"""Tests for AnnealedPerBranchMultiBasinLyapunovCfCCell (round 256)."""

from __future__ import annotations

import math
import unittest

import torch

from lnn.core.annealed_per_branch_aux_cfc import (
    AnnealedPerBranchMultiBasinLyapunovCfCCell,
)


class TestAnnealedPerBranchAux(unittest.TestCase):
    def test_inherits_per_branch(self):
        cell = AnnealedPerBranchMultiBasinLyapunovCfCCell(
            3, 8, n_branches=4, n_basin=3,
        )
        param_names = [n for n, _ in cell.named_parameters()]
        self.assertIn("basin_centers", param_names)
        self.assertEqual(cell.anneal_schedule, "linear")

    def test_default_lyap_lambda_max(self):
        cell = AnnealedPerBranchMultiBasinLyapunovCfCCell(
            3, 8, lyap_lambda_max=0.1,
        )
        self.assertAlmostEqual(cell.default_lyap_lambda_max, 0.1, places=4)

    def test_linear_anneal_at_epoch_0(self):
        """At epoch 0, λ = λ_max."""
        cell = AnnealedPerBranchMultiBasinLyapunovCfCCell(
            3, 8, lyap_lambda_max=0.1, anneal_epochs=50,
            anneal_schedule="linear",
        )
        cell.set_epoch(0)
        self.assertAlmostEqual(cell.get_lambda(), 0.1, places=4)

    def test_linear_anneal_at_epoch_half(self):
        """At epoch = T_anneal/2, λ = λ_max/2."""
        cell = AnnealedPerBranchMultiBasinLyapunovCfCCell(
            3, 8, lyap_lambda_max=0.1, anneal_epochs=50,
            anneal_schedule="linear",
        )
        cell.set_epoch(25)
        self.assertAlmostEqual(cell.get_lambda(), 0.05, places=4)

    def test_linear_anneal_at_epoch_end(self):
        """At epoch = T_anneal, λ = 0."""
        cell = AnnealedPerBranchMultiBasinLyapunovCfCCell(
            3, 8, lyap_lambda_max=0.1, anneal_epochs=50,
            anneal_schedule="linear",
        )
        cell.set_epoch(50)
        self.assertAlmostEqual(cell.get_lambda(), 0.0, places=4)

    def test_linear_anneal_past_end(self):
        """Past T_anneal, λ = 0."""
        cell = AnnealedPerBranchMultiBasinLyapunovCfCCell(
            3, 8, lyap_lambda_max=0.1, anneal_epochs=50,
            anneal_schedule="linear",
        )
        cell.set_epoch(100)
        self.assertAlmostEqual(cell.get_lambda(), 0.0, places=4)

    def test_cosine_anneal_at_epoch_end(self):
        """At epoch = T_anneal, cosine λ = 0."""
        cell = AnnealedPerBranchMultiBasinLyapunovCfCCell(
            3, 8, lyap_lambda_max=0.1, anneal_epochs=50,
            anneal_schedule="cosine",
        )
        cell.set_epoch(50)
        self.assertAlmostEqual(cell.get_lambda(), 0.0, places=4)

    def test_exp_anneal_at_epoch_end(self):
        """At epoch = T_anneal, exp λ ≈ λ_max * e^-3."""
        cell = AnnealedPerBranchMultiBasinLyapunovCfCCell(
            3, 8, lyap_lambda_max=0.1, anneal_epochs=50,
            anneal_schedule="exp",
        )
        cell.set_epoch(50)
        expected = 0.1 * math.exp(-3.0)
        self.assertAlmostEqual(cell.get_lambda(), expected, places=4)

    def test_current_lambda_in_aux(self):
        d_in, d_h, B, K, B_ = 3, 8, 4, 4, 3
        cell = AnnealedPerBranchMultiBasinLyapunovCfCCell(
            d_in, d_h, n_branches=K, n_basin=B_,
            lyap_lambda_max=0.1, anneal_epochs=10,
        )
        cell.set_epoch(5)  # mid-anneal
        h_list = cell.init_state(B)
        x_t = torch.randn(B, d_in)
        _, _, aux = cell.forward_with_aux(x_t, h_list)
        # current_lambda should be ~0.05 (half).
        self.assertAlmostEqual(aux["current_lambda"].item(), 0.05, places=4)

    def test_grad_flows_to_basin_centers(self):
        d_in, d_h, B, K, B_ = 3, 8, 4, 4, 3
        cell = AnnealedPerBranchMultiBasinLyapunovCfCCell(
            d_in, d_h, n_branches=K, n_basin=B_,
            lyap_lambda_max=0.1, anneal_epochs=10,
        )
        cell.set_epoch(0)
        h_list = cell.init_state(B)
        x_t = torch.randn(B, d_in)
        _, _, aux = cell.forward_with_aux(x_t, h_list)
        aux["lyap_loss_total"].backward()
        self.assertIsNotNone(cell.basin_centers.grad)
        self.assertGreater(cell.basin_centers.grad.abs().sum().item(), 0.0)

    def test_forward_shape(self):
        d_in, d_h, T, B, K = 3, 8, 12, 4, 4
        cell = AnnealedPerBranchMultiBasinLyapunovCfCCell(
            d_in, d_h, n_branches=K, n_basin=3,
        )
        x = torch.randn(T, B, d_in)
        h_list = cell.init_state(B)
        outputs = []
        for t in range(T):
            h, h_list = cell(x[t], h_list)
            outputs.append(h)
        self.assertEqual(outputs[-1].shape, (B, d_h))


if __name__ == "__main__":
    unittest.main()
