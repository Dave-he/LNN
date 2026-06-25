"""Tests for InterBasinDistanceCfCCell (round 257)."""

from __future__ import annotations

import unittest

import torch

from lnn.core.inter_basin_distance_cfc import (
    InterBasinDistanceCfCCell,
    cross_branch_repulsion_loss,
    inter_basin_repulsion_loss,
)


class TestInterBasinRepulsion(unittest.TestCase):
    def test_close_centers_have_nonzero_loss(self):
        c = torch.zeros(3, 4)
        loss = inter_basin_repulsion_loss(c, d_min=1.0)
        # All centers at origin → all distances 0 → max(0, 1-0)^2 = 1 per pair
        # 3 pairs → 3.0
        self.assertAlmostEqual(loss.item(), 3.0, places=4)

    def test_far_centers_have_zero_loss(self):
        c = torch.zeros(3, 4)
        c[1, 0] = 5.0
        c[2, 0] = -5.0
        loss = inter_basin_repulsion_loss(c, d_min=1.0)
        self.assertAlmostEqual(loss.item(), 0.0, places=4)

    def test_partial_loss_intermediate(self):
        c = torch.zeros(3, 4)
        c[1, 0] = 0.5  # distance 0.5 from c[0]
        c[2, 0] = 5.0  # distance 5.0 from c[0] and 4.5 from c[1]
        loss = inter_basin_repulsion_loss(c, d_min=1.0)
        # pair (0,1): max(0, 1-0.5)^2 = 0.25
        # pair (0,2): 0
        # pair (1,2): 0
        self.assertAlmostEqual(loss.item(), 0.25, places=4)

    def test_n_basin_1_returns_zero(self):
        c = torch.zeros(1, 4)
        loss = inter_basin_repulsion_loss(c, d_min=1.0)
        self.assertEqual(loss.item(), 0.0)

    def test_d_min_scaling(self):
        c = torch.zeros(3, 4)
        c[1, 0] = 0.5
        c[2, 0] = 5.0
        loss_d1 = inter_basin_repulsion_loss(c, d_min=1.0)
        loss_d2 = inter_basin_repulsion_loss(c, d_min=2.0)
        # 3 pairs: (0,1)=0.5, (0,2)=5, (1,2)=4.5
        # d_min=1.0: only (0,1) close → (1-0.5)^2 = 0.25
        # d_min=2.0: only (0,1) close → (2-0.5)^2 = 2.25
        self.assertAlmostEqual(loss_d1.item(), 0.25, places=4)
        self.assertAlmostEqual(loss_d2.item(), 2.25, places=4)


class TestCrossBranchRepulsion(unittest.TestCase):
    def test_identical_centers_have_nonzero(self):
        c = torch.zeros(2, 3, 4)  # 2 branches, 3 basins, dim 4
        loss = cross_branch_repulsion_loss(c, d_min=1.0)
        # 1 branch-pair (a, b) × 3 same-index basins = 3 pairs
        # each pair at distance 0 → 1.0^2 = 1.0 → total ~3
        self.assertAlmostEqual(loss.item(), 3.0, places=2)

    def test_far_centers_zero(self):
        c = torch.zeros(2, 3, 4)
        c[1, :, 0] = 5.0
        loss = cross_branch_repulsion_loss(c, d_min=1.0)
        self.assertAlmostEqual(loss.item(), 0.0, places=4)

    def test_single_branch_zero(self):
        c = torch.zeros(1, 3, 4)
        loss = cross_branch_repulsion_loss(c, d_min=1.0)
        self.assertEqual(loss.item(), 0.0)


class TestInterBasinDistanceCell(unittest.TestCase):
    def test_inherits_per_branch(self):
        cell = InterBasinDistanceCfCCell(3, 8, n_branches=4, n_basin=3)
        param_names = [n for n, _ in cell.named_parameters()]
        self.assertIn("basin_centers", param_names)
        self.assertAlmostEqual(cell.d_min, 1.0, places=4)

    def test_inter_basin_loss_in_aux(self):
        cell = InterBasinDistanceCfCCell(3, 8, d_min=1.0)
        h_list = cell.init_state(4)
        x_t = torch.randn(4, 3)
        _, _, aux = cell.forward_with_aux(x_t, h_list)
        self.assertIn("inter_basin_loss", aux)
        self.assertIn("cross_branch_loss", aux)
        self.assertGreaterEqual(aux["inter_basin_loss"].item(), 0.0)

    def test_grad_flows_to_basin_centers(self):
        cell = InterBasinDistanceCfCCell(3, 8, d_min=1.0)
        h_list = cell.init_state(4)
        x_t = torch.randn(4, 3)
        _, _, aux = cell.forward_with_aux(
            x_t, h_list, dist_lambda=1.0,
        )
        loss = aux["inter_basin_loss_total"]
        loss.backward()
        self.assertIsNotNone(cell.basin_centers.grad)
        self.assertGreater(cell.basin_centers.grad.abs().sum().item(), 0.0)

    def test_grad_sign_pushes_apart(self):
        """Inter-basin loss should INCREASE distance between close centers."""
        torch.manual_seed(0)
        cell = InterBasinDistanceCfCCell(
            3, 8, n_branches=2, n_basin=2, d_min=2.0,
        )
        # Force basin centers to be very close (branch 0, basin 0 and 1).
        with torch.no_grad():
            cell.basin_centers.zero_()
            cell.basin_centers[0, 1, 0] = 0.1
        h_list = cell.init_state(4)
        x_t = torch.randn(4, 3)
        _, _, aux = cell.forward_with_aux(
            x_t, h_list, dist_lambda=1.0,
        )
        aux["inter_basin_loss_total"].backward()
        grad = cell.basin_centers.grad
        self.assertFalse(torch.isnan(grad).any().item())
        self.assertGreater(grad.abs().sum().item(), 0.0)

    def test_forward_shape(self):
        d_in, d_h, T, B, K = 3, 8, 12, 4, 4
        cell = InterBasinDistanceCfCCell(d_in, d_h, n_branches=K, n_basin=3)
        x = torch.randn(T, B, d_in)
        h_list = cell.init_state(B)
        outputs = []
        for t in range(T):
            h, h_list = cell(x[t], h_list)
            outputs.append(h)
        self.assertEqual(outputs[-1].shape, (B, d_h))


if __name__ == "__main__":
    unittest.main()
