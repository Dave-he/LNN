"""Tests for FrozenRandomBasinCfCCell (round 250)."""

from __future__ import annotations

import unittest

import torch

from lnn.core.frozen_random_basin_cfc import FrozenRandomBasinCfCCell


class TestFrozenRandomBasinCfCCell(unittest.TestCase):
    def test_tau_frozen(self):
        cell = FrozenRandomBasinCfCCell(3, 8, n_branches=4)
        param_names = [n for n, _ in cell.named_parameters()]
        self.assertNotIn("tau_frozen", param_names)
        buffer_names = [n for n, _ in cell.named_buffers()]
        self.assertIn("tau_frozen", buffer_names)

    def test_branch_cells_use_frozen_tau(self):
        cell = FrozenRandomBasinCfCCell(3, 8, n_branches=4)
        for k in range(cell.n_branches):
            tau_actual = cell.cells[k].time_scale.mean().item()
            tau_frozen = cell.tau_frozen[k].item()
            self.assertAlmostEqual(tau_actual, tau_frozen, places=4)

    def test_basin_centers_frozen(self):
        """basin_centers must be buffer (not parameter)."""
        cell = FrozenRandomBasinCfCCell(3, 8, n_branches=4, n_basin=3)
        param_names = [n for n, _ in cell.named_parameters()]
        self.assertNotIn("basin_centers", param_names)
        buffer_names = [n for n, _ in cell.named_buffers()]
        self.assertIn("basin_centers", buffer_names)

    def test_basin_centers_shape(self):
        cell = FrozenRandomBasinCfCCell(3, 8, n_branches=4, n_basin=3)
        self.assertEqual(cell.basin_centers.shape, (4, 3, 8))

    def test_basin_centers_randomized_per_branch(self):
        cell = FrozenRandomBasinCfCCell(3, 8, n_branches=4, n_basin=3,
                                         basin_seed=137)
        # Different branches should have different basin centers.
        diffs = []
        for k1 in range(4):
            for k2 in range(k1 + 1, 4):
                d = (cell.basin_centers[k1] - cell.basin_centers[k2]).abs().max()
                diffs.append(d.item())
        self.assertGreater(max(diffs), 0.0)

    def test_basin_centers_dont_change_with_training(self):
        cell = FrozenRandomBasinCfCCell(3, 8, n_branches=4, n_basin=3)
        before = cell.basin_centers.clone()
        opt = torch.optim.Adam(cell.parameters(), lr=1e-2)
        x = torch.randn(8, 2, 3)
        for _ in range(3):
            opt.zero_grad()
            h_list = cell.init_state(2)
            h = None
            for t in range(8):
                h, h_list = cell(x[t], h_list)
            loss = (h ** 2).sum()
            loss.backward()
            opt.step()
        self.assertTrue(torch.allclose(cell.basin_centers, before))

    def test_forward_shape(self):
        d_in, d_h, T, B, K = 3, 8, 12, 4, 4
        cell = FrozenRandomBasinCfCCell(d_in, d_h, n_branches=K)
        x = torch.randn(T, B, d_in)
        h_list = cell.init_state(B)
        outputs = []
        for t in range(T):
            h, h_list = cell(x[t], h_list)
            outputs.append(h)
        self.assertEqual(outputs[-1].shape, (B, d_h))

    def test_forward_with_aux(self):
        d_in, d_h, B, K = 3, 8, 4, 4
        cell = FrozenRandomBasinCfCCell(d_in, d_h, n_branches=K)
        h_list = cell.init_state(B)
        x_t = torch.randn(B, d_in)
        h, _, aux = cell.forward_with_aux(
            x_t, h_list, lyap_lambda=0.1,
        )
        self.assertEqual(h.shape, (B, d_h))
        for k in ("alpha_mix", "per_branch_V_next", "per_branch_basin_H",
                  "mean_basin_H", "lyap_loss", "lyap_loss_total"):
            self.assertIn(k, aux)
        self.assertEqual(aux["per_branch_V_next"].shape, (K,))
        self.assertEqual(aux["per_branch_basin_H"].shape, (K,))

    def test_alpha_init_uniform(self):
        cell = FrozenRandomBasinCfCCell(3, 8, n_branches=4, learn_mix=True)
        for v in cell.alpha_mix.tolist():
            self.assertAlmostEqual(v, 0.25, places=4)


if __name__ == "__main__":
    unittest.main()