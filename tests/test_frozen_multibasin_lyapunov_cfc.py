"""Tests for FrozenMultiBasinLyapunovCfCCell (round 247, composition of 246 × 244)."""

from __future__ import annotations

import unittest

import torch

from lnn.core.frozen_multibasin_lyapunov_cfc import (
    FrozenMultiBasinLyapunovCfCCell,
)


class TestFrozenMultiBasinLyapunovCfCCell(unittest.TestCase):
    def test_tau_frozen(self):
        cell = FrozenMultiBasinLyapunovCfCCell(3, 8, n_branches=4, seed=42)
        # tau_frozen is buffer (not parameter).
        param_names = [n for n, _ in cell.named_parameters()]
        self.assertNotIn("tau_frozen", param_names)
        buffer_names = [n for n, _ in cell.named_buffers()]
        self.assertIn("tau_frozen", buffer_names)

    def test_branch_cells_use_frozen_tau(self):
        cell = FrozenMultiBasinLyapunovCfCCell(3, 8, n_branches=4, seed=42)
        for k in range(cell.n_branches):
            tau_actual = cell.cells[k].time_scale.mean().item()
            tau_frozen = cell.tau_frozen[k].item()
            self.assertAlmostEqual(tau_actual, tau_frozen, places=4)

    def test_basin_centers_learnable(self):
        cell = FrozenMultiBasinLyapunovCfCCell(3, 8, n_branches=4, n_basin=3,
                                                seed=42)
        param_names = [n for n, _ in cell.named_parameters()]
        self.assertIn("basin_centers", param_names)

    def test_forward_shape(self):
        d_in, d_h, T, B, K = 3, 8, 12, 4, 4
        cell = FrozenMultiBasinLyapunovCfCCell(d_in, d_h, n_branches=K,
                                                n_basin=3, seed=42)
        x = torch.randn(T, B, d_in)
        h_list = cell.init_state(B)
        outputs = []
        for t in range(T):
            h, h_list = cell(x[t], h_list)
            outputs.append(h)
        self.assertEqual(outputs[-1].shape, (B, d_h))

    def test_forward_with_aux(self):
        d_in, d_h, B, K = 3, 8, 4, 4
        cell = FrozenMultiBasinLyapunovCfCCell(d_in, d_h, n_branches=K,
                                                n_basin=3, seed=42)
        h_list = cell.init_state(B)
        x_t = torch.randn(B, d_in)
        h, _, aux = cell.forward_with_aux(
            x_t, h_list, lyap_lambda=0.1, sep_lambda=0.01,
        )
        self.assertEqual(h.shape, (B, d_h))
        for k in ("alpha_mix", "V_h", "V_next", "basin_assign",
                  "basin_entropy", "lyap_loss", "lyap_loss_total",
                  "sep_loss_total"):
            self.assertIn(k, aux)
        self.assertEqual(aux["basin_assign"].shape, (B, 3))
        self.assertEqual(aux["alpha_mix"].shape, (K,))

    def test_separation_loss(self):
        cell = FrozenMultiBasinLyapunovCfCCell(2, 4, n_branches=4, n_basin=3,
                                                pd_eps=0.5, seed=42)
        sep = cell.basin_separation_loss().item()
        self.assertGreaterEqual(sep, 0.0)

    def test_alpha_init_uniform(self):
        cell = FrozenMultiBasinLyapunovCfCCell(3, 8, n_branches=4, learn_mix=True)
        for v in cell.alpha_mix.tolist():
            self.assertAlmostEqual(v, 0.25, places=4)


if __name__ == "__main__":
    unittest.main()