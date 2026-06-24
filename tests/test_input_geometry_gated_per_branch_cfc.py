"""Tests for InputGeometryGatedPerBranchCfCCell (round 249)."""

from __future__ import annotations

import unittest

import torch

from lnn.core.input_geometry_gated_per_branch_cfc import (
    InputGeometryGatedPerBranchCfCCell,
)


class TestInputGeometryGatedPerBranchCfCCell(unittest.TestCase):
    def test_tau_frozen(self):
        cell = InputGeometryGatedPerBranchCfCCell(3, 8, n_branches=4, seed=42)
        param_names = [n for n, _ in cell.named_parameters()]
        self.assertNotIn("tau_frozen", param_names)
        buffer_names = [n for n, _ in cell.named_buffers()]
        self.assertIn("tau_frozen", buffer_names)

    def test_branch_cells_use_frozen_tau(self):
        cell = InputGeometryGatedPerBranchCfCCell(3, 8, n_branches=4, seed=42)
        for k in range(cell.n_branches):
            tau_actual = cell.cells[k].time_scale.mean().item()
            tau_frozen = cell.tau_frozen[k].item()
            self.assertAlmostEqual(tau_actual, tau_frozen, places=4)

    def test_gate_network_shape(self):
        cell = InputGeometryGatedPerBranchCfCCell(3, 8, n_branches=4, seed=42)
        # gate input is (d_in + n_branches), output (n_branches).
        self.assertEqual(cell.gate.in_features, 3 + 4)
        self.assertEqual(cell.gate.out_features, 4)

    def test_forward_shape(self):
        d_in, d_h, T, B, K = 3, 8, 12, 4, 4
        cell = InputGeometryGatedPerBranchCfCCell(d_in, d_h, n_branches=K,
                                                  seed=42)
        x = torch.randn(T, B, d_in)
        h_list = cell.init_state(B)
        outputs = []
        for t in range(T):
            h, h_list, alpha = cell(x[t], h_list)
            outputs.append(h)
            self.assertEqual(alpha.shape, (K,))
        self.assertEqual(outputs[-1].shape, (B, d_h))

    def test_alpha_depends_on_input(self):
        """Two very different inputs should produce different alpha."""
        torch.manual_seed(0)
        cell = InputGeometryGatedPerBranchCfCCell(2, 6, n_branches=4, seed=42)
        h_list = cell.init_state(2)
        x_a = torch.tensor([[10.0, 0.0], [10.0, 0.0]])
        x_b = torch.tensor([[0.0, 10.0], [0.0, 10.0]])
        # Run for a few steps first so state isn't zero.
        for _ in range(3):
            torch.manual_seed(0)
            h_a, h_list_a, _ = cell(x_a, cell.init_state(2))
            torch.manual_seed(0)
            h_b, h_list_b, _ = cell(x_b, cell.init_state(2))
        torch.manual_seed(0)
        _, _, alpha_a = cell(x_a, h_list_a)
        torch.manual_seed(0)
        _, _, alpha_b = cell(x_b, h_list_b)
        diff = (alpha_a - alpha_b).abs().max().item()
        self.assertGreater(diff, 0.0)

    def test_forward_with_aux(self):
        d_in, d_h, B, K = 3, 8, 4, 4
        cell = InputGeometryGatedPerBranchCfCCell(d_in, d_h, n_branches=K,
                                                  seed=42)
        h_list = cell.init_state(B)
        x_t = torch.randn(B, d_in)
        h, _, aux = cell.forward_with_aux(
            x_t, h_list, lyap_lambda=0.1, sep_lambda=0.01,
        )
        self.assertEqual(h.shape, (B, d_h))
        for k in ("alpha_mix", "per_branch_V_next", "per_branch_basin_H",
                  "mean_basin_H", "lyap_loss", "lyap_loss_total",
                  "sep_loss_total"):
            self.assertIn(k, aux)
        self.assertEqual(aux["per_branch_V_next"].shape, (K,))
        self.assertEqual(aux["per_branch_basin_H"].shape, (K,))

    def test_per_branch_separation_loss(self):
        cell = InputGeometryGatedPerBranchCfCCell(2, 4, n_branches=4,
                                                  pd_eps=0.5, seed=42)
        sep = cell.per_branch_separation_loss().item()
        self.assertGreaterEqual(sep, 0.0)

    def test_alpha_sums_to_one(self):
        cell = InputGeometryGatedPerBranchCfCCell(3, 8, n_branches=4, seed=42)
        h_list = cell.init_state(4)
        x_t = torch.randn(4, 3)
        _, _, alpha = cell(x_t, h_list)
        self.assertAlmostEqual(alpha.sum().item(), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()