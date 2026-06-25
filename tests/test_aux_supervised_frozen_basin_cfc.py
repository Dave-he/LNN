"""Tests for AuxSupervisedFrozenRandomBasinCfCCell (round 251)."""

from __future__ import annotations

import unittest

import torch

from lnn.core.aux_supervised_frozen_basin_cfc import (
    AuxSupervisedFrozenRandomBasinCfCCell,
)


class TestAuxSupervisedFrozenRandomBasinCfCCell(unittest.TestCase):
    def test_inherits_frozen_random(self):
        """basin_centers and tau_frozen should still be buffers."""
        cell = AuxSupervisedFrozenRandomBasinCfCCell(3, 8, n_branches=4,
                                                     n_basin=3)
        param_names = [n for n, _ in cell.named_parameters()]
        self.assertNotIn("basin_centers", param_names)
        self.assertNotIn("tau_frozen", param_names)
        buffer_names = [n for n, _ in cell.named_buffers()]
        self.assertIn("basin_centers", buffer_names)
        self.assertIn("tau_frozen", buffer_names)

    def test_default_lyap_lambda(self):
        cell = AuxSupervisedFrozenRandomBasinCfCCell(3, 8, lyap_lambda=0.1)
        self.assertAlmostEqual(cell.default_lyap_lambda, 0.1, places=4)

    def test_forward_with_aux_applies_default(self):
        d_in, d_h, B, K = 3, 8, 4, 4
        cell = AuxSupervisedFrozenRandomBasinCfCCell(d_in, d_h, n_branches=K,
                                                     n_basin=3, lyap_lambda=0.1)
        h_list = cell.init_state(B)
        x_t = torch.randn(B, d_in)
        h, _, aux = cell.forward_with_aux(x_t, h_list)
        # lyap_loss_total must be present (default lyap_lambda=0.1).
        self.assertIn("lyap_loss_total", aux)
        # lyap_loss_total should equal lyap_lambda * lyap_loss.
        expected = 0.1 * aux["lyap_loss"]
        self.assertAlmostEqual(aux["lyap_loss_total"].item(),
                                expected.item(), places=4)

    def test_forward_with_aux_explicit_zero(self):
        d_in, d_h, B, K = 3, 8, 4, 4
        cell = AuxSupervisedFrozenRandomBasinCfCCell(d_in, d_h, n_branches=K,
                                                     n_basin=3, lyap_lambda=0.1)
        h_list = cell.init_state(B)
        x_t = torch.randn(B, d_in)
        h, _, aux = cell.forward_with_aux(x_t, h_list, lyap_lambda=0.0)
        self.assertNotIn("lyap_loss_total", aux)

    def test_grad_flows_through_lyap(self):
        """Confirm gradient flows through the aux loss path into cell params."""
        d_in, d_h, B, K = 3, 8, 4, 4
        cell = AuxSupervisedFrozenRandomBasinCfCCell(d_in, d_h, n_branches=K,
                                                     n_basin=3, lyap_lambda=0.1)
        h_list = cell.init_state(B)
        x_t = torch.randn(B, d_in)
        h, _, aux = cell.forward_with_aux(x_t, h_list)
        aux["lyap_loss_total"].backward()
        # Lyap loss goes through h_next_k -> cell_k params.
        # mix_param does NOT appear in Lyap loss graph (basin_centers buffer).
        any_grad = False
        for k in range(cell.n_branches):
            for name, p in cell.cells[k].named_parameters():
                if p.grad is not None and p.grad.abs().sum().item() > 0.0:
                    any_grad = True
                    break
            if any_grad:
                break
        self.assertTrue(any_grad)

    def test_forward_shape(self):
        d_in, d_h, T, B, K = 3, 8, 12, 4, 4
        cell = AuxSupervisedFrozenRandomBasinCfCCell(d_in, d_h, n_branches=K,
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