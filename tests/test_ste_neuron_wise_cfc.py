"""Unit tests for STE-NeuronWiseCfCCell (round 265)."""

from __future__ import annotations

import unittest

import torch

from lnn.core.ste_neuron_wise_cfc import STENeuronWiseCfCCell


class TestSTENeuronWiseCfCCellBasics(unittest.TestCase):

    def test_init_defaults(self):
        cell = STENeuronWiseCfCCell(input_size=4, hidden_size=8)
        self.assertEqual(cell.input_size, 4)
        self.assertEqual(cell.hidden_size, 8)
        self.assertEqual(cell.density, 0.3)
        self.assertEqual(cell.ste_temperature, 1.0)

    def test_init_ste_temperature_validation(self):
        with self.assertRaises(ValueError):
            STENeuronWiseCfCCell(input_size=4, hidden_size=8, ste_temperature=0.0)
        with self.assertRaises(ValueError):
            STENeuronWiseCfCCell(input_size=4, hidden_size=8, ste_temperature=-1.0)

    def test_inherits_per_neuron_params(self):
        cell = STENeuronWiseCfCCell(input_size=4, hidden_size=8)
        # Inherited from r263's NeuronWiseCfCCell.
        for name in [
            "W_rec", "neighbor_logits", "tau_per_neuron",
            "alpha_per_neuron", "bias_per_neuron",
            "input_strength_per_neuron", "W_in",
        ]:
            self.assertTrue(hasattr(cell, name), f"missing param: {name}")


class TestSTENeuronWiseCfCCellMask(unittest.TestCase):

    def test_ste_mask_forward_is_binary(self):
        """Forward pass of mask must be exactly 0 or 1 (true sparsity)."""
        cell = STENeuronWiseCfCCell(input_size=4, hidden_size=8, density=0.5)
        mask = cell.get_neighborhood_mask()
        unique_vals = torch.unique(mask)
        # Should only be 0.0 and 1.0.
        for v in unique_vals.tolist():
            self.assertIn(v, [0.0, 1.0])

    def test_ste_mask_topk_count_per_row(self):
        """Hard top-k: each row has exactly k = round(density * d_h) ones."""
        n = 10
        k = 3
        cell = STENeuronWiseCfCCell(
            input_size=4, hidden_size=n, density=k / n
        )
        mask = cell.get_neighborhood_mask()
        row_sums = mask.sum(dim=-1)
        # All rows should have exactly k ones.
        for i in range(n):
            self.assertEqual(int(row_sums[i].item()), k)

    def test_ste_mask_backward_flows_grad(self):
        """The STE must propagate grad to neighbor_logits."""
        cell = STENeuronWiseCfCCell(input_size=4, hidden_size=8, density=0.5)
        # Forward mask is detached; backward uses soft (not detached).
        mask = cell.get_neighborhood_mask()
        # Verify that autograd wires soft into the graph.
        self.assertTrue(mask.requires_grad)

    def test_ste_hard_mask_matches_topk(self):
        cell = STENeuronWiseCfCCell(input_size=4, hidden_size=8, density=0.5)
        hard = cell.get_ste_hard_mask()
        self.assertTrue(((hard == 0.0) | (hard == 1.0)).all())
        # Each row should have 4 ones (50% of 8).
        row_sums = hard.sum(dim=-1)
        for i in range(8):
            self.assertEqual(int(row_sums[i].item()), 4)

    def test_ste_soft_mask_in_unit_interval(self):
        cell = STENeuronWiseCfCCell(input_size=4, hidden_size=8, density=0.5)
        soft = cell.get_ste_soft_mask()
        self.assertTrue((soft >= 0.0).all())
        self.assertTrue((soft <= 1.0).all())


class TestSTENeuronWiseCfCCellForward(unittest.TestCase):

    def test_forward_shape(self):
        cell = STENeuronWiseCfCCell(input_size=4, hidden_size=8, density=0.5)
        x = torch.randn(2, 16, 4)
        out, h = cell(x)
        self.assertEqual(out.shape, (2, 16, 8))
        self.assertEqual(h.shape, (2, 8))
        del out, h

    def test_forward_with_h0(self):
        cell = STENeuronWiseCfCCell(input_size=4, hidden_size=8, density=0.5)
        x = torch.randn(2, 16, 4)
        h0 = torch.randn(2, 8)
        out, h = cell(x, h0=h0)
        self.assertEqual(out.shape, (2, 16, 8))
        self.assertFalse(torch.allclose(h, h0))
        del out

    def test_forward_output_finite(self):
        cell = STENeuronWiseCfCCell(input_size=4, hidden_size=8, density=0.5)
        x = torch.randn(2, 16, 4)
        out, h = cell(x)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.isfinite(h).all())
        del out, h

    def test_gradients_flow_including_neighbor_logits(self):
        """KEY TEST: neighbor_logits IS grad-trained via STE."""
        cell = STENeuronWiseCfCCell(input_size=4, hidden_size=8, density=0.5)
        x = torch.randn(2, 16, 4)
        target = torch.randn(2, 16, 8)
        out, _ = cell(x)
        loss = (out - target).pow(2).mean()
        loss.backward()
        # neighbor_logits must receive a finite, non-zero grad.
        self.assertIsNotNone(cell.neighbor_logits.grad)
        self.assertTrue(torch.isfinite(cell.neighbor_logits.grad).all())
        # At least some grad entries must be non-zero.
        self.assertGreater(cell.neighbor_logits.grad.abs().sum().item(), 0.0)

    def test_other_params_grad(self):
        cell = STENeuronWiseCfCCell(input_size=4, hidden_size=8, density=0.5)
        x = torch.randn(2, 16, 4)
        target = torch.randn(2, 16, 8)
        out, _ = cell(x)
        loss = (out - target).pow(2).mean()
        loss.backward()
        for name in ["W_rec", "tau_per_neuron", "alpha_per_neuron",
                     "bias_per_neuron", "input_strength_per_neuron"]:
            p = getattr(cell, name)
            self.assertIsNotNone(p.grad, f"no grad for {name}")


class TestSTENeuronWiseCfCCellDiagnostics(unittest.TestCase):

    def test_get_tau_bounded(self):
        cell = STENeuronWiseCfCCell(input_size=4, hidden_size=8, density=0.5)
        tau = cell.get_tau()
        self.assertTrue((tau >= cell.tau_min).all())
        self.assertTrue((tau <= cell.tau_max).all())

    def test_get_alpha_bounded(self):
        cell = STENeuronWiseCfCCell(input_size=4, hidden_size=8, density=0.5)
        with torch.no_grad():
            cell.alpha_per_neuron.fill_(100.0)
        alpha = cell.get_alpha()
        self.assertTrue((alpha <= cell.alpha_max + 1e-6).all())

    def test_neighborhood_density_within_d(self):
        cell = STENeuronWiseCfCCell(
            input_size=4, hidden_size=8, density=0.5
        )
        d = cell.neighborhood_density()
        self.assertLessEqual(d, 0.5 + 1e-6)

    def test_ste_temperature_low_produces_sharp_soft(self):
        cell_cold = STENeuronWiseCfCCell(
            input_size=4, hidden_size=8, density=0.5, ste_temperature=0.1
        )
        cell_warm = STENeuronWiseCfCCell(
            input_size=4, hidden_size=8, density=0.5, ste_temperature=10.0
        )
        # Same neighbor_logits (same seed).
        cell_warm.neighbor_logits.data = cell_cold.neighbor_logits.data.clone()
        soft_cold = cell_cold.get_ste_soft_mask()
        soft_warm = cell_warm.get_ste_soft_mask()
        # Cold should be more peaked (closer to 0 or 1).
        # Variance of cold is higher (more extreme values).
        # Mean should still be ~0.5 (sigmoid symmetric around 0).
        # So compare extremes: fraction near 0 or 1.
        frac_extreme_cold = ((soft_cold < 0.1) | (soft_cold > 0.9)).float().mean()
        frac_extreme_warm = ((soft_warm < 0.1) | (soft_warm > 0.9)).float().mean()
        self.assertGreater(frac_extreme_cold.item(), frac_extreme_warm.item())


class TestSTENeuronWiseCfCCellBatchIndependence(unittest.TestCase):

    def test_batch_elems_independent(self):
        cell = STENeuronWiseCfCCell(input_size=4, hidden_size=8, density=0.5)
        x1 = torch.zeros(1, 16, 4)
        x2 = torch.ones(1, 16, 4)
        x = torch.cat([x1, x2], dim=0)
        out, _ = cell(x)
        self.assertFalse(torch.allclose(out[0], out[1]))


if __name__ == "__main__":
    unittest.main()
