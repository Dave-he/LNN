"""Unit tests for STEWithEntropy (round 267)."""

from __future__ import annotations

import math
import unittest

import torch

from lnn.core.ste_entropy_neuron_wise_cfc import STEWithEntropy


class TestSTEWithEntropyBasics(unittest.TestCase):

    def test_init_defaults(self):
        cell = STEWithEntropy(input_size=4, hidden_size=8)
        self.assertEqual(cell.entropy_lambda, 0.0)
        self.assertEqual(cell.ste_temperature, 1.0)

    def test_init_entropy_validation(self):
        with self.assertRaises(ValueError):
            STEWithEntropy(input_size=4, hidden_size=8, entropy_lambda=-0.1)

    def test_inherits_ste_params(self):
        cell = STEWithEntropy(input_size=4, hidden_size=8, entropy_lambda=0.1)
        for name in [
            "W_rec", "neighbor_logits", "tau_per_neuron",
            "alpha_per_neuron", "bias_per_neuron",
            "input_strength_per_neuron", "W_in",
            "ste_temperature",
        ]:
            self.assertTrue(hasattr(cell, name), f"missing: {name}")


class TestSTEWithEntropyLoss(unittest.TestCase):

    def test_extra_loss_zero_when_lambda_zero(self):
        cell = STEWithEntropy(input_size=4, hidden_size=8, entropy_lambda=0.0)
        loss = cell.extra_loss()
        self.assertEqual(float(loss.item()), 0.0)

    def test_extra_loss_positive_when_lambda_positive(self):
        cell = STEWithEntropy(input_size=4, hidden_size=8, entropy_lambda=0.1)
        loss = cell.extra_loss()
        self.assertGreater(float(loss.item()), 0.0)

    def test_extra_loss_proportional_to_lambda(self):
        cell1 = STEWithEntropy(input_size=4, hidden_size=8, entropy_lambda=0.1)
        cell2 = STEWithEntropy(input_size=4, hidden_size=8, entropy_lambda=0.5)
        # Same neighbor_logits (same seed).
        cell2.neighbor_logits.data = cell1.neighbor_logits.data.clone()
        e1 = float(cell1.extra_loss().item())
        e2 = float(cell2.extra_loss().item())
        self.assertAlmostEqual(e2 / e1, 5.0, places=4)

    def test_extra_loss_grad_to_neighbor_logits(self):
        cell = STEWithEntropy(input_size=4, hidden_size=8, entropy_lambda=0.1)
        cell.extra_loss().backward()
        self.assertIsNotNone(cell.neighbor_logits.grad)
        self.assertTrue(torch.isfinite(cell.neighbor_logits.grad).all())

    def test_entropy_bounded_by_log_d(self):
        """Entropy must be in [0, log(d_h)]."""
        cell = STEWithEntropy(input_size=4, hidden_size=8, entropy_lambda=0.1)
        h = float(cell.entropy_value())
        self.assertGreaterEqual(h, 0.0)
        self.assertLessEqual(h, math.log(8) + 1e-4)

    def test_max_entropy_helper(self):
        cell = STEWithEntropy(input_size=4, hidden_size=16, entropy_lambda=0.1)
        self.assertAlmostEqual(cell.max_entropy(), math.log(16), places=4)


class TestSTEWithEntropyForward(unittest.TestCase):

    def test_forward_shape(self):
        cell = STEWithEntropy(input_size=4, hidden_size=8, entropy_lambda=0.1)
        x = torch.randn(2, 16, 4)
        out, h = cell(x)
        self.assertEqual(out.shape, (2, 16, 8))
        self.assertEqual(h.shape, (2, 8))
        del out, h

    def test_forward_with_h0(self):
        cell = STEWithEntropy(input_size=4, hidden_size=8, entropy_lambda=0.1)
        x = torch.randn(2, 16, 4)
        h0 = torch.randn(2, 8)
        out, h = cell(x, h0=h0)
        self.assertEqual(out.shape, (2, 16, 8))
        self.assertFalse(torch.allclose(h, h0))
        del out

    def test_gradients_flow(self):
        """Both task loss and entropy reg should flow to neighbor_logits."""
        cell = STEWithEntropy(input_size=4, hidden_size=8, entropy_lambda=0.1)
        x = torch.randn(2, 16, 4)
        target = torch.randn(2, 16, 8)
        out, _ = cell(x)
        task_loss = (out - target).pow(2).mean()
        e_loss = cell.extra_loss()
        total = task_loss + e_loss
        total.backward()
        # neighbor_logits must have a finite grad.
        self.assertIsNotNone(cell.neighbor_logits.grad)
        self.assertTrue(torch.isfinite(cell.neighbor_logits.grad).all())


class TestSTEWithEntropyMask(unittest.TestCase):

    def test_ste_mask_forward_is_binary(self):
        """The STE mask is still binary in forward (entropy reg doesn't
        change the mask mechanism)."""
        cell = STEWithEntropy(input_size=4, hidden_size=8, entropy_lambda=0.5)
        mask = cell.get_neighborhood_mask()
        unique_vals = torch.unique(mask)
        for v in unique_vals.tolist():
            self.assertIn(v, [0.0, 1.0])

    def test_ste_hard_mask_count(self):
        cell = STEWithEntropy(input_size=4, hidden_size=10, entropy_lambda=0.5, density=0.3)
        hard = cell.get_ste_hard_mask()
        row_sums = hard.sum(dim=-1)
        for i in range(10):
            self.assertEqual(int(row_sums[i].item()), 3)


if __name__ == "__main__":
    unittest.main()