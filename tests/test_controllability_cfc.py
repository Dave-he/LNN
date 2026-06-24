"""Tests for ControllabilityCfCCell (arXiv:2606.08431 response, round 241)."""

from __future__ import annotations

import unittest

import torch

from lnn.core.controllability_cfc import (
    ControllabilityCfCCell,
    controllability_loss,
    input_jacobian_norm,
    input_sensitivity,
)


class TestInputSensitivity(unittest.TestCase):
    def test_high_when_input_drives_cell(self):
        """A large input delta should produce a large sensitivity."""
        d_in, d_h, B = 3, 8, 4
        cell = ControllabilityCfCCell(d_in, d_h).cell
        h = torch.zeros(B, d_h)
        x_t = torch.randn(B, d_in) * 5.0  # large input
        c_t = input_sensitivity(x_t, h, cell)
        # Each sample must be non-negative (it's a ratio of norms).
        self.assertTrue((c_t >= -1e-5).all())
        # Mean should be substantial.
        self.assertGreater(c_t.mean().item(), 0.05)

    def test_low_when_input_is_zero(self):
        d_in, d_h, B = 2, 6, 4
        cell = ControllabilityCfCCell(d_in, d_h).cell
        h = torch.zeros(B, d_h)
        x_t = torch.zeros(B, d_in)  # zero input
        c_t = input_sensitivity(x_t, h, cell)
        # x_t == 0 means h_with == h_without, so diff is 0 and c_t = 0.
        self.assertTrue(torch.allclose(c_t, torch.zeros(B), atol=1e-5))


class TestControllabilityLoss(unittest.TestCase):
    def test_zero_when_above_margin(self):
        d_in, d_h, B = 3, 8, 4
        cell = ControllabilityCfCCell(d_in, d_h).cell
        h = torch.zeros(B, d_h)
        x_t = torch.randn(B, d_in) * 5.0
        loss = controllability_loss(x_t, h, cell, margin=0.01)
        # Sensitivity > 0.01 -> loss = 0.
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_positive_when_below_margin(self):
        d_in, d_h, B = 2, 6, 4
        cell = ControllabilityCfCCell(d_in, d_h).cell
        h = torch.zeros(B, d_h)
        x_t = torch.zeros(B, d_in)
        loss = controllability_loss(x_t, h, cell, margin=0.5)
        # Zero input -> sensitivity = 0 -> loss = margin.
        self.assertAlmostEqual(loss.item(), 0.5, places=5)


class TestInputJacobianNorm(unittest.TestCase):
    def test_positive_for_random_input(self):
        d_in, d_h, B = 2, 4, 2
        cell = ControllabilityCfCCell(d_in, d_h).cell
        h = torch.zeros(B, d_h)
        x_t = torch.randn(B, d_in)
        jn = input_jacobian_norm(x_t, h, cell)
        # Some gradient must exist (parameters are random, inputs random).
        self.assertGreater(jn.mean().item(), 0.0)


class TestControllabilityCfCCell(unittest.TestCase):
    def test_forward_shape(self):
        d_in, d_h, B = 3, 8, 4
        cell = ControllabilityCfCCell(d_in, d_h)
        h = torch.zeros(B, d_h)
        x_t = torch.randn(B, d_in)
        h_next = cell(x_t, h)
        self.assertEqual(h_next.shape, (B, d_h))

    def test_forward_with_aux(self):
        d_in, d_h, B = 3, 8, 4
        cell = ControllabilityCfCCell(d_in, d_h)
        h = torch.zeros(B, d_h)
        x_t = torch.randn(B, d_in) * 3.0
        h_next, aux = cell.forward_with_aux(x_t, h, ctrl_lambda=0.5)
        self.assertEqual(h_next.shape, (B, d_h))
        for k in ("h_next", "c_t", "ctrl_loss", "ctrl_loss_total", "jacobian_norm"):
            self.assertIn(k, aux)
        self.assertAlmostEqual(
            aux["ctrl_loss_total"].item(),
            0.5 * aux["ctrl_loss"].item(),
            places=5,
        )

    def test_multi_tau(self):
        d_in, d_h, n_tau = 3, 6, 3
        cell = ControllabilityCfCCell(d_in, d_h, n_tau=n_tau, tau_scales=(0.1, 1.0, 10.0))
        h = torch.zeros(2, d_h)
        x_t = torch.randn(2, d_in)
        h_next, aux = cell.forward_with_aux(x_t, h)
        self.assertEqual(h_next.shape, (2, d_h))
        self.assertEqual(cell.cell.n_tau, 3)

    def test_margin_propagates(self):
        cell = ControllabilityCfCCell(2, 4, margin=0.123)
        self.assertAlmostEqual(cell.margin, 0.123, places=6)


if __name__ == "__main__":
    unittest.main()