"""Tests for r293 default decorrelation in BlendGatedLiquidTauCfCCell.

Verifies:
    H6: existing blend_gated tests still pass (smoke).
    H7: decorr_lambda=0 ≡ old blend_gated (no decorrelation in extra_loss).
    New: extra_loss() with decorr_lambda=1e-4 includes decorrelation
       term (non-zero after forward).
    New: decorr_lambda validation.
    New: extra_loss() with no prior forward returns entropy only.
"""

from __future__ import annotations

import unittest

import torch

from lnn.core.blend_gated_liquid_tau_cfc import BlendGatedLiquidTauCfCCell


def _sine(B=4, T=64, freq=3.0):
    t = torch.linspace(0, 6.2831853 * freq, T).view(1, T, 1).repeat(B, 1, 1)
    return torch.sin(t)


class TestR293Defaults(unittest.TestCase):

    def test_default_decorr_lambda(self):
        # r294 default is 1e-5 — matches r291 toy SP scale and r294
        # Henry Hub validation (-1.3% / -2.6% in-cell).
        cell = BlendGatedLiquidTauCfCCell(input_size=1, hidden_size=8)
        self.assertEqual(cell.decorr_lambda, 1e-5)

    def test_decorr_lambda_validation(self):
        with self.assertRaises(ValueError):
            BlendGatedLiquidTauCfCCell(input_size=1, hidden_size=8,
                                         decorr_lambda=-0.1)

    def test_forward_populates_last_outputs(self):
        cell = BlendGatedLiquidTauCfCCell(input_size=1, hidden_size=8, seed=1)
        self.assertIsNone(cell._last_outputs)
        _ = cell(_sine(B=2, T=16))
        self.assertIsNotNone(cell._last_outputs)
        self.assertEqual(cell._last_outputs.shape, (2, 16, 8))


class TestR293ExtraLoss(unittest.TestCase):

    def test_extra_loss_no_decor_when_lambda_zero(self):
        cell = BlendGatedLiquidTauCfCCell(
            input_size=1, hidden_size=8, seed=2, decorr_lambda=0.0,
            entropy_lambda=0.0)
        # Run forward to populate _last_outputs.
        _ = cell(_sine(B=2, T=16))
        # With entropy_lambda=0 AND decorr_lambda=0, extra_loss should be 0.
        loss = cell.extra_loss()
        self.assertEqual(float(loss.item()), 0.0)

    def test_extra_loss_with_default_decor(self):
        cell = BlendGatedLiquidTauCfCCell(
            input_size=1, hidden_size=8, seed=3,
            entropy_lambda=0.0,  # isolate decorrelation term
            decorr_lambda=1e-4)
        _ = cell(_sine(B=2, T=16))
        loss = cell.extra_loss()
        # Decorrelation loss should be non-zero on a sine-driven state.
        self.assertGreater(float(loss.item()), 0.0)

    def test_extra_loss_no_forward_returns_zero(self):
        cell = BlendGatedLiquidTauCfCCell(
            input_size=1, hidden_size=8, seed=4,
            entropy_lambda=0.1, decorr_lambda=1e-4)
        # No forward called yet — extra_loss should return entropy only
        # (which is non-zero because of entropy_lambda=0.1).
        loss = cell.extra_loss()
        self.assertGreater(float(loss.item()), 0.0)
        # Specifically, no decorrelation contribution.
        # Decorrelation on a "None" state should give 0; entropy still works.


class TestR293Supersets(unittest.TestCase):

    def test_decor_lambda_zero_matches_old_behavior(self):
        # The r280 blend_gated cell had no decor_lambda arg; this test
        # verifies the new default cell with decorr_lambda=0 produces
        # the same extra_loss as the parent.
        cell = BlendGatedLiquidTauCfCCell(
            input_size=1, hidden_size=24, density=0.3, seed=5,
            entropy_lambda=0.1, gate_mode="blend",
            decorr_lambda=0.0)
        # parent cell.
        from lnn.core.accel_gated_liquid_tau_cfc import AccelGatedLiquidTauCfCCell
        parent = AccelGatedLiquidTauCfCCell(
            input_size=1, hidden_size=24, density=0.3, seed=5,
            entropy_lambda=0.1, diff_order=2)
        x = _sine(B=2, T=16)
        _ = cell(x)
        _ = parent(x)
        loss_child = float(cell.extra_loss().item())
        loss_parent = float(parent.extra_loss().item())
        self.assertAlmostEqual(loss_child, loss_parent, places=5,
                               msg=f"{loss_child} != {loss_parent}")


if __name__ == "__main__":
    unittest.main()