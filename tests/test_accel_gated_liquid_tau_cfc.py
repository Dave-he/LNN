"""Unit tests for AccelGatedLiquidTauCfCCell (round 279).

Verifies the acceleration-gated liquid τ: gate on the 2nd difference
(|Δ²x|) instead of the 1st (|Δ¹x|), so smooth-but-fast signals (a sine)
read as predictable while erratic noise collapses the gate. diff_order=1
recovers r278 exactly; inherited liquid/STE machinery is preserved.
"""

from __future__ import annotations

import unittest

import torch

from lnn.core.accel_gated_liquid_tau_cfc import AccelGatedLiquidTauCfCCell
from lnn.core.pred_gated_liquid_tau_cfc import PredictabilityGatedLiquidTauCfCCell
from lnn.core.liquid_tau_ste_cfc import LiquidTauSTECfCCell


def _sine(B=4, T=64, freq=3.0):
    t = torch.linspace(0, 6.2831853 * freq, T).view(1, T, 1).repeat(B, 1, 1)
    return torch.sin(t)


def _noise(B=4, T=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, 1, generator=g)


class TestAccelGatedBasics(unittest.TestCase):

    def test_init_defaults(self):
        cell = AccelGatedLiquidTauCfCCell(input_size=2, hidden_size=8)
        self.assertEqual(cell.diff_order, 2)
        self.assertEqual(cell.pred_gate_beta, 4.0)
        self.assertEqual(cell.ema_gamma, 0.5)

    def test_is_subclass_of_r278_and_liquid(self):
        cell = AccelGatedLiquidTauCfCCell(input_size=2, hidden_size=8)
        self.assertIsInstance(cell, PredictabilityGatedLiquidTauCfCCell)
        self.assertIsInstance(cell, LiquidTauSTECfCCell)

    def test_diff_order_validation(self):
        with self.assertRaises(ValueError):
            AccelGatedLiquidTauCfCCell(input_size=2, hidden_size=8, diff_order=0)
        with self.assertRaises(ValueError):
            AccelGatedLiquidTauCfCCell(input_size=2, hidden_size=8, diff_order=3)

    def test_inherits_beta_validation(self):
        with self.assertRaises(ValueError):
            AccelGatedLiquidTauCfCCell(
                input_size=2, hidden_size=8, pred_gate_beta=-1.0)


class TestForwardShapes(unittest.TestCase):

    def test_forward_shape(self):
        cell = AccelGatedLiquidTauCfCCell(input_size=1, hidden_size=16, seed=1)
        x = _sine(3, 20)
        out, h = cell(x)
        self.assertEqual(out.shape, (3, 20, 16))
        self.assertEqual(h.shape, (3, 16))

    def test_forward_aux_keys(self):
        cell = AccelGatedLiquidTauCfCCell(input_size=1, hidden_size=16, seed=1)
        _, _, aux = cell(_sine(2, 20), return_aux=True)
        for k in ("gate_mean", "gate_min", "gate_max", "diff_order",
                  "tau_temporal_std", "pred_gate_beta"):
            self.assertIn(k, aux)
        self.assertEqual(aux["diff_order"], 2)

    def test_finite_outputs(self):
        cell = AccelGatedLiquidTauCfCCell(input_size=1, hidden_size=16, seed=1)
        out, h = cell(_noise(4, 32, seed=7))
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.isfinite(h).all())


class TestSupersetProperty(unittest.TestCase):
    """diff_order=1 must reproduce r278 (velocity gate) forward exactly."""

    def test_order1_matches_r278_forward(self):
        cell = AccelGatedLiquidTauCfCCell(
            input_size=1, hidden_size=16, seed=1, diff_order=1)
        x = _sine(3, 24)
        out_accel, h_accel = cell.forward(x)
        # Parent (r278) forward on the SAME object = same weights.
        out_par, h_par = PredictabilityGatedLiquidTauCfCCell.forward(cell, x)
        self.assertLess(float((out_accel - out_par).abs().max()), 1e-6)
        self.assertLess(float((h_accel - h_par).abs().max()), 1e-6)

    def test_order2_differs_from_order1(self):
        # On a signal with velocity but low acceleration (sine), the two
        # gates must produce different trajectories.
        x = _sine(3, 48)
        c1 = AccelGatedLiquidTauCfCCell(
            input_size=1, hidden_size=16, seed=2, diff_order=1)
        c2 = AccelGatedLiquidTauCfCCell(
            input_size=1, hidden_size=16, seed=2, diff_order=2)
        # same seed → weights differ across instances (r278 W_in quirk),
        # so compare the GATE signal instead of the output.
        _, _, a1 = c1(x, return_aux=True)
        _, _, a2 = c2(x, return_aux=True)
        # order-2 (accel) gate on a sine should be HIGHER than order-1.
        self.assertGreater(a2["gate_mean"], a1["gate_mean"])


class TestGateSemantics(unittest.TestCase):
    """The headline mechanism: sine passes, noise collapses."""

    def test_sine_gate_high(self):
        cell = AccelGatedLiquidTauCfCCell(input_size=1, hidden_size=16, seed=1)
        _, _, aux = cell(_sine(8, 64), return_aux=True)
        self.assertGreater(aux["gate_mean"], 0.6)

    def test_noise_gate_low(self):
        cell = AccelGatedLiquidTauCfCCell(input_size=1, hidden_size=16, seed=1)
        _, _, aux = cell(_noise(8, 64, seed=3), return_aux=True)
        self.assertLess(aux["gate_mean"], 0.15)

    def test_sine_gate_beats_noise_gate(self):
        cell = AccelGatedLiquidTauCfCCell(input_size=1, hidden_size=16, seed=1)
        _, _, a_sin = cell(_sine(8, 64), return_aux=True)
        _, _, a_noise = cell(_noise(8, 64, seed=5), return_aux=True)
        # The whole point of r279: sine >> noise (r278 velocity gate had
        # them nearly equal).
        self.assertGreater(a_sin["gate_mean"], a_noise["gate_mean"] + 0.4)

    def test_beta_zero_gate_is_one(self):
        cell = AccelGatedLiquidTauCfCCell(
            input_size=1, hidden_size=16, seed=1, pred_gate_beta=0.0)
        _, _, aux = cell(_noise(4, 32, seed=9), return_aux=True)
        self.assertAlmostEqual(aux["gate_mean"], 1.0, places=5)
        self.assertAlmostEqual(aux["gate_min"], 1.0, places=5)


class TestGradientFlow(unittest.TestCase):

    def test_gradients_flow(self):
        cell = AccelGatedLiquidTauCfCCell(input_size=1, hidden_size=16, seed=1)
        x = _sine(4, 32)
        out, _ = cell(x)
        loss = out.pow(2).mean()
        loss.backward()
        # W_tau (liquid) and W_rec (recurrent) must receive gradient.
        self.assertIsNotNone(cell.W_tau.weight.grad)
        self.assertGreater(float(cell.W_tau.weight.grad.abs().sum()), 0.0)
        self.assertIsNotNone(cell.W_rec.grad)

    def test_gate_is_not_learnable(self):
        # The predictability gate has no parameters — the only τ-path
        # params come from W_tau (inherited). Verify no new Parameters.
        cell = AccelGatedLiquidTauCfCCell(input_size=1, hidden_size=8)
        names = {n for n, _ in cell.named_parameters()}
        # diff_order / beta / ema are plain floats, not Parameters.
        self.assertNotIn("diff_order", names)
        self.assertNotIn("pred_gate_beta", names)


if __name__ == "__main__":
    unittest.main()
