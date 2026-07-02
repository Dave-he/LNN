"""Unit tests for BlendGatedLiquidTauCfCCell (round 280).

Verifies the blend gate max(velocity, acceleration): recovers the high
gate of whichever predictability signal fires on predictable data,
collapses on noise (both high), and reproduces r278 (velocity) / r279
(acceleration) exactly in the pure modes.
"""

from __future__ import annotations

import unittest

import torch

from lnn.core.blend_gated_liquid_tau_cfc import BlendGatedLiquidTauCfCCell
from lnn.core.accel_gated_liquid_tau_cfc import AccelGatedLiquidTauCfCCell
from lnn.core.pred_gated_liquid_tau_cfc import PredictabilityGatedLiquidTauCfCCell
from lnn.core.liquid_tau_ste_cfc import LiquidTauSTECfCCell


def _sine(B=4, T=64, freq=3.0):
    t = torch.linspace(0, 6.2831853 * freq, T).view(1, T, 1).repeat(B, 1, 1)
    return torch.sin(t)


def _noise(B=4, T=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, 1, generator=g)


def _structured(B=8, T=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(B, T, 1)
    for b in range(B):
        lv = 0.0
        for i in range(T):
            if i % 16 == 0:
                lv = float(torch.randint(0, 4, (1,), generator=g)) - 1.5
            x[b, i, 0] = lv
    return x


class TestBlendBasics(unittest.TestCase):

    def test_init_defaults(self):
        cell = BlendGatedLiquidTauCfCCell(input_size=2, hidden_size=8)
        self.assertEqual(cell.gate_mode, "blend")
        self.assertEqual(cell.pred_gate_beta, 4.0)

    def test_subclass_chain(self):
        cell = BlendGatedLiquidTauCfCCell(input_size=2, hidden_size=8)
        self.assertIsInstance(cell, AccelGatedLiquidTauCfCCell)
        self.assertIsInstance(cell, PredictabilityGatedLiquidTauCfCCell)
        self.assertIsInstance(cell, LiquidTauSTECfCCell)

    def test_gate_mode_validation(self):
        with self.assertRaises(ValueError):
            BlendGatedLiquidTauCfCCell(
                input_size=2, hidden_size=8, gate_mode="bogus")

    def test_inherits_beta_validation(self):
        with self.assertRaises(ValueError):
            BlendGatedLiquidTauCfCCell(
                input_size=2, hidden_size=8, pred_gate_beta=-1.0)


class TestForwardShapes(unittest.TestCase):

    def test_forward_shape(self):
        cell = BlendGatedLiquidTauCfCCell(input_size=1, hidden_size=16, seed=1)
        out, h = cell(_sine(3, 20))
        self.assertEqual(out.shape, (3, 20, 16))
        self.assertEqual(h.shape, (3, 16))

    def test_forward_aux_keys(self):
        cell = BlendGatedLiquidTauCfCCell(input_size=1, hidden_size=16, seed=1)
        _, _, aux = cell(_sine(2, 20), return_aux=True)
        for k in ("gate_mean", "gate_min", "gate_max", "gate_mode",
                  "tau_temporal_std"):
            self.assertIn(k, aux)
        self.assertEqual(aux["gate_mode"], "blend")

    def test_finite_on_noise(self):
        cell = BlendGatedLiquidTauCfCCell(input_size=1, hidden_size=16, seed=1)
        out, h = cell(_noise(4, 32, seed=7))
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.isfinite(h).all())


class TestSupersetProperty(unittest.TestCase):
    """velocity ≡ r278, acceleration ≡ r279 (bit-identical on same object)."""

    def test_velocity_matches_r278(self):
        cell = BlendGatedLiquidTauCfCCell(
            input_size=1, hidden_size=16, seed=1, gate_mode="velocity")
        x = _sine(3, 24)
        o_blend, h_blend = cell.forward(x)
        o_par, h_par = PredictabilityGatedLiquidTauCfCCell.forward(cell, x)
        self.assertLess(float((o_blend - o_par).abs().max()), 1e-6)
        self.assertLess(float((h_blend - h_par).abs().max()), 1e-6)

    def test_acceleration_matches_r279(self):
        cell = BlendGatedLiquidTauCfCCell(
            input_size=1, hidden_size=16, seed=1, gate_mode="acceleration")
        x = _sine(3, 24)
        o_blend, h_blend = cell.forward(x)
        o_par, h_par = AccelGatedLiquidTauCfCCell.forward(cell, x)
        self.assertLess(float((o_blend - o_par).abs().max()), 1e-6)
        self.assertLess(float((h_blend - h_par).abs().max()), 1e-6)


class TestBlendSemantics(unittest.TestCase):
    """The headline mechanism: blend gate = best-of-both, collapses on noise."""

    def test_blend_ge_both_components(self):
        # On predictable data, blend gate_mean ≥ each component gate_mean.
        x = _sine(8, 64)
        c_blend = BlendGatedLiquidTauCfCCell(
            input_size=1, hidden_size=16, seed=1, gate_mode="blend")
        c_vel = BlendGatedLiquidTauCfCCell(
            input_size=1, hidden_size=16, seed=1, gate_mode="velocity")
        c_acc = BlendGatedLiquidTauCfCCell(
            input_size=1, hidden_size=16, seed=1, gate_mode="acceleration")
        _, _, ab = c_blend(x, return_aux=True)
        _, _, av = c_vel(x, return_aux=True)
        _, _, aa = c_acc(x, return_aux=True)
        # max ⇒ blend ≥ both (small tolerance for EMA warmup edge).
        self.assertGreaterEqual(ab["gate_mean"] + 1e-6, av["gate_mean"])
        self.assertGreaterEqual(ab["gate_mean"] + 1e-6, aa["gate_mean"])

    def test_sine_gate_high(self):
        cell = BlendGatedLiquidTauCfCCell(input_size=1, hidden_size=16, seed=1)
        _, _, aux = cell(_sine(8, 64), return_aux=True)
        self.assertGreater(aux["gate_mean"], 0.7)

    def test_structured_gate_high(self):
        cell = BlendGatedLiquidTauCfCCell(input_size=1, hidden_size=16, seed=1)
        _, _, aux = cell(_structured(8, 64), return_aux=True)
        self.assertGreater(aux["gate_mean"], 0.8)

    def test_noise_gate_low(self):
        cell = BlendGatedLiquidTauCfCCell(input_size=1, hidden_size=16, seed=1)
        _, _, aux = cell(_noise(8, 64, seed=3), return_aux=True)
        self.assertLess(aux["gate_mean"], 0.15)

    def test_beta_zero_gate_is_one(self):
        cell = BlendGatedLiquidTauCfCCell(
            input_size=1, hidden_size=16, seed=1, pred_gate_beta=0.0)
        _, _, aux = cell(_noise(4, 32, seed=9), return_aux=True)
        self.assertAlmostEqual(aux["gate_mean"], 1.0, places=5)


class TestGradientFlow(unittest.TestCase):

    def test_gradients_flow(self):
        cell = BlendGatedLiquidTauCfCCell(input_size=1, hidden_size=16, seed=1)
        out, _ = cell(_sine(4, 32))
        out.pow(2).mean().backward()
        self.assertIsNotNone(cell.W_tau.weight.grad)
        self.assertGreater(float(cell.W_tau.weight.grad.abs().sum()), 0.0)
        self.assertIsNotNone(cell.W_rec.grad)

    def test_gate_is_not_learnable(self):
        cell = BlendGatedLiquidTauCfCCell(input_size=1, hidden_size=8)
        names = {n for n, _ in cell.named_parameters()}
        self.assertNotIn("gate_mode", names)
        self.assertNotIn("pred_gate_beta", names)


if __name__ == "__main__":
    unittest.main()
