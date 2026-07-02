"""Unit tests for PredictabilityGatedLiquidTauCfCCell (round 278).

Verifies the parameter-free predictability gate that scales the liquid
τ contribution: gate ∈ (0,1], high on smooth input / low on noisy input,
beta=0 recovers r277, and inherited liquid/STE machinery is preserved.
"""

from __future__ import annotations

import math
import unittest

import torch

from lnn.core.pred_gated_liquid_tau_cfc import PredictabilityGatedLiquidTauCfCCell
from lnn.core.liquid_tau_ste_cfc import LiquidTauSTECfCCell


class TestPredGatedBasics(unittest.TestCase):

    def test_init_defaults(self):
        cell = PredictabilityGatedLiquidTauCfCCell(input_size=2, hidden_size=8)
        self.assertEqual(cell.pred_gate_beta, 4.0)
        self.assertEqual(cell.ema_gamma, 0.5)
        self.assertEqual(cell.liquid_tau_strength, 1.0)

    def test_is_subclass_of_liquid(self):
        cell = PredictabilityGatedLiquidTauCfCCell(input_size=2, hidden_size=8)
        self.assertIsInstance(cell, LiquidTauSTECfCCell)

    def test_beta_validation(self):
        with self.assertRaises(ValueError):
            PredictabilityGatedLiquidTauCfCCell(
                input_size=2, hidden_size=8, pred_gate_beta=-1.0)

    def test_ema_gamma_validation(self):
        with self.assertRaises(ValueError):
            PredictabilityGatedLiquidTauCfCCell(
                input_size=2, hidden_size=8, ema_gamma=1.0)
        with self.assertRaises(ValueError):
            PredictabilityGatedLiquidTauCfCCell(
                input_size=2, hidden_size=8, ema_gamma=-0.1)


class TestPredictabilityGate(unittest.TestCase):

    def test_gate_range(self):
        cell = PredictabilityGatedLiquidTauCfCCell(input_size=2, hidden_size=8)
        vol = torch.tensor([0.0, 0.5, 2.0, 10.0])
        g = cell.predictability_gate(vol)
        self.assertEqual(g.shape, (4, 1))
        self.assertTrue(torch.all(g > 0.0))
        self.assertTrue(torch.all(g <= 1.0 + 1e-6))

    def test_gate_one_at_zero_vol(self):
        cell = PredictabilityGatedLiquidTauCfCCell(input_size=2, hidden_size=8)
        g = cell.predictability_gate(torch.zeros(3))
        self.assertTrue(torch.allclose(g, torch.ones(3, 1), atol=1e-6))

    def test_gate_monotonic_decreasing(self):
        cell = PredictabilityGatedLiquidTauCfCCell(input_size=2, hidden_size=8)
        vol = torch.tensor([0.0, 0.5, 1.0, 2.0])
        g = cell.predictability_gate(vol).squeeze(-1)
        # gate should strictly decrease as volatility rises.
        self.assertTrue(torch.all(g[:-1] > g[1:]))

    def test_gate_collapses_under_high_vol(self):
        cell = PredictabilityGatedLiquidTauCfCCell(
            input_size=2, hidden_size=8, pred_gate_beta=4.0)
        g = cell.predictability_gate(torch.tensor([5.0]))
        self.assertLess(float(g.item()), 0.01)


class TestGatedTau(unittest.TestCase):

    def test_gated_tau_shape(self):
        cell = PredictabilityGatedLiquidTauCfCCell(input_size=2, hidden_size=8)
        x_t = torch.randn(5, 2)
        h = torch.randn(5, 8)
        gate = torch.rand(5, 1)
        tau = cell.get_tau_gated(x_t, h, gate)
        self.assertEqual(tau.shape, (5, 8))

    def test_gated_tau_bounded(self):
        cell = PredictabilityGatedLiquidTauCfCCell(
            input_size=2, hidden_size=8, tau_min=0.05, tau_max=0.95)
        with torch.no_grad():
            cell.W_tau.weight.normal_(0, 5.0)
        x_t = torch.randn(16, 2) * 10
        h = torch.randn(16, 8) * 10
        gate = torch.ones(16, 1)
        tau = cell.get_tau_gated(x_t, h, gate)
        self.assertTrue(torch.all(tau >= 0.05 - 1e-6))
        self.assertTrue(torch.all(tau <= 0.95 + 1e-6))

    def test_gate_zero_collapses_to_static(self):
        """gate=0 ⇒ τ == static per-neuron τ regardless of input."""
        cell = PredictabilityGatedLiquidTauCfCCell(input_size=2, hidden_size=8)
        with torch.no_grad():
            cell.W_tau.weight.normal_(0, 5.0)
        x_t = torch.randn(4, 2)
        h = torch.randn(4, 8)
        gate = torch.zeros(4, 1)
        tau = cell.get_tau_gated(x_t, h, gate)
        static = cell.get_tau().unsqueeze(0).expand(4, -1)
        self.assertTrue(torch.allclose(tau, static, atol=1e-6))


class TestForward(unittest.TestCase):

    def test_forward_shape(self):
        cell = PredictabilityGatedLiquidTauCfCCell(input_size=2, hidden_size=8)
        x = torch.randn(4, 10, 2)
        out, h = cell(x)
        self.assertEqual(out.shape, (4, 10, 8))
        self.assertEqual(h.shape, (4, 8))

    def test_forward_aux_gate_keys(self):
        cell = PredictabilityGatedLiquidTauCfCCell(input_size=2, hidden_size=8)
        x = torch.randn(4, 10, 2)
        out, h, aux = cell(x, return_aux=True)
        for key in ("gate_mean", "gate_min", "gate_max",
                    "tau_temporal_std", "pred_gate_beta"):
            self.assertIn(key, aux)

    def test_gate_high_on_smooth_low_on_noisy(self):
        """H3 mechanism: smooth input → high gate, noisy input → low gate."""
        cell = PredictabilityGatedLiquidTauCfCCell(
            input_size=1, hidden_size=8, pred_gate_beta=4.0)
        T = 40
        t = torch.linspace(0, 1, T).view(1, T, 1).repeat(4, 1, 1)
        smooth = torch.sin(2 * math.pi * t)
        noisy = torch.randn(4, T, 1)
        _, _, aux_s = cell(smooth, return_aux=True)
        _, _, aux_n = cell(noisy, return_aux=True)
        self.assertGreater(aux_s["gate_mean"], aux_n["gate_mean"])

    def test_beta_zero_matches_r277(self):
        """H4 superset: beta=0 ⇒ gate≡1 ⇒ exactly r277 liquid cell."""
        kw = dict(input_size=1, hidden_size=8, density=0.3, seed=7)
        gated = PredictabilityGatedLiquidTauCfCCell(pred_gate_beta=0.0, **kw)
        liquid = LiquidTauSTECfCCell(**kw)
        # Copy shared params so only the gate mechanism can differ.
        gated.load_state_dict(
            {k: v for k, v in liquid.state_dict().items()
             if k in gated.state_dict()},
            strict=False,
        )
        # Break zero-init so the liquid path is actually exercised.
        with torch.no_grad():
            w = torch.randn_like(gated.W_tau.weight) * 0.5
            gated.W_tau.weight.copy_(w)
            liquid.W_tau.weight.copy_(w)
        x = torch.randn(3, 15, 1)
        out_g, _ = gated(x)
        out_l, _ = liquid(x)
        self.assertTrue(torch.allclose(out_g, out_l, atol=1e-5))

    def test_gradients_flow_to_gate_weight(self):
        cell = PredictabilityGatedLiquidTauCfCCell(input_size=2, hidden_size=8)
        with torch.no_grad():
            cell.W_tau.weight.normal_(0, 0.5)
        x = torch.randn(4, 10, 2)
        out, _ = cell(x)
        out.pow(2).mean().backward()
        self.assertIsNotNone(cell.W_tau.weight.grad)
        self.assertGreater(cell.W_tau.weight.grad.abs().sum().item(), 0.0)


class TestInheritedMachinery(unittest.TestCase):

    def test_inherits_entropy_loss(self):
        cell = PredictabilityGatedLiquidTauCfCCell(
            input_size=2, hidden_size=8, entropy_lambda=0.1)
        self.assertGreater(float(cell.extra_loss().item()), 0.0)

    def test_inherits_ste_hard_mask_binary(self):
        cell = PredictabilityGatedLiquidTauCfCCell(input_size=2, hidden_size=8)
        hard = cell.get_ste_hard_mask()
        self.assertTrue(set(hard.unique().tolist()).issubset({0.0, 1.0}))


if __name__ == "__main__":
    unittest.main()
