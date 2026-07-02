"""Unit tests for LiquidTauSTECfCCell (round 277).

Verifies the input-dependent (liquid) τ mechanism layered on top of
STEWithEntropy: strict-superset behaviour at init (zero-init gate),
per-timestep τ variation, and that inherited STE / entropy machinery
is preserved.
"""

from __future__ import annotations

import unittest

import torch

from lnn.core.liquid_tau_ste_cfc import LiquidTauSTECfCCell
from lnn.core.ste_entropy_neuron_wise_cfc import STEWithEntropy


class TestLiquidTauBasics(unittest.TestCase):

    def test_init_defaults(self):
        cell = LiquidTauSTECfCCell(input_size=4, hidden_size=8)
        self.assertEqual(cell.liquid_tau_strength, 1.0)
        self.assertEqual(cell.entropy_lambda, 0.0)
        self.assertEqual(cell.ste_temperature, 1.0)

    def test_is_subclass_of_ste_entropy(self):
        cell = LiquidTauSTECfCCell(input_size=4, hidden_size=8)
        self.assertIsInstance(cell, STEWithEntropy)

    def test_strength_validation(self):
        with self.assertRaises(ValueError):
            LiquidTauSTECfCCell(input_size=4, hidden_size=8,
                                liquid_tau_strength=-0.1)

    def test_gate_is_zero_initialised(self):
        cell = LiquidTauSTECfCCell(input_size=4, hidden_size=8)
        self.assertTrue(torch.all(cell.W_tau.weight == 0.0))

    def test_gate_shape(self):
        cell = LiquidTauSTECfCCell(input_size=3, hidden_size=8)
        self.assertEqual(cell.W_tau.weight.shape, (8, 3 + 8))


class TestLiquidTauDynamic(unittest.TestCase):

    def test_dynamic_tau_shape(self):
        cell = LiquidTauSTECfCCell(input_size=4, hidden_size=8)
        x_t = torch.randn(5, 4)
        h = torch.randn(5, 8)
        tau = cell.get_tau_dynamic(x_t, h)
        self.assertEqual(tau.shape, (5, 8))

    def test_dynamic_tau_bounded(self):
        cell = LiquidTauSTECfCCell(input_size=4, hidden_size=8,
                                   tau_min=0.05, tau_max=0.95)
        # Force a big non-zero gate to push logits to extremes.
        with torch.no_grad():
            cell.W_tau.weight.normal_(0, 5.0)
        x_t = torch.randn(16, 4) * 10
        h = torch.randn(16, 8) * 10
        tau = cell.get_tau_dynamic(x_t, h)
        self.assertTrue(torch.all(tau >= 0.05 - 1e-6))
        self.assertTrue(torch.all(tau <= 0.95 + 1e-6))

    def test_zero_init_matches_static_tau(self):
        """At init (zero gate), dynamic τ == inherited static τ."""
        cell = LiquidTauSTECfCCell(input_size=4, hidden_size=8)
        x_t = torch.randn(7, 4)
        h = torch.randn(7, 8)
        dyn = cell.get_tau_dynamic(x_t, h)  # (B, d_h)
        static = cell.get_tau().unsqueeze(0)  # (1, d_h)
        self.assertTrue(torch.allclose(dyn, static.expand_as(dyn), atol=1e-6))

    def test_strength_zero_kills_liquid(self):
        """liquid_tau_strength=0 ⇒ τ is input-independent even with a
        non-zero gate."""
        cell = LiquidTauSTECfCCell(input_size=4, hidden_size=8,
                                   liquid_tau_strength=0.0)
        with torch.no_grad():
            cell.W_tau.weight.normal_(0, 5.0)
        x1 = torch.randn(3, 4)
        x2 = torch.randn(3, 4)
        h = torch.randn(3, 8)
        tau1 = cell.get_tau_dynamic(x1, h)
        tau2 = cell.get_tau_dynamic(x2, h)
        self.assertTrue(torch.allclose(tau1, tau2, atol=1e-6))

    def test_nonzero_gate_makes_tau_input_dependent(self):
        cell = LiquidTauSTECfCCell(input_size=4, hidden_size=8)
        with torch.no_grad():
            cell.W_tau.weight.normal_(0, 2.0)
        x1 = torch.randn(3, 4)
        x2 = torch.randn(3, 4)
        h = torch.zeros(3, 8)
        tau1 = cell.get_tau_dynamic(x1, h)
        tau2 = cell.get_tau_dynamic(x2, h)
        self.assertFalse(torch.allclose(tau1, tau2, atol=1e-4))


class TestLiquidTauForward(unittest.TestCase):

    def test_forward_shape(self):
        cell = LiquidTauSTECfCCell(input_size=2, hidden_size=8)
        x = torch.randn(4, 10, 2)
        out, h = cell(x)
        self.assertEqual(out.shape, (4, 10, 8))
        self.assertEqual(h.shape, (4, 8))

    def test_forward_with_h0(self):
        cell = LiquidTauSTECfCCell(input_size=2, hidden_size=8)
        x = torch.randn(4, 10, 2)
        h0 = torch.randn(4, 8)
        out, h = cell(x, h0=h0)
        self.assertEqual(out.shape, (4, 10, 8))

    def test_forward_aux_keys(self):
        cell = LiquidTauSTECfCCell(input_size=2, hidden_size=8)
        x = torch.randn(4, 10, 2)
        out, h, aux = cell(x, return_aux=True)
        for key in ("tau_temporal_std", "tau_dynamic_mean",
                    "tau_dynamic_min", "tau_dynamic_max",
                    "liquid_tau_strength", "mask"):
            self.assertIn(key, aux)

    def test_zero_init_forward_matches_static(self):
        """A zero-init liquid cell must produce the same forward output
        as an equivalent static-τ STEWithEntropy cell (superset)."""
        kw = dict(input_size=2, hidden_size=8, density=0.3, seed=42)
        liquid = LiquidTauSTECfCCell(**kw)
        static = STEWithEntropy(**kw)
        # Copy shared params so only the τ mechanism can differ.
        static.load_state_dict(
            {k: v for k, v in liquid.state_dict().items()
             if k in static.state_dict()},
            strict=True,
        )
        x = torch.randn(3, 12, 2)
        out_l, _ = liquid(x)
        out_s, _ = static(x)
        self.assertTrue(torch.allclose(out_l, out_s, atol=1e-5))

    def test_gradients_flow_to_gate(self):
        cell = LiquidTauSTECfCCell(input_size=2, hidden_size=8)
        # Break the zero-init so the gate has a non-trivial Jacobian.
        with torch.no_grad():
            cell.W_tau.weight.normal_(0, 0.5)
        x = torch.randn(4, 10, 2)
        out, _ = cell(x)
        out.pow(2).mean().backward()
        self.assertIsNotNone(cell.W_tau.weight.grad)
        self.assertGreater(cell.W_tau.weight.grad.abs().sum().item(), 0.0)

    def test_tau_flows_after_learning_gate(self):
        """With a trained (non-zero) gate, τ should vary across time."""
        cell = LiquidTauSTECfCCell(input_size=2, hidden_size=8)
        with torch.no_grad():
            cell.W_tau.weight.normal_(0, 1.0)
        # Sequence with real temporal variation.
        t = torch.linspace(0, 1, 20).view(1, 20, 1).repeat(4, 1, 2)
        x = torch.sin(2 * 3.14159 * t) + 0.1 * torch.randn(4, 20, 2)
        _, _, aux = cell(x, return_aux=True)
        self.assertGreater(aux["tau_temporal_std"], 0.0)


class TestLiquidTauInheritedSTE(unittest.TestCase):

    def test_inherits_entropy_loss(self):
        cell = LiquidTauSTECfCCell(input_size=2, hidden_size=8,
                                   entropy_lambda=0.1)
        loss = cell.extra_loss()
        self.assertGreater(float(loss.item()), 0.0)

    def test_inherits_ste_hard_mask_binary(self):
        cell = LiquidTauSTECfCCell(input_size=2, hidden_size=8)
        hard = cell.get_ste_hard_mask()
        uniq = set(hard.unique().tolist())
        self.assertTrue(uniq.issubset({0.0, 1.0}))

    def test_density_respected(self):
        cell = LiquidTauSTECfCCell(input_size=2, hidden_size=16, density=0.3)
        d = cell.neighborhood_density()
        # Allow slack: top-k rounding vs exact fraction.
        self.assertLessEqual(d, 0.5)


if __name__ == "__main__":
    unittest.main()
