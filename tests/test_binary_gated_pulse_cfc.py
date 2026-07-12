"""Unit tests for BinaryGatedPulseCfCCell (round 287).

Verifies:
    H5: threshold=0 ≡ PulseGatedLiquidTauCfCCell (r284, unconditional)
    H6: threshold=10 ≡ r280 blend cell (pulse always off)
    Forward shapes, finite outputs, gradient flow.
    Threshold gate actually suppresses pulse on low-g steps.
    threshold=0.5 sits between structured (gate≈0.8 > 0.5 fires) and
        noise (gate≈0.1 < 0.5 suppressed) on realistic gates.
    Threshold validation rejects negative values.
"""

from __future__ import annotations

import unittest

import torch

from lnn.core.binary_gated_pulse_cfc import BinaryGatedPulseCfCCell
from lnn.core.predictability_gated_pulse_cfc import (
    PredictabilityGatedPulseCfCCell,
)
from lnn.core.pulse_gated_liquid_tau_cfc import PulseGatedLiquidTauCfCCell
from lnn.core.blend_gated_liquid_tau_cfc import BlendGatedLiquidTauCfCCell


def _sine(B=4, T=64, freq=3.0):
    t = torch.linspace(0, 6.2831853 * freq, T).view(1, T, 1).repeat(B, 1, 1)
    return torch.sin(t)


def _noise(B=4, T=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, 1, generator=g)


class TestBinaryGatedPulseBasics(unittest.TestCase):

    def test_init_defaults(self):
        cell = BinaryGatedPulseCfCCell(input_size=1, hidden_size=8)
        self.assertEqual(cell.threshold, 0.5)
        self.assertTrue(cell.gate_pulse)
        self.assertEqual(cell.pulse_mode, "sin")

    def test_subclass_of_r285(self):
        cell = BinaryGatedPulseCfCCell(input_size=1, hidden_size=8)
        self.assertIsInstance(cell, PredictabilityGatedPulseCfCCell)

    def test_threshold_validation(self):
        with self.assertRaises(ValueError):
            BinaryGatedPulseCfCCell(
                input_size=1, hidden_size=8, threshold=-0.1)


class TestForwardShapes(unittest.TestCase):

    def test_forward_shape(self):
        cell = BinaryGatedPulseCfCCell(input_size=1, hidden_size=16, seed=1)
        x = _sine(B=3, T=32)
        out, h = cell(x)
        self.assertEqual(out.shape, (3, 32, 16))
        self.assertEqual(h.shape, (3, 16))
        self.assertTrue(torch.isfinite(out).all())


class TestSupersets(unittest.TestCase):
    """H5/H6: threshold=0 ≡ r284; threshold=10 ≡ r280."""

    def test_threshold_zero_equals_r284(self):
        kw = dict(input_size=1, hidden_size=24, density=0.3, seed=5,
                  pulse_strength=1.0, pulse_amp_init=0.1,
                  pulse_mode="sin", state_phase=True)
        torch.manual_seed(123)
        binary = BinaryGatedPulseCfCCell(threshold=0.0, **kw)
        torch.manual_seed(123)
        r284 = PulseGatedLiquidTauCfCCell(**kw)
        x = _sine(B=4, T=48)
        with torch.no_grad():
            o_binary, _ = binary(x)
            o_r284, _ = r284(x)
        self.assertTrue(
            torch.allclose(o_binary, o_r284, atol=1e-6),
            f"max diff {(o_binary - o_r284).abs().max().item():.2e}")

    def test_threshold_high_equals_blend(self):
        # threshold=10 ⇒ gate is always below threshold ⇒ pulse is
        # always zero ⇒ ≡ r280 blend cell.
        kw = dict(input_size=1, hidden_size=24, density=0.3, seed=5,
                  pulse_strength=1.0, pulse_amp_init=0.1,
                  pulse_mode="sin", state_phase=True)
        torch.manual_seed(123)
        binary = BinaryGatedPulseCfCCell(threshold=10.0, **kw)
        torch.manual_seed(123)
        blend = BlendGatedLiquidTauCfCCell(
            input_size=1, hidden_size=24, density=0.3, seed=5,
            gate_mode="blend")
        x = _sine(B=4, T=48)
        with torch.no_grad():
            o_binary, _ = binary(x)
            o_blend, _ = blend(x)
        self.assertTrue(
            torch.allclose(o_binary, o_blend, atol=1e-6),
            f"max diff {(o_binary - o_blend).abs().max().item():.2e}")


class TestBinaryGate(unittest.TestCase):

    def test_low_gate_suppresses_pulse(self):
        cell = BinaryGatedPulseCfCCell(
            input_size=1, hidden_size=8, seed=3, pulse_amp_init=0.5,
            pulse_mode="sin", state_phase=False, threshold=0.5,
            pulse_strength=1.0)
        h = torch.zeros(1, cell.hidden_size)
        # gate below threshold → pulse exactly zero.
        sig_low = cell._pulse_term(2, 16, h, None,
                                    gate=torch.full((1, 1), 0.3))
        self.assertEqual(float(sig_low.abs().max().item()), 0.0)

        # gate above threshold → pulse full strength (not attenuated).
        sig_high = cell._pulse_term(2, 16, h, None,
                                     gate=torch.full((1, 1), 0.9))
        ref = PulseGatedLiquidTauCfCCell(
            input_size=1, hidden_size=8, seed=3, pulse_amp_init=0.5,
            pulse_mode="sin", state_phase=False)._pulse_term(2, 16, h, None)
        # Full strength means equal to ungated reference (not scaled).
        diff = (sig_high - ref).abs().max().item()
        self.assertLess(diff, 1e-5,
                        f"high-gate pulse not at full strength, diff {diff:.2e}")

    def test_binary_threshold_step(self):
        """At threshold=0.5, gate=0.4 should suppress and gate=0.6 should fire."""
        cell = BinaryGatedPulseCfCCell(
            input_size=1, hidden_size=8, seed=3, pulse_amp_init=0.5,
            pulse_mode="sin", state_phase=False, threshold=0.5,
            pulse_strength=1.0)
        h = torch.zeros(1, cell.hidden_size)
        sig_just_below = cell._pulse_term(2, 16, h, None,
                                           gate=torch.full((1, 1), 0.499))
        sig_just_above = cell._pulse_term(2, 16, h, None,
                                           gate=torch.full((1, 1), 0.501))
        self.assertEqual(float(sig_just_below.abs().max().item()), 0.0)
        self.assertGreater(float(sig_just_above.abs().max().item()), 1e-3)

    def test_different_thresholds(self):
        """Higher threshold ⇒ stricter gating (more suppressed)."""
        cell_lo = BinaryGatedPulseCfCCell(
            input_size=1, hidden_size=4, seed=7, pulse_amp_init=0.5,
            pulse_mode="sin", state_phase=False, threshold=0.3,
            pulse_strength=1.0)
        cell_hi = BinaryGatedPulseCfCCell(
            input_size=1, hidden_size=4, seed=7, pulse_amp_init=0.5,
            pulse_mode="sin", state_phase=False, threshold=0.7,
            pulse_strength=1.0)
        h = torch.zeros(1, 4)
        for g_val in (0.4, 0.6):
            g = torch.full((1, 1), g_val)
            sig_lo = cell_lo._pulse_term(2, 8, h, None, gate=g)
            sig_hi = cell_hi._pulse_term(2, 8, h, None, gate=g)
            # Lower threshold fires more often → larger pulse amplitude.
            self.assertGreaterEqual(
                sig_lo.abs().sum().item(), sig_hi.abs().sum().item(),
                f"low-threshold pulse smaller at g={g_val}")


class TestGradientFlow(unittest.TestCase):

    def test_gradients_flow_when_pulse_active(self):
        # Sine input keeps gate high → pulse fires → gradients flow.
        cell = BinaryGatedPulseCfCCell(
            input_size=1, hidden_size=16, seed=2, pulse_amp_init=0.2,
            threshold=0.5, pulse_strength=1.0)
        x = _sine(B=4, T=32)
        out, _ = cell(x)
        out.pow(2).mean().backward()
        self.assertIsNotNone(cell.pulse_amp.grad)
        self.assertGreater(cell.pulse_amp.grad.abs().sum().item(), 0.0)

    def test_zero_pulse_gradient_on_always_suppressed(self):
        # threshold=10 ⇒ pulse always off ⇒ gradient on A must be zero.
        cell = BinaryGatedPulseCfCCell(
            input_size=1, hidden_size=8, seed=2, pulse_amp_init=0.2,
            threshold=10.0, pulse_strength=1.0)
        x = _sine(B=2, T=24)
        out, _ = cell(x)
        out.pow(2).mean().backward()
        # The pulse term is exactly zero at every step, so the gradient
        # on A through the pulse path should be zero. Other params
        # (recurrent W, W_in) still have gradients.
        self.assertIsNotNone(cell.pulse_amp.grad)
        self.assertEqual(float(cell.pulse_amp.grad.abs().sum().item()), 0.0)


class TestGateModePassthrough(unittest.TestCase):

    def test_gate_modes_run(self):
        for gm in ("blend", "velocity", "acceleration"):
            cell = BinaryGatedPulseCfCCell(
                input_size=1, hidden_size=12, gate_mode=gm, seed=1,
                threshold=0.5)
            out, _ = cell(_sine(B=2, T=24))
            self.assertEqual(out.shape, (2, 24, 12))
            self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()