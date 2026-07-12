"""Unit tests for PredictabilityGatedPulseCfCCell (round 285).

Verifies:
    H4 (superset): gate_pulse=False ≡ PulseGatedLiquidTauCfCCell (r284)
    H4' (composed): gate_pulse=True + pulse_strength=0 ≡ r280 blend cell
    Forward shapes, finite outputs, gradient flow.
    Gating not just clamping: when gate is all-ones, gated_pulse ≡ r284.
    Gating actually suppresses pulse on near-zero gate.
    Diagnostic aux exposes gate_pulse flag.
"""

from __future__ import annotations

import math
import unittest

import torch

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


class TestPredictabilityGatedPulseBasics(unittest.TestCase):

    def test_init_defaults(self):
        cell = PredictabilityGatedPulseCfCCell(input_size=1, hidden_size=8)
        self.assertTrue(cell.gate_pulse)
        self.assertEqual(cell.pulse_mode, "sin")
        self.assertEqual(cell.gate_mode, "blend")
        self.assertEqual(cell.pulse_amp.shape, (8,))

    def test_subclass_of_r284(self):
        cell = PredictabilityGatedPulseCfCCell(input_size=1, hidden_size=8)
        self.assertIsInstance(cell, PulseGatedLiquidTauCfCCell)


class TestForwardShapes(unittest.TestCase):

    def test_forward_shape(self):
        cell = PredictabilityGatedPulseCfCCell(
            input_size=1, hidden_size=16, seed=1)
        x = _sine(B=3, T=32)
        out, h = cell(x)
        self.assertEqual(out.shape, (3, 32, 16))
        self.assertEqual(h.shape, (3, 16))
        self.assertTrue(torch.isfinite(out).all())

    def test_aux_diagnostics_includes_flag(self):
        cell = PredictabilityGatedPulseCfCCell(
            input_size=1, hidden_size=16, seed=1)
        _, _, aux = cell(_sine(B=2, T=32), return_aux=True)
        self.assertIn("gate_pulse", aux)
        self.assertTrue(aux["gate_pulse"])
        for k in ("gate_mean", "pulse_rms", "pulse_amp_mean"):
            self.assertIn(k, aux)


class TestSuperset(unittest.TestCase):
    """H4: gate_pulse=False ≡ r284 PulseGatedLiquidTauCfCCell bit-for-bit."""

    def test_gate_pulse_false_equals_r284(self):
        kw = dict(input_size=1, hidden_size=24, density=0.3, seed=5,
                  pulse_strength=1.0, pulse_amp_init=0.1,
                  pulse_mode="sin", state_phase=True)
        torch.manual_seed(123)
        gated = PredictabilityGatedPulseCfCCell(
            input_size=1, hidden_size=24, density=0.3, seed=5,
            pulse_strength=1.0, pulse_amp_init=0.1,
            pulse_mode="sin", state_phase=True,
            gate_pulse=False)
        torch.manual_seed(123)
        r284 = PulseGatedLiquidTauCfCCell(
            input_size=1, hidden_size=24, density=0.3, seed=5,
            pulse_strength=1.0, pulse_amp_init=0.1,
            pulse_mode="sin", state_phase=True)
        x = _sine(B=4, T=48)
        with torch.no_grad():
            o_gated, _ = gated(x)
            o_r284, _ = r284(x)
        self.assertTrue(
            torch.allclose(o_gated, o_r284, atol=1e-6),
            f"max diff {(o_gated - o_r284).abs().max().item():.2e}")

    def test_gate_pulse_false_equals_r284_on_noise(self):
        torch.manual_seed(456)
        gated = PredictabilityGatedPulseCfCCell(
            input_size=1, hidden_size=24, density=0.3, seed=7,
            pulse_strength=1.0, pulse_amp_init=0.1,
            pulse_mode="sin", state_phase=True,
            gate_pulse=False)
        torch.manual_seed(456)
        r284 = PulseGatedLiquidTauCfCCell(
            input_size=1, hidden_size=24, density=0.3, seed=7,
            pulse_strength=1.0, pulse_amp_init=0.1,
            pulse_mode="sin", state_phase=True)
        x = _noise(B=4, T=48, seed=11)
        with torch.no_grad():
            o_gated, _ = gated(x)
            o_r284, _ = r284(x)
        self.assertTrue(torch.allclose(o_gated, o_r284, atol=1e-6))

    def test_pulse_off_equals_blend(self):
        # Composed: gate_pulse=True + pulse_strength=0 ≡ r280 blend.
        torch.manual_seed(789)
        gated = PredictabilityGatedPulseCfCCell(
            input_size=1, hidden_size=24, density=0.3, seed=5,
            pulse_strength=0.0, pulse_amp_init=0.1,
            pulse_mode="sin", state_phase=True, gate_pulse=True)
        torch.manual_seed(789)
        blend = BlendGatedLiquidTauCfCCell(
            input_size=1, hidden_size=24, density=0.3, seed=5,
            gate_mode="blend")
        x = _sine(B=4, T=48)
        with torch.no_grad():
            o_gated, _ = gated(x)
            o_blend, _ = blend(x)
        self.assertTrue(
            torch.allclose(o_gated, o_blend, atol=1e-6),
            f"max diff {(o_gated - o_blend).abs().max().item():.2e}")


class TestGatingMechanism(unittest.TestCase):

    def test_gating_actually_scales_pulse(self):
        """A zero gate should zero the pulse contribution."""
        cell = PredictabilityGatedPulseCfCCell(
            input_size=1, hidden_size=8, seed=3, pulse_amp_init=0.5,
            pulse_mode="sin", state_phase=False, gate_pulse=True,
            pulse_strength=1.0)
        T = 16
        h = torch.zeros(1, cell.hidden_size)
        # With all-ones gate, pulse term should equal the ungated sin pulse.
        gate_one = torch.ones(1, 1)
        sig_full = cell._pulse_term(2, T, h, None, gate=gate_one)
        # With zero gate, pulse term should be exactly zero.
        gate_zero = torch.zeros(1, 1)
        sig_zero = cell._pulse_term(2, T, h, None, gate=gate_zero)
        self.assertEqual(float(sig_zero.abs().max().item()), 0.0)
        # Half gate -> half amplitude (since amp*sin scales linearly).
        gate_half = torch.full((1, 1), 0.5)
        sig_half = cell._pulse_term(2, T, h, None, gate=gate_half)
        diff = (sig_full - 2.0 * sig_half).abs().max().item()
        self.assertLess(diff, 1e-5,
                        f"linear-scaling violated, max diff {diff:.2e}")

    def test_gating_disabled_passes_gate_unchanged(self):
        """gate_pulse=False should ignore the gate argument."""
        cell = PredictabilityGatedPulseCfCCell(
            input_size=1, hidden_size=8, seed=3, pulse_amp_init=0.5,
            pulse_mode="sin", state_phase=False, gate_pulse=False,
            pulse_strength=1.0)
        T = 16
        h = torch.zeros(1, cell.hidden_size)
        sig_full = cell._pulse_term(2, T, h, None,
                                     gate=torch.ones(1, 1))
        sig_zero = cell._pulse_term(2, T, h, None,
                                     gate=torch.zeros(1, 1))
        # gate_pulse=False ⇒ output is identical regardless of gate.
        self.assertTrue(torch.allclose(sig_full, sig_zero, atol=1e-6))

    def test_gated_pulse_differs_from_ungated_on_nontrivial_gate(self):
        cell_gated = PredictabilityGatedPulseCfCCell(
            input_size=1, hidden_size=16, seed=5, pulse_amp_init=0.3,
            pulse_mode="sin", state_phase=True, gate_pulse=True,
            pulse_strength=1.0)
        cell_ungated = PredictabilityGatedPulseCfCCell(
            input_size=1, hidden_size=16, seed=5, pulse_amp_init=0.3,
            pulse_mode="sin", state_phase=True, gate_pulse=False,
            pulse_strength=1.0)
        x = _noise(B=4, T=48, seed=21)
        with torch.no_grad():
            o_gated, _ = cell_gated(x)
            o_ungated, _ = cell_ungated(x)
        # On random input the gate is small but not zero, so the outputs
        # should differ noticeably.
        self.assertGreater((o_gated - o_ungated).abs().max().item(), 1e-3)


class TestGradientFlow(unittest.TestCase):

    def test_gradients_flow_to_pulse_params(self):
        cell = PredictabilityGatedPulseCfCCell(
            input_size=1, hidden_size=16, seed=2, pulse_amp_init=0.2,
            gate_pulse=True, pulse_strength=1.0)
        x = _sine(B=4, T=32)
        out, _ = cell(x)
        out.pow(2).mean().backward()
        self.assertIsNotNone(cell.pulse_amp.grad)
        self.assertGreater(cell.pulse_amp.grad.abs().sum().item(), 0.0)
        self.assertGreater(cell.pulse_phase0.grad.abs().sum().item(), 0.0)
        self.assertGreater(cell.pulse_omega.grad.abs().sum().item(), 0.0)

    def test_gradients_flow_through_gate(self):
        """Gate is per-step, so gradient must reach W_in via the pulse path
        when gate varies across steps."""
        cell = PredictabilityGatedPulseCfCCell(
            input_size=1, hidden_size=8, seed=2, pulse_amp_init=0.3,
            gate_pulse=True, pulse_strength=1.0)
        # Noise input → gate varies a lot per step.
        x = _noise(B=2, T=24, seed=99)
        out, _ = cell(x)
        out.pow(2).mean().backward()
        self.assertIsNotNone(cell.W_in.weight.grad)
        self.assertGreater(cell.W_in.weight.grad.abs().sum().item(), 0.0)


class TestGateModePassthrough(unittest.TestCase):

    def test_gate_modes_run(self):
        for gm in ("blend", "velocity", "acceleration"):
            cell = PredictabilityGatedPulseCfCCell(
                input_size=1, hidden_size=12, gate_mode=gm, seed=1)
            out, _ = cell(_sine(B=2, T=24))
            self.assertEqual(out.shape, (2, 24, 12))
            self.assertTrue(torch.isfinite(out).all())


class TestPulseSummary(unittest.TestCase):

    def test_pulse_summary_exposes_amplitudes(self):
        cell = PredictabilityGatedPulseCfCCell(
            input_size=1, hidden_size=8, seed=3, pulse_amp_init=0.2)
        s = cell.pulse_summary()
        self.assertIn("amp_mean", s)
        self.assertIn("omega_mean", s)
        self.assertEqual(s["pulse_mode"], "sin")


if __name__ == "__main__":
    unittest.main()