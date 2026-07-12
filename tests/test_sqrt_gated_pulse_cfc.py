"""Unit tests for SqrtGatedPulseCfCCell (round 286).

Verifies:
    H5: gate_pulse_shape='none' ≡ PulseGatedLiquidTauCfCCell (r284)
    H6: gate_pulse_shape='linear' ≡ PredictabilityGatedPulseCfCCell (r285)
    Forward shapes, finite outputs, gradient flow.
    Sqrt shape: pulse with sqrt(g) is between ungated and linear-gated
        on the geometric-mean axis (sqrt(0.25)=0.5, between 1.0 and 0.25).
    Sqrt gate actually rescales pulse: pulse_sqrt(g)·A vs pulse_A
        differs when gate != 1.
    gate_pulse_shape validation.
    pulse_strength=0 ≡ r280 blend cell.
"""

from __future__ import annotations

import unittest

import torch

from lnn.core.sqrt_gated_pulse_cfc import SqrtGatedPulseCfCCell
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


class TestSqrtGatedPulseBasics(unittest.TestCase):

    def test_init_default_shape(self):
        cell = SqrtGatedPulseCfCCell(input_size=1, hidden_size=8)
        self.assertEqual(cell.gate_pulse_shape, "sqrt")
        self.assertTrue(cell.gate_pulse)  # sqrt implies gating enabled

    def test_subclass_of_r285(self):
        cell = SqrtGatedPulseCfCCell(input_size=1, hidden_size=8)
        self.assertIsInstance(cell, PredictabilityGatedPulseCfCCell)

    def test_shape_validation(self):
        with self.assertRaises(ValueError):
            SqrtGatedPulseCfCCell(
                input_size=1, hidden_size=8, gate_pulse_shape="bogus")


class TestForwardShapes(unittest.TestCase):

    def test_forward_shape(self):
        cell = SqrtGatedPulseCfCCell(input_size=1, hidden_size=16, seed=1)
        x = _sine(B=3, T=32)
        out, h = cell(x)
        self.assertEqual(out.shape, (3, 32, 16))
        self.assertEqual(h.shape, (3, 16))
        self.assertTrue(torch.isfinite(out).all())

    def test_cell_records_shape(self):
        cell = SqrtGatedPulseCfCCell(input_size=1, hidden_size=16, seed=1)
        self.assertEqual(cell.gate_pulse_shape, "sqrt")
        # aux exposes gate_pulse from r285; gate_pulse_shape is a
        # static attribute on the cell (parent aux schema is fixed).
        _, _, aux = cell(_sine(B=2, T=32), return_aux=True)
        self.assertIn("gate_pulse", aux)
        self.assertTrue(aux["gate_pulse"])


class TestSupersets(unittest.TestCase):
    """H5/H6: shape='none' ≡ r284, shape='linear' ≡ r285."""

    def test_none_shape_equals_r284(self):
        kw = dict(input_size=1, hidden_size=24, density=0.3, seed=5,
                  pulse_strength=1.0, pulse_amp_init=0.1,
                  pulse_mode="sin", state_phase=True)
        torch.manual_seed(123)
        sqrt_cell = SqrtGatedPulseCfCCell(gate_pulse_shape="none", **kw)
        torch.manual_seed(123)
        r284 = PulseGatedLiquidTauCfCCell(**kw)
        x = _sine(B=4, T=48)
        with torch.no_grad():
            o_sqrt, _ = sqrt_cell(x)
            o_r284, _ = r284(x)
        self.assertTrue(
            torch.allclose(o_sqrt, o_r284, atol=1e-6),
            f"max diff {(o_sqrt - o_r284).abs().max().item():.2e}")

    def test_linear_shape_equals_r285(self):
        kw = dict(input_size=1, hidden_size=24, density=0.3, seed=5,
                  pulse_strength=1.0, pulse_amp_init=0.1,
                  pulse_mode="sin", state_phase=True)
        torch.manual_seed(123)
        sqrt_cell = SqrtGatedPulseCfCCell(gate_pulse_shape="linear", **kw)
        torch.manual_seed(123)
        r285 = PredictabilityGatedPulseCfCCell(gate_pulse=True, **kw)
        x = _sine(B=4, T=48)
        with torch.no_grad():
            o_sqrt, _ = sqrt_cell(x)
            o_r285, _ = r285(x)
        self.assertTrue(
            torch.allclose(o_sqrt, o_r285, atol=1e-6),
            f"max diff {(o_sqrt - o_r285).abs().max().item():.2e}")

    def test_pulse_off_equals_blend(self):
        # Composed: any shape + pulse_strength=0 ≡ r280 blend cell.
        for shape in ("sqrt", "linear", "none"):
            torch.manual_seed(789)
            gated = SqrtGatedPulseCfCCell(
                input_size=1, hidden_size=24, density=0.3, seed=5,
                pulse_strength=0.0, pulse_amp_init=0.1,
                pulse_mode="sin", state_phase=True,
                gate_pulse_shape=shape)
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
                f"shape={shape} max diff "
                f"{(o_gated - o_blend).abs().max().item():.2e}")


class TestSqrtShape(unittest.TestCase):

    def test_sqrt_gate_scales_pulse_correctly(self):
        """sqrt(0.25) = 0.5 — between linear(0.25) and identity(1.0)."""
        cell = SqrtGatedPulseCfCCell(
            input_size=1, hidden_size=8, seed=3, pulse_amp_init=0.5,
            pulse_mode="sin", state_phase=False,
            gate_pulse_shape="sqrt", pulse_strength=1.0)
        T = 16
        h = torch.zeros(1, cell.hidden_size)
        # g = 0.25 → sqrt(g) = 0.5, so sqrt-pulse should be 0.5× ungated.
        gate = torch.full((1, 1), 0.25)
        sig = cell._pulse_term(2, T, h, None, gate=gate)
        # Compare to ungated (gate=1) reference at same t, h.
        ref = PulseGatedLiquidTauCfCCell(
            input_size=1, hidden_size=8, seed=3, pulse_amp_init=0.5,
            pulse_mode="sin", state_phase=False)._pulse_term(2, T, h, None)
        # sqrt-pulse with g=0.25 should equal 0.5× ungated.
        diff = (sig - 0.5 * ref).abs().max().item()
        self.assertLess(diff, 1e-5,
                        f"sqrt scaling violated, max diff {diff:.2e}")

    def test_sqrt_pulse_between_linear_and_none(self):
        """For any gate g in (0,1]: sqrt(g) > g (closer to 1)."""
        cell_sqrt = SqrtGatedPulseCfCCell(
            input_size=1, hidden_size=4, seed=7, pulse_amp_init=0.5,
            pulse_mode="sin", state_phase=False, gate_pulse_shape="sqrt")
        cell_linear = SqrtGatedPulseCfCCell(
            input_size=1, hidden_size=4, seed=7, pulse_amp_init=0.5,
            pulse_mode="sin", state_phase=False, gate_pulse_shape="linear")
        T = 8
        h = torch.zeros(1, 4)
        for g_val in (0.1, 0.25, 0.5, 0.8):
            g = torch.full((1, 1), g_val)
            sig_sqrt = cell_sqrt._pulse_term(2, T, h, None, gate=g)
            sig_linear = cell_linear._pulse_term(2, T, h, None, gate=g)
            # sqrt pulse amplitude should be larger than linear pulse amp.
            self.assertGreater(
                sig_sqrt.abs().sum().item(),
                sig_linear.abs().sum().item(),
                f"sqrt pulse smaller than linear at g={g_val}")

    def test_zero_gate_zeros_sqrt_pulse(self):
        """sqrt(0) = 0 — pulse is exactly zero when gate is zero."""
        cell = SqrtGatedPulseCfCCell(
            input_size=1, hidden_size=8, seed=3, pulse_amp_init=0.5,
            pulse_mode="sin", state_phase=False, gate_pulse_shape="sqrt",
            pulse_strength=1.0)
        h = torch.zeros(1, cell.hidden_size)
        sig_zero = cell._pulse_term(2, 16, h, None,
                                    gate=torch.zeros(1, 1))
        self.assertEqual(float(sig_zero.abs().max().item()), 0.0)


class TestGradientFlow(unittest.TestCase):

    def test_gradients_flow_to_pulse_params(self):
        cell = SqrtGatedPulseCfCCell(
            input_size=1, hidden_size=16, seed=2, pulse_amp_init=0.2,
            gate_pulse_shape="sqrt", pulse_strength=1.0)
        x = _sine(B=4, T=32)
        out, _ = cell(x)
        out.pow(2).mean().backward()
        self.assertIsNotNone(cell.pulse_amp.grad)
        self.assertGreater(cell.pulse_amp.grad.abs().sum().item(), 0.0)
        self.assertGreater(cell.pulse_omega.grad.abs().sum().item(), 0.0)

    def test_gradients_flow_to_W_in_via_pulse(self):
        """W_in must receive gradient through the gated pulse path."""
        cell = SqrtGatedPulseCfCCell(
            input_size=1, hidden_size=8, seed=2, pulse_amp_init=0.3,
            gate_pulse_shape="sqrt", pulse_strength=1.0)
        x = _noise(B=2, T=24, seed=99)
        out, _ = cell(x)
        out.pow(2).mean().backward()
        self.assertIsNotNone(cell.W_in.weight.grad)
        self.assertGreater(cell.W_in.weight.grad.abs().sum().item(), 0.0)


class TestGateModePassthrough(unittest.TestCase):

    def test_gate_modes_run(self):
        for gm in ("blend", "velocity", "acceleration"):
            cell = SqrtGatedPulseCfCCell(
                input_size=1, hidden_size=12, gate_mode=gm, seed=1,
                gate_pulse_shape="sqrt")
            out, _ = cell(_sine(B=2, T=24))
            self.assertEqual(out.shape, (2, 24, 12))
            self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()