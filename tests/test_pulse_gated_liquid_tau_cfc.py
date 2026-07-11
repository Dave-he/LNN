"""Unit tests for PulseGatedLiquidTauCfCCell (round 284).

Grounded in arXiv:2603.00153 (Pulse-Driven Neural Architecture). Verifies
the learnable oscillatory pulse: it is a strict superset of the r280 blend
gate (pulse off ≡ r280), gradients flow to the pulse parameters, the sin
pulse is a genuine oscillator (autocorrelated in time), and the 'noise'
mechanism control is RMS-matched but NOT oscillatory.
"""

from __future__ import annotations

import math
import unittest

import torch

from lnn.core.pulse_gated_liquid_tau_cfc import PulseGatedLiquidTauCfCCell
from lnn.core.blend_gated_liquid_tau_cfc import BlendGatedLiquidTauCfCCell


def _sine(B=4, T=64, freq=3.0):
    t = torch.linspace(0, 6.2831853 * freq, T).view(1, T, 1).repeat(B, 1, 1)
    return torch.sin(t)


def _noise(B=4, T=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, 1, generator=g)


class TestPulseBasics(unittest.TestCase):

    def test_init_defaults(self):
        cell = PulseGatedLiquidTauCfCCell(input_size=1, hidden_size=8)
        self.assertEqual(cell.pulse_mode, "sin")
        self.assertEqual(cell.gate_mode, "blend")
        self.assertTrue(cell.state_phase)
        self.assertEqual(cell.pulse_amp.shape, (8,))
        self.assertEqual(cell.pulse_omega.shape, (8,))

    def test_subclass_of_blend(self):
        cell = PulseGatedLiquidTauCfCCell(input_size=1, hidden_size=8)
        self.assertIsInstance(cell, BlendGatedLiquidTauCfCCell)

    def test_pulse_mode_validation(self):
        with self.assertRaises(ValueError):
            PulseGatedLiquidTauCfCCell(
                input_size=1, hidden_size=8, pulse_mode="bogus")

    def test_off_mode_zeros_strength(self):
        cell = PulseGatedLiquidTauCfCCell(
            input_size=1, hidden_size=8, pulse_mode="off", pulse_strength=1.0)
        self.assertEqual(cell.pulse_strength, 0.0)


class TestForwardShapes(unittest.TestCase):

    def test_forward_shape(self):
        cell = PulseGatedLiquidTauCfCCell(input_size=1, hidden_size=16, seed=1)
        x = _sine(B=3, T=32)
        out, h = cell(x)
        self.assertEqual(out.shape, (3, 32, 16))
        self.assertEqual(h.shape, (3, 16))
        self.assertTrue(torch.isfinite(out).all())

    def test_aux_diagnostics(self):
        cell = PulseGatedLiquidTauCfCCell(input_size=1, hidden_size=16, seed=1)
        _, _, aux = cell(_sine(B=2, T=32), return_aux=True)
        for k in ("gate_mean", "pulse_rms", "pulse_amp_mean",
                  "pulse_omega_mean", "pulse_mode"):
            self.assertIn(k, aux)
        self.assertGreater(aux["pulse_rms"], 0.0)


class TestSuperset(unittest.TestCase):
    """pulse_strength=0 ⇒ bit-for-bit identical to r280 blend cell."""

    def test_pulse_off_equals_blend(self):
        kw = dict(input_size=1, hidden_size=24, density=0.3, seed=5)
        torch.manual_seed(123)
        pulse = PulseGatedLiquidTauCfCCell(pulse_strength=0.0, **kw)
        torch.manual_seed(123)
        blend = BlendGatedLiquidTauCfCCell(gate_mode="blend", **kw)
        x = _sine(B=4, T=48)
        with torch.no_grad():
            o_pulse, _ = pulse(x)
            o_blend, _ = blend(x)
        self.assertTrue(torch.allclose(o_pulse, o_blend, atol=1e-6),
                        f"max diff {(o_pulse - o_blend).abs().max().item():.2e}")

    def test_off_mode_equals_blend(self):
        kw = dict(input_size=1, hidden_size=24, density=0.3, seed=5)
        torch.manual_seed(321)
        pulse = PulseGatedLiquidTauCfCCell(pulse_mode="off", **kw)
        torch.manual_seed(321)
        blend = BlendGatedLiquidTauCfCCell(gate_mode="blend", **kw)
        x = _noise(B=4, T=48)
        with torch.no_grad():
            o_pulse, _ = pulse(x)
            o_blend, _ = blend(x)
        self.assertTrue(torch.allclose(o_pulse, o_blend, atol=1e-6))


class TestPulseChangesOutput(unittest.TestCase):

    def test_pulse_on_differs_from_off(self):
        kw = dict(input_size=1, hidden_size=24, seed=5, pulse_amp_init=0.3)
        on = PulseGatedLiquidTauCfCCell(pulse_strength=1.0, **kw)
        off = PulseGatedLiquidTauCfCCell(pulse_strength=0.0, **kw)
        x = _sine(B=4, T=48)
        with torch.no_grad():
            o_on, _ = on(x)
            o_off, _ = off(x)
        self.assertGreater((o_on - o_off).abs().max().item(), 1e-3)

    def test_gradients_flow_to_pulse_params(self):
        cell = PulseGatedLiquidTauCfCCell(
            input_size=1, hidden_size=16, seed=2, pulse_amp_init=0.2)
        x = _sine(B=4, T=32)
        out, _ = cell(x)
        out.pow(2).mean().backward()
        self.assertIsNotNone(cell.pulse_amp.grad)
        self.assertGreater(cell.pulse_amp.grad.abs().sum().item(), 0.0)
        self.assertGreater(cell.pulse_phase0.grad.abs().sum().item(), 0.0)
        # omega gets gradient because it modulates the sin argument.
        self.assertGreater(cell.pulse_omega.grad.abs().sum().item(), 0.0)


class TestOscillatorStructure(unittest.TestCase):
    """The sin pulse is a real oscillator; the noise control is not."""

    def _pulse_signal(self, cell, T=64):
        # Isolate the raw pulse by reading aux pulse_rms is not enough; probe
        # the pulse term directly through a zero-state, single-batch call.
        h = torch.zeros(1, cell.hidden_size)
        vals = []
        for t in range(T):
            vals.append(cell._pulse_term(t, T, h, cell_noise(cell, T)).squeeze(0))
        return torch.stack(vals, dim=0)  # (T, d_h)

    def test_sin_is_autocorrelated(self):
        cell = PulseGatedLiquidTauCfCCell(
            input_size=1, hidden_size=8, seed=3, pulse_amp_init=0.5,
            state_phase=False)
        T = 64
        h = torch.zeros(1, cell.hidden_size)
        sig = torch.stack(
            [cell._pulse_term(t, T, h, None).squeeze(0) for t in range(T)], 0)
        # lag-1 autocorrelation of a smooth oscillator is high (>0.8).
        s = sig[:, 0]
        s = s - s.mean()
        ac1 = (s[1:] * s[:-1]).sum() / (s.pow(2).sum() + 1e-9)
        self.assertGreater(float(ac1), 0.8)

    def test_noise_is_not_autocorrelated(self):
        cell = PulseGatedLiquidTauCfCCell(
            input_size=1, hidden_size=8, seed=3, pulse_amp_init=0.5,
            pulse_mode="noise")
        T = 64
        gen = torch.Generator().manual_seed(cell.pulse_seed)
        nd = torch.randn(T, cell.hidden_size, generator=gen)
        h = torch.zeros(1, cell.hidden_size)
        sig = torch.stack(
            [cell._pulse_term(t, T, h, nd).squeeze(0) for t in range(T)], 0)
        s = sig[:, 0]
        s = s - s.mean()
        ac1 = (s[1:] * s[:-1]).sum() / (s.pow(2).sum() + 1e-9)
        # white noise has near-zero lag-1 autocorrelation.
        self.assertLess(abs(float(ac1)), 0.4)

    def test_noise_rms_matches_sin_scale(self):
        # With equal amplitude, the noise control RMS is comparable to sin.
        T = 64
        cs = PulseGatedLiquidTauCfCCell(
            input_size=1, hidden_size=32, seed=9, pulse_amp_init=0.4,
            pulse_mode="sin", state_phase=False)
        cn = PulseGatedLiquidTauCfCCell(
            input_size=1, hidden_size=32, seed=9, pulse_amp_init=0.4,
            pulse_mode="noise")
        h = torch.zeros(1, 32)
        sin_sig = torch.stack(
            [cs._pulse_term(t, T, h, None).squeeze(0) for t in range(T)], 0)
        gen = torch.Generator().manual_seed(cn.pulse_seed)
        nd = torch.randn(T, 32, generator=gen)
        noise_sig = torch.stack(
            [cn._pulse_term(t, T, h, nd).squeeze(0) for t in range(T)], 0)
        r_sin = sin_sig.pow(2).mean().sqrt().item()
        r_noise = noise_sig.pow(2).mean().sqrt().item()
        self.assertLess(abs(r_sin - r_noise) / max(r_sin, 1e-6), 0.6)


class TestGateModePassthrough(unittest.TestCase):

    def test_gate_modes_run(self):
        for gm in ("blend", "velocity", "acceleration"):
            cell = PulseGatedLiquidTauCfCCell(
                input_size=1, hidden_size=12, gate_mode=gm, seed=1)
            out, _ = cell(_sine(B=2, T=24))
            self.assertEqual(out.shape, (2, 24, 12))
            self.assertTrue(torch.isfinite(out).all())


def cell_noise(cell, T):
    if cell.pulse_mode != "noise":
        return None
    gen = torch.Generator().manual_seed(cell.pulse_seed)
    return torch.randn(T, cell.hidden_size, generator=gen)


if __name__ == "__main__":
    unittest.main()
