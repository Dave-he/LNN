"""Unit tests for EmaGatedPulseCfCCell (round 288).

Verifies:
    H5: ema_alpha=1.0 ≡ BinaryGatedPulseCfCCell (r287)
    H6: threshold=0 ≡ r284
    Forward shapes, finite outputs, gradient flow.
    EMA smoothing: g_ema tracks a smooth path, not the raw per-step gate.
    Initial state: g_ema starts at g_ema_init (default 1.0).
    EMA validation rejects out-of-range alpha.
"""

from __future__ import annotations

import unittest

import torch

from lnn.core.ema_gated_pulse_cfc import EmaGatedPulseCfCCell
from lnn.core.binary_gated_pulse_cfc import BinaryGatedPulseCfCCell
from lnn.core.pulse_gated_liquid_tau_cfc import PulseGatedLiquidTauCfCCell


def _sine(B=4, T=64, freq=3.0):
    t = torch.linspace(0, 6.2831853 * freq, T).view(1, T, 1).repeat(B, 1, 1)
    return torch.sin(t)


def _noise(B=4, T=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, 1, generator=g)


class TestEmaGatedPulseBasics(unittest.TestCase):

    def test_init_defaults(self):
        cell = EmaGatedPulseCfCCell(input_size=1, hidden_size=8)
        self.assertEqual(cell.threshold, 0.5)
        self.assertEqual(cell.ema_alpha, 0.3)
        self.assertEqual(cell.g_ema_init, 1.0)
        self.assertTrue(cell.gate_pulse)

    def test_subclass_of_r287(self):
        cell = EmaGatedPulseCfCCell(input_size=1, hidden_size=8)
        self.assertIsInstance(cell, BinaryGatedPulseCfCCell)

    def test_alpha_validation(self):
        with self.assertRaises(ValueError):
            EmaGatedPulseCfCCell(input_size=1, hidden_size=8, ema_alpha=0.0)
        with self.assertRaises(ValueError):
            EmaGatedPulseCfCCell(input_size=1, hidden_size=8, ema_alpha=1.5)


class TestForwardShapes(unittest.TestCase):

    def test_forward_shape(self):
        cell = EmaGatedPulseCfCCell(input_size=1, hidden_size=16, seed=1)
        x = _sine(B=3, T=32)
        out, h = cell(x)
        self.assertEqual(out.shape, (3, 32, 16))
        self.assertEqual(h.shape, (3, 16))
        self.assertTrue(torch.isfinite(out).all())

    def test_ema_state_resets_between_forewards(self):
        # After forward, the EMA state should exist (the forward resets
        # it at the start and updates it through the sequence).
        cell = EmaGatedPulseCfCCell(input_size=1, hidden_size=8, seed=2)
        x = _sine(B=2, T=16)
        _ = cell(x)
        self.assertIsNotNone(cell._g_ema_state)
        # State value is whatever the EMA converged to during the
        # sequence — we just want to verify the reset happens.
        v1 = float(cell._g_ema_state.mean().item())

        # Second forward: state is reset to g_ema_init=1.0 at the
        # start. The final state will again be whatever the EMA
        # converges to from the same input, so v2 should equal v1.
        _ = cell(x)
        self.assertIsNotNone(cell._g_ema_state)
        v2 = float(cell._g_ema_state.mean().item())
        # Both forwards start from the same initial state and process
        # the same input, so the final state must match.
        self.assertAlmostEqual(v1, v2, places=5,
                               msg=f"reset failed: {v1} != {v2}")


class TestSupersets(unittest.TestCase):

    def test_alpha_one_equals_r287(self):
        # ema_alpha=1.0 ⇒ g_ema_t = 1·g_t + 0·g_ema_{t-1} = g_t ⇒ r287.
        torch.manual_seed(123)
        ema = EmaGatedPulseCfCCell(
            input_size=1, hidden_size=24, density=0.3, seed=5,
            pulse_strength=1.0, pulse_amp_init=0.1,
            pulse_mode="sin", state_phase=True,
            threshold=0.5, ema_alpha=1.0, g_ema_init=0.5)
        torch.manual_seed(123)
        r287 = BinaryGatedPulseCfCCell(
            input_size=1, hidden_size=24, density=0.3, seed=5,
            pulse_strength=1.0, pulse_amp_init=0.1,
            pulse_mode="sin", state_phase=True, threshold=0.5)
        x = _sine(B=4, T=48)
        with torch.no_grad():
            o_ema, _ = ema(x)
            o_r287, _ = r287(x)
        self.assertTrue(
            torch.allclose(o_ema, o_r287, atol=1e-6),
            f"max diff {(o_ema - o_r287).abs().max().item():.2e}")

    def test_threshold_zero_equals_r284(self):
        # threshold=0 ⇒ pulse always on (regardless of gate) ⇒ r284.
        kw = dict(input_size=1, hidden_size=24, density=0.3, seed=5,
                  pulse_strength=1.0, pulse_amp_init=0.1,
                  pulse_mode="sin", state_phase=True)
        torch.manual_seed(123)
        ema = EmaGatedPulseCfCCell(
            threshold=0.0, ema_alpha=0.3, g_ema_init=1.0, **kw)
        torch.manual_seed(123)
        r284 = PulseGatedLiquidTauCfCCell(**kw)
        x = _sine(B=4, T=48)
        with torch.no_grad():
            o_ema, _ = ema(x)
            o_r284, _ = r284(x)
        self.assertTrue(
            torch.allclose(o_ema, o_r284, atol=1e-6),
            f"max diff {(o_ema - o_r284).abs().max().item():.2e}")


class TestEmaSmoothing(unittest.TestCase):

    def test_ema_tracks_smoothed_path(self):
        """g_ema should lag the raw g_t and stay high during a 1-step dip."""
        # Manually probe the EMA update path with a synthetic gate
        # sequence: [1.0, 1.0, 0.0, 1.0, 1.0].
        cell = EmaGatedPulseCfCCell(
            input_size=1, hidden_size=8, seed=3, pulse_amp_init=0.5,
            pulse_mode="sin", state_phase=False,
            ema_alpha=0.3, g_ema_init=1.0, threshold=0.5,
            pulse_strength=1.0)
        h = torch.zeros(1, cell.hidden_size)
        cell._g_ema_state = torch.full((1, 1), 1.0)
        gates = [1.0, 1.0, 0.0, 1.0, 1.0]
        sigs = []
        for g_val in gates:
            g = torch.full((1, 1), g_val)
            sigs.append(cell._pulse_term(2, 16, h, None, gate=g).abs().sum().item())
        # First step: g_ema starts at 1.0, gate=1.0 → g_ema stays 1.0 → on.
        # After dip (g=0): g_ema = 0.3·0 + 0.7·1.0 = 0.7 → still on.
        # After recovery (g=1): g_ema = 0.3·1 + 0.7·0.7 = 0.79 → still on.
        # All pulses should be non-zero because g_ema stays > 0.5.
        for i, s in enumerate(sigs):
            self.assertGreater(s, 0.0,
                               f"step {i} pulse=0, gate={gates[i]}")

    def test_ema_decays_on_persistent_low_gate(self):
        """If g_t is consistently low, g_ema decays below threshold."""
        cell = EmaGatedPulseCfCCell(
            input_size=1, hidden_size=8, seed=3, pulse_amp_init=0.5,
            pulse_mode="sin", state_phase=False,
            ema_alpha=0.3, g_ema_init=1.0, threshold=0.5,
            pulse_strength=1.0)
        h = torch.zeros(1, cell.hidden_size)
        cell._g_ema_state = torch.full((1, 1), 1.0)
        # Apply low gate for 10 steps — should decay to ~0.1 < 0.5.
        for _ in range(10):
            cell._pulse_term(2, 16, h, None,
                              gate=torch.full((1, 1), 0.1))
        # After 10 steps of g_t=0.1 with α=0.3:
        # g_ema = 1·0.7^10 + 0.1·(1 - 0.7^10) ≈ 0.028 + 0.094 ≈ 0.122.
        sig = cell._pulse_term(2, 16, h, None,
                                gate=torch.full((1, 1), 0.1))
        self.assertEqual(float(sig.abs().max().item()), 0.0,
                         "EMA should have decayed below threshold")


class TestGradientFlow(unittest.TestCase):

    def test_gradients_flow_when_pulse_active(self):
        # Sine input → EMA stays high → pulse fires → gradients flow.
        cell = EmaGatedPulseCfCCell(
            input_size=1, hidden_size=16, seed=2, pulse_amp_init=0.2,
            ema_alpha=0.3, threshold=0.5, pulse_strength=1.0)
        x = _sine(B=4, T=32)
        out, _ = cell(x)
        out.pow(2).mean().backward()
        self.assertIsNotNone(cell.pulse_amp.grad)
        self.assertGreater(cell.pulse_amp.grad.abs().sum().item(), 0.0)


class TestGateModePassthrough(unittest.TestCase):

    def test_gate_modes_run(self):
        for gm in ("blend", "velocity", "acceleration"):
            cell = EmaGatedPulseCfCCell(
                input_size=1, hidden_size=12, gate_mode=gm, seed=1,
                ema_alpha=0.3, threshold=0.5)
            out, _ = cell(_sine(B=2, T=24))
            self.assertEqual(out.shape, (2, 24, 12))
            self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()