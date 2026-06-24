"""Tests for FrozenSampledMultiTauCfCCell (arXiv:2606.15571 L-RFM response, round 246)."""

from __future__ import annotations

import math
import unittest

import torch

from lnn.core.frozen_sampled_multitau_cfc import (
    FrozenSampledMultiTauCfCCell,
    sample_log_uniform,
)


class TestSampleLogUniform(unittest.TestCase):
    def test_returns_correct_length(self):
        out = sample_log_uniform(8, 0.1, 10.0, seed=0)
        self.assertEqual(out.shape, (8,))

    def test_within_bounds(self):
        out = sample_log_uniform(20, 0.05, 20.0, seed=0)
        self.assertTrue((out >= 0.05).all())
        self.assertTrue((out <= 20.0).all())

    def test_deterministic_with_seed(self):
        a = sample_log_uniform(5, 0.1, 1.0, seed=7)
        b = sample_log_uniform(5, 0.1, 1.0, seed=7)
        self.assertTrue(torch.allclose(a, b))

    def test_log_uniform_coverage(self):
        out = sample_log_uniform(50, 0.1, 100.0, seed=0)
        # log10(100/0.1) = 3 decades
        ratio = out.max() / out.min()
        self.assertGreater(float(torch.log10(ratio).item()), 2.0)


class TestFrozenSampledMultiTauCfCCell(unittest.TestCase):
    def test_tau_frozen_not_learnable(self):
        cell = FrozenSampledMultiTauCfCCell(3, 8, n_branches=4, seed=42)
        # tau_frozen is a buffer (not a parameter).
        param_names = [n for n, _ in cell.named_parameters()]
        self.assertNotIn("tau_frozen", param_names)
        # tau_frozen is in the buffer set.
        buffer_names = [n for n, _ in cell.named_buffers()]
        self.assertIn("tau_frozen", buffer_names)

    def test_branch_cells_use_frozen_tau(self):
        cell = FrozenSampledMultiTauCfCCell(3, 8, n_branches=4, seed=42)
        for k in range(cell.n_branches):
            tau_actual = cell.cells[k].time_scale.mean().item()
            tau_frozen = cell.tau_frozen[k].item()
            self.assertAlmostEqual(tau_actual, tau_frozen, places=4)

    def test_log_coverage(self):
        cell = FrozenSampledMultiTauCfCCell(3, 8, n_branches=4,
                                             tau_min=0.05, tau_max=20.0,
                                             seed=42)
        # log10(tau_max / tau_min) = log10(400) ≈ 2.6 decades max
        cov = cell.log_coverage()
        self.assertGreater(cov, 1.4)

    def test_forward_shape(self):
        d_in, d_h, T, B, K = 3, 8, 12, 4, 4
        cell = FrozenSampledMultiTauCfCCell(d_in, d_h, n_branches=K, seed=42)
        x = torch.randn(T, B, d_in)
        h_list = cell.init_state(B)
        outputs = []
        for t in range(T):
            h, h_list = cell(x[t], h_list)
            outputs.append(h)
        self.assertEqual(outputs[-1].shape, (B, d_h))

    def test_forward_with_aux(self):
        d_in, d_h, B, K = 3, 8, 4, 4
        cell = FrozenSampledMultiTauCfCCell(d_in, d_h, n_branches=K, seed=42)
        h_list = cell.init_state(B)
        x_t = torch.randn(B, d_in)
        h, _, aux = cell.forward_with_aux(x_t, h_list)
        self.assertEqual(h.shape, (B, d_h))
        for k in ("alpha", "alpha_entropy", "tau_frozen", "log_coverage"):
            self.assertIn(k, aux)
        self.assertEqual(aux["alpha"].shape, (K,))
        # α sums to 1.
        self.assertAlmostEqual(aux["alpha"].sum().item(), 1.0, places=5)

    def test_alpha_init_uniform(self):
        cell = FrozenSampledMultiTauCfCCell(3, 8, n_branches=4, learn_mix=True)
        a = cell.alpha
        # softmax(0) = 1/K
        for v in a.tolist():
            self.assertAlmostEqual(v, 0.25, places=4)

    def test_alpha_max_entropy_when_initialized(self):
        cell = FrozenSampledMultiTauCfCCell(3, 8, n_branches=4, learn_mix=True)
        ent = cell.alpha_entropy().item() if hasattr(cell, "alpha_entropy") else None
        # Compute manually: should be log(4) ≈ 1.386.
        a = cell.alpha
        eps = 1e-8
        manual = float((-a * (a + eps).log()).sum().item())
        self.assertAlmostEqual(manual, math.log(4), places=4)

    def test_fixed_mix_returns_uniform(self):
        cell = FrozenSampledMultiTauCfCCell(3, 8, n_branches=4, learn_mix=False)
        for v in cell.alpha.tolist():
            self.assertAlmostEqual(v, 0.25, places=4)


if __name__ == "__main__":
    unittest.main()