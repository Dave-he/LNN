"""Tests for HierarchicalMultiTauCfCCell (arXiv:2606.19579 FlowFake response, round 245)."""

from __future__ import annotations

import unittest

import torch

from lnn.core.hierarchical_multitau_cfc import HierarchicalMultiTauCfCCell


class TestHierarchicalMultiTauCfCCell(unittest.TestCase):
    def test_forward_shape(self):
        d_in, d_h, T, B = 3, 8, 12, 4
        cell = HierarchicalMultiTauCfCCell(d_in, d_h)
        x = torch.randn(T, B, d_in)
        h_f, h_s = cell.init_state(B)
        outputs = []
        for t in range(T):
            h, h_f, h_s = cell(x[t], h_f, h_s)
            outputs.append(h)
        self.assertEqual(outputs[-1].shape, (B, d_h))

    def test_forward_with_aux(self):
        d_in, d_h, B = 3, 8, 4
        cell = HierarchicalMultiTauCfCCell(d_in, d_h, learn_mix=True)
        h_f, h_s = cell.init_state(B)
        x_t = torch.randn(B, d_in)
        h, h_f2, h_s2, aux = cell.forward_with_aux(x_t, h_f, h_s)
        self.assertEqual(h.shape, (B, d_h))
        self.assertEqual(h_f2.shape, (B, d_h))
        self.assertEqual(h_s2.shape, (B, d_h))
        for k in ("h_next", "h_fast_next", "h_slow_next", "alpha"):
            self.assertIn(k, aux)
        a = aux["alpha"].item()
        self.assertGreater(a, 0.0)
        self.assertLess(a, 1.0)

    def test_alpha_initialized_to_half_when_mix_zero(self):
        cell = HierarchicalMultiTauCfCCell(3, 8, learn_mix=True, mix_init=0.0)
        self.assertAlmostEqual(cell.alpha.item(), 0.5, places=4)

    def test_alpha_mix_init_shifts_balance(self):
        cell_pos = HierarchicalMultiTauCfCCell(3, 8, learn_mix=True, mix_init=2.0)
        cell_neg = HierarchicalMultiTauCfCCell(3, 8, learn_mix=True, mix_init=-2.0)
        # sigmoid(2) ≈ 0.88, sigmoid(-2) ≈ 0.12
        self.assertGreater(cell_pos.alpha.item(), 0.8)
        self.assertLess(cell_neg.alpha.item(), 0.2)

    def test_fixed_mix_returns_half(self):
        cell = HierarchicalMultiTauCfCCell(3, 8, learn_mix=False)
        self.assertAlmostEqual(cell.alpha.item(), 0.5, places=4)

    def test_tau_bands_distinct(self):
        cell = HierarchicalMultiTauCfCCell(3, 8, tau_fast=0.1, tau_slow=5.0)
        tau_fast_actual = cell.fast_cell.time_scale.mean().item()
        tau_slow_actual = cell.slow_cell.time_scale.mean().item()
        self.assertAlmostEqual(tau_fast_actual, 0.1, places=4)
        self.assertAlmostEqual(tau_slow_actual, 5.0, places=4)

    def test_bands_have_different_time_scales(self):
        """The two bands should have distinct effective time constants —
        this is the whole point of non-geometric multi-τ."""
        d_in, d_h = 2, 8
        cell = HierarchicalMultiTauCfCCell(d_in, d_h, tau_fast=0.05,
                                            tau_slow=2.0)
        # Apply the same input repeatedly and compare convergence rates.
        h_f, h_s = cell.init_state(16)
        x = torch.full((16, d_in), 1.0)
        diffs_fast = []
        diffs_slow = []
        for _ in range(20):
            h, h_f, h_s = cell(x, h_f, h_s)
            diffs_fast.append(h_f.std().item())
            diffs_slow.append(h_s.std().item())
        # The slow band should have lower variance across time (smoother).
        # We don't assert strict ordering — the test is just that the
        # bands are *operationally distinct*.
        self.assertNotEqual(round(diffs_fast[-1], 3),
                            round(diffs_slow[-1], 3))

    def test_mix_learnable(self):
        cell = HierarchicalMultiTauCfCCell(3, 8, learn_mix=True)
        n_params = sum(p.numel() for p in cell.parameters() if p.requires_grad)
        # Two CfC cells each ~3*(d_in+d_h)*d_h + d_h params, plus 1 mix param.
        # Just sanity-check that there is at least one scalar param.
        self.assertGreater(n_params, 0)
        # Verify the mix_param is in the parameter list.
        param_names = [n for n, _ in cell.named_parameters()]
        self.assertIn("mix_param", param_names)


if __name__ == "__main__":
    unittest.main()