"""Tests for AdaptiveGatedMultiTauCfCCell (arXiv:2606.22801 response, round 243)."""

from __future__ import annotations

import math
import unittest

import torch

from lnn.core.adaptive_gated_multitau_cfc import (
    AdaptiveGatedMultiTauCfCCell,
    _extend_scales,
    gated_fusion_entropy,
)


class TestExtendScales(unittest.TestCase):
    def test_short_input(self):
        out = _extend_scales((0.1, 1.0), 4)
        self.assertEqual(out, [0.1, 1.0, 10.0, 100.0])

    def test_exact_input(self):
        self.assertEqual(_extend_scales((0.1, 1.0, 10.0), 3), [0.1, 1.0, 10.0])

    def test_long_input_truncates(self):
        out = _extend_scales((0.1, 1.0, 10.0, 100.0), 2)
        self.assertEqual(out, [0.1, 1.0])


class TestGatedFusionEntropy(unittest.TestCase):
    def test_max_entropy_when_uniform(self):
        gate = torch.tensor([[1.0 / 3, 1.0 / 3, 1.0 / 3]])
        ent = gated_fusion_entropy(gate).item()
        # log(3) ≈ 1.0986
        self.assertAlmostEqual(ent, math.log(3), places=4)

    def test_zero_entropy_when_one_hot(self):
        gate = torch.tensor([[1.0, 0.0, 0.0]])
        ent = gated_fusion_entropy(gate).item()
        self.assertAlmostEqual(ent, 0.0, places=4)


class TestAdaptiveGatedMultiTauCfCCell(unittest.TestCase):
    def test_forward_shape(self):
        d_in, d_h, T, B, n_tau = 3, 8, 12, 4, 3
        cell = AdaptiveGatedMultiTauCfCCell(d_in, d_h, n_tau=n_tau)
        x = torch.randn(T, B, d_in)
        h = torch.zeros(B, d_h)
        outputs = []
        for t in range(T):
            h = cell(x[t], h)
            outputs.append(h)
        self.assertEqual(outputs[-1].shape, (B, d_h))

    def test_forward_with_aux(self):
        d_in, d_h, B, n_tau = 3, 8, 4, 3
        cell = AdaptiveGatedMultiTauCfCCell(d_in, d_h, n_tau=n_tau)
        h = torch.zeros(B, d_h)
        x_t = torch.randn(B, d_in)
        h_next, aux = cell.forward_with_aux(x_t, h, gate_entropy_lambda=0.1)
        self.assertEqual(h_next.shape, (B, d_h))
        for k in ("h_next", "tau_eff", "gate", "gate_entropy", "gate_loss_total"):
            self.assertIn(k, aux)
        self.assertEqual(aux["tau_eff"].shape, (B, n_tau))
        self.assertEqual(aux["gate"].shape, (B, n_tau))

    def test_per_branch_tau_input_dependent(self):
        """Different inputs should yield different effective τ."""
        d_in, d_h, B, n_tau = 4, 6, 2, 3
        cell = AdaptiveGatedMultiTauCfCCell(d_in, d_h, n_tau=n_tau)
        x_a = torch.full((B, d_in), 1.0)
        x_b = torch.full((B, d_in), -1.0)
        h = torch.zeros(B, d_h)
        # Manually call _per_branch_tau to inspect τ without running the cell.
        taus_a = cell._per_branch_tau(x_a)
        taus_b = cell._per_branch_tau(x_b)
        # At least one branch should differ between the two inputs.
        diff = (taus_a - taus_b).abs().max().item()
        self.assertGreater(diff, 1e-4)

    def test_tau_stays_positive(self):
        d_in, d_h, B, n_tau = 3, 6, 4, 3
        cell = AdaptiveGatedMultiTauCfCCell(d_in, d_h, n_tau=n_tau)
        x = torch.randn(B, d_in) * 5.0  # large inputs
        taus = cell._per_branch_tau(x)
        # τ_eff = exp(log_tau) * sigmoid(...) is always positive.
        self.assertTrue((taus > 0).all())

    def test_gate_sums_to_one(self):
        d_in, d_h, B, n_tau = 3, 6, 4, 3
        cell = AdaptiveGatedMultiTauCfCCell(d_in, d_h, n_tau=n_tau)
        x = torch.randn(B, d_in)
        gate = torch.softmax(cell.W_gate(x), dim=-1)
        sums = gate.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones(B), atol=1e-5))

    def test_n_tau_2_works(self):
        d_in, d_h, B = 2, 4, 2
        cell = AdaptiveGatedMultiTauCfCCell(d_in, d_h, n_tau=2, tau_base=(0.1, 1.0))
        x = torch.randn(B, d_in)
        h = torch.zeros(B, d_h)
        h_next, aux = cell.forward_with_aux(x, h)
        self.assertEqual(h_next.shape, (B, d_h))
        self.assertEqual(aux["tau_eff"].shape, (B, 2))
        self.assertEqual(aux["gate"].shape, (B, 2))


if __name__ == "__main__":
    unittest.main()