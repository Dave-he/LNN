"""Tests for MultiBasinLyapunovStableCfCCell (arXiv:2606.18315 response, round 244)."""

from __future__ import annotations

import math
import unittest

import torch

from lnn.core.multi_basin_lyapunov_cfc import (
    MultiBasinLyapunovStableCfCCell,
    basin_assignment_entropy,
    multi_basin_distance,
    multi_basin_iss_decay_loss,
    multi_basin_lyap_decay_loss,
    multi_basin_lyapunov_value,
)


class TestMultiBasinDistance(unittest.TestCase):
    def test_distance_shape(self):
        h = torch.randn(4, 6)
        c = torch.randn(3, 6)
        d = multi_basin_distance(h, c)
        self.assertEqual(d.shape, (4, 3))

    def test_zero_distance_at_center(self):
        h = torch.tensor([[1.0, 0.0, 0.0]])
        c = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        d = multi_basin_distance(h, c)
        self.assertAlmostEqual(d[0, 0].item(), 0.0)
        self.assertAlmostEqual(d[0, 1].item(), 1.0)


class TestMultiBasinLyapunovValue(unittest.TestCase):
    def test_low_value_at_center(self):
        h = torch.tensor([[1.0, 0.0]])
        c = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
        V = multi_basin_lyapunov_value(h, c, beta_v=10.0)
        # Closest distance is 0 → V → 0
        self.assertLess(V.item(), 1e-3)

    def test_higher_value_away_from_all_centers(self):
        h_near = torch.tensor([[1.0, 0.0]])
        h_far = torch.tensor([[5.0, 5.0]])
        c = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
        V_near = multi_basin_lyapunov_value(h_near, c).item()
        V_far = multi_basin_lyapunov_value(h_far, c).item()
        self.assertGreater(V_far, V_near)

    def test_soft_min_temperature_monotone(self):
        # As β_v → ∞, V(h) → min_k d_k² from below.
        # As β_v → 0, V(h) → average d_k².
        h = torch.tensor([[0.5, 0.0]])
        c = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
        # d² = [0.25, 2.25] → min = 0.25 (β→∞), mean = 1.25 (β→0).
        V_low = multi_basin_lyapunov_value(h, c, beta_v=0.5).item()
        V_high = multi_basin_lyapunov_value(h, c, beta_v=100.0).item()
        # V should be ≤ min(d²) for any β_v.
        self.assertLessEqual(V_high, 0.25 + 1e-3)
        self.assertLessEqual(V_low, 0.25 + 1e-3)
        # V(high β) is closer to min = 0.25 than V(low β).
        self.assertLess(abs(V_high - 0.25), abs(V_low - 0.25))


class TestMultiBasinDecayLoss(unittest.TestCase):
    def test_loss_shape(self):
        h = torch.randn(4, 6)
        h_next = torch.randn(4, 6)
        c = torch.randn(3, 6)
        loss = multi_basin_lyap_decay_loss(h, h_next, c)
        self.assertEqual(loss.shape, ())
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_zero_loss_when_contracting(self):
        # If h_next → c[0] (closest basin) and h is at c[0], loss should be 0.
        h = torch.tensor([[0.0, 0.0]])
        h_next = torch.tensor([[0.0, 0.0]])
        c = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        loss = multi_basin_lyap_decay_loss(h, h_next, c)
        self.assertLess(loss.item(), 1e-3)


class TestMultiBasinISS(unittest.TestCase):
    def test_input_relaxes_bound(self):
        h = torch.tensor([[0.0, 0.0]])
        h_next = torch.tensor([[2.0, 2.0]])
        c = torch.tensor([[0.0, 0.0]])
        x_zero = torch.zeros(1, 1)
        x_large = torch.tensor([[10.0]])
        L_zero = multi_basin_iss_decay_loss(h, h_next, x_zero, c).item()
        L_large = multi_basin_iss_decay_loss(h, h_next, x_large, c).item()
        # Larger ||x||² should give a smaller (or equal) ISS loss.
        self.assertLessEqual(L_large, L_zero)


class TestBasinAssignmentEntropy(unittest.TestCase):
    def test_max_entropy_when_uniform(self):
        h = torch.tensor([[0.0, 0.0]])
        c = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
        ent = basin_assignment_entropy(h, c, beta_v=0.0).item()
        # With β_v=0 all distances are zero → uniform over 3 basins.
        self.assertAlmostEqual(ent, math.log(3), places=4)

    def test_zero_entropy_when_one_center_dominates(self):
        h = torch.tensor([[1.0, 0.0]])
        c = torch.tensor([[1.0, 0.0], [-100.0, -100.0]])
        ent = basin_assignment_entropy(h, c, beta_v=10.0).item()
        self.assertLess(ent, 0.1)


class TestMultiBasinLyapunovStableCfCCell(unittest.TestCase):
    def test_forward_shape(self):
        d_in, d_h, T, B, K = 3, 8, 12, 4, 3
        cell = MultiBasinLyapunovStableCfCCell(d_in, d_h, n_basin=K)
        x = torch.randn(T, B, d_in)
        h = torch.zeros(B, d_h)
        outputs = []
        for t in range(T):
            h = cell(x[t], h)
            outputs.append(h)
        self.assertEqual(outputs[-1].shape, (B, d_h))

    def test_forward_with_aux(self):
        d_in, d_h, B, K = 3, 8, 4, 3
        cell = MultiBasinLyapunovStableCfCCell(d_in, d_h, n_basin=K)
        h = torch.zeros(B, d_h)
        x_t = torch.randn(B, d_in)
        h_next, aux = cell.forward_with_aux(
            x_t, h, lyap_lambda=0.1, sep_lambda=0.01, iss_lambda=0.05,
        )
        self.assertEqual(h_next.shape, (B, d_h))
        for k in ("V_h", "V_next", "basin_assign", "basin_entropy",
                  "lyap_loss", "lyap_loss_total", "iss_loss_total",
                  "sep_loss_total"):
            self.assertIn(k, aux)
        self.assertEqual(aux["basin_assign"].shape, (B, K))
        # Basin assignment should sum to 1.
        sums = aux["basin_assign"].sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones(B), atol=1e-5))

    def test_separation_loss(self):
        d_in, d_h, K = 2, 4, 3
        cell = MultiBasinLyapunovStableCfCCell(d_in, d_h, n_basin=K,
                                               pd_eps=0.5)
        # Initially centres are random — separation may be 0 or positive.
        sep = cell.basin_separation_loss().item()
        self.assertGreaterEqual(sep, 0.0)


if __name__ == "__main__":
    unittest.main()