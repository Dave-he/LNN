"""Tests for LyapunovStableCfCCell (arXiv:2606.19109 response, round 240)."""

from __future__ import annotations

import unittest

import torch

from lnn.core.lyapunov_stable_cfc import (
    LyapunovStableCfCCell,
    lyapunov_decay_loss,
    lyapunov_value,
    make_lyapunov_matrix,
    positive_definite_loss,
)


class TestLyapunovValue(unittest.TestCase):
    def test_zero_state(self):
        d = 4
        P = make_lyapunov_matrix(d, scale=1.0)
        h = torch.zeros(3, d)
        V = lyapunov_value(h, P)
        # V(0) = 0 (Lyapunov axiom).
        self.assertTrue(torch.allclose(V, torch.zeros(3), atol=1e-6))

    def test_identity_gives_squared_norm(self):
        d = 5
        P = torch.eye(d)
        h = torch.randn(7, d)
        V = lyapunov_value(h, P)
        expected = (h * h).sum(dim=-1)
        self.assertTrue(torch.allclose(V, expected, atol=1e-5))

    def test_positive_definite_diag(self):
        d = 3
        P = 2.0 * torch.eye(d)
        h = torch.randn(4, d)
        V = lyapunov_value(h, P)
        # V must be non-negative everywhere.
        self.assertTrue((V >= -1e-5).all())


class TestLyapunovDecayLoss(unittest.TestCase):
    def test_decay_when_v_shrinks(self):
        d = 4
        P = torch.eye(d)
        h = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]])
        # Shrink each state by sqrt(0.9) -> V drops by 10% per step.
        h_next = h * (0.9 ** 0.5)
        loss = lyapunov_decay_loss(h, h_next, P, alpha=0.05)
        # 10% drop satisfies 5% contraction, so loss = 0.
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_nonzero_when_v_grows(self):
        d = 4
        P = torch.eye(d)
        h = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        # Expand by sqrt(1.1) -> V grows by 10%.
        h_next = h * (1.1 ** 0.5)
        loss = lyapunov_decay_loss(h, h_next, P, alpha=0.05)
        # Should be positive (V grows beyond contraction).
        self.assertGreater(loss.item(), 0.0)


class TestPositiveDefiniteLoss(unittest.TestCase):
    def test_pd_loss_zero_for_identity(self):
        d = 5
        P = torch.eye(d)
        loss = positive_definite_loss(P, eps=1e-3)
        # lambda_min(I) = 1 > eps, so loss = 0.
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_pd_loss_positive_for_non_pd(self):
        d = 3
        # Build a non-PD symmetric matrix with a negative eigenvalue.
        P = torch.tensor([[1.0, 2.0, 0.0], [2.0, 1.0, 2.0], [0.0, 2.0, 1.0]])
        loss = positive_definite_loss(P, eps=1e-3)
        self.assertGreater(loss.item(), 0.0)


class TestLyapunovStableCfCCell(unittest.TestCase):
    def test_forward_shape(self):
        d_in, d_h, T, B = 3, 8, 12, 4
        cell = LyapunovStableCfCCell(d_in, d_h)
        x = torch.randn(T, B, d_in)
        h = torch.zeros(B, d_h)
        outputs = []
        for t in range(T):
            h = cell(x[t], h)
            outputs.append(h)
        self.assertEqual(outputs[-1].shape, (B, d_h))

    def test_forward_with_aux(self):
        d_in, d_h, B = 2, 5, 3
        cell = LyapunovStableCfCCell(d_in, d_h)
        x_t = torch.randn(B, d_in)
        h = torch.randn(B, d_h)
        h_next, aux = cell.forward_with_aux(x_t, h, lyap_lambda=0.1, pd_lambda=0.01)
        self.assertEqual(h_next.shape, (B, d_h))
        # All keys must be present.
        for k in ("h", "h_next", "V_h", "V_next", "lyap_decay_loss", "pd_loss",
                  "lyap_loss_total", "pd_loss_total"):
            self.assertIn(k, aux)
        # Total = lambda * raw.
        self.assertAlmostEqual(
            aux["lyap_loss_total"].item(),
            0.1 * aux["lyap_decay_loss"].item(),
            places=6,
        )
        self.assertAlmostEqual(
            aux["pd_loss_total"].item(),
            0.01 * aux["pd_loss"].item(),
            places=6,
        )

    def test_v_satisfied_for_strong_contraction(self):
        """If CfC drives V -> 0, the decay loss should be near zero."""
        d_in, d_h, B = 1, 4, 6
        cell = LyapunovStableCfCCell(d_in, d_h, alpha=0.05)
        # Force the next-state to be very small so V drops sharply.
        h = torch.randn(B, d_h)
        x_t = torch.zeros(B, d_in)
        h_next, aux = cell.forward_with_aux(x_t, h)
        # V must be non-negative.
        self.assertTrue((aux["V_h"] >= -1e-5).all())
        self.assertTrue((aux["V_next"] >= -1e-5).all())

    def test_multi_tau(self):
        d_in, d_h, n_tau = 2, 6, 3
        cell = LyapunovStableCfCCell(d_in, d_h, n_tau=n_tau, tau_scales=(0.1, 1.0, 10.0))
        h = torch.zeros(2, d_h)
        x_t = torch.randn(2, d_in)
        h_next, aux = cell.forward_with_aux(x_t, h)
        self.assertEqual(h_next.shape, (2, d_h))
        self.assertEqual(cell.cell.n_tau, 3)


if __name__ == "__main__":
    unittest.main()