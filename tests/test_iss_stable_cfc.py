"""Tests for ISSStableCfCCell (arXiv:2606.14136 response, round 242)."""

from __future__ import annotations

import unittest

import torch

from lnn.core.iss_stable_cfc import (
    ISSStableCfCCell,
    input_bound_ratio,
    iss_decay_loss,
)
from lnn.core.lyapunov_stable_cfc import (
    lyapunov_value,
    make_lyapunov_matrix,
    positive_definite_loss,
)


class TestISSDecayLoss(unittest.TestCase):
    def test_zero_when_contraction_holds_with_input(self):
        """If V shrinks AND the input is large enough, ISS loss = 0."""
        d = 4
        P = torch.eye(d)
        h = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        h_next = h * 0.5  # V drops by 75%
        x_t = torch.tensor([[2.0, 0.0, 0.0, 0.0]])  # ||x||^2 = 4
        # Need: V_next <= (1 - alpha) * V - beta * ||x||^2
        # 0.25 <= 0.95 * 1.0 - 0.01 * 4 = 0.91 -> OK
        loss = iss_decay_loss(h, h_next, x_t, P, alpha=0.05, beta=0.01)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_positive_when_V_grows_too_much(self):
        d = 4
        P = torch.eye(d)
        h = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        # V_next > (1 - alpha) * V - beta * ||x||^2
        h_next = h * 1.5  # V grows by 125%
        x_t = torch.tensor([[0.1, 0.0, 0.0, 0.0]])  # ||x||^2 = 0.01 (tiny)
        loss = iss_decay_loss(h, h_next, x_t, P, alpha=0.05, beta=0.01)
        # 2.25 - 0.95 + 0.0001 = 1.30 -> loss > 0
        self.assertGreater(loss.item(), 0.0)

    def test_input_can_offset_growth(self):
        """A large input should reduce the loss relative to a small input."""
        d = 4
        P = torch.eye(d)
        h = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        h_next = h * 1.2  # V grows by 44%
        x_small = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
        x_large = torch.tensor([[10.0, 0.0, 0.0, 0.0]])  # ||x||^2 = 100
        loss_small = iss_decay_loss(h, h_next, x_small, P, alpha=0.05, beta=0.01)
        loss_large = iss_decay_loss(h, h_next, x_large, P, alpha=0.05, beta=0.01)
        # Large input -> lower (more negative) bound -> loss smaller
        self.assertLess(loss_large.item(), loss_small.item())


class TestInputBoundRatio(unittest.TestCase):
    def test_ratio_below_one_when_iss_holds(self):
        d = 4
        P = torch.eye(d)
        h = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        h_next = h * 0.5
        x_t = torch.tensor([[0.5, 0.0, 0.0, 0.0]])
        ratio = input_bound_ratio(h, h_next, x_t, P, alpha=0.05, beta=0.01)
        # bound = 0.95*1 + 0.01*0.25 = 0.9525; V_next = 0.25; ratio = 0.262 < 1
        self.assertLess(ratio.item(), 1.0)


class TestPositiveDefiniteLoss(unittest.TestCase):
    def test_inherited_from_round_240(self):
        d = 5
        P = torch.eye(d)
        loss = positive_definite_loss(P, eps=1e-3)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)


class TestISSStableCfCCell(unittest.TestCase):
    def test_forward_shape(self):
        d_in, d_h, T, B = 3, 8, 12, 4
        cell = ISSStableCfCCell(d_in, d_h)
        x = torch.randn(T, B, d_in)
        h = torch.zeros(B, d_h)
        outputs = []
        for t in range(T):
            h = cell(x[t], h)
            outputs.append(h)
        self.assertEqual(outputs[-1].shape, (B, d_h))

    def test_forward_with_aux(self):
        d_in, d_h, B = 2, 5, 3
        cell = ISSStableCfCCell(d_in, d_h)
        x_t = torch.randn(B, d_in)
        h = torch.randn(B, d_h)
        h_next, aux = cell.forward_with_aux(x_t, h, iss_lambda=0.1, pd_lambda=0.01)
        self.assertEqual(h_next.shape, (B, d_h))
        for k in ("h", "h_next", "x_t", "V_h", "V_next", "x_norm_sq",
                  "iss_loss", "pd_loss", "bound_ratio",
                  "iss_loss_total", "pd_loss_total"):
            self.assertIn(k, aux)
        self.assertAlmostEqual(
            aux["iss_loss_total"].item(),
            0.1 * aux["iss_loss"].item(),
            places=6,
        )
        self.assertAlmostEqual(
            aux["pd_loss_total"].item(),
            0.01 * aux["pd_loss"].item(),
            places=6,
        )

    def test_alpha_beta_propagate(self):
        cell = ISSStableCfCCell(2, 4, alpha=0.123, beta=0.456)
        self.assertAlmostEqual(cell.alpha, 0.123, places=6)
        self.assertAlmostEqual(cell.beta, 0.456, places=6)

    def test_multi_tau(self):
        d_in, d_h, n_tau = 2, 6, 3
        cell = ISSStableCfCCell(d_in, d_h, n_tau=n_tau, tau_scales=(0.1, 1.0, 10.0))
        h = torch.zeros(2, d_h)
        x_t = torch.randn(2, d_in)
        h_next, aux = cell.forward_with_aux(x_t, h)
        self.assertEqual(h_next.shape, (2, d_h))
        self.assertEqual(cell.cell.n_tau, 3)

    def test_lyapunov_matrix_inherited(self):
        """Round 240 and 242 share the lyapunov_P module API."""
        cell = ISSStableCfCCell(2, 4)
        self.assertTrue(hasattr(cell, "lyapunov_P"))
        self.assertEqual(cell.lyapunov_P.shape, (4, 4))


if __name__ == "__main__":
    unittest.main()