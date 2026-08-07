"""Unit tests for MidpointCfC cell + network (Round 305, non-anchor parallel scan)."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from lnn.core.midpoint_cfc import MidpointCfCCell, MidpointCfCNetwork


def _seed(s: int = 0) -> None:
    torch.manual_seed(s)
    np.random.seed(s)


class TestMidpointCfCCellWindow1(unittest.TestCase):
    def test_init_window_1(self):
        cell = MidpointCfCCell(input_size=3, hidden_size=8, window=1)
        self.assertEqual(cell.window, 1)
        self.assertEqual(cell.hidden_size, 8)

    def test_window_1_forward_shape(self):
        _seed(0)
        cell = MidpointCfCCell(input_size=4, hidden_size=12, window=1)
        out = cell(torch.randn(2, 4), torch.randn(2, 12), dt=torch.tensor(1.0))
        self.assertEqual(out.shape, (2, 12))
        self.assertTrue(torch.isfinite(out).all())

    def test_window_1_with_batch_dt(self):
        cell = MidpointCfCCell(input_size=2, hidden_size=4, window=1)
        out = cell(
            torch.randn(3, 2), torch.randn(3, 4),
            dt=torch.tensor([1.0, 0.5, 2.0]),
        )
        self.assertEqual(out.shape, (3, 4))
        self.assertTrue(torch.isfinite(out).all())

    def test_window_1_gradient_flows(self):
        _seed(1)
        cell = MidpointCfCCell(input_size=3, hidden_size=8, window=1)
        out = cell(torch.randn(4, 3), torch.randn(4, 8), dt=torch.tensor(1.0))
        out.sum().backward()
        self.assertIsNotNone(cell.time_scale.grad)
        ts_grad = cell.time_scale.grad
        assert ts_grad is not None
        self.assertTrue(torch.isfinite(ts_grad).all())
        self.assertGreater(ts_grad.abs().sum().item(), 0.0)


class TestMidpointCfCCellParallel(unittest.TestCase):
    def test_init_window_4(self):
        cell = MidpointCfCCell(input_size=3, hidden_size=8, window=4)
        self.assertEqual(cell.window, 4)
        for name in ("f_gate", "g_branch", "h_branch", "time_scale"):
            self.assertTrue(hasattr(cell, name))

    def test_window_4_forward_shape(self):
        _seed(2)
        cell = MidpointCfCCell(input_size=3, hidden_size=8, window=4)
        out = cell(torch.randn(2, 4, 3), torch.randn(2, 8), dt=torch.tensor(1.0))
        self.assertEqual(out.shape, (2, 8))
        self.assertTrue(torch.isfinite(out).all())

    def test_window_4_midpoint_differs_from_anchor(self):
        """Midpoint corrector should produce a different output than pure
        anchor at h_0. We test that the result is *not exactly equal* to a
        single anchor pass — i.e. the corrector is doing real work."""
        _seed(3)
        cell = MidpointCfCCell(input_size=3, hidden_size=8, window=4)
        x = torch.randn(2, 4, 3)
        h = torch.randn(2, 8)
        out_mid = cell(x, h, dt=torch.tensor(1.0))
        h_pred = cell._parallel_eval(x, h, dt=torch.tensor(1.0))
        self.assertFalse(torch.allclose(out_mid, h_pred, atol=1e-6))

    def test_window_4_window_mismatch_raises(self):
        cell = MidpointCfCCell(input_size=3, hidden_size=8, window=4)
        with self.assertRaises(AssertionError):
            cell(torch.randn(2, 3, 3), torch.randn(2, 8), dt=torch.tensor(1.0))

    def test_window_4_with_batch_dt(self):
        cell = MidpointCfCCell(input_size=2, hidden_size=4, window=4)
        out = cell(
            torch.randn(3, 4, 2), torch.randn(3, 4),
            dt=torch.tensor([1.0, 0.5, 2.0]),
        )
        self.assertEqual(out.shape, (3, 4))
        self.assertTrue(torch.isfinite(out).all())

    def test_window_4_gradient_flows(self):
        _seed(4)
        cell = MidpointCfCCell(input_size=3, hidden_size=8, window=4)
        out = cell(torch.randn(2, 4, 3), torch.randn(2, 8), dt=torch.tensor(1.0))
        out.sum().backward()
        for p in cell.f_gate.parameters():
            self.assertIsNotNone(p.grad)
            p_grad = p.grad
            assert p_grad is not None
            self.assertTrue(torch.isfinite(p_grad).all())
            self.assertGreater(p_grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(cell.time_scale.grad)
        ts_grad = cell.time_scale.grad
        assert ts_grad is not None
        self.assertTrue(torch.isfinite(ts_grad).all())

    def test_window_8_output_finite(self):
        _seed(5)
        cell = MidpointCfCCell(input_size=2, hidden_size=4, window=8, tau_init=0.5)
        out = cell(
            torch.randn(2, 8, 2), torch.randn(2, 4),
            dt=torch.tensor(2.0),
        )
        self.assertTrue(torch.isfinite(out).all())


class TestMidpointCfCNetworkBasics(unittest.TestCase):
    def test_init(self):
        net = MidpointCfCNetwork(
            input_size=2, hidden_size=8, output_size=1, num_layers=1, window=4
        )
        self.assertEqual(len(net.cells), 1)
        self.assertEqual(net.window, 4)

    def test_window_1_sequential(self):
        _seed(0)
        net = MidpointCfCNetwork(
            input_size=2, hidden_size=8, output_size=1, window=1
        )
        out = net(torch.randn(2, 16, 2))
        self.assertEqual(out.shape, (2, 1))
        self.assertTrue(torch.isfinite(out).all())

    def test_window_4_chunked(self):
        _seed(1)
        net = MidpointCfCNetwork(
            input_size=2, hidden_size=8, output_size=1, window=4
        )
        out = net(torch.randn(2, 16, 2))
        self.assertEqual(out.shape, (2, 1))
        self.assertTrue(torch.isfinite(out).all())

    def test_window_4_non_divisible_T_raises(self):
        net = MidpointCfCNetwork(
            input_size=2, hidden_size=8, output_size=1, window=4
        )
        with self.assertRaises(AssertionError):
            net(torch.randn(2, 15, 2))

    def test_multilayer(self):
        _seed(2)
        net = MidpointCfCNetwork(
            input_size=2, hidden_size=8, output_size=1, num_layers=2, window=4
        )
        out = net(torch.randn(2, 8, 2))
        self.assertEqual(out.shape, (2, 1))
        self.assertTrue(torch.isfinite(out).all())

    def test_return_sequences(self):
        _seed(3)
        net = MidpointCfCNetwork(
            input_size=2, hidden_size=8, output_size=3, window=4,
            return_sequences=True,
        )
        out = net(torch.randn(2, 8, 2))
        self.assertEqual(out.shape, (2, 8, 3))
        self.assertTrue(torch.isfinite(out).all())

    def test_grad_flows(self):
        _seed(4)
        net = MidpointCfCNetwork(
            input_size=2, hidden_size=8, output_size=1, window=4
        )
        out = net(torch.randn(2, 8, 2))
        out.sum().backward()
        for cell in net.cells:
            self.assertIsNotNone(cell.time_scale.grad)
            ts_grad = cell.time_scale.grad
            assert ts_grad is not None
            self.assertTrue(torch.isfinite(ts_grad).all())
            self.assertGreater(ts_grad.abs().sum().item(), 0.0)


class TestMidpointCfCWindowSweep(unittest.TestCase):
    def test_window_sweep(self):
        _seed(0)
        for W in (1, 2, 4, 8):
            net = MidpointCfCNetwork(
                input_size=2, hidden_size=8, output_size=1, window=W
            )
            out = net(torch.randn(2, 16, 2))
            self.assertEqual(out.shape, (2, 1))
            self.assertTrue(torch.isfinite(out).all())


class TestMidpointCfCParameterCount(unittest.TestCase):
    def test_param_count_formula(self):
        cell = MidpointCfCCell(input_size=4, hidden_size=8, window=4)
        n_params = sum(p.numel() for p in cell.parameters())
        # 3 * 12 * 8 + 3 * 8 + 8 = 320 — same as ParallelCfCCell.
        self.assertEqual(n_params, 320)


if __name__ == "__main__":
    unittest.main()
