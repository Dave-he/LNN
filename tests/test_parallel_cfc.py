"""Unit tests for ParallelCfC cell + network (Round 301, PLAN-inspired).

Validates:
  - window=1 path is shape-compatible with vanilla CfCCell.
  - window>1 path returns finite outputs and is differentiable.
  - chunked T-divisibility contract holds.
  - layer stacking (num_layers>1) is well-defined.
"""
from __future__ import annotations

import unittest

import numpy as np
import torch

from lnn.core.parallel_cfc import ParallelCfCCell, ParallelCfCNetwork


def _seed(s: int = 0) -> None:
    torch.manual_seed(s)
    np.random.seed(s)


class TestParallelCfCCellWindow1(unittest.TestCase):
    """window=1 must behave like a single CfC step."""

    def test_init_window_1(self):
        cell = ParallelCfCCell(input_size=3, hidden_size=8, window=1)
        self.assertEqual(cell.window, 1)
        self.assertEqual(cell.hidden_size, 8)
        self.assertEqual(cell.input_size, 3)

    def test_window_1_forward_shape(self):
        _seed(0)
        cell = ParallelCfCCell(input_size=4, hidden_size=12, window=1)
        x_t = torch.randn(2, 4)
        h = torch.randn(2, 12)
        out = cell(x_t, h, dt=torch.tensor(1.0))
        self.assertEqual(out.shape, (2, 12))
        self.assertTrue(torch.isfinite(out).all())

    def test_window_1_with_2d_dt(self):
        cell = ParallelCfCCell(input_size=2, hidden_size=4, window=1)
        x_t = torch.randn(3, 2)
        h = torch.randn(3, 4)
        dt = torch.tensor([1.0, 0.5, 2.0])
        out = cell(x_t, h, dt=dt)
        self.assertEqual(out.shape, (3, 4))
        self.assertTrue(torch.isfinite(out).all())

    def test_window_1_gradient_flows(self):
        _seed(1)
        cell = ParallelCfCCell(input_size=3, hidden_size=8, window=1)
        x_t = torch.randn(4, 3)
        h = torch.randn(4, 8)
        out = cell(x_t, h, dt=torch.tensor(1.0))
        out.sum().backward()
        self.assertIsNotNone(cell.time_scale.grad)
        ts_grad = cell.time_scale.grad
        assert ts_grad is not None  # for type-checker
        self.assertTrue(torch.isfinite(ts_grad).all())
        self.assertGreater(ts_grad.abs().sum().item(), 0.0)


class TestParallelCfCCellParallel(unittest.TestCase):
    """window>1 parallel path."""

    def test_init_window_4(self):
        cell = ParallelCfCCell(input_size=3, hidden_size=8, window=4)
        self.assertEqual(cell.window, 4)
        # Same parameter names as CfCCell.
        self.assertTrue(hasattr(cell, "f_gate"))
        self.assertTrue(hasattr(cell, "g_branch"))
        self.assertTrue(hasattr(cell, "h_branch"))
        self.assertTrue(hasattr(cell, "time_scale"))

    def test_window_4_forward_shape(self):
        _seed(2)
        cell = ParallelCfCCell(input_size=3, hidden_size=8, window=4)
        x = torch.randn(2, 4, 3)  # (B, W, d_in)
        h = torch.randn(2, 8)
        out = cell(x, h, dt=torch.tensor(1.0))
        self.assertEqual(out.shape, (2, 8))
        self.assertTrue(torch.isfinite(out).all())

    def test_window_4_window_mismatch_raises(self):
        cell = ParallelCfCCell(input_size=3, hidden_size=8, window=4)
        x = torch.randn(2, 3, 3)  # W=3, but cell.window=4
        h = torch.randn(2, 8)
        with self.assertRaises(AssertionError):
            cell(x, h, dt=torch.tensor(1.0))

    def test_window_4_with_batch_dt(self):
        cell = ParallelCfCCell(input_size=2, hidden_size=4, window=4)
        x = torch.randn(3, 4, 2)
        h = torch.randn(3, 4)
        dt = torch.tensor([1.0, 0.5, 2.0])
        out = cell(x, h, dt=dt)
        self.assertEqual(out.shape, (3, 4))
        self.assertTrue(torch.isfinite(out).all())

    def test_window_4_gradient_flows(self):
        _seed(3)
        cell = ParallelCfCCell(input_size=3, hidden_size=8, window=4)
        x = torch.randn(2, 4, 3)
        h = torch.randn(2, 8)
        out = cell(x, h, dt=torch.tensor(1.0))
        out.sum().backward()
        # f_gate, g_branch, h_branch, time_scale all must receive non-zero grad.
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

    def test_window_8_output_bounded(self):
        _seed(4)
        cell = ParallelCfCCell(input_size=2, hidden_size=4, window=8, tau_init=0.5)
        x = torch.randn(2, 8, 2)
        h = torch.randn(2, 4)
        out = cell(x, h, dt=torch.tensor(2.0))
        self.assertTrue(torch.isfinite(out).all())
        self.assertLess(out.abs().max().item(), 100.0)


class TestParallelCfCNetworkBasics(unittest.TestCase):
    """Network-level: chunked sequential across chunks, parallel within."""

    def test_init(self):
        net = ParallelCfCNetwork(
            input_size=2, hidden_size=8, output_size=1, num_layers=1, window=4
        )
        self.assertEqual(len(net.cells), 1)
        self.assertEqual(net.window, 4)

    def test_window_1_sequential_path(self):
        _seed(0)
        net = ParallelCfCNetwork(
            input_size=2, hidden_size=8, output_size=1, window=1
        )
        x = torch.randn(2, 16, 2)
        out = net(x)
        self.assertEqual(out.shape, (2, 1))
        self.assertTrue(torch.isfinite(out).all())

    def test_window_4_chunked_path(self):
        _seed(1)
        net = ParallelCfCNetwork(
            input_size=2, hidden_size=8, output_size=1, window=4
        )
        x = torch.randn(2, 16, 2)  # 16 = 4 * 4 chunks
        out = net(x)
        self.assertEqual(out.shape, (2, 1))
        self.assertTrue(torch.isfinite(out).all())

    def test_window_4_non_divisible_T_raises(self):
        net = ParallelCfCNetwork(
            input_size=2, hidden_size=8, output_size=1, window=4
        )
        x = torch.randn(2, 15, 2)  # 15 not divisible by 4
        with self.assertRaises(AssertionError):
            net(x)

    def test_multilayer(self):
        _seed(2)
        net = ParallelCfCNetwork(
            input_size=2, hidden_size=8, output_size=1, num_layers=2, window=4
        )
        x = torch.randn(2, 8, 2)
        out = net(x)
        self.assertEqual(out.shape, (2, 1))
        self.assertTrue(torch.isfinite(out).all())

    def test_return_sequences(self):
        _seed(3)
        net = ParallelCfCNetwork(
            input_size=2,
            hidden_size=8,
            output_size=3,
            window=4,
            return_sequences=True,
        )
        x = torch.randn(2, 8, 2)
        out = net(x)
        self.assertEqual(out.shape, (2, 8, 3))
        self.assertTrue(torch.isfinite(out).all())

    def test_grad_flows_through_chunks(self):
        _seed(4)
        net = ParallelCfCNetwork(
            input_size=2, hidden_size=8, output_size=1, window=4
        )
        x = torch.randn(2, 8, 2)
        out = net(x)
        out.sum().backward()
        for cell in net.cells:
            self.assertIsNotNone(cell.time_scale.grad)
            ts_grad = cell.time_scale.grad
            assert ts_grad is not None
            self.assertTrue(torch.isfinite(ts_grad).all())
            self.assertGreater(ts_grad.abs().sum().item(), 0.0)


class TestParallelCfCWindowSweep(unittest.TestCase):
    """window ∈ {1, 2, 4, 8} all produce shape-valid outputs."""

    def test_window_sweep(self):
        _seed(0)
        for W in (1, 2, 4, 8):
            net = ParallelCfCNetwork(
                input_size=2, hidden_size=8, output_size=1, window=W
            )
            T = 16
            x = torch.randn(2, T, 2)
            out = net(x)
            self.assertEqual(out.shape, (2, 1), f"window={W} shape wrong")
            self.assertTrue(torch.isfinite(out).all(), f"window={W} non-finite")


class TestParallelCfCCellInvalidInit(unittest.TestCase):
    def test_window_zero_raises(self):
        with self.assertRaises(AssertionError):
            ParallelCfCCell(input_size=2, hidden_size=4, window=0)

    def test_invalid_mode_raises(self):
        with self.assertRaises(AssertionError):
            ParallelCfCCell(input_size=2, hidden_size=4, window=4, mode="bogus")


class TestParallelCfCParameterCount(unittest.TestCase):
    """Parameter count: 3 * (d_in + hidden) * hidden + hidden per cell."""

    def test_param_count_formula(self):
        cell = ParallelCfCCell(input_size=4, hidden_size=8, window=4)
        # 3 linear layers, each (4+8) * 8 + 8 = 104 params.  Plus time_scale (8).
        n_params = sum(p.numel() for p in cell.parameters())
        self.assertEqual(n_params, 3 * 12 * 8 + 3 * 8 + 8)


if __name__ == "__main__":
    unittest.main()
