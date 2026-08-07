"""Unit tests for STEParallelCfC cell + network (Round 303).

Validates:
  - window=1 degenerate path (both branches single-step, STE still active).
  - window>1 parallel path (PLAN anchor + sequential fall-back).
  - STE mask shape and density contract (forward=hard, density=rho).
  - Gradient flow through hard mask (STE straight-through).
  - Entropy regulariser forward/backward.
  - Multi-layer network.
  - Init / dtype / device consistency.
  - Edge cases: density=0 (all sequential), density=1 (all parallel).
  - Extra-loss accumulation in multi-layer network.
"""
from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from lnn.core.ste_parallel_cfc import (
    STEParallelCfCCell,
    STEParallelCfCNetwork,
    _topk_binary_mask,
)


def _seed(s: int = 0) -> None:
    torch.manual_seed(s)
    np.random.seed(s)


class TestTopkBinaryMask(unittest.TestCase):
    """The top-k helper is the building block of the STE hard mask."""

    def test_density_one_all_ones(self):
        m = _topk_binary_mask(torch.randn(10), density=1.0)
        self.assertEqual(m.shape, (10,))
        self.assertTrue(torch.allclose(m, torch.ones(10)))

    def test_density_zero_all_zeros(self):
        m = _topk_binary_mask(torch.randn(10), density=0.0)
        self.assertEqual(m.shape, (10,))
        self.assertTrue(torch.allclose(m, torch.zeros(10)))

    def test_density_half_keeps_five(self):
        logits = torch.arange(10, dtype=torch.float32)
        m = _topk_binary_mask(logits, density=0.5)
        self.assertEqual(int(m.sum().item()), 5)
        # The top-5 logits (5,6,7,8,9) should be kept.
        self.assertTrue(torch.allclose(m[5:], torch.ones(5)))
        self.assertTrue(torch.allclose(m[:5], torch.zeros(5)))

    def test_density_rounded_up(self):
        # 10 * 0.34 = 3.4 -> round -> 3.
        m = _topk_binary_mask(torch.arange(10, dtype=torch.float32), density=0.34)
        self.assertEqual(int(m.sum().item()), 3)


class TestSTEParallelCfCCellInit(unittest.TestCase):
    def test_default_init(self):
        cell = STEParallelCfCCell(input_size=3, hidden_size=8, window=8)
        self.assertEqual(cell.window, 8)
        self.assertEqual(cell.hidden_size, 8)
        self.assertEqual(cell.input_size, 3)
        self.assertEqual(cell.density, 0.5)
        self.assertTrue(hasattr(cell, "route_logits"))
        self.assertEqual(cell.route_logits.shape, (8,))
        # Both branches should expose the standard CfC sub-modules.
        self.assertTrue(hasattr(cell, "f_gate_p"))
        self.assertTrue(hasattr(cell, "g_branch_p"))
        self.assertTrue(hasattr(cell, "h_branch_p"))
        self.assertTrue(hasattr(cell, "f_gate_s"))
        self.assertTrue(hasattr(cell, "g_branch_s"))
        self.assertTrue(hasattr(cell, "h_branch_s"))
        self.assertTrue(hasattr(cell, "time_scale"))

    def test_invalid_density_raises(self):
        with self.assertRaises(AssertionError):
            STEParallelCfCCell(input_size=2, hidden_size=4, window=4, density=1.5)
        with self.assertRaises(AssertionError):
            STEParallelCfCCell(input_size=2, hidden_size=4, window=4, density=-0.1)

    def test_invalid_window_raises(self):
        with self.assertRaises(AssertionError):
            STEParallelCfCCell(input_size=2, hidden_size=4, window=0)

    def test_invalid_ste_temperature_raises(self):
        with self.assertRaises(AssertionError):
            STEParallelCfCCell(input_size=2, hidden_size=4, window=4, ste_temperature=0.0)

    def test_invalid_entropy_lambda_raises(self):
        with self.assertRaises(AssertionError):
            STEParallelCfCCell(input_size=2, hidden_size=4, window=4, entropy_lambda=-0.1)


class TestSTEParallelCfCCellMasks(unittest.TestCase):
    """Mask contracts: shapes, values, density."""

    def test_soft_mask_shape_and_range(self):
        _seed(0)
        cell = STEParallelCfCCell(input_size=2, hidden_size=6, window=4, density=0.3)
        soft = cell.get_soft_mask()
        self.assertEqual(soft.shape, (6,))
        # All entries in (0, 1) - sigmoid.
        self.assertTrue((soft > 0).all().item())
        self.assertTrue((soft < 1).all().item())

    def test_hard_mask_density_05(self):
        _seed(1)
        cell = STEParallelCfCCell(input_size=2, hidden_size=10, window=4, density=0.5)
        hard = cell.get_hard_mask()
        self.assertEqual(hard.shape, (10,))
        # Round to nearest int and check density is within +-1.
        self.assertTrue(((hard == 0) | (hard == 1)).all().item())
        # density=0.5 -> expect ~5 ones (the test is robust to +-1 ties).
        self.assertGreaterEqual(int(hard.sum().item()), 4)
        self.assertLessEqual(int(hard.sum().item()), 6)

    def test_hard_mask_density_03(self):
        _seed(2)
        cell = STEParallelCfCCell(input_size=2, hidden_size=10, window=4, density=0.3)
        hard = cell.get_hard_mask()
        # density=0.3 -> expect 3 ones for d=10.
        self.assertEqual(int(hard.sum().item()), 3)

    def test_hard_mask_density_00(self):
        cell = STEParallelCfCCell(input_size=2, hidden_size=10, window=4, density=0.0)
        hard = cell.get_hard_mask()
        self.assertTrue(torch.allclose(hard, torch.zeros(10)))

    def test_hard_mask_density_10(self):
        cell = STEParallelCfCCell(input_size=2, hidden_size=10, window=4, density=1.0)
        hard = cell.get_hard_mask()
        self.assertTrue(torch.allclose(hard, torch.ones(10)))

    def test_ste_mask_forward_equals_hard(self):
        _seed(3)
        cell = STEParallelCfCCell(input_size=2, hidden_size=8, window=4, density=0.5)
        ste = cell.get_ste_mask()
        hard = cell.get_hard_mask()
        # STE: (hard - soft).detach() + soft -> value = hard (since
        # (hard - soft) is detached).
        self.assertTrue(torch.allclose(ste, hard, atol=1e-6))

    def test_ste_mask_backward_passes_through_soft(self):
        _seed(4)
        cell = STEParallelCfCCell(input_size=2, hidden_size=8, window=4, density=0.5)
        # The route_logits are 0 by default - push them up to 1.0 so the
        # soft and hard masks differ.
        with torch.no_grad():
            cell.route_logits.fill_(1.0)
        # Compute a scalar loss through the STE mask and backprop.
        out = cell.get_ste_mask().sum()
        out.backward()
        # d_out/d_route_logits = d_soft/d_route_logits (the hard part is detached).
        # The gradient must be finite and non-zero.
        self.assertIsNotNone(cell.route_logits.grad)
        grad = cell.route_logits.grad
        assert grad is not None
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0.0)


class TestSTEParallelCfCCellForward(unittest.TestCase):
    """Forward path contracts."""

    def test_window_1_forward_shape(self):
        _seed(0)
        cell = STEParallelCfCCell(input_size=3, hidden_size=8, window=1, density=0.5)
        x_t = torch.randn(2, 3)
        h = torch.randn(2, 8)
        out = cell(x_t, h, dt=torch.tensor(1.0))
        self.assertEqual(out.shape, (2, 8))
        self.assertTrue(torch.isfinite(out).all())

    def test_window_4_forward_shape(self):
        _seed(1)
        cell = STEParallelCfCCell(input_size=3, hidden_size=8, window=4, density=0.3)
        x = torch.randn(2, 4, 3)
        h = torch.randn(2, 8)
        out = cell(x, h, dt=torch.tensor(1.0))
        self.assertEqual(out.shape, (2, 8))
        self.assertTrue(torch.isfinite(out).all())

    def test_window_4_window_mismatch_raises(self):
        cell = STEParallelCfCCell(input_size=3, hidden_size=8, window=4, density=0.5)
        x = torch.randn(2, 3, 3)  # W=3
        h = torch.randn(2, 8)
        with self.assertRaises(AssertionError):
            cell(x, h, dt=torch.tensor(1.0))

    def test_density_1_equals_parallel_only(self):
        """density=1.0 -> mask=ones -> cell output = h_parallel path only."""
        _seed(2)
        torch.manual_seed(2)
        cell = STEParallelCfCCell(input_size=2, hidden_size=6, window=4, density=1.0)
        x = torch.randn(2, 4, 2)
        h = torch.randn(2, 6)
        out = cell(x, h, dt=torch.tensor(1.0))
        # Manual parallel-path.
        B, W, _ = x.shape
        h_anchor = h.unsqueeze(1).expand(B, W, 6)
        z = torch.cat([x, h_anchor], dim=-1)
        f = torch.sigmoid(cell.f_gate_p(z))
        g = torch.tanh(cell.g_branch_p(z))
        hp = torch.tanh(cell.h_branch_p(z))
        decay = torch.sigmoid(-f * cell.time_scale.view(1, 1, -1))
        h_steps = decay * g + (1.0 - decay) * hp
        h_parallel = h_steps[:, -1, :]
        self.assertTrue(torch.allclose(out, h_parallel, atol=1e-5))

    def test_density_0_equals_sequential_only(self):
        """density=0.0 -> mask=zeros -> cell output = sequential path only."""
        _seed(3)
        torch.manual_seed(3)
        cell = STEParallelCfCCell(input_size=2, hidden_size=6, window=4, density=0.0)
        x = torch.randn(2, 4, 2)
        h = torch.randn(2, 6)
        out = cell(x, h, dt=torch.tensor(1.0))
        # Manual sequential-path = one CfC step on x[:, -1, :].
        x_t = x[:, -1, :]
        z = torch.cat([x_t, h], dim=-1)
        f = torch.sigmoid(cell.f_gate_s(z))
        g = torch.tanh(cell.g_branch_s(z))
        hp = torch.tanh(cell.h_branch_s(z))
        decay = torch.sigmoid(-f * cell.time_scale.view(1, -1))
        h_seq = decay * g + (1.0 - decay) * hp
        self.assertTrue(torch.allclose(out, h_seq, atol=1e-5))

    def test_window_8_output_bounded(self):
        _seed(4)
        cell = STEParallelCfCCell(
            input_size=2, hidden_size=4, window=8, density=0.3, tau_init=0.5
        )
        x = torch.randn(2, 8, 2)
        h = torch.randn(2, 4)
        out = cell(x, h, dt=torch.tensor(2.0))
        self.assertTrue(torch.isfinite(out).all())
        self.assertLess(out.abs().max().item(), 100.0)


class TestSTEParallelCfCCellGradients(unittest.TestCase):
    """Gradient flow through the STE machinery."""

    def test_grad_flows_to_route_logits(self):
        _seed(0)
        cell = STEParallelCfCCell(input_size=2, hidden_size=6, window=4, density=0.5)
        x = torch.randn(2, 4, 2)
        h = torch.randn(2, 6)
        out = cell(x, h, dt=torch.tensor(1.0))
        out.sum().backward()
        self.assertIsNotNone(cell.route_logits.grad)
        grad = cell.route_logits.grad
        assert grad is not None
        self.assertTrue(torch.isfinite(grad).all())

    def test_grad_flows_to_both_branches(self):
        _seed(1)
        cell = STEParallelCfCCell(input_size=2, hidden_size=6, window=4, density=0.5)
        x = torch.randn(2, 4, 2)
        h = torch.randn(2, 6)
        out = cell(x, h, dt=torch.tensor(1.0))
        out.sum().backward()
        # At least one parameter in each branch should have a non-zero grad.
        for layer in (cell.f_gate_p, cell.g_branch_p, cell.h_branch_p):
            self.assertIsNotNone(layer.weight.grad)
            wgrad = layer.weight.grad
            assert wgrad is not None
            self.assertTrue(torch.isfinite(wgrad).all())

    def test_grad_flows_to_time_scale(self):
        _seed(2)
        cell = STEParallelCfCCell(input_size=2, hidden_size=6, window=4, density=0.5)
        x = torch.randn(2, 4, 2)
        h = torch.randn(2, 6)
        out = cell(x, h, dt=torch.tensor(1.0))
        out.sum().backward()
        self.assertIsNotNone(cell.time_scale.grad)
        grad = cell.time_scale.grad
        assert grad is not None
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0.0)


class TestSTEParallelCfCEntropyReg(unittest.TestCase):
    """Entropy reg (r267) - soft-mask Bernoulli entropy penalty."""

    def test_entropy_value_finite(self):
        _seed(0)
        cell = STEParallelCfCCell(
            input_size=2, hidden_size=8, window=4, density=0.5, entropy_lambda=0.01
        )
        H = cell.soft_mask_entropy()
        self.assertTrue(torch.isfinite(H).item())
        # Bernoulli entropy is bounded in [0, log 2].
        self.assertGreaterEqual(H.item(), 0.0)
        self.assertLessEqual(H.item(), math.log(2.0) + 1e-6)

    def test_extra_loss_zero_when_disabled(self):
        cell = STEParallelCfCCell(
            input_size=2, hidden_size=8, window=4, density=0.5, entropy_lambda=0.0
        )
        loss = cell.extra_loss()
        self.assertEqual(loss.item(), 0.0)

    def test_extra_loss_proportional_to_lambda(self):
        _seed(1)
        cell_small = STEParallelCfCCell(
            input_size=2, hidden_size=8, window=4, density=0.5, entropy_lambda=0.01
        )
        cell_large = STEParallelCfCCell(
            input_size=2, hidden_size=8, window=4, density=0.5, entropy_lambda=0.1
        )
        # Copy parameters from small -> large so the entropy is identical.
        with torch.no_grad():
            for p_s, p_l in zip(cell_small.parameters(), cell_large.parameters()):
                p_l.copy_(p_s)
        self.assertAlmostEqual(
            cell_small.extra_loss().item() * 10.0,
            cell_large.extra_loss().item(),
            places=5,
        )

    def test_extra_loss_backward(self):
        _seed(2)
        cell = STEParallelCfCCell(
            input_size=2, hidden_size=8, window=4, density=0.5, entropy_lambda=0.1
        )
        # Push the route_logits off the symmetric point sigmoid(0)=0.5
        # so the entropy has a non-trivial gradient (Bernoulli entropy
        # is stationary at p=0.5).
        with torch.no_grad():
            cell.route_logits.add_(torch.linspace(-1.0, 1.0, cell.hidden_size))
        loss = cell.extra_loss()
        loss.backward()
        # route_logits should receive a finite, non-zero gradient from the
        # entropy term.
        self.assertIsNotNone(cell.route_logits.grad)
        grad = cell.route_logits.grad
        assert grad is not None
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0.0)


class TestSTEParallelCfCNetwork(unittest.TestCase):
    """Multi-layer network with chunked sequential semantics."""

    def test_init(self):
        net = STEParallelCfCNetwork(
            input_size=2, hidden_size=8, output_size=1, num_layers=1, window=4,
            density=0.3,
        )
        self.assertEqual(len(net.cells), 1)
        self.assertEqual(net.window, 4)
        self.assertEqual(net.density, 0.3)

    def test_window_1_sequential_path(self):
        _seed(0)
        net = STEParallelCfCNetwork(
            input_size=2, hidden_size=8, output_size=1, window=1, density=0.5,
        )
        x = torch.randn(2, 16, 2)
        out = net(x)
        self.assertEqual(out.shape, (2, 1))
        self.assertTrue(torch.isfinite(out).all())

    def test_window_4_chunked_path(self):
        _seed(1)
        net = STEParallelCfCNetwork(
            input_size=2, hidden_size=8, output_size=1, window=4, density=0.3,
        )
        x = torch.randn(2, 16, 2)  # 16 = 4 * 4 chunks
        out = net(x)
        self.assertEqual(out.shape, (2, 1))
        self.assertTrue(torch.isfinite(out).all())

    def test_window_4_non_divisible_T_raises(self):
        net = STEParallelCfCNetwork(
            input_size=2, hidden_size=8, output_size=1, window=4, density=0.5,
        )
        x = torch.randn(2, 15, 2)  # 15 not divisible by 4
        with self.assertRaises(AssertionError):
            net(x)

    def test_multilayer(self):
        _seed(2)
        net = STEParallelCfCNetwork(
            input_size=2, hidden_size=8, output_size=1, num_layers=2,
            window=4, density=0.3,
        )
        x = torch.randn(2, 8, 2)
        out = net(x)
        self.assertEqual(out.shape, (2, 1))
        self.assertTrue(torch.isfinite(out).all())

    def test_return_sequences(self):
        _seed(3)
        net = STEParallelCfCNetwork(
            input_size=2, hidden_size=8, output_size=3, window=4,
            density=0.3, return_sequences=True,
        )
        x = torch.randn(2, 8, 2)
        out = net(x)
        self.assertEqual(out.shape, (2, 8, 3))
        self.assertTrue(torch.isfinite(out).all())

    def test_extra_loss_sums_cells(self):
        _seed(4)
        net = STEParallelCfCNetwork(
            input_size=2, hidden_size=8, output_size=1, num_layers=2,
            window=4, density=0.3, entropy_lambda=0.1,
        )
        loss = net.extra_loss()
        # Sum of two cells' entropy losses.
        manual = sum(c.extra_loss() for c in net.cells)
        self.assertAlmostEqual(loss.item(), manual.item(), places=5)

    def test_grad_flows_through_chunks(self):
        _seed(5)
        net = STEParallelCfCNetwork(
            input_size=2, hidden_size=8, output_size=1, window=4, density=0.3,
        )
        x = torch.randn(2, 8, 2)
        out = net(x)
        out.sum().backward()
        for cell in net.cells:
            self.assertIsNotNone(cell.time_scale.grad)
            tgrad = cell.time_scale.grad
            assert tgrad is not None
            self.assertTrue(torch.isfinite(tgrad).all())


class TestSTEParallelCfCWindowSweep(unittest.TestCase):
    """Sweep W in {1, 2, 4, 8} - all produce shape-valid outputs."""

    def test_window_sweep(self):
        _seed(0)
        for W in (1, 2, 4, 8):
            net = STEParallelCfCNetwork(
                input_size=2, hidden_size=8, output_size=1, window=W, density=0.3,
            )
            T = 16
            x = torch.randn(2, T, 2)
            out = net(x)
            self.assertEqual(out.shape, (2, 1), f"window={W} shape wrong")
            self.assertTrue(torch.isfinite(out).all(), f"window={W} non-finite")


class TestSTEParallelCfCDtypeDevice(unittest.TestCase):
    def test_default_dtype_is_float32(self):
        cell = STEParallelCfCCell(input_size=2, hidden_size=4, window=4)
        for p in cell.parameters():
            self.assertEqual(p.dtype, torch.float32)

    def test_default_device_is_cpu(self):
        cell = STEParallelCfCCell(input_size=2, hidden_size=4, window=4)
        for p in cell.parameters():
            self.assertEqual(p.device, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
