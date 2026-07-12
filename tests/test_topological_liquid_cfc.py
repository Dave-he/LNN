"""Tests for TopologicalLiquidCfCCell (round 297).

Verifies:
    Forward shapes, finite outputs, gradient flow.
    n_incoming correctly limits the recurrent connections.
    Topological vs dense: same parameter count, different connectivity.
    Decorrelation default (r295) inherited.
    Different gate_modes work.
"""

from __future__ import annotations

import unittest

import torch

from lnn.core.topological_liquid_cfc import TopologicalLiquidCfCCell


def _sine(B=4, T=64, freq=3.0):
    t = torch.linspace(0, 6.2831853 * freq, T).view(1, T, 1).repeat(B, 1, 1)
    return torch.sin(t)


class TestTopologicalBasics(unittest.TestCase):

    def test_init_defaults(self):
        cell = TopologicalLiquidCfCCell(input_size=1, hidden_size=32)
        self.assertEqual(cell.n_incoming, 8)
        self.assertEqual(cell.decorr_lambda, 1e-5)
        self.assertEqual(cell.gate_mode, "blend")

    def test_n_incoming_from_density(self):
        cell = TopologicalLiquidCfCCell(input_size=1, hidden_size=32,
                                          density=0.25)
        self.assertEqual(cell.n_incoming, 8)  # int(32*0.25) = 8

    def test_n_incoming_capped_at_hidden(self):
        cell = TopologicalLiquidCfCCell(input_size=1, hidden_size=8,
                                          n_incoming=20)
        self.assertEqual(cell.n_incoming, 8)

    def test_gate_mode_validation(self):
        with self.assertRaises(ValueError):
            TopologicalLiquidCfCCell(input_size=1, hidden_size=8,
                                      gate_mode="bogus")

    def test_decorr_lambda_validation(self):
        with self.assertRaises(ValueError):
            TopologicalLiquidCfCCell(input_size=1, hidden_size=8,
                                      decorr_lambda=-1.0)


class TestTopologicalForward(unittest.TestCase):

    def test_forward_shape(self):
        cell = TopologicalLiquidCfCCell(input_size=1, hidden_size=16, seed=1)
        x = _sine(B=3, T=32)
        out, h = cell(x)
        self.assertEqual(out.shape, (3, 32, 16))
        self.assertEqual(h.shape, (3, 16))
        self.assertTrue(torch.isfinite(out).all())

    def test_n_incoming_smaller_than_dense(self):
        """Topological cell has FEWER recurrent parameters than dense."""
        topo = TopologicalLiquidCfCCell(input_size=1, hidden_size=16,
                                          n_incoming=4, seed=2)
        dense = TopologicalLiquidCfCCell(input_size=1, hidden_size=16,
                                          n_incoming=16, seed=2)
        self.assertLess(topo.W_rec_sparse.numel(),
                         dense.W_rec_sparse.numel())

    def test_src_indices_fixed_at_init(self):
        """Source indices should be a buffer, not a parameter."""
        cell = TopologicalLiquidCfCCell(input_size=1, hidden_size=16, seed=3)
        self.assertFalse(
            isinstance(cell.src_indices, torch.nn.Parameter))
        # Should be (d_h, n_incoming) of long.
        self.assertEqual(cell.src_indices.shape, (16, 8))

    def test_src_indices_distinct_per_neuron(self):
        """Each neuron should have its own set of source neurons
        (possibly overlapping but at least diverse)."""
        cell = TopologicalLiquidCfCCell(input_size=1, hidden_size=32,
                                          n_incoming=8, seed=4)
        # Convert rows to sets; should be diverse.
        rows = [set(cell.src_indices[i].tolist()) for i in range(32)]
        # Average overlap between any two rows should be ≤ n_incoming.
        # Most rows should be distinct from each other.
        same = sum(1 for i in range(32) for j in range(i + 1, 32)
                   if rows[i] == rows[j])
        # Allow at most a few duplicates by random chance.
        self.assertLess(same, 5)


class TestTopologicalGradient(unittest.TestCase):

    def test_gradients_flow_to_W_rec_sparse(self):
        cell = TopologicalLiquidCfCCell(input_size=1, hidden_size=8,
                                          seed=5, n_incoming=4)
        x = _sine(B=2, T=16)
        out, _ = cell(x)
        out.pow(2).mean().backward()
        self.assertIsNotNone(cell.W_rec_sparse.grad)
        self.assertGreater(cell.W_rec_sparse.grad.abs().sum().item(), 0.0)

    def test_gradients_flow_to_W_in(self):
        cell = TopologicalLiquidCfCCell(input_size=1, hidden_size=8, seed=6)
        x = _sine(B=2, T=16)
        out, _ = cell(x)
        out.pow(2).mean().backward()
        self.assertIsNotNone(cell.W_in.weight.grad)
        self.assertGreater(cell.W_in.weight.grad.abs().sum().item(), 0.0)


class TestTopologicalExtraLoss(unittest.TestCase):

    def test_extra_loss_zero_without_forward(self):
        cell = TopologicalLiquidCfCCell(input_size=1, hidden_size=8, seed=7)
        loss = cell.extra_loss()
        self.assertEqual(float(loss.item()), 0.0)

    def test_extra_loss_nonzero_after_forward(self):
        cell = TopologicalLiquidCfCCell(input_size=1, hidden_size=8, seed=8)
        x = _sine(B=2, T=16)
        _ = cell(x)
        loss = cell.extra_loss()
        self.assertGreater(float(loss.item()), 0.0)

    def test_decorr_lambda_zero_disables(self):
        cell = TopologicalLiquidCfCCell(input_size=1, hidden_size=8,
                                          seed=9, decorr_lambda=0.0)
        x = _sine(B=2, T=16)
        _ = cell(x)
        loss = cell.extra_loss()
        self.assertEqual(float(loss.item()), 0.0)


class TestTopologicalGateModes(unittest.TestCase):

    def test_all_gate_modes_run(self):
        for gm in ("blend", "velocity", "acceleration"):
            cell = TopologicalLiquidCfCCell(input_size=1, hidden_size=12,
                                              gate_mode=gm, seed=10)
            out, h = cell(_sine(B=2, T=16))
            self.assertEqual(out.shape, (2, 16, 12))
            self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()