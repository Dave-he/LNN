"""Tests for MixStaticInterBasinGraphCfCCell (round 261)."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from lnn.core.mix_static_inter_basin_graph_cfc import (
    MixStaticInterBasinGraphCfCCell,
    mix_static_and_input,
)


class TestMixStaticAndInput(unittest.TestCase):

    def test_alpha_zero_returns_static(self):
        A_static = torch.tensor([[0.7, 0.2, 0.1]])
        A_input = torch.tensor([[0.1, 0.8, 0.1]])
        out = mix_static_and_input(A_static, A_input, torch.tensor(0.0))
        # alpha=0 → pure static
        self.assertTrue(torch.allclose(out, A_static, atol=1e-6))

    def test_alpha_one_returns_input(self):
        A_static = torch.tensor([[0.7, 0.2, 0.1]])
        A_input = torch.tensor([[0.1, 0.8, 0.1]])
        out = mix_static_and_input(A_static, A_input, torch.tensor(1.0))
        self.assertTrue(torch.allclose(out, A_input, atol=1e-6))

    def test_alpha_half_is_convex(self):
        A_static = torch.tensor([[0.7, 0.2, 0.1]])
        A_input = torch.tensor([[0.1, 0.8, 0.1]])
        out = mix_static_and_input(A_static, A_input, torch.tensor(0.5))
        expected = 0.5 * A_static + 0.5 * A_input
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_batch_broadcast(self):
        """Static (K, K) should broadcast to (B, K, K) when added with input."""
        A_static = torch.tensor([[0.5, 0.3, 0.2], [0.6, 0.2, 0.2], [0.1, 0.1, 0.8]])
        # B=4, K=3, K=3 input
        A_input = torch.full((4, 3, 3), 1.0 / 3)
        A_static_b = A_static.unsqueeze(0).expand(4, 3, 3)
        out = mix_static_and_input(A_static_b, A_input, torch.tensor(0.5))
        self.assertEqual(out.shape, (4, 3, 3))
        # Each row should sum to 1.
        self.assertTrue(torch.allclose(
            out.sum(dim=-1), torch.ones(4, 3), atol=1e-5,
        ))


class TestMixStaticCell(unittest.TestCase):

    def test_inherits_per_step(self):
        from lnn.core.per_step_inter_basin_graph_cfc import (
            PerStepInterBasinGraphCfCCell,
        )
        cell = MixStaticInterBasinGraphCfCCell(
            input_size=2, hidden_size=4,
            n_branches=2, n_basin=2,
        )
        self.assertIsInstance(cell, PerStepInterBasinGraphCfCCell)

    def test_has_alpha_logit(self):
        cell = MixStaticInterBasinGraphCfCCell(
            input_size=2, hidden_size=4,
            n_branches=3, n_basin=2,
        )
        self.assertTrue(hasattr(cell, "alpha_logit"))
        self.assertEqual(cell.alpha_logit.shape, (3,))

    def test_init_alpha(self):
        import math
        cell = MixStaticInterBasinGraphCfCCell(
            input_size=2, hidden_size=4,
            n_branches=2, n_basin=2, init_alpha=0.3,
        )
        alpha = cell.per_branch_alpha()
        # Should be ~0.3 at init.
        self.assertTrue(torch.allclose(alpha, torch.full((2,), 0.3), atol=1e-3))

    def test_per_step_adjacency_shape_and_stochastic(self):
        cell = MixStaticInterBasinGraphCfCCell(
            input_size=2, hidden_size=4,
            n_branches=2, n_basin=3, init_alpha=0.5,
        )
        x = torch.randn(4, 2)
        A = cell.per_step_adjacency(x, k=0)
        self.assertEqual(A.shape, (4, 3, 3))
        # Row stochastic.
        sums = A.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    def test_alpha_extreme_zero_dominates_static(self):
        """If alpha_logit is very negative, A should ≈ static A."""
        cell = MixStaticInterBasinGraphCfCCell(
            input_size=2, hidden_size=4,
            n_branches=2, n_basin=2, init_alpha=0.001,
        )
        x = torch.randn(4, 2)
        A = cell.per_step_adjacency(x, k=0)
        # alpha ≈ 0.001 → mostly static.
        # Static is initialized near identity + 0.1 noise.
        # Each row should be close to the static row.
        A_static_k = cell.adjacency_stochastic()[0]  # (K, K)
        A_static_b = A_static_k.unsqueeze(0).expand(4, 2, 2)
        diff = (A - A_static_b).abs().max().item()
        self.assertLess(diff, 0.05)

    def test_alpha_extreme_one_dominates_input(self):
        """If alpha_logit is very positive, A should ≈ input MLP output."""
        cell = MixStaticInterBasinGraphCfCCell(
            input_size=2, hidden_size=4,
            n_branches=2, n_basin=2, init_alpha=0.999,
        )
        x = torch.randn(4, 2)
        A = cell.per_step_adjacency(x, k=0)
        # alpha ≈ 1 → mostly input-dependent.
        # We can't compare directly to MLP output (softmax), but the
        # diff from static should be much larger than in test_alpha_extreme_zero.
        A_static_k = cell.adjacency_stochastic()[0]
        A_static_b = A_static_k.unsqueeze(0).expand(4, 2, 2)
        diff = (A - A_static_b).abs().max().item()
        self.assertGreater(diff, 0.01)

    def test_forward_with_aux_exposes_alpha(self):
        cell = MixStaticInterBasinGraphCfCCell(
            input_size=2, hidden_size=4,
            n_branches=2, n_basin=3, init_alpha=0.4,
        )
        x = torch.randn(4, 2)
        h_list = [torch.zeros(4, 4) for _ in range(2)]
        _, _, aux = cell.forward_with_aux(
            x, h_list, lyap_lambda=0.0, sep_lambda=0.0,
            dist_lambda=1.0, graph_lambda=0.0,
        )
        self.assertIn("alpha_per_branch", aux)
        self.assertIn("alpha_mean", aux)
        self.assertEqual(aux["alpha_per_branch"].shape, (2,))
        # Mean should be near init_alpha=0.4 at start.
        self.assertAlmostEqual(aux["alpha_mean"].item(), 0.4, places=2)
        # Both adjacency types exposed.
        self.assertIn("adjacency_per_step", aux)
        self.assertIn("adjacency_input_only", aux)
        self.assertEqual(aux["adjacency_per_step"].shape, (2, 4, 3, 3))
        self.assertEqual(aux["adjacency_input_only"].shape, (2, 4, 3, 3))

    def test_grad_flows_to_alpha_logit(self):
        cell = MixStaticInterBasinGraphCfCCell(
            input_size=2, hidden_size=4,
            n_branches=2, n_basin=2, init_alpha=0.5,
        )
        x = torch.randn(4, 2)
        h_list = [torch.zeros(4, 4) for _ in range(2)]
        _, _, aux = cell.forward_with_aux(
            x, h_list, lyap_lambda=0.0, sep_lambda=0.0,
            dist_lambda=1.0, graph_lambda=0.0,
        )
        loss = aux["alpha_mean"] + aux["mean_basin_H"]
        loss.backward()
        self.assertIsNotNone(cell.alpha_logit.grad)
        self.assertGreater(cell.alpha_logit.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()