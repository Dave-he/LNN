"""Tests for MultiHopInterBasinGraphCfCCell (round 259)."""

from __future__ import annotations

import unittest

import torch

from lnn.core.inter_basin_graph_cfc import (
    InterBasinGraphCfCCell,
    basin_assignment_prob,
    inter_basin_graph_mix,
)
from lnn.core.multi_hop_inter_basin_graph_cfc import (
    MultiHopInterBasinGraphCfCCell,
)


class TestMultiHopBehavior(unittest.TestCase):
    def test_n_hops_1_equals_single_mix(self):
        h = torch.randn(4, 5)
        c = torch.randn(3, 5)
        p = basin_assignment_prob(h, c)
        A = torch.softmax(torch.randn(3, 3), dim=-1)
        # Single mix.
        q1 = inter_basin_graph_mix(p, A)
        # 1-hop from multi_hop_mix.
        cell = MultiHopInterBasinGraphCfCCell(
            3, 8, n_branches=2, n_basin=3, n_hops=1,
        )
        with torch.no_grad():
            cell.adjacency.zero_()
            cell.adjacency[0] = A.log()  # invertible via log
        q_mh = cell.multi_hop_mix(p, A)
        # Should be very close (within float precision).
        diff = (q1 - q_mh).abs().max().item()
        self.assertLess(diff, 1e-4)

    def test_n_hops_3_converges_toward_uniform(self):
        """A K-hop with weakly-mixing A should drive q toward uniform."""
        h = torch.randn(4, 3)
        c = torch.randn(3, 3)
        p = basin_assignment_prob(h, c, beta_v=10.0)
        # A uniform-ish adjacency (low temperature softmax).
        A_logits = torch.zeros(3, 3) + 0.01 * torch.randn(3, 3)
        A = torch.softmax(A_logits, dim=-1)
        # After many hops q → stationary distribution of A (uniform since
        # A is uniform up to small noise).
        cell = MultiHopInterBasinGraphCfCCell(
            3, 8, n_branches=2, n_basin=3, n_hops=20,
        )
        q_mh = cell.multi_hop_mix(p, A)
        for i in range(q_mh.shape[0]):
            for j in range(3):
                # q_mh[i, j] should be ~1/3 (uniform stationary).
                self.assertAlmostEqual(q_mh[i, j].item(), 1.0 / 3, places=1)

    def test_q_renormalized_after_each_hop(self):
        h = torch.randn(4, 5)
        c = torch.randn(3, 5)
        p = basin_assignment_prob(h, c)
        # Adversarial A.
        A = torch.softmax(torch.tensor([
            [10.0, -10.0, 0.0],
            [0.0, 10.0, -10.0],
            [-10.0, 0.0, 10.0],
        ]), dim=-1)
        cell = MultiHopInterBasinGraphCfCCell(
            3, 8, n_branches=2, n_basin=3, n_hops=5,
        )
        q = cell.multi_hop_mix(p, A)
        sums = q.sum(dim=-1)
        for s in sums:
            self.assertAlmostEqual(s.item(), 1.0, places=5)


class TestMultiHopCell(unittest.TestCase):
    def test_inherits_inter_basin_graph(self):
        cell = MultiHopInterBasinGraphCfCCell(3, 8, n_hops=3)
        self.assertIsInstance(cell, InterBasinGraphCfCCell)
        self.assertEqual(cell.n_hops, 3)
        param_names = [n for n, _ in cell.named_parameters()]
        self.assertIn("basin_centers", param_names)
        self.assertIn("adjacency", param_names)

    def test_n_hops_1_falls_back_to_r258(self):
        # Aux with n_hops=1 should match r258's aux (within numerical tol).
        torch.manual_seed(0)
        cell259 = MultiHopInterBasinGraphCfCCell(
            3, 8, n_branches=4, n_basin=3, n_hops=1, seed=42,
        )
        torch.manual_seed(0)
        cell258 = InterBasinGraphCfCCell(
            3, 8, n_branches=4, n_basin=3, seed=42,
        )
        h_list_259 = cell259.init_state(4)
        h_list_258 = cell258.init_state(4)
        x_t = torch.randn(4, 3)
        _, _, aux_259 = cell259.forward_with_aux(x_t, h_list_259)
        _, _, aux_258 = cell258.forward_with_aux(x_t, h_list_258)
        # q_per_branch (final hop = same as r258's q) should be identical.
        self.assertTrue(
            torch.allclose(aux_259["q_per_branch"],
                           aux_258["q_per_branch"], atol=1e-5),
        )

    def test_aux_keys(self):
        cell = MultiHopInterBasinGraphCfCCell(3, 8, n_hops=3)
        h_list = cell.init_state(4)
        x_t = torch.randn(4, 3)
        _, _, aux = cell.forward_with_aux(x_t, h_list)
        self.assertIn("n_hops", aux)
        self.assertEqual(aux["n_hops"].item() if hasattr(aux["n_hops"], "item")
                         else aux["n_hops"], 3)
        self.assertIn("q_per_branch", aux)
        self.assertIn("p_per_branch", aux)
        self.assertIn("adjacency_stochastic", aux)

    def test_grad_flows_to_adjacency(self):
        cell = MultiHopInterBasinGraphCfCCell(
            3, 8, n_branches=4, n_basin=3, n_hops=2,
            sym_lambda=1.0, sparse_lambda=0.0,
        )
        h_list = cell.init_state(4)
        x_t = torch.randn(4, 3)
        _, _, aux = cell.forward_with_aux(
            x_t, h_list, graph_lambda=1.0,
        )
        aux["graph_loss_total"].backward()
        self.assertIsNotNone(cell.adjacency.grad)
        self.assertGreater(cell.adjacency.grad.abs().sum().item(), 0.0)

    def test_forward_shape(self):
        d_in, d_h, T, B, K = 3, 8, 12, 4, 4
        cell = MultiHopInterBasinGraphCfCCell(
            d_in, d_h, n_branches=K, n_basin=3, n_hops=3,
        )
        x = torch.randn(T, B, d_in)
        h_list = cell.init_state(B)
        outputs = []
        for t in range(T):
            h, h_list = cell(x[t], h_list)
            outputs.append(h)
        self.assertEqual(outputs[-1].shape, (B, d_h))

    def test_hop3_H_greater_or_equal_hop1(self):
        """More hops should NEVER decrease H (graph mix is a Markov
        diffusion; entropy is non-decreasing for stochastic A)."""
        torch.manual_seed(0)
        cell1 = MultiHopInterBasinGraphCfCCell(
            3, 8, n_branches=2, n_basin=3, n_hops=1, seed=42,
        )
        torch.manual_seed(0)
        cell3 = MultiHopInterBasinGraphCfCCell(
            3, 8, n_branches=2, n_basin=3, n_hops=3, seed=42,
        )
        # Force same initial state.
        with torch.no_grad():
            for p1, p3 in zip(cell1.parameters(), cell3.parameters()):
                p3.data.copy_(p1.data)
        h_list_1 = cell1.init_state(8)
        h_list_3 = cell3.init_state(8)
        x_t = torch.randn(8, 3)
        _, _, aux1 = cell1.forward_with_aux(x_t, h_list_1)
        _, _, aux3 = cell3.forward_with_aux(x_t, h_list_3)
        H1 = aux1["mean_basin_H"].item()
        H3 = aux3["mean_basin_H"].item()
        # Hop3 should have >= H1 (more mixing → more spread).
        self.assertGreaterEqual(H3, H1 - 1e-4)

    def test_off_diag_sparsity_helper(self):
        """Verify the off-diagonal sparsity concept (used in the fix
        idea from r258's memory)."""
        # A is row-stochastic; ||A||_1 = K = 3 always.
        # ||A_off_diag||_1 = K - trace(A) → can vary.
        A = torch.softmax(torch.randn(3, 3), dim=-1)
        off_diag_sum = A.sum() - A.diagonal().sum()
        self.assertGreater(off_diag_sum.item(), 0.0)


if __name__ == "__main__":
    unittest.main()