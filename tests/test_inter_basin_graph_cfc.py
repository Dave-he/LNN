"""Tests for InterBasinGraphCfCCell (round 258)."""

from __future__ import annotations

import unittest

import torch

from lnn.core.inter_basin_graph_cfc import (
    InterBasinGraphCfCCell,
    basin_assignment_prob,
    inter_basin_graph_mix,
    inter_basin_graph_regularizer,
)


class TestBasinAssignmentProb(unittest.TestCase):
    def test_close_basin_high_prob(self):
        # h[0] exactly at c[1], but c[0]/c[2] far away.
        h = torch.zeros(2, 4)
        c = torch.zeros(3, 4)
        c[1] = h[0]                       # c[1] = origin, c[0],c[2] far
        c[0] = torch.tensor([10., 10., 10., 10.])
        c[2] = torch.tensor([-10., -10., -10., -10.])
        p = basin_assignment_prob(h, c, beta_v=10.0)
        self.assertGreater(p[0, 1].item(), 0.99)
        # h[1] at c[0] → p[0] dominates.
        h[1] = c[0].clone()
        p_close = basin_assignment_prob(h[1:2], c, beta_v=10.0)
        self.assertGreater(p_close[0, 0].item(), 0.99)

    def test_probabilities_sum_to_one(self):
        h = torch.randn(8, 4)
        c = torch.randn(3, 4)
        p = basin_assignment_prob(h, c)
        sums = p.sum(dim=-1)
        for s in sums:
            self.assertAlmostEqual(s.item(), 1.0, places=5)


class TestGraphMix(unittest.TestCase):
    def test_identity_matrix_preserves(self):
        h = torch.randn(4, 5)
        c = torch.randn(3, 5)
        p = basin_assignment_prob(h, c)
        A = torch.eye(3)
        q = inter_basin_graph_mix(p, A)
        diff = (q - p).abs().max().item()
        self.assertLess(diff, 1e-5)

    def test_uniform_graph_uniformizes(self):
        h = torch.randn(4, 3)
        c = torch.randn(3, 3)
        p = basin_assignment_prob(h, c, beta_v=10.0)
        # If p[0, 1] is high (very close), uniform A should make q uniform.
        A = torch.ones(3, 3) / 3.0
        q = inter_basin_graph_mix(p, A)
        # After uniform mix + renorm, every q[i] should be ~1/3.
        for i in range(q.shape[0]):
            for j in range(3):
                self.assertAlmostEqual(q[i, j].item(), 1.0 / 3, places=2)

    def test_row_stochastic_input_preserves_sum(self):
        h = torch.randn(4, 5)
        c = torch.randn(3, 5)
        p = basin_assignment_prob(h, c)
        A = torch.softmax(torch.randn(3, 3), dim=-1)
        q = inter_basin_graph_mix(p, A)
        sums = q.sum(dim=-1)
        for s in sums:
            self.assertAlmostEqual(s.item(), 1.0, places=5)


class TestGraphRegularizer(unittest.TestCase):
    def test_identity_zero_symmetry(self):
        A = torch.eye(3)
        r = inter_basin_graph_regularizer(A)
        self.assertAlmostEqual(r["symmetry_break"].item(), 0.0, places=6)

    def test_asymmetric_nonzero_symmetry(self):
        A = torch.zeros(3, 3)
        A[0, 1] = 1.0  # asymmetric
        A[0, 0] = 1.0
        r = inter_basin_graph_regularizer(A)
        self.assertGreater(r["symmetry_break"].item(), 0.0)

    def test_sparsity_zero_for_zero_matrix(self):
        A = torch.zeros(3, 3)
        r = inter_basin_graph_regularizer(A)
        self.assertAlmostEqual(r["sparsity"].item(), 0.0, places=6)

    def test_sparsity_positive_for_nonzero(self):
        A = torch.ones(3, 3)
        r = inter_basin_graph_regularizer(A)
        self.assertAlmostEqual(r["sparsity"].item(), 9.0, places=6)


class TestInterBasinGraphCell(unittest.TestCase):
    def test_inherits_inter_basin(self):
        cell = InterBasinGraphCfCCell(3, 8, n_branches=4, n_basin=3)
        param_names = [n for n, _ in cell.named_parameters()]
        self.assertIn("basin_centers", param_names)
        self.assertIn("adjacency", param_names)

    def test_adjacency_stochastic_shape(self):
        cell = InterBasinGraphCfCCell(3, 8, n_branches=4, n_basin=3)
        A = cell.adjacency_stochastic()
        self.assertEqual(A.shape, (4, 3, 3))
        # Row-stochastic: rows sum to 1.
        sums = A.sum(dim=-1)
        for s in sums.flatten():
            self.assertAlmostEqual(s.item(), 1.0, places=5)

    def test_adjacency_init_near_identity(self):
        # Initial adjacency should be softmax of (I + 0.1 * randn) → ~identity.
        cell = InterBasinGraphCfCCell(3, 8, n_branches=2, n_basin=3)
        A = cell.adjacency_stochastic()
        # Diagonal entries should dominate over off-diagonal at init.
        for k in range(2):
            for i in range(3):
                diag = A[k, i, i].item()
                for j in range(3):
                    if i != j:
                        self.assertGreater(diag, A[k, i, j].item())

    def test_graph_mix_in_aux(self):
        cell = InterBasinGraphCfCCell(3, 8)
        h_list = cell.init_state(4)
        x_t = torch.randn(4, 3)
        _, _, aux = cell.forward_with_aux(x_t, h_list)
        self.assertIn("adjacency_stochastic", aux)
        self.assertIn("graph_symmetry", aux)
        self.assertIn("graph_sparsity", aux)
        self.assertIn("p_per_branch", aux)
        self.assertIn("q_per_branch", aux)
        self.assertIn("per_branch_basin_H", aux)
        self.assertIn("per_branch_basin_H_raw", aux)

    def test_grad_flows_to_adjacency(self):
        cell = InterBasinGraphCfCCell(
            3, 8, n_branches=4, n_basin=3,
            sym_lambda=1.0, sparse_lambda=0.1,
        )
        h_list = cell.init_state(4)
        x_t = torch.randn(4, 3)
        _, _, aux = cell.forward_with_aux(
            x_t, h_list, graph_lambda=1.0,
        )
        aux["graph_loss_total"].backward()
        self.assertIsNotNone(cell.adjacency.grad)
        self.assertGreater(cell.adjacency.grad.abs().sum().item(), 0.0)
        # The aux also includes the stochastic adjacency for inspection.
        A = aux["adjacency_stochastic"]
        self.assertEqual(A.shape, (4, 3, 3))

    def test_q_renormalized(self):
        # Even with arbitrary A, q should always sum to 1 per row.
        cell = InterBasinGraphCfCCell(
            3, 8, n_branches=2, n_basin=3,
        )
        # Set adversarial A (non-uniform).
        with torch.no_grad():
            cell.adjacency.zero_()
            cell.adjacency[0] = torch.tensor([
                [10.0, -10.0, 0.0],
                [0.0, 10.0, -10.0],
                [-10.0, 0.0, 10.0],
            ])
        A = cell.adjacency_stochastic()
        h_list = cell.init_state(4)
        x_t = torch.randn(4, 3)
        _, _, aux = cell.forward_with_aux(x_t, h_list)
        # q_per_branch is averaged over batch — sum should be 1 per branch.
        for q in aux["q_per_branch"]:
            self.assertAlmostEqual(q.sum().item(), 1.0, places=5)

    def test_graph_lambda_off_no_total(self):
        cell = InterBasinGraphCfCCell(
            3, 8, sym_lambda=1.0, sparse_lambda=1.0,
        )
        h_list = cell.init_state(4)
        x_t = torch.randn(4, 3)
        _, _, aux = cell.forward_with_aux(
            x_t, h_list, graph_lambda=0.0,
        )
        self.assertNotIn("graph_loss_applied", aux)

    def test_graph_lambda_on_includes_total(self):
        cell = InterBasinGraphCfCCell(
            3, 8, sym_lambda=1.0, sparse_lambda=1.0,
        )
        h_list = cell.init_state(4)
        x_t = torch.randn(4, 3)
        _, _, aux = cell.forward_with_aux(
            x_t, h_list, graph_lambda=1.0,
        )
        self.assertIn("graph_loss_applied", aux)
        self.assertGreater(aux["graph_loss_applied"].item(), 0.0)

    def test_forward_shape(self):
        d_in, d_h, T, B, K = 3, 8, 12, 4, 4
        cell = InterBasinGraphCfCCell(
            d_in, d_h, n_branches=K, n_basin=3,
        )
        x = torch.randn(T, B, d_in)
        h_list = cell.init_state(B)
        outputs = []
        for t in range(T):
            h, h_list = cell(x[t], h_list)
            outputs.append(h)
        self.assertEqual(outputs[-1].shape, (B, d_h))


if __name__ == "__main__":
    unittest.main()