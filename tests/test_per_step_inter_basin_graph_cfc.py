"""Tests for PerStepInterBasinGraphCfCCell (round 260).

Verifies:
  * input_dependent_adjacency returns row-stochastic tensors
  * PerStepInterBasinGraphCfCCell inherits r258 correctly
  * A_t changes with x_t (not constant)
  * grad flows through a_mlp
  * forward_with_aux exposes adjacency_per_step (B, n_branches, K, K)
  * A_diversity tracks deviation from static A
"""

from __future__ import annotations

import unittest

import torch

from lnn.core.per_step_inter_basin_graph_cfc import (
    PerStepInterBasinGraphCfCCell,
    input_dependent_adjacency,
)


def _zero_mlp(d_in: int, k_sq: int):
    """MLP that returns zero logits — adjacency ≈ uniform."""
    import torch.nn as nn

    return nn.Linear(d_in, k_sq)


class TestInputDependentAdjacency(unittest.TestCase):

    def test_shape(self):
        import torch.nn as nn

        mlp = nn.Linear(2, 9)  # 3*3
        x = torch.randn(4, 2)
        A = input_dependent_adjacency(x, mlp, n_basin=3)
        self.assertEqual(A.shape, (4, 3, 3))

    def test_row_stochastic(self):
        import torch.nn as nn

        mlp = nn.Linear(2, 9)
        x = torch.randn(8, 2)
        A = input_dependent_adjacency(x, mlp, n_basin=3)
        sums = A.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    def test_changes_with_x(self):
        import torch.nn as nn

        # Use a NON-ZERO weight so different inputs → different A.
        mlp = nn.Linear(2, 9)
        with torch.no_grad():
            mlp.weight.copy_(torch.randn_like(mlp.weight) * 0.5)
            mlp.bias.zero_()
        x1 = torch.tensor([[1.0, 0.0]])
        x2 = torch.tensor([[0.0, 1.0]])
        A1 = input_dependent_adjacency(x1, mlp, n_basin=3)
        A2 = input_dependent_adjacency(x2, mlp, n_basin=3)
        diff = (A1 - A2).abs().sum().item()
        self.assertGreater(diff, 0.1)

    def test_uniform_when_mlp_zero(self):
        """With zero-weight MLP and zero bias, A should be uniform."""
        mlp = _zero_mlp(2, 9)
        with torch.no_grad():
            mlp.weight.zero_()
            mlp.bias.zero_()
        x = torch.randn(4, 2)
        A = input_dependent_adjacency(x, mlp, n_basin=3)
        # Each row should be ~[1/3, 1/3, 1/3].
        self.assertTrue(torch.allclose(
            A, torch.full_like(A, 1.0 / 3), atol=1e-5,
        ))


class TestPerStepCell(unittest.TestCase):

    def test_inherits_inter_basin_graph(self):
        from lnn.core.inter_basin_graph_cfc import InterBasinGraphCfCCell
        cell = PerStepInterBasinGraphCfCCell(
            input_size=2, hidden_size=4,
            n_branches=2, n_basin=2,
        )
        self.assertIsInstance(cell, InterBasinGraphCfCCell)

    def test_has_a_mlp(self):
        cell = PerStepInterBasinGraphCfCCell(
            input_size=2, hidden_size=4,
            n_branches=2, n_basin=3,
        )
        self.assertTrue(hasattr(cell, "a_mlp"))

    def test_per_step_adjacency_shape(self):
        cell = PerStepInterBasinGraphCfCCell(
            input_size=2, hidden_size=4,
            n_branches=2, n_basin=3,
        )
        x = torch.randn(4, 2)
        A = cell.per_step_adjacency(x, k=0)
        self.assertEqual(A.shape, (4, 3, 3))

    def test_per_step_adjacency_row_stochastic(self):
        cell = PerStepInterBasinGraphCfCCell(
            input_size=2, hidden_size=4,
            n_branches=2, n_basin=3,
        )
        x = torch.randn(8, 2)
        A = cell.per_step_adjacency(x, k=0)
        sums = A.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    def test_grad_flows_to_a_mlp(self):
        cell = PerStepInterBasinGraphCfCCell(
            input_size=2, hidden_size=4,
            n_branches=2, n_basin=3,
        )
        x = torch.randn(4, 2)
        A = cell.per_step_adjacency(x, k=0)
        loss = A.sum()
        loss.backward()
        # MLP weight should have non-zero grad.
        self.assertIsNotNone(cell.a_mlp.weight.grad)
        self.assertGreater(cell.a_mlp.weight.grad.abs().sum().item(), 0.0)

    def test_forward_with_aux_keys(self):
        cell = PerStepInterBasinGraphCfCCell(
            input_size=2, hidden_size=4,
            n_branches=2, n_basin=3,
        )
        x = torch.randn(4, 2)
        h_list = [torch.zeros(4, 4) for _ in range(2)]
        _, _, aux = cell.forward_with_aux(
            x, h_list, lyap_lambda=0.0, sep_lambda=0.0,
            dist_lambda=1.0, graph_lambda=0.0,
        )
        self.assertIn("adjacency_per_step", aux)
        # Shape: (n_branches, B, K, K).
        A_per_step = aux["adjacency_per_step"]
        self.assertEqual(A_per_step.shape, (2, 4, 3, 3))
        # Row-stochastic.
        sums = A_per_step.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    def test_adjacency_per_step_varies_with_x(self):
        cell = PerStepInterBasinGraphCfCCell(
            input_size=2, hidden_size=4,
            n_branches=2, n_basin=3,
        )
        B = 4
        h_list = [torch.zeros(B, 4) for _ in range(2)]
        x1 = torch.tensor([[1.0, 0.0]] * B)
        x2 = torch.tensor([[0.0, 1.0]] * B)
        _, _, aux1 = cell.forward_with_aux(
            x1, h_list, lyap_lambda=0.0, sep_lambda=0.0,
            dist_lambda=1.0, graph_lambda=0.0,
        )
        _, _, aux2 = cell.forward_with_aux(
            x2, h_list, lyap_lambda=0.0, sep_lambda=0.0,
            dist_lambda=1.0, graph_lambda=0.0,
        )
        A1 = aux1["adjacency_per_step"]
        A2 = aux2["adjacency_per_step"]
        diff = (A1 - A2).abs().mean().item()
        self.assertGreater(diff, 1e-3)

    def test_a_diversity_non_negative(self):
        cell = PerStepInterBasinGraphCfCCell(
            input_size=2, hidden_size=4,
            n_branches=2, n_basin=3,
        )
        x = torch.randn(4, 2)
        h_list = [torch.zeros(4, 4) for _ in range(2)]
        _, _, aux = cell.forward_with_aux(
            x, h_list, lyap_lambda=0.0, sep_lambda=0.0,
            dist_lambda=1.0, graph_lambda=0.0,
        )
        self.assertIn("A_diversity", aux)
        self.assertGreaterEqual(aux["A_diversity"].item(), 0.0)

    def test_graph_loss_uses_static(self):
        """sym_lambda + sparse_lambda should apply to static A, not A_t."""
        cell = PerStepInterBasinGraphCfCCell(
            input_size=2, hidden_size=4,
            n_branches=2, n_basin=3,
            sym_lambda=0.1, sparse_lambda=0.1,
        )
        x = torch.randn(4, 2)
        h_list = [torch.zeros(4, 4) for _ in range(2)]
        _, _, aux = cell.forward_with_aux(
            x, h_list, lyap_lambda=0.0, sep_lambda=0.0,
            dist_lambda=1.0, graph_lambda=1.0,
        )
        self.assertIn("graph_loss_applied", aux)
        self.assertGreater(aux["graph_loss_applied"].item(), 0.0)

    def test_mlp_hidden_creates_sequential(self):
        cell = PerStepInterBasinGraphCfCCell(
            input_size=2, hidden_size=4,
            n_branches=2, n_basin=3, mlp_hidden=8,
        )
        import torch.nn as nn
        self.assertIsInstance(cell.a_mlp, nn.Sequential)
        x = torch.randn(4, 2)
        A = cell.per_step_adjacency(x, k=0)
        self.assertEqual(A.shape, (4, 3, 3))


if __name__ == "__main__":
    unittest.main()