"""Tests for ChannelProjectionCfCCell (round 262)."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from lnn.core.channel_projection_cfc import ChannelProjectionCfCCell


class TestChannelProjectionCell(unittest.TestCase):

    def test_inherits_per_step(self):
        from lnn.core.per_step_inter_basin_graph_cfc import (
            PerStepInterBasinGraphCfCCell,
        )
        cell = ChannelProjectionCfCCell(
            input_size=4, hidden_size=4,
            n_branches=2, n_basin=2, d_ctx=8,
        )
        self.assertIsInstance(cell, PerStepInterBasinGraphCfCCell)

    def test_has_channel_proj(self):
        cell = ChannelProjectionCfCCell(
            input_size=4, hidden_size=4,
            n_branches=2, n_basin=2, d_ctx=8,
        )
        self.assertTrue(hasattr(cell, "channel_proj"))
        self.assertEqual(cell.channel_proj.in_features, 4)
        self.assertEqual(cell.channel_proj.out_features, 8)

    def test_project_input_shape(self):
        cell = ChannelProjectionCfCCell(
            input_size=4, hidden_size=4,
            n_branches=2, n_basin=2, d_ctx=8,
        )
        x = torch.randn(4, 4)
        c = cell.project_input(x)
        self.assertEqual(c.shape, (4, 8))

    def test_per_step_adjacency_shape(self):
        cell = ChannelProjectionCfCCell(
            input_size=4, hidden_size=4,
            n_branches=2, n_basin=3, d_ctx=8,
        )
        x = torch.randn(4, 4)
        A = cell.per_step_adjacency(x, k=0)
        self.assertEqual(A.shape, (4, 3, 3))

    def test_per_step_adjacency_row_stochastic(self):
        cell = ChannelProjectionCfCCell(
            input_size=4, hidden_size=4,
            n_branches=2, n_basin=3, d_ctx=8,
        )
        x = torch.randn(8, 4)
        A = cell.per_step_adjacency(x, k=0)
        sums = A.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    def test_adjacency_changes_with_x(self):
        cell = ChannelProjectionCfCCell(
            input_size=4, hidden_size=4,
            n_branches=2, n_basin=2, d_ctx=8,
        )
        x1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4)
        x2 = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 4)
        A1 = cell.per_step_adjacency(x1, k=0)
        A2 = cell.per_step_adjacency(x2, k=0)
        diff = (A1 - A2).abs().mean().item()
        # Channel projection is initialized small (std=0.1) so change is small.
        # We just need to confirm A is sensitive to x_t (i.e. not pure noise).
        self.assertGreater(diff, 1e-4)

    def test_grad_flows_to_channel_proj(self):
        cell = ChannelProjectionCfCCell(
            input_size=4, hidden_size=4,
            n_branches=2, n_basin=2, d_ctx=8,
        )
        x = torch.randn(4, 4)
        c = cell.project_input(x)
        loss = c.sum()
        loss.backward()
        self.assertIsNotNone(cell.channel_proj.weight.grad)
        self.assertGreater(
            cell.channel_proj.weight.grad.abs().sum().item(), 0.0,
        )

    def test_forward_with_aux_exposes_routing_context(self):
        cell = ChannelProjectionCfCCell(
            input_size=4, hidden_size=4,
            n_branches=2, n_basin=3, d_ctx=8,
        )
        x = torch.randn(4, 4)
        h_list = [torch.zeros(4, 4) for _ in range(2)]
        _, _, aux = cell.forward_with_aux(
            x, h_list, lyap_lambda=0.0, sep_lambda=0.0,
            dist_lambda=1.0, graph_lambda=0.0,
        )
        self.assertIn("routing_context", aux)
        self.assertIn("routing_context_var", aux)
        self.assertEqual(aux["routing_context"].shape, (4, 8))
        self.assertGreaterEqual(aux["routing_context_var"].item(), 0.0)

    def test_mlp_hidden_creates_sequential(self):
        cell = ChannelProjectionCfCCell(
            input_size=4, hidden_size=4,
            n_branches=2, n_basin=2, d_ctx=8, mlp_hidden=4,
        )
        self.assertIsInstance(cell.a_mlp, nn.Sequential)

    def test_d_ctx_configurable(self):
        cell = ChannelProjectionCfCCell(
            input_size=4, hidden_size=4,
            n_branches=2, n_basin=2, d_ctx=16,
        )
        self.assertEqual(cell.d_ctx, 16)
        x = torch.randn(4, 4)
        c = cell.project_input(x)
        self.assertEqual(c.shape, (4, 16))


if __name__ == "__main__":
    unittest.main()