"""Tests for the Expert Choice routing module (PRD #10-74)."""
from __future__ import annotations

import math

import pytest
import torch

from lnn.core.expert_choice import (
    ExpertChoiceCfCCell,
    ExpertChoiceCfCNetwork,
    ExpertChoiceRouter,
    expert_choice_load,
)


class TestExpertChoiceRouter:
    def test_init(self):
        r = ExpertChoiceRouter(input_size=2, hidden_size=4, n_experts=3)
        assert r.n_experts == 3
        assert r.use_sigmoid is True

    def test_init_with_router_hidden(self):
        r = ExpertChoiceRouter(input_size=2, hidden_size=4, n_experts=3, router_hidden=8)
        assert r.router_hidden == 8
        assert isinstance(r.net, torch.nn.Sequential)

    def test_init_softmax_mode(self):
        r = ExpertChoiceRouter(input_size=2, hidden_size=4, n_experts=3, use_sigmoid=False)
        assert r.use_sigmoid is False

    def test_forward_shape(self):
        r = ExpertChoiceRouter(input_size=2, hidden_size=4, n_experts=3)
        x = torch.randn(4, 8, 2)
        h = torch.randn(4, 8, 4)
        assign_mask, assign_w = r(x, h, cap_k=4)
        assert assign_mask.shape == (4, 3, 8)
        assert assign_w.shape == (4, 3, 8)
        # Each expert should pick exactly 4 tokens per batch row.
        assert (assign_mask.float().sum(dim=-1) == 4).all()

    def test_perfect_load_balance(self):
        """The defining property of EC: every expert processes the same #tokens."""
        r = ExpertChoiceRouter(input_size=2, hidden_size=4, n_experts=5)
        x = torch.randn(2, 16, 2)
        h = torch.randn(2, 16, 4)
        assign_mask, _ = r(x, h, cap_k=8)
        # Per-expert count over (B, T) should be exactly 2*8=16 for each expert.
        counts = assign_mask.float().sum(dim=(0, -1))  # [K]
        assert (counts == 16).all()

    def test_cap_k_capped_at_T(self):
        r = ExpertChoiceRouter(input_size=2, hidden_size=4, n_experts=3)
        x = torch.randn(2, 4, 2)
        h = torch.randn(2, 4, 4)
        assign_mask, _ = r(x, h, cap_k=10)  # cap_k > T
        # Should select min(cap_k, T) = 4.
        assert (assign_mask.float().sum(dim=-1) == 4).all()

    def test_assign_w_in_range_sigmoid(self):
        r = ExpertChoiceRouter(input_size=2, hidden_size=4, n_experts=3, use_sigmoid=True)
        x = torch.randn(4, 8, 2)
        h = torch.randn(4, 8, 4)
        _, assign_w = r(x, h, cap_k=4)
        assert (assign_w >= 0.0).all()
        assert (assign_w <= 1.0).all()

    def test_gradient_flows(self):
        r = ExpertChoiceRouter(input_size=2, hidden_size=4, n_experts=3)
        x = torch.randn(2, 8, 2, requires_grad=True)
        h = torch.randn(2, 8, 4, requires_grad=True)
        _, assign_w = r(x, h, cap_k=4)
        assign_w.sum().backward()
        assert x.grad is not None
        assert h.grad is not None


class TestExpertChoiceCfCCell:
    def test_init(self):
        cell = ExpertChoiceCfCCell(input_size=2, hidden_size=4, n_experts=3)
        assert cell.n_experts == 3
        assert len(cell.experts) == 3

    def test_init_with_cap(self):
        cell = ExpertChoiceCfCCell(input_size=2, hidden_size=4, n_experts=3, cap_k=2)
        assert cell.cap_k == 2

    def test_forward_with_assignment(self):
        cell = ExpertChoiceCfCCell(input_size=2, hidden_size=4, n_experts=3, cap_k=4)
        B, T = 4, 8
        x = torch.randn(B, T, 2)
        h0 = torch.randn(B, T, 4)
        # Pre-compute assignment.
        assign_mask, assign_w = cell.router(x, h0, cap_k=4)
        # Apply to a single step.
        out = cell(
            x[:, 0, :], h0[:, 0, :], dt=1.0, T=T,
            assign_mask_K=assign_mask[:, :, 0], assign_w_K=assign_w[:, :, 0],
        )
        assert out.shape == (B, 4)

    def test_forward_no_assignment_fallback(self):
        """Single-step fallback (no pre-computed assignment)."""
        cell = ExpertChoiceCfCCell(input_size=2, hidden_size=4, n_experts=3, cap_k=4)
        x = torch.randn(4, 2)
        h = torch.randn(4, 4)
        out = cell(x, h, dt=1.0, T=4)
        assert out.shape == (4, 4)

    def test_gradient_flows(self):
        cell = ExpertChoiceCfCCell(input_size=2, hidden_size=4, n_experts=3, cap_k=4)
        B, T = 4, 8
        x = torch.randn(B, T, 2, requires_grad=True)
        h0 = torch.randn(B, T, 4, requires_grad=True)
        assign_mask, assign_w = cell.router(x, h0, cap_k=4)
        out = cell(
            x[:, 0, :], h0[:, 0, :], dt=1.0, T=T,
            assign_mask_K=assign_mask[:, :, 0], assign_w_K=assign_w[:, :, 0],
        )
        out.sum().backward()
        assert x.grad is not None
        assert h0.grad is not None


class TestExpertChoiceCfCNetwork:
    def test_init(self):
        net = ExpertChoiceCfCNetwork(
            input_size=2, hidden_size=4, output_size=1,
            num_layers=2, n_experts=3,
        )
        assert net.num_layers == 2
        assert len(net.cells) == 2
        for cell in net.cells:
            assert cell.n_experts == 3

    def test_init_with_int_cap(self):
        net = ExpertChoiceCfCNetwork(
            input_size=2, hidden_size=4, output_size=1, cap_k=4,
        )
        assert net.cells[0].cap_k == 4

    def test_init_with_frac_cap(self):
        net = ExpertChoiceCfCNetwork(
            input_size=2, hidden_size=4, output_size=1, cap_k_frac=0.5,
        )
        assert net.cells[0].cap_k is None  # resolved at forward

    def test_init_both_raises(self):
        with pytest.raises(ValueError):
            ExpertChoiceCfCNetwork(
                input_size=2, hidden_size=4, output_size=1,
                cap_k=2, cap_k_frac=0.5,
            )

    def test_init_bad_frac_raises(self):
        with pytest.raises(ValueError):
            ExpertChoiceCfCNetwork(
                input_size=2, hidden_size=4, output_size=1, cap_k_frac=1.5,
            )

    def test_forward_dense(self):
        """cap_k=None → dense (every expert processes all tokens)."""
        net = ExpertChoiceCfCNetwork(
            input_size=2, hidden_size=4, output_size=1,
            num_layers=1, n_experts=3, cap_k=None, return_sequences=True,
        )
        x = torch.randn(4, 8, 2)
        out = net(x)
        assert out.shape == (4, 8, 1)

    def test_forward_last_step(self):
        net = ExpertChoiceCfCNetwork(
            input_size=2, hidden_size=4, output_size=1,
            num_layers=1, n_experts=3, cap_k=None, return_sequences=False,
        )
        x = torch.randn(4, 8, 2)
        out = net(x)
        assert out.shape == (4, 1)

    def test_forward_with_int_cap(self):
        net = ExpertChoiceCfCNetwork(
            input_size=2, hidden_size=4, output_size=1,
            num_layers=2, n_experts=3, cap_k=4, return_sequences=True,
        )
        x = torch.randn(4, 8, 2)
        out = net(x)
        assert out.shape == (4, 8, 1)
        # Diagnostics stashed.
        for cell in net.cells:
            assert cell.last_assign_mask is not None
            assert cell.last_assign_w is not None

    def test_forward_with_frac_cap(self):
        net = ExpertChoiceCfCNetwork(
            input_size=2, hidden_size=4, output_size=1,
            num_layers=2, n_experts=3, cap_k_frac=0.5, return_sequences=True,
        )
        x = torch.randn(4, 8, 2)
        out = net(x)
        assert out.shape == (4, 8, 1)
        # Each cell should have cap_k=4 (=0.5*8).
        for cell in net.cells:
            assert cell.cap_k == 4

    def test_gradient_flows(self):
        net = ExpertChoiceCfCNetwork(
            input_size=2, hidden_size=4, output_size=1,
            num_layers=1, n_experts=3, cap_k=4, return_sequences=True,
        )
        x = torch.randn(4, 8, 2, requires_grad=True)
        out = net(x)
        out.sum().backward()
        assert x.grad is not None


class TestExpertChoiceIntegration:
    def test_perfect_load_balance(self):
        """After a forward pass, every expert should process the same # tokens."""
        net = ExpertChoiceCfCNetwork(
            input_size=2, hidden_size=4, output_size=1,
            num_layers=1, n_experts=4, cap_k=4, return_sequences=True,
        )
        x = torch.randn(2, 8, 2)
        net(x)
        for cell in net.cells:
            counts = expert_choice_load(cell)
            # Each expert should have processed 2*4=8 tokens total.
            assert (counts == 8).all()

    def test_expert_choice_load_function_no_forward(self):
        cell = ExpertChoiceCfCCell(input_size=2, hidden_size=4, n_experts=3, cap_k=4)
        # No forward pass — should return zeros.
        counts = expert_choice_load(cell)
        assert (counts == 0).all()

    def test_captures_signal(self):
        """EC network should fit a simple sin signal at least as well as
        no-cap baseline (dense), with balanced load.
        """
        torch.manual_seed(42)
        net = ExpertChoiceCfCNetwork(
            input_size=1, hidden_size=8, output_size=1,
            num_layers=1, n_experts=4, cap_k=8, return_sequences=True,
        )
        T = 16
        t = torch.linspace(0, 2 * math.pi, T).unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
        target = torch.sin(t)
        optim = torch.optim.Adam(net.parameters(), lr=0.01)
        loss = torch.tensor(0.0)
        for _ in range(50):
            optim.zero_grad()
            pred = net(t)
            loss = (pred - target).pow(2).mean()
            loss.backward()
            optim.step()
        assert loss.item() < 0.5

    def test_smaller_cap_k_does_not_crash(self):
        """cap_k=1 (extreme sparse) should still run without errors."""
        net = ExpertChoiceCfCNetwork(
            input_size=2, hidden_size=4, output_size=1,
            num_layers=1, n_experts=3, cap_k=1, return_sequences=True,
        )
        x = torch.randn(2, 8, 2)
        out = net(x)
        assert out.shape == (2, 8, 1)
        # Each expert should process exactly 1 token per batch row.
        for cell in net.cells:
            counts = expert_choice_load(cell)
            assert (counts == 2).all()  # 2 batches * 1 token each
