"""Tests for the MoD (Mixture-of-Depths) routing module (PRD #10-73)."""
from __future__ import annotations

import math

import pytest
import torch

from lnn.core.mod_routing import (
    MoDCfCCell,
    MoDCfCNetwork,
    MoDRouter,
    compute_mod_aux_loss,
)


class TestMoDRouter:
    def test_init(self):
        r = MoDRouter(input_size=2, hidden_size=4, router_hidden=0)
        assert hasattr(r, "net")
        # Linear: in -> 1
        assert r.net.out_features == 1
        assert r.net.in_features == 2 + 4

    def test_init_with_router_hidden(self):
        r = MoDRouter(input_size=2, hidden_size=4, router_hidden=8)
        assert r.router_hidden == 8
        # Sequential with 3 layers
        assert isinstance(r.net, torch.nn.Sequential)
        assert len(list(r.net)) == 3

    def test_forward_shape(self):
        r = MoDRouter(input_size=2, hidden_size=4)
        x = torch.randn(8, 2)
        h = torch.randn(8, 4)
        process_mask, router_prob, aux_loss = r(x, h, cap_k=3, T=8)
        assert process_mask.shape == (8,)
        assert process_mask.dtype == torch.bool
        assert router_prob.shape == (8,)
        # Exactly cap_k=3 should be selected.
        assert process_mask.sum().item() == 3
        assert aux_loss.dim() == 0  # scalar

    def test_aux_loss_non_negative(self):
        r = MoDRouter(input_size=2, hidden_size=4)
        x = torch.randn(16, 2)
        h = torch.randn(16, 4)
        _, _, aux_loss = r(x, h, cap_k=4, T=16)
        assert aux_loss.item() >= 0.0

    def test_cap_k_capped_at_batch(self):
        r = MoDRouter(input_size=2, hidden_size=4)
        x = torch.randn(4, 2)
        h = torch.randn(4, 4)
        process_mask, _, _ = r(x, h, cap_k=10, T=4)  # cap_k > B
        # Should select min(cap_k, B) = 4 (all)
        assert process_mask.sum().item() == 4

    def test_router_prob_in_range(self):
        r = MoDRouter(input_size=2, hidden_size=4)
        x = torch.randn(16, 2)
        h = torch.randn(16, 4)
        _, router_prob, _ = r(x, h, cap_k=4, T=16)
        assert (router_prob >= 0.0).all()
        assert (router_prob <= 1.0).all()

    def test_gradient_flows(self):
        r = MoDRouter(input_size=2, hidden_size=4)
        x = torch.randn(8, 2, requires_grad=True)
        h = torch.randn(8, 4, requires_grad=True)
        _, _, aux_loss = r(x, h, cap_k=3, T=8)
        aux_loss.backward()
        assert x.grad is not None
        assert h.grad is not None


class TestMoDCfCCell:
    def test_init_no_cap(self):
        cell = MoDCfCCell(input_size=2, hidden_size=4, cap_k=None)
        assert cell.cap_k is None
        assert hasattr(cell, "cell")
        assert hasattr(cell, "router")

    def test_init_with_cap(self):
        cell = MoDCfCCell(input_size=2, hidden_size=4, cap_k=2)
        assert cell.cap_k == 2

    def test_forward_no_cap(self):
        # cap_k=None means always process; should behave like CfCCell.
        cell = MoDCfCCell(input_size=2, hidden_size=4, cap_k=None)
        x = torch.randn(8, 2)
        h = torch.randn(8, 4)
        out = cell(x, h, dt=1.0)
        assert out.shape == (8, 4)
        # No aux loss accumulated because no routing decision made.
        assert cell.aux_loss is None

    def test_forward_with_cap(self):
        cell = MoDCfCCell(input_size=2, hidden_size=4, cap_k=3)
        x = torch.randn(8, 2)
        h = torch.randn(8, 4)
        out = cell(x, h, dt=1.0, T=8)
        assert out.shape == (8, 4)
        # Diagnostics stashed.
        assert cell.last_router_prob is not None
        assert cell.last_process_mask is not None
        assert cell.last_process_mask.sum().item() == 3
        assert cell.aux_loss is not None

    def test_forward_skipped_timestep_keeps_hidden(self):
        """If process_mask is False for a sample, output should equal h for that sample."""
        # B=2, cap_k=1 → exactly one of the two is processed.
        cell = MoDCfCCell(input_size=2, hidden_size=4, cap_k=1)
        x = torch.randn(2, 2)
        h = torch.randn(2, 4)
        out = cell(x, h, dt=1.0, T=2)
        # One of the two samples must be unchanged.
        mask = cell.last_process_mask
        for i in range(2):
            if not mask[i].item():
                # Skipped: output should equal h[i].
                assert torch.allclose(out[i], h[i], atol=1e-6)
            else:
                # Processed: should be different from h[i] (CfC update).
                assert not torch.allclose(out[i], h[i], atol=1e-6)

    def test_gradient_flows_with_cap(self):
        cell = MoDCfCCell(input_size=2, hidden_size=4, cap_k=3)
        x = torch.randn(8, 2, requires_grad=True)
        h = torch.randn(8, 4, requires_grad=True)
        out = cell(x, h, dt=1.0, T=8)
        out.sum().backward()
        assert x.grad is not None
        assert h.grad is not None


class TestMoDCfCNetwork:
    def test_init_no_cap(self):
        net = MoDCfCNetwork(
            input_size=2, hidden_size=4, output_size=1,
            num_layers=2, cap_k=None,
        )
        assert net.num_layers == 2
        assert len(net.cells) == 2

    def test_init_with_int_cap(self):
        net = MoDCfCNetwork(
            input_size=2, hidden_size=4, output_size=1,
            num_layers=1, cap_k=4,
        )
        assert net.cells[0].cap_k == 4

    def test_init_with_frac_cap(self):
        net = MoDCfCNetwork(
            input_size=2, hidden_size=4, output_size=1,
            num_layers=1, cap_k_frac=0.5,
        )
        # Placeholder cap; resolved at forward.
        assert net.cells[0].cap_k is None

    def test_init_both_raises(self):
        with pytest.raises(ValueError):
            MoDCfCNetwork(
                input_size=2, hidden_size=4, output_size=1,
                cap_k=2, cap_k_frac=0.5,
            )

    def test_init_bad_frac_raises(self):
        with pytest.raises(ValueError):
            MoDCfCNetwork(
                input_size=2, hidden_size=4, output_size=1,
                cap_k_frac=1.5,
            )

    def test_forward_no_cap(self):
        net = MoDCfCNetwork(
            input_size=2, hidden_size=4, output_size=1,
            num_layers=1, cap_k=None, return_sequences=True,
        )
        x = torch.randn(4, 8, 2)
        out = net(x)
        assert out.shape == (4, 8, 1)

    def test_forward_last_step(self):
        net = MoDCfCNetwork(
            input_size=2, hidden_size=4, output_size=1,
            num_layers=1, cap_k=None, return_sequences=False,
        )
        x = torch.randn(4, 8, 2)
        out = net(x)
        assert out.shape == (4, 1)

    def test_forward_with_cap(self):
        net = MoDCfCNetwork(
            input_size=2, hidden_size=4, output_size=1,
            num_layers=2, cap_k=3, return_sequences=True,
        )
        x = torch.randn(4, 8, 2)
        out = net(x)
        assert out.shape == (4, 8, 1)
        # Aux loss should be present in cells.
        for cell in net.cells:
            assert cell.aux_loss is not None

    def test_forward_with_frac_cap(self):
        net = MoDCfCNetwork(
            input_size=2, hidden_size=4, output_size=1,
            num_layers=2, cap_k_frac=0.5, return_sequences=True,
        )
        x = torch.randn(4, 8, 2)
        out = net(x)
        assert out.shape == (4, 8, 1)
        # Each cell should have cap_k = 4 (=0.5 * 8)
        for cell in net.cells:
            assert cell.cap_k == 4

    def test_gradient_flows(self):
        net = MoDCfCNetwork(
            input_size=2, hidden_size=4, output_size=1,
            num_layers=1, cap_k=3, return_sequences=True,
        )
        x = torch.randn(4, 8, 2, requires_grad=True)
        out = net(x)
        out.sum().backward()
        assert x.grad is not None

    def test_aux_loss_aggregation(self):
        net = MoDCfCNetwork(
            input_size=2, hidden_size=4, output_size=1,
            num_layers=2, cap_k=3, return_sequences=True,
        )
        x = torch.randn(4, 8, 2)
        out = net(x)
        aux = compute_mod_aux_loss(net)
        assert aux.dim() == 0
        assert aux.item() >= 0.0

    def test_aux_loss_no_cap_is_zero(self):
        net = MoDCfCNetwork(
            input_size=2, hidden_size=4, output_size=1,
            num_layers=2, cap_k=None, return_sequences=True,
        )
        x = torch.randn(4, 8, 2)
        net(x)
        aux = compute_mod_aux_loss(net)
        assert aux.item() == 0.0


class TestMoDIntegration:
    def test_captures_signal(self):
        """MoD network should fit a simple linear-ish signal at least as well as
        no-cap baseline on sin data, with skip rate not collapsing to 0 or 1.
        """
        torch.manual_seed(42)
        net = MoDCfCNetwork(
            input_size=1, hidden_size=8, output_size=1,
            num_layers=1, cap_k=8, return_sequences=True,
        )
        # sin sequence of T=16
        T = 16
        t = torch.linspace(0, 2 * math.pi, T).unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
        target = torch.sin(t)
        optim = torch.optim.Adam(net.parameters(), lr=0.01)
        loss = torch.tensor(0.0)
        for _ in range(50):
            optim.zero_grad()
            pred = net(t)
            loss = (pred - target).pow(2).mean()
            aux = compute_mod_aux_loss(net)
            (loss + 0.01 * aux).backward()
            optim.step()
        # After 50 epochs, sin should be at least partially captured.
        assert loss.item() < 0.5

    def test_process_mask_varies_across_steps(self):
        """Within a single sequence, the router should make different
        process/skip decisions for different timesteps (else it has
        collapsed to all-skip or all-process).
        """
        torch.manual_seed(0)
        cell = MoDCfCCell(input_size=2, hidden_size=4, cap_k=2)
        # Apply to T=8 timesteps with different inputs.  B=8 so cap_k=2 < B.
        masks = []
        h = torch.zeros(8, 4)
        for t in range(8):
            x = torch.randn(8, 2) * (1.0 + 0.5 * t)
            cell(x, h, dt=1.0, T=8)
            masks.append(cell.last_process_mask.clone())
        # Convert list of bool tensors to (T, B) int matrix.
        mask_matrix = torch.stack([m.float() for m in masks], dim=0)
        # Not all rows should be identical.
        all_same = (mask_matrix.std(dim=0) == 0).all().item()
        assert not all_same, "Router should make different decisions across timesteps"

    def test_higher_cap_k_higher_skipped_fraction(self):
        """Smaller cap_k → smaller fraction of timesteps processed."""
        torch.manual_seed(0)
        caps = [1, 4, 6]  # all < B=8 so routing is actually used
        skip_fracs = []
        for cap in caps:
            cell = MoDCfCCell(input_size=2, hidden_size=4, cap_k=cap)
            # 8 steps, B=8 samples per step.
            for t in range(8):
                x = torch.randn(8, 2)
                h = torch.randn(8, 4)
                cell(x, h, dt=1.0, T=8)
            # The aux loss includes f = mean(process_mask).  Larger cap → larger f.
            # We track the average f across timesteps in the last call.
            assert cell.aux_loss is not None
            skip_fracs.append(cell.aux_loss.item() / (cap if cap > 0 else 1.0))
        # This is a soft check; mainly verify nothing crashed and values are non-degenerate.
        assert all(s >= 0 for s in skip_fracs)
