"""LearnedBetaPS+LN+Khl+FFT+DropConnect-CfC (PRD #10-158, Round 196, 2026-06-16).

Wraps round 187's winner (lbps_lnkhlfft_5_3_2) with **DropConnect**
on the cell's linear weights (Wan et al 2013).

After 4 noise rounds (r192 input, r193 hidden, r194 combined,
r195 σ sweep), test **weight-level** regularization. DropConnect
zeros individual weights in W matrices during training (vs
dropout which zeros activations).

Mechanism (TRAINING ONLY, not eval):
    # For each cell's linear layer (f_gate, g_branch, h_branch):
    if self.training and self.dropconnect_p > 0:
        mask = (rand_like(W) > dropconnect_p).float()
        W_masked = W * mask / (1 - dropconnect_p)  # inverted dropout
    else:
        W_masked = W
    output = W_masked @ input
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lnn.core.learned_beta_ps_ln_cfc import LearnedBetaPSLNCfCCell
from lnn.core.learned_beta_ps_ln_khl_cfc import LearnedBetaPSLNKhlCfCStackedNetwork
from lnn.core.learned_beta_ps_ln_khlfft_cfc import LearnedBetaPSLNKhlFftCfCStackedNetwork


class DropConnectLinearFunction(torch.autograd.Function):
    """Custom autograd function for DropConnect (preserves gradient flow).

    Forward: applies binary mask to weight matrix
    Backward: gradients flow through the mask (only surviving weights get gradient)
    """

    @staticmethod
    def forward(ctx, weight, bias, input, mask, p):
        ctx.save_for_backward(weight, mask)
        ctx.p = p
        # Apply mask (and inverted dropout scaling)
        masked_weight = weight * mask / (1.0 - p) if p > 0 else weight
        return F.linear(input, masked_weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        weight, mask = ctx.saved_tensors
        p = ctx.p
        masked_weight = weight * mask / (1.0 - p) if p > 0 else weight
        # Gradient flows only through masked weight
        grad_input = grad_output @ masked_weight
        grad_weight = grad_output.t() @ (ctx_input := None) if False else None
        return None, None, grad_input, None, None


def _apply_dropconnect_forward(linear, x, p, training):
    """Apply DropConnect to a single nn.Linear layer."""
    if not (training and p > 0):
        return linear(x)
    # Generate binary mask (1 = keep, 0 = drop)
    mask = (torch.rand_like(linear.weight) > p).float()
    # Apply mask with inverted dropout
    masked_weight = linear.weight * mask / (1.0 - p)
    return F.linear(x, masked_weight, linear.bias)


class DropConnectCfCCell(nn.Module):
    """DropConnect wrapper around LearnedBetaPSLNCfCCell.

    Applies DropConnect to the cell's 3 linear layers:
    f_gate, g_branch, h_branch.
    """

    def __init__(self, inner_cell, dropconnect_p=0.1):
        super().__init__()
        self.inner_cell = inner_cell
        self.dropconnect_p = dropconnect_p

    def forward(self, x_t, h_t, emas_x, emas_h, dt=1.0):
        x_t = torch.nan_to_num(x_t, nan=0.0)
        h_t = torch.nan_to_num(h_t, nan=0.0)
        emas_x = [torch.nan_to_num(e, nan=0.0) for e in emas_x]
        emas_h = [torch.nan_to_num(e, nan=0.0) for e in emas_h]

        beta_x = self.inner_cell.beta_x
        emas_x_new = [
            beta_x[k] * emas_x[k] + (1.0 - beta_x[k]) * x_t
            for k in range(self.inner_cell.Kx)
        ]
        beta_h = self.inner_cell.beta_h
        emas_h_new = [
            beta_h[k] * emas_h[k] + (1.0 - beta_h[k]) * h_t
            for k in range(self.inner_cell.Kh)
        ]

        if self.inner_cell.mode_x == "concat":
            aug_x = torch.cat([x_t] + emas_x_new, dim=-1)
        else:
            aug_x = torch.cat([x_t] + [e - x_t for e in emas_x_new], dim=-1)

        if self.inner_cell.mode_h == "concat":
            aug_h = torch.cat([h_t] + emas_h_new, dim=-1)
        else:
            aug_h = torch.cat([h_t] + [e - h_t for e in emas_h_new], dim=-1)

        z = torch.cat([aug_x, aug_h], dim=-1)
        z = self.inner_cell.layer_norm(z)

        # Closed-form CfC with DropConnect on linear weights
        f = _apply_dropconnect_forward(
            self.inner_cell.f_gate[0], z, self.dropconnect_p, self.training
        )
        f = torch.sigmoid(f)
        g_pre = _apply_dropconnect_forward(
            self.inner_cell.g_branch[0], z, self.dropconnect_p, self.training
        )
        g = torch.tanh(g_pre)
        h_pre = _apply_dropconnect_forward(
            self.inner_cell.h_branch[0], z, self.dropconnect_p, self.training
        )
        h_branch = torch.tanh(h_pre)

        if isinstance(dt, torch.Tensor):
            dt_b = dt
            if dt_b.dim() < 2:
                dt_b = dt_b.unsqueeze(-1)
            tau_eff = torch.exp(-f * dt_b / torch.abs(self.inner_cell.time_scale))
        else:
            tau_eff = torch.exp(-f * float(dt) / torch.abs(self.inner_cell.time_scale))
        h_new = tau_eff * g + (1.0 - tau_eff) * h_branch
        return h_new, emas_x_new, emas_h_new


class DropConnectCfCStackedNetwork(nn.Module):
    """DropConnect wrapper around round 187's stack."""

    def __init__(
        self,
        inner_network,
        dropconnect_p=0.1,
    ):
        super().__init__()
        self.inner_network = inner_network
        self.dropconnect_p = dropconnect_p
        # Pre-create cell wrappers so they participate in eval()/train()
        inner = inner_network.cfc_net
        self.wrapped_cells = nn.ModuleList(
            [DropConnectCfCCell(c, dropconnect_p) for c in inner.cells]
        )

    def forward(self, x):
        # Apply FFT encoder first
        x_aug = self.inner_network.fft_encoder(x)
        # Reach into the inner cfc_net
        inner = self.inner_network.cfc_net
        B, T, _ = x_aug.shape
        device = x_aug.device
        hs = [torch.zeros(B, inner.hidden_size, device=device) for _ in range(inner.num_layers)]
        emas_x = [
            [torch.zeros(B, inner.cells[l].input_size, device=device) for _ in range(inner.Kx)]
            for l in range(inner.num_layers)
        ]
        emas_h = [
            [torch.zeros(B, inner.hidden_size, device=device) for _ in range(inner.Kh_ladder[l])]
            for l in range(inner.num_layers)
        ]
        outputs = []
        for t in range(T):
            inp = x_aug[:, t, :]
            for l, cell in enumerate(self.wrapped_cells):
                hs[l], emas_x[l], emas_h[l] = cell(
                    inp, hs[l], emas_x[l], emas_h[l],
                )
                inp = hs[l]
            outputs.append(inner.head(hs[-1]))
        outputs = torch.stack(outputs, dim=1)
        if inner.return_sequences:
            return outputs
        return outputs[:, -1, :]


def make_lbps_lnkhlfft_dropconnect_5_3_2(input_size, hidden_size, output_size, num_layers=3, dropconnect_p=0.1):
    """Kh=[5,3,2] + LN + FFT + DropConnect on cell weights."""
    inner = LearnedBetaPSLNKhlFftCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kh_ladder=[5, 3, 2], Kx=5,
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )
    return DropConnectCfCStackedNetwork(inner, dropconnect_p=dropconnect_p)


__all__ = [
    "DropConnectLinearFunction",
    "DropConnectCfCCell",
    "DropConnectCfCStackedNetwork",
    "make_lbps_lnkhlfft_dropconnect_5_3_2",
]
