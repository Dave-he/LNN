"""MR-MoE: Multi-Rate Mixture of Experts for Liquid Neural Networks (PRD #10-24, 2026-06-14).

Implements the minimum-viable cell-level instantiation of the
Multi-Rate Mixture-of-Experts (MR-MoE) pattern from
arXiv:2606.12240 (Zong et al., 2026) and the pattern-routed
heterogeneous-experts idea from arXiv:2606.13024 (CausalMoE, 2026-06-11).

Each cell wraps ``K`` independent ``CfCCell`` experts and a softmax
router that produces per-step mixture weights ``g ∈ Δ^K``.  The cell
output is ``Σ_k g_k · expert_k(x_t, h_prev)``.

This module is intentionally a *cell-level* MR-MoE: each expert is a
full ``CfCCell`` (with its own optional ``n_tau`` from round 76) and
the mixture is a soft attention.  It does **not** include:

- Causal discovery heads (see PRD #10-33 / CausalMoE paper)
- Adaptive token compression (see PRD #10-34)
- Load-balancing auxiliary loss (research remains open in MR-MoE v1)
- Multi-rate gating in time (each expert is a single ``CfCCell``;
  combining ``n_experts=K`` with ``n_tau_per_expert=K'`` gives the
  paper's K×K' effective multi-scale regime)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lnn.core.cfc import CfCCell
from lnn.core.sequence_utils import select_step_delta, select_step_mask


class MRMoECfCCell(nn.Module):
    """Multi-Rate Mixture-of-Experts wrapper around ``K`` CfCCell experts.

    Args:
        input_size: Input feature dimension.
        hidden_size: Hidden state dimension.
        n_experts: Number of parallel ``CfCCell`` experts (K ≥ 1).
            ``n_experts=1`` reduces to a single ``CfCCell`` with a
            no-op router (softmax of a single logit is 1.0); the
            forward is numerically equivalent within float32 eps.
        n_tau_per_expert: Per-expert ``n_tau`` value (forwarded to each
            underlying ``CfCCell``).  Default 1 keeps the original
            single-τ behaviour per expert; set ≥ 2 to enable
            intra-expert multi-rate (round 76 + MR-MoE combination).
        tau_scales: Per-branch initial time constants, forwarded to
            every expert's underlying ``CfCCell``.
        router_hidden: Hidden size of the router MLP.  ``0`` (default)
            uses a single linear layer ``Linear(input+hidden → K)``;
            ``>0`` uses a 2-layer MLP with the given width.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_experts: int = 3,
        n_tau_per_expert: int = 1,
        tau_scales: tuple = (0.1, 1.0, 10.0),
        router_hidden: int = 0,
    ):
        super().__init__()
        assert n_experts >= 1, f"n_experts must be >= 1, got {n_experts}"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_experts = int(n_experts)
        self.n_tau_per_expert = int(n_tau_per_expert)
        self.tau_scales = tuple(tau_scales)
        self.router_hidden = int(router_hidden)

        # K independent CfC experts.  Each can have its own n_tau.
        self.experts = nn.ModuleList(
            [
                CfCCell(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    n_tau=self.n_tau_per_expert,
                    tau_scales=self.tau_scales,
                )
                for _ in range(self.n_experts)
            ]
        )

        # Router: produces K logits from [x_t; h_prev].
        router_in = input_size + hidden_size
        if self.router_hidden > 0:
            self.router = nn.Sequential(
                nn.Linear(router_in, self.router_hidden),
                nn.Tanh(),
                nn.Linear(self.router_hidden, self.n_experts),
            )
        else:
            self.router = nn.Linear(router_in, self.n_experts)

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """One step of soft-routed multi-expert CfC.

        Args:
            x_t: [B, input_size] input at this step.
            h:   [B, hidden_size] previous hidden state.
            dt:  scalar or [B] per-sample time delta.

        Returns:
            h_new: [B, hidden_size] mixed expert output.
            g:     [B, n_experts]  mixture weights (returned via attribute ``last_g``
                                   for introspection — pure side-channel, the
                                   function's return is the new hidden state to
                                   keep the contract aligned with ``CfCCell``).
        """
        combined = torch.cat([x_t, h], dim=-1)
        g = F.softmax(self.router(combined), dim=-1)  # [B, K]
        # Stash mixture weights for diagnostics (no grad impact).
        self.last_g = g.detach()
        # Run each expert; per CfCCell contract each takes (x_t, h, dt) → [B, H].
        outs = [expert(x_t, h, dt=dt) for expert in self.experts]  # K × [B, H]
        stacked = torch.stack(outs, dim=1)  # [B, K, H]
        # Weighted sum: g [B, K] × stacked [B, K, H] → [B, H]
        return (g.unsqueeze(-1) * stacked).sum(dim=1)


class MRMoECfCNetwork(nn.Module):
    """Stacked MR-MoE CfC network.

    Mirrors the ``CfCNetwork`` API (return_sequences, mask, dt) but
    swaps every layer's ``CfCCell`` for a ``MRMoECfCCell``.

    Args:
        input_size: Input feature dimension.
        hidden_size: Hidden state dimension.
        output_size: Output dimension.
        num_layers: Number of stacked MR-MoE CfC layers.
        return_sequences: If True, return the full sequence; else last step.
        n_experts: Number of CfC experts per layer (K).
        n_tau_per_expert: Per-expert ``n_tau`` (round 76 compatibility).
        tau_scales: Per-branch τ init, forwarded to every expert.
        router_hidden: Router MLP width (``0`` = linear).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        return_sequences: bool = True,
        n_experts: int = 3,
        n_tau_per_expert: int = 1,
        tau_scales: tuple = (0.1, 1.0, 10.0),
        router_hidden: int = 0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.return_sequences = return_sequences
        self.n_experts = int(n_experts)
        self.n_tau_per_expert = int(n_tau_per_expert)
        self.tau_scales = tuple(tau_scales)
        self.router_hidden = int(router_hidden)

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            self.cells.append(
                MRMoECfCCell(
                    input_size=in_dim,
                    hidden_size=hidden_size,
                    n_experts=self.n_experts,
                    n_tau_per_expert=self.n_tau_per_expert,
                    tau_scales=self.tau_scales,
                    router_hidden=self.router_hidden,
                )
            )
        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
        dt: float | torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Process a batch of sequences.

        Args:
            x: [B, T, F] input sequence.
            h0: Optional [num_layers, B, H] initial hidden state.
            dt: Same per-step time-delta shapes as ``CfCNetwork``.
            mask: Same mask shapes as ``CfCNetwork``.

        Returns:
            [B, T, output_size] (return_sequences=True) or [B, output_size].
        """
        batch_size, seq_len, _ = x.shape
        if h0 is None:
            h0 = torch.zeros(
                self.num_layers, batch_size, self.hidden_size,
                device=x.device, dtype=x.dtype,
            )

        h = h0
        layer_input = x
        for i, cell in enumerate(self.cells):
            outputs = []
            h_i = h[i]
            for t in range(seq_len):
                dt_t = select_step_delta(dt, t, batch_size, seq_len, x.device, x.dtype)
                input_mask, update_mask = select_step_mask(
                    mask, t, batch_size, seq_len, self.input_size, x.device, x.dtype,
                )
                x_t = torch.nan_to_num(layer_input[:, t, :])
                if i == 0 and input_mask is not None:
                    x_t = x_t * input_mask
                h_candidate = cell(x_t, h_i, dt=dt_t)
                h_i = h_candidate if update_mask is None else update_mask * h_candidate + (1.0 - update_mask) * h_i
                outputs.append(h_i)
            layer_input = torch.stack(outputs, dim=1)
            h = torch.cat(
                [h_i.unsqueeze(0) if j == i else h[j].unsqueeze(0) for j in range(self.num_layers)],
                dim=0,
            )

        if self.return_sequences:
            return self.output_proj(layer_input)
        return self.output_proj(layer_input[:, -1, :])
