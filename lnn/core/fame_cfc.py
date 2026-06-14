"""FAME-style top-K sparse MoE wrapper around K CfCCell experts (PRD #10-36, 2026-06-14).

Wraps ``K`` independent ``CfCCell`` experts behind a
``ForecastabilityRouter`` (FAME, arXiv:2606.08896).  The cell output
is ``Σ_k g_k · expert_k(x_t, h_prev)`` where ``g`` has at most
``top_k`` non-zero entries.

Reuses the round 77 ``CfCCell(n_tau)`` interface and the round 76
multi-time-scale machinery; the only change vs ``MRMoECfCCell`` is
the router (sparse top-K instead of dense softmax).

This module is intentionally a *cell-level* FAME implementation:
- No production-data replay simulator (FAME §4.2).
- No cost-aware router training (FAME §3.4 mines expert-suitability
  targets from validation; we just use the router logits directly).
- No multi-modal fingerprinting (FAME §3.2 uses a 6-d fingerprint;
  we use ``[x_t; h_prev]`` as a proxy).
- No load-balancing auxiliary loss.

The follow-up PRD #10-37 (orthogonality constraint) and #10-38
(K×n_tau×top_K sweep) extend this base.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.cfc import CfCCell
from lnn.core.forecastability_router import ForecastabilityRouter
from lnn.core.sequence_utils import select_step_delta, select_step_mask


class FAMECfCCell(nn.Module):
    """FAME-style top-K sparse MoE wrapper around ``K`` CfCCell experts.

    Args:
        input_size: Input feature dimension.
        hidden_size: Hidden state dimension.
        n_experts: Number of experts (K ≥ 1).
        top_k: Number of experts activated per step (K' ∈ [1, K]).
            Default 2 matches the FAME paper's empirical choice
            (1.92 experts/series on the production vending dataset).
        n_tau_per_expert: Per-expert ``n_tau`` (round 76 compatibility).
        tau_scales: Per-branch initial time constants, forwarded to every expert.
        router_hidden: Width of the optional 2-layer router MLP (``0`` = linear).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_experts: int = 3,
        top_k: int = 2,
        n_tau_per_expert: int = 1,
        tau_scales: tuple = (0.1, 1.0, 10.0),
        router_hidden: int = 0,
    ):
        super().__init__()
        assert n_experts >= 1
        assert 1 <= top_k <= n_experts
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_experts = int(n_experts)
        self.top_k = int(top_k)
        self.n_tau_per_expert = int(n_tau_per_expert)
        self.tau_scales = tuple(tau_scales)
        self.router_hidden = int(router_hidden)

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
        self.router = ForecastabilityRouter(
            input_size=input_size,
            hidden_size=hidden_size,
            n_experts=self.n_experts,
            top_k=self.top_k,
            router_hidden=self.router_hidden,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """One step of top-K sparse FAME routing over CfC experts.

        Args:
            x_t: [B, input_size] input at this step.
            h:   [B, hidden_size] previous hidden state.
            dt:  scalar or [B] per-sample time delta.

        Returns:
            h_new: [B, hidden_size] mixed expert output.
        """
        g = self.router(x_t, h)  # [B, K] with K' nonzeros
        # Diagnostics side-channel: mixture weights and top-K indices.
        self.last_g = g.detach()
        self.last_top_idx = self.router.last_top_idx.detach()
        # Run all K experts but only the top-K contribute via g.
        # (Masking rather than skipping the non-top-K forward keeps
        # autograd simple and ensures gradient flows only to activated experts.)
        outs = [expert(x_t, h, dt=dt) for expert in self.experts]  # K × [B, H]
        stacked = torch.stack(outs, dim=1)  # [B, K, H]
        return (g.unsqueeze(-1) * stacked).sum(dim=1)  # [B, H]


class FAMECfCNetwork(nn.Module):
    """Stacked FAME-style top-K sparse MoE CfC network.

    Mirrors the ``CfCNetwork`` / ``MRMoECfCNetwork`` API
    (return_sequences, mask, dt) but swaps every layer's ``CfCCell``
    for a ``FAMECfCCell``.

    Args:
        input_size: Input feature dimension.
        hidden_size: Hidden state dimension.
        output_size: Output dimension.
        num_layers: Number of stacked FAME CfC layers.
        return_sequences: If True, return the full sequence; else last step.
        n_experts: Number of experts per layer (K).
        top_k: Number of experts activated per step (K').
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
        top_k: int = 2,
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
        self.top_k = int(top_k)
        self.n_tau_per_expert = int(n_tau_per_expert)
        self.tau_scales = tuple(tau_scales)
        self.router_hidden = int(router_hidden)

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            self.cells.append(
                FAMECfCCell(
                    input_size=in_dim,
                    hidden_size=hidden_size,
                    n_experts=self.n_experts,
                    top_k=self.top_k,
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
