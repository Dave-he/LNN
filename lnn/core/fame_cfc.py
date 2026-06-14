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
from lnn.core.cosine_router import CosineRouter
from lnn.core.forecastability_router import ForecastabilityRouter
from lnn.core.phi_balancing import PhiBalancer
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
        phi_balance: If True, attach a ``PhiBalancer`` (PRD #10-40) to
            the router and update it on every training step.  Default
            ``False`` (back-compat with round 80).
        ema_alpha: φ-balancing EMA decay (forwarded to ``PhiBalancer``).
        phi_step_size: φ-balancing mirror-descent step size η.
        router_type: ``"learned"`` (default, back-compat) uses
            ``ForecastabilityRouter``; ``"cosine"`` uses the
            parameter-free ``CosineRouter`` (PRD #10-41, arXiv:2605.12476).
            ``phi_balance`` is ignored when ``router_type="cosine"``
            (no learned logits to bias).
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
        phi_balance: bool = False,
        ema_alpha: float = 0.01,
        phi_step_size: float = 0.01,
        router_type: str = "learned",
    ):
        super().__init__()
        assert n_experts >= 1
        assert 1 <= top_k <= n_experts
        assert router_type in ("learned", "cosine"), (
            f"router_type must be 'learned' or 'cosine', got {router_type!r}"
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_experts = int(n_experts)
        self.top_k = int(top_k)
        self.n_tau_per_expert = int(n_tau_per_expert)
        self.tau_scales = tuple(tau_scales)
        self.router_hidden = int(router_hidden)
        self.phi_balance = bool(phi_balance)
        self.ema_alpha = float(ema_alpha)
        self.phi_step_size = float(phi_step_size)
        self.router_type = router_type

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
        # φ-balancing (PRD #10-40): per-layer balancer instance shared
        # between the router (for bias add) and the cell (for EMA update).
        # We expose it as a submodule so .to(device) etc. move the buffers.
        # Note: φ-balancing only makes sense with the learned router.
        if self.phi_balance and router_type == "learned":
            self.balancer = PhiBalancer(
                n_experts=self.n_experts,
                ema_alpha=self.ema_alpha,
                step_size=self.phi_step_size,
            )
        else:
            self.balancer = None
        if router_type == "learned":
            self.router = ForecastabilityRouter(
                input_size=input_size,
                hidden_size=hidden_size,
                n_experts=self.n_experts,
                top_k=self.top_k,
                router_hidden=self.router_hidden,
                balancer=self.balancer,  # may be None — back-compat
            )
        else:  # "cosine" — parameter-free
            self.router = CosineRouter(
                input_size=input_size,
                hidden_size=hidden_size,
                n_experts=self.n_experts,
                top_k=self.top_k,
                ema_alpha=self.ema_alpha,
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
        h_new, expert_outs = self.forward_with_aux(x_t, h, dt=dt)
        del expert_outs  # forward() doesn't expose them; use forward_with_aux
        return h_new

    def forward_with_aux(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Same as ``forward`` but also returns the per-expert outputs.

        Used by ``FAMECfCNetwork.forward_with_aux`` to compute the
        geometric orthogonality constraint (PRD #10-37, AnchorMoE
        arXiv:2606.03631) over the K expert hidden states.

        Returns:
            (h_new, outs):
                h_new: [B, hidden_size] mixed expert output.
                outs:  K-element list of [B, hidden_size] per-expert outputs,
                       in the same order as ``self.experts``.  Useful for
                       the orthogonality loss.
        """
        g = self.router(x_t, h)  # [B, K] with K' nonzeros
        # Diagnostics side-channel: mixture weights and top-K indices.
        self.last_g = g.detach()
        self.last_top_idx = self.router.last_top_idx.detach()
        # φ-balancing (PRD #10-40): update the EMA of per-expert assignment
        # from the hard top-K indices.  Skipped in eval mode (the bias is
        # frozen, matching the paper's protocol).  Note: the bias was
        # already added inside router.forward() via the same instance.
        if self.balancer is not None and self.training:
            self.balancer.update(self.last_top_idx)
        # CosineRouter (PRD #10-41): update per-expert running hidden-state
        # mean from the just-routed [x_t; h].  Same train/eval gate.
        if self.router_type == "cosine" and self.training:
            combined = torch.cat([x_t, h], dim=-1)
            self.router.update(combined, self.last_top_idx)
        # Run all K experts but only the top-K contribute via g.
        # (Masking rather than skipping the non-top-K forward keeps
        # autograd simple and ensures gradient flows only to activated experts.)
        outs = [expert(x_t, h, dt=dt) for expert in self.experts]  # K × [B, H]
        stacked = torch.stack(outs, dim=1)  # [B, K, H]
        h_new = (g.unsqueeze(-1) * stacked).sum(dim=1)  # [B, H]
        return h_new, outs


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
        phi_balance: Forward to every layer's ``FAMECfCCell``.
        ema_alpha: Forward to every layer's balancer / CosineRouter.
        phi_step_size: Forward to every layer's balancer.
        router_type: ``"learned"`` (default) or ``"cosine"`` (PRD #10-41).
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
        phi_balance: bool = False,
        ema_alpha: float = 0.01,
        phi_step_size: float = 0.01,
        router_type: str = "learned",
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
        self.phi_balance = bool(phi_balance)
        self.ema_alpha = float(ema_alpha)
        self.phi_step_size = float(phi_step_size)
        self.router_type = router_type

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
                    phi_balance=self.phi_balance,
                    ema_alpha=self.ema_alpha,
                    phi_step_size=self.phi_step_size,
                    router_type=self.router_type,
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

    def forward_with_aux(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
        dt: float | torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[list[list[torch.Tensor]]]]:
        """Like ``forward`` but also returns per-step, per-layer, per-expert outputs.

        The shape of the returned nested list is
        ``[num_layers][T][K]`` of ``[B, hidden_size]`` tensors.  The
        final-step expert outputs (T-1) per layer are the most useful
        for the orthogonality loss, but we keep the full trace so
        callers can choose.

        Returns:
            (y_pred, expert_outputs):
                y_pred: same shape as ``forward`` would return.
                expert_outputs: nested list ``[num_layers][T][K]`` of
                    ``[B, hidden_size]`` tensors.
        """
        batch_size, seq_len, _ = x.shape
        if h0 is None:
            h0 = torch.zeros(
                self.num_layers, batch_size, self.hidden_size,
                device=x.device, dtype=x.dtype,
            )

        h = h0
        layer_input = x
        # expert_outputs[layer_idx][t_idx] = list of K [B, H] tensors
        expert_outputs: list[list[list[torch.Tensor]]] = [[] for _ in range(self.num_layers)]
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
                h_candidate, outs_t = cell.forward_with_aux(x_t, h_i, dt=dt_t)
                expert_outputs[i].append(outs_t)
                h_i = h_candidate if update_mask is None else update_mask * h_candidate + (1.0 - update_mask) * h_i
                outputs.append(h_i)
            layer_input = torch.stack(outputs, dim=1)
            h = torch.cat(
                [h_i.unsqueeze(0) if j == i else h[j].unsqueeze(0) for j in range(self.num_layers)],
                dim=0,
            )

        if self.return_sequences:
            y_pred = self.output_proj(layer_input)
        else:
            y_pred = self.output_proj(layer_input[:, -1, :])
        return y_pred, expert_outputs
