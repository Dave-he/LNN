"""DeepSeekMoE-style Shared Expert Isolation wrapper around CfCCell experts (PRD #10-75, 2026-06-15).

Response to arXiv:2401.06066 (DeepSeek-AI, January 2024) — *DeepSeekMoE: Towards
Ultimate Expert Specialization in Mixture-of-Experts Language Models*.

Wraps ``K_s`` shared experts (always active, no routing) and ``K_r`` routed
experts (FAME-style top-K_r sparse routing) behind a single cell. The
combination is **additive** (not averaged), making the shared path a stable
residual anchor that does not break the recurrent CfC dynamics.

Forward pass::

    shared_out = mean([shared_e(x_t, h, dt=dt) for e in shared_experts])  # [B, H]
    g = router(x_t, h)                                                   # [B, K_r]
    routed_out = sum_k g_k * routed_k(x_t, h, dt=dt)                      # [B, H]
    h_new = shared_out + routed_out                                       # [B, H]

The shared path:
- Always processes every step (no routing, no failure mode)
- Outputs are MEAN-aggregated across K_s shared experts, then ADDED to routed path
- Acts as a "common knowledge sink" that never collapses

The routed path:
- FAME-style sparse top-K_r routing
- Softmax-weighted combination of activated experts

This module is intentionally minimal:
- No fine-grained expert segmentation (the paper's 2nd contribution)
- No shared-expert isolation loss (the paper uses an aux loss to encourage
  routed experts to be different from shared; we use the additive structure)
- No aux load-balancing loss for shared (always active by construction)
- No capacity factor (EC's failure mode - we don't drop tokens)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.cfc import CfCCell
from lnn.core.forecastability_router import ForecastabilityRouter
from lnn.core.sequence_utils import select_step_delta, select_step_mask


class DeepSeekCfCCell(nn.Module):
    """DeepSeekMoE-style cell: K_s shared (always-on) + K_r routed (top-K_r) experts.

    Args:
        input_size: Input feature dimension.
        hidden_size: Hidden state dimension.
        n_shared: Number of shared experts (K_s, always active).
            Default 1 matches the original DeepSeekMoE default.
        n_routed: Number of routed experts (K_r, top-K_r per step).
            Default 3 gives a K=4 total expert budget.
        top_k: Number of routed experts activated per step (K_r').
            Default 2 matches FAME's empirical sweet spot.
        n_tau_per_expert: Per-expert ``n_tau`` (round 76 compatibility).
        tau_scales: Per-branch initial time constants.
        router_hidden: Width of the optional 2-layer router MLP (``0`` = linear).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_shared: int = 1,
        n_routed: int = 3,
        top_k: int = 2,
        n_tau_per_expert: int = 1,
        tau_scales: tuple = (0.1, 1.0, 10.0),
        router_hidden: int = 0,
    ):
        super().__init__()
        assert n_shared >= 0, f"n_shared must be >= 0, got {n_shared}"
        assert n_routed >= 0, f"n_routed must be >= 0, got {n_routed}"
        assert n_shared + n_routed >= 1, "need at least one expert (shared or routed)"
        if n_routed > 0:
            assert 1 <= top_k <= n_routed, (
                f"top_k must be in [1, n_routed={n_routed}], got {top_k}"
            )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_shared = int(n_shared)
        self.n_routed = int(n_routed)
        self.top_k = int(top_k) if n_routed > 0 else 0
        self.n_tau_per_expert = int(n_tau_per_expert)
        self.tau_scales = tuple(tau_scales)
        self.router_hidden = int(router_hidden)

        # Shared experts (always active).
        if self.n_shared > 0:
            self.shared_experts = nn.ModuleList(
                [
                    CfCCell(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        n_tau=self.n_tau_per_expert,
                        tau_scales=self.tau_scales,
                    )
                    for _ in range(self.n_shared)
                ]
            )
        else:
            self.shared_experts = nn.ModuleList()

        # Routed experts (top-K_r per step).
        if self.n_routed > 0:
            self.routed_experts = nn.ModuleList(
                [
                    CfCCell(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        n_tau=self.n_tau_per_expert,
                        tau_scales=self.tau_scales,
                    )
                    for _ in range(self.n_routed)
                ]
            )
            self.router = ForecastabilityRouter(
                input_size=input_size,
                hidden_size=hidden_size,
                n_experts=self.n_routed,
                top_k=self.top_k,
                router_hidden=self.router_hidden,
            )
        else:
            self.routed_experts = nn.ModuleList()
            self.router = None

        # Diagnostic stash.
        self.last_g: torch.Tensor | None = None
        self.last_top_idx: torch.Tensor | None = None
        self.last_shared_util: torch.Tensor | None = None  # [K_s] always 1.0

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """One step of DeepSeekMoE-style shared+routed cell.

        Args:
            x_t: [B, input_size] input at this step.
            h:   [B, hidden_size] previous hidden state.
            dt:  scalar or [B] per-sample time delta.

        Returns:
            h_new: [B, hidden_size] = shared_out + routed_out.
        """
        # 1) Shared expert path: ALWAYS active.  Mean across K_s.
        if self.n_shared > 0:
            shared_outs = [expert(x_t, h, dt=dt) for expert in self.shared_experts]
            # Stash shared utilization diagnostic: always 1.0 (always active).
            self.last_shared_util = torch.ones(self.n_shared)
            shared_out = torch.stack(shared_outs, dim=1).mean(dim=1)  # [B, H]
        else:
            shared_out = torch.zeros(
                x_t.shape[0], self.hidden_size, device=x_t.device, dtype=x_t.dtype,
            )
            self.last_shared_util = torch.zeros(0)

        # 2) Routed expert path: FAME-style top-K_r.
        if self.n_routed > 0:
            g = self.router(x_t, h)  # [B, K_r]
            self.last_g = g.detach()
            self.last_top_idx = self.router.last_top_idx.detach()
            routed_outs = [
                expert(x_t, h, dt=dt) for expert in self.routed_experts
            ]  # K_r × [B, H]
            stacked_routed = torch.stack(routed_outs, dim=1)  # [B, K_r, H]
            routed_out = (g.unsqueeze(-1) * stacked_routed).sum(dim=1)  # [B, H]
        else:
            routed_out = torch.zeros(
                x_t.shape[0], self.hidden_size, device=x_t.device, dtype=x_t.dtype,
            )
            self.last_g = None
            self.last_top_idx = None

        # 3) Additive combination (DeepSeekMoE key insight).
        h_new = shared_out + routed_out
        return h_new


class DeepSeekCfCNetwork(nn.Module):
    """Stacked DeepSeekMoE-style shared+routed CfC network.

    Mirrors the ``CfCNetwork`` / ``FAMECfCNetwork`` API (return_sequences, mask,
    dt) but uses ``DeepSeekCfCCell`` for every layer.

    Args:
        input_size: Input feature dimension.
        hidden_size: Hidden state dimension.
        output_size: Output dimension.
        num_layers: Number of stacked DeepSeek cells.
        return_sequences: If True, return the full sequence; else last step.
        n_shared: Number of shared experts per layer (K_s, always active).
        n_routed: Number of routed experts per layer (K_r).
        top_k: Number of routed experts activated per step.
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
        n_shared: int = 1,
        n_routed: int = 3,
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
        self.n_shared = int(n_shared)
        self.n_routed = int(n_routed)
        self.top_k = int(top_k)
        self.n_tau_per_expert = int(n_tau_per_expert)
        self.tau_scales = tuple(tau_scales)
        self.router_hidden = int(router_hidden)

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            self.cells.append(
                DeepSeekCfCCell(
                    input_size=in_dim,
                    hidden_size=hidden_size,
                    n_shared=self.n_shared,
                    n_routed=self.n_routed,
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


def deepseek_utilization(cell: DeepSeekCfCCell) -> dict:
    """Diagnostic for DeepSeek cell's expert utilization.

    Returns:
        Dict with:
            - "shared_util": [K_s] tensor (all 1.0 by construction).
            - "routed_util": [K_r] tensor (mean of last_g per expert, or zeros).
    """
    shared_util = (
        cell.last_shared_util.cpu()
        if cell.last_shared_util is not None
        else torch.ones(cell.n_shared)
    )
    if cell.last_g is not None and cell.n_routed > 0:
        routed_util = cell.last_g.mean(dim=0).cpu()
    else:
        routed_util = torch.zeros(cell.n_routed)
    return {
        "shared_util": shared_util,
        "routed_util": routed_util,
    }
