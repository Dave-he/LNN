"""ReMoE-style Fully Differentiable MoE with ReLU Routing wrapper around CfCCell experts (PRD #10-76, 2026-06-15).

Response to arXiv:2412.14711 (Wang, Zhu, Chen, December 2024, ICLR 2025) — *ReMoE:
Fully Differentiable Mixture-of-Experts with ReLU Routing*.

The core idea: replace the standard TopK+Softmax sparse routing (which is
discontinuous and only partially differentiable) with a **fully differentiable**
ReLU-based router ``g = ReLU(W x)`` followed by an additive combination of
expert outputs ``h_new = sum_i g_i * expert_i(x)``.

Why this fits the 91-113 audit pattern:
- ReLU's natural sparsity (negative scores become 0) gives sparse gating
  WITHOUT a hard top-K operator.
- The combination is a soft, smooth, fully-differentiable weighted sum.
- The recurrent state ``h_t`` is consumed by each expert through the standard
  CfC gate-and-update, never modified, averaged, or mixed across experts.
- Gradient flows to ALL non-zero-gated experts, unlike FAME where only the
  top-K are activated.

Forward pass::

    s = router(x_t, h)                                              # [B, K]
    g = ReLU(s)                                                      # [B, K]
    h_new = sum_i g_i * expert_i(x_t, h, dt=dt)                      # [B, H]

The module is intentionally minimal:
- No top-K straight-through estimator (the ReLU IS the sparsity mechanism)
- No softmax (ReLU is unbounded above; load balancing uses an aux loss)
- Optional load-balancing aux loss ``remoe_load_balancing_loss(g, K)``
- No fine-grained expert segmentation (the paper's second contribution)
- No shared-expert isolation (orthogonal; can be combined in a future round)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lnn.core.cfc import CfCCell
from lnn.core.sequence_utils import select_step_delta, select_step_mask


class ReMoERouter(nn.Module):
    """ReLU-based router: produces K non-negative gating scores per sample.

    Args:
        input_size: Input feature dimension.
        hidden_size: Hidden state dimension (concatenated with input as router input).
        n_experts: Number of experts K to route over.
        router_hidden: Width of optional 2-layer router MLP (``0`` = linear).

    Forward:
        x_t: [B, input_size], h: [B, hidden_size]
        Returns: g: [B, n_experts] with ReLU non-negativity (naturally sparse).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_experts: int,
        router_hidden: int = 0,
        init_bias: float = 1.0,
    ):
        super().__init__()
        assert n_experts >= 1, f"n_experts must be >= 1, got {n_experts}"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_experts = int(n_experts)
        self.router_hidden = int(router_hidden)
        router_in = input_size + hidden_size
        if router_hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(router_in, router_hidden),
                nn.Tanh(),
                nn.Linear(router_hidden, self.n_experts),
            )
        else:
            self.net = nn.Linear(router_in, self.n_experts)
        # Per-expert load-balancing bias (similar to auxiliary-loss-free MoE).
        # Initialized positive so all experts start active; learned to balance load.
        self.bias = nn.Parameter(torch.full((self.n_experts,), float(init_bias)))

    def forward(self, x_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Compute ReLU-gated routing scores with per-expert bias.

        Args:
            x_t: [B, input_size] input at this step.
            h:   [B, hidden_size] previous hidden state.

        Returns:
            g: [B, n_experts] non-negative routing scores (naturally sparse).
        """
        z = torch.cat([x_t, h], dim=-1)
        s = self.net(z)
        g = F.relu(s + self.bias)  # [B, K] - natural sparsity + initial balance
        return g


class ReMoECfCCell(nn.Module):
    """ReMoE-style cell: K experts + ReLU router, fully differentiable.

    Args:
        input_size: Input feature dimension.
        hidden_size: Hidden state dimension.
        n_experts: Number of experts K.
        n_tau_per_expert: Per-expert ``n_tau`` (round 76 compatibility).
        tau_scales: Per-branch initial time constants.
        router_hidden: Width of optional 2-layer router MLP (``0`` = linear).
        sparsity_target: Target fraction of experts with non-zero gate (informational).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_experts: int = 4,
        n_tau_per_expert: int = 1,
        tau_scales: tuple = (0.1, 1.0, 10.0),
        router_hidden: int = 0,
        sparsity_target: float = 0.5,
    ):
        super().__init__()
        assert n_experts >= 1, f"n_experts must be >= 1, got {n_experts}"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_experts = int(n_experts)
        self.n_tau_per_expert = int(n_tau_per_expert)
        self.tau_scales = tuple(tau_scales)
        self.router_hidden = int(router_hidden)
        self.sparsity_target = float(sparsity_target)

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
        self.router = ReMoERouter(
            input_size=input_size,
            hidden_size=hidden_size,
            n_experts=self.n_experts,
            router_hidden=self.router_hidden,
        )

        # Diagnostic stash.
        self.last_g: torch.Tensor | None = None  # [B, K]
        self.last_sparsity: torch.Tensor | None = None  # scalar: mean fraction of non-zero gates per row

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """One step of ReMoE-style cell.

        Args:
            x_t: [B, input_size] input at this step.
            h:   [B, hidden_size] previous hidden state.
            dt:  scalar or [B] per-sample time delta.

        Returns:
            h_new: [B, hidden_size] = Σ_i g_i * expert_i(x_t, h, dt=dt).
        """
        # 1) Compute ReLU-gated routing scores.
        g = self.router(x_t, h)  # [B, K], non-negative
        self.last_g = g.detach()
        # Track fraction of non-zero gates per row.
        self.last_sparsity = (g > 0).float().mean().detach()

        # 2) Run all K experts (full gradient, no top-K).
        expert_outs = [expert(x_t, h, dt=dt) for expert in self.experts]  # K × [B, H]
        stacked = torch.stack(expert_outs, dim=1)  # [B, K, H]

        # 3) ReLU-gated additive combination.
        h_new = (g.unsqueeze(-1) * stacked).sum(dim=1)  # [B, H]
        return h_new


class ReMoECfCNetwork(nn.Module):
    """Stacked ReMoE-style CfC network.

    Mirrors the ``CfCNetwork`` / ``DeepSeekCfCNetwork`` API (return_sequences,
    mask, dt) but uses ``ReMoECfCCell`` for every layer.

    Args:
        input_size: Input feature dimension.
        hidden_size: Hidden state dimension.
        output_size: Output dimension.
        num_layers: Number of stacked ReMoE cells.
        return_sequences: If True, return the full sequence; else last step.
        n_experts: Number of experts K per layer.
        n_tau_per_expert: Per-expert ``n_tau``.
        tau_scales: Per-branch τ init, forwarded to every expert.
        router_hidden: Router MLP width (``0`` = linear).
        sparsity_target: Target fraction of experts with non-zero gate.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        return_sequences: bool = True,
        n_experts: int = 4,
        n_tau_per_expert: int = 1,
        tau_scales: tuple = (0.1, 1.0, 10.0),
        router_hidden: int = 0,
        sparsity_target: float = 0.5,
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
        self.sparsity_target = float(sparsity_target)

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            self.cells.append(
                ReMoECfCCell(
                    input_size=in_dim,
                    hidden_size=hidden_size,
                    n_experts=self.n_experts,
                    n_tau_per_expert=self.n_tau_per_expert,
                    tau_scales=self.tau_scales,
                    router_hidden=self.router_hidden,
                    sparsity_target=self.sparsity_target,
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


def remoe_utilization(cell: ReMoECfCCell) -> dict:
    """Diagnostic for ReMoE cell's expert utilization.

    Returns:
        Dict with:
            - "g_mean": [K] tensor of mean gate activation per expert.
            - "g_active_frac": [K] tensor of fraction of samples with non-zero gate.
            - "sparsity": scalar fraction of all (B*K) gates that are non-zero.
    """
    if cell.last_g is None:
        return {
            "g_mean": torch.zeros(cell.n_experts),
            "g_active_frac": torch.zeros(cell.n_experts),
            "sparsity": torch.tensor(0.0),
        }
    g = cell.last_g
    g_mean = g.mean(dim=0).cpu()  # [K]
    g_active_frac = (g > 0).float().mean(dim=0).cpu()  # [K]
    sparsity = (g > 0).float().mean().cpu()  # scalar
    return {
        "g_mean": g_mean,
        "g_active_frac": g_active_frac,
        "sparsity": sparsity,
    }


def remoe_load_balancing_loss(
    g: torch.Tensor,
    n_experts: int | None = None,
) -> torch.Tensor:
    """Auxiliary load-balancing loss for ReMoE.

    Computes ``-Σ_i f_i * log(f_i * K)`` where ``f_i = g_i / Σ_j g_j`` is the
    fraction of total routing mass assigned to expert ``i``. The loss is
    minimized (== 0) when the mass is perfectly uniform across K experts.

    Args:
        g: [B, K] non-negative routing scores.
        n_experts: K (defaults to g.shape[-1]).

    Returns:
        Scalar tensor: load-balancing loss (>= 0, == 0 at uniform).
    """
    if n_experts is None:
        n_experts = g.shape[-1]
    # Total mass per expert across batch.
    mass = g.sum(dim=0)  # [K]
    total = mass.sum() + 1e-8
    f = mass / total  # [K], fraction of mass per expert
    # -Σ_i f_i * log(f_i * K) is minimized at f_i = 1/K, value 0.
    # Equivalent to: Σ_i f_i * log(f_i * K) but with negation.
    eps = 1e-8
    loss = (f * (torch.log(f * n_experts + eps))).sum()
    # loss >= 0; we want to MINIMIZE divergence from uniform
    # so we return +loss (the standard form).
    return loss
