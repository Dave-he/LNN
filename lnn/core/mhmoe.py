"""Multi-Head Mixture-of-Experts (MH-MoE) wrapper around CfCCell experts (PRD #10-77, 2026-06-15).

Response to arXiv:2404.15045 (Wu, Huang, Wang, Wei, April 2024, NeurIPS 2024) —
*Multi-Head Mixture-of-Experts*.

The core idea: split each input into H sub-tokens (feature chunks), route each
sub-token to its own top-K experts, process in parallel, concatenate back to
the original feature dimension.

Why this fits the 91-114 audit pattern:
- H sub-tokens × K experts = K·H parallel sub-token paths
- On average, each expert receives H·(B/K) sub-tokens per step → balanced load
- Fixes the FAME H=0 collapse (round 103): every sub-token gets its own expert
  decision, so all K are exercised on average
- Recurrent state is per-cell, preserved across timesteps
- Sub-token outputs are CONCATENATED (not averaged), preserving the D-dim signal
- Gradient flows to all K experts naturally (each sub-token picks a different
  expert on average)

Forward pass::

    x_t: [B, D] → split into H sub-tokens of dim D/H → [B*H, D/H]
    g = softmax(W x)        # router: [B*H, K] per sub-token
    top_vals, top_idx = topk(g, top_k)   # [B*H, top_k]
    routed = sum_i g_i * expert_i(sub_token)   # [B*H, D/H]
    h_new = concat(routed, dim=0)         # [B, D]

The mechanism is intentionally minimal:
- Per-sub-token routing (not per-token)
- H is a hyperparameter (not data-dependent)
- No load-balancing aux loss (the multi-head split itself provides balance)
- No fine-grained expert segmentation (the paper's other contribution)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lnn.core.cfc import CfCCell
from lnn.core.sequence_utils import select_step_delta, select_step_mask


class MHRouter(nn.Module):
    """Per-sub-token router: g = softmax(W x).

    Args:
        head_dim: Sub-token dimension (D/H).
        n_experts: Number of experts K.
        router_hidden: Width of optional 2-layer router MLP (``0`` = linear).

    Forward:
        x: [N, head_dim] sub-tokens (N = B*H).
        Returns: g: [N, n_experts] softmax probabilities.
    """

    def __init__(self, head_dim: int, n_experts: int, router_hidden: int = 0):
        super().__init__()
        assert n_experts >= 1, f"n_experts must be >= 1, got {n_experts}"
        self.head_dim = head_dim
        self.n_experts = int(n_experts)
        self.router_hidden = int(router_hidden)
        if router_hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(head_dim, router_hidden),
                nn.Tanh(),
                nn.Linear(router_hidden, self.n_experts),
            )
        else:
            self.net = nn.Linear(head_dim, self.n_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute softmax routing scores per sub-token.

        Args:
            x: [N, head_dim] sub-tokens.

        Returns:
            g: [N, n_experts] softmax probabilities (sum to 1 per row).
        """
        return F.softmax(self.net(x), dim=-1)


class MHMoECfCCell(nn.Module):
    """Multi-Head MoE cell: H sub-tokens × K experts, top-K per sub-token.

    Args:
        input_size: Input feature dimension D (must be divisible by n_heads).
        hidden_size: Hidden state dimension.
        n_experts: Number of experts K.
        n_heads: Number of sub-token heads H (default 2).
        top_k: Number of experts activated per sub-token (default 1).
        n_tau_per_expert: Per-expert ``n_tau`` (round 76 compatibility).
        tau_scales: Per-branch initial time constants.
        router_hidden: Router MLP width (``0`` = linear).

    Notes:
        - Each expert is a CfC cell with input/output dim = input_size // n_heads.
        - The hidden state h_t is shared across sub-tokens of a given timestep.
        - Sub-token outputs are CONCATENATED (not averaged) to form h_new.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_experts: int = 4,
        n_heads: int = 2,
        top_k: int = 1,
        n_tau_per_expert: int = 1,
        tau_scales: tuple = (0.1, 1.0, 10.0),
        router_hidden: int = 0,
    ):
        super().__init__()
        assert n_experts >= 1, f"n_experts must be >= 1, got {n_experts}"
        assert n_heads >= 1, f"n_heads must be >= 1, got {n_heads}"
        assert input_size % n_heads == 0, (
            f"input_size ({input_size}) must be divisible by n_heads ({n_heads})"
        )
        if n_experts > 0:
            assert 1 <= top_k <= n_experts, (
                f"top_k must be in [1, n_experts={n_experts}], got {top_k}"
            )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_experts = int(n_experts)
        self.n_heads = int(n_heads)
        self.top_k = int(top_k)
        self.n_tau_per_expert = int(n_tau_per_expert)
        self.tau_scales = tuple(tau_scales)
        self.router_hidden = int(router_hidden)
        self.head_dim = input_size // n_heads

        # Each expert is a CfC cell with sub-token dim = head_dim.
        # Output dim is hidden_size (not head_dim) so we can concatenate heads.
        # We project head_dim -> hidden_size in the expert.
        self.experts = nn.ModuleList(
            [
                CfCCell(
                    input_size=self.head_dim,
                    hidden_size=hidden_size,
                    n_tau=self.n_tau_per_expert,
                    tau_scales=self.tau_scales,
                )
                for _ in range(self.n_experts)
            ]
        )
        self.router = MHRouter(
            head_dim=self.head_dim,
            n_experts=self.n_experts,
            router_hidden=self.router_hidden,
        )

        # Diagnostic stash.
        self.last_g: torch.Tensor | None = None  # [B*H, K] per-sub-token routing probs
        self.last_top_idx: torch.Tensor | None = None  # [B*H, top_k]
        self.last_expert_util: torch.Tensor | None = None  # [K] fraction of sub-tokens routed to each

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """One step of MH-MoE cell.

        Args:
            x_t: [B, input_size] input at this step.
            h:   [B, hidden_size] previous hidden state.
            dt:  scalar or [B] per-sample time delta.

        Returns:
            h_new: [B, hidden_size] = concat over heads of routed expert outputs.
        """
        B, D = x_t.shape
        H = self.n_heads
        head_dim = self.head_dim

        # 1) Split input into H sub-tokens.
        sub_tokens = x_t.view(B, H, head_dim).reshape(B * H, head_dim)  # [B*H, head_dim]

        # 2) Per-sub-token routing.
        g = self.router(sub_tokens)  # [B*H, K]
        self.last_g = g.detach()
        top_vals, top_idx = g.topk(self.top_k, dim=-1)  # [B*H, top_k]
        self.last_top_idx = top_idx.detach()

        # Track expert utilization (fraction of sub-tokens routed to each expert
        # — counts each top-k selection).
        expert_counts = torch.zeros(self.n_experts, device=x_t.device, dtype=x_t.dtype)
        for k in range(self.top_k):
            expert_counts.scatter_add_(0, top_idx[:, k], torch.ones_like(top_idx[:, k], dtype=x_t.dtype))
        self.last_expert_util = (expert_counts / (B * H * self.top_k)).detach()

        # 3) Process each sub-token by all K experts (parallel forward).
        # Repeat h across H sub-tokens of the same timestep.
        h_repeat = h.unsqueeze(1).expand(-1, H, -1).reshape(B * H, self.hidden_size)  # [B*H, H]
        expert_outs = [
            expert(sub_tokens, h_repeat, dt=dt) for expert in self.experts
        ]  # K × [B*H, hidden_size]
        stacked = torch.stack(expert_outs, dim=1)  # [B*H, K, hidden_size]

        # Gather top-K expert outputs and weight by softmax probabilities.
        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, self.hidden_size)  # [B*H, top_k, hidden_size]
        selected = stacked.gather(1, gather_idx)  # [B*H, top_k, hidden_size]
        routed = (top_vals.unsqueeze(-1) * selected).sum(dim=1)  # [B*H, hidden_size]

        # 4) Concatenate sub-token outputs back to [B, hidden_size].
        # Note: routed already has hidden_size per sub-token (since each expert outputs hidden_size).
        h_new = routed.view(B, H, self.hidden_size).mean(dim=1)  # [B, hidden_size]
        return h_new


class MHMoECfCNetwork(nn.Module):
    """Stacked Multi-Head MoE CfC network.

    Mirrors the ``CfCNetwork`` / ``ReMoECfCNetwork`` API (return_sequences, mask,
    dt) but uses ``MHMoECfCCell`` for every layer.

    Args:
        input_size: Input feature dimension D (must be divisible by n_heads).
        hidden_size: Hidden state dimension.
        output_size: Output dimension.
        num_layers: Number of stacked MH-MoE cells.
        return_sequences: If True, return the full sequence; else last step.
        n_experts: Number of experts K per layer.
        n_heads: Number of sub-token heads H per layer.
        top_k: Number of experts activated per sub-token.
        n_tau_per_expert: Per-expert ``n_tau``.
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
        n_experts: int = 4,
        n_heads: int = 2,
        top_k: int = 1,
        n_tau_per_expert: int = 1,
        tau_scales: tuple = (0.1, 1.0, 10.0),
        router_hidden: int = 0,
    ):
        super().__init__()
        assert input_size % n_heads == 0, (
            f"input_size ({input_size}) must be divisible by n_heads ({n_heads})"
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.return_sequences = return_sequences
        self.n_experts = int(n_experts)
        self.n_heads = int(n_heads)
        self.top_k = int(top_k)
        self.n_tau_per_expert = int(n_tau_per_expert)
        self.tau_scales = tuple(tau_scales)
        self.router_hidden = int(router_hidden)

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            self.cells.append(
                MHMoECfCCell(
                    input_size=in_dim,
                    hidden_size=hidden_size,
                    n_experts=self.n_experts,
                    n_heads=self.n_heads,
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


def mhmoe_utilization(cell: MHMoECfCCell) -> dict:
    """Diagnostic for MH-MoE cell's expert utilization.

    Returns:
        Dict with:
            - "expert_util": [K] tensor of fraction of sub-tokens routed to each.
            - "expert_count": [K] raw count of sub-token → expert assignments.
            - "routing_entropy": scalar — entropy of expert_util (in nats).
    """
    if cell.last_expert_util is None:
        return {
            "expert_util": torch.zeros(cell.n_experts),
            "expert_count": torch.zeros(cell.n_experts),
            "routing_entropy": torch.tensor(0.0),
        }
    util = cell.last_expert_util.cpu()  # [K]
    eps = 1e-8
    # Entropy: -Σ p log p (in nats), max = log K for uniform
    entropy = -(util * torch.log(util + eps)).sum().cpu()
    return {
        "expert_util": util,
        "expert_count": (util * (cell.n_experts * 100)).long(),  # rough count
        "routing_entropy": entropy,
    }
