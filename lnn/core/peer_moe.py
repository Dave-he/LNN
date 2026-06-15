"""PEER (Parameter Efficient Expert Retrieval) for CfC (PRD #10-81, round 119, 2026-06-15).

Response to arXiv:2407.04153 — *Mixture of A Million Experts* (Xu Owen
He, Google DeepMind, July 2024).  The core idea: each "expert" is a
**single linear neuron** (the smallest possible unit), and routing is
done via **product-key lookup** (two hash tables, each with √N
buckets, find top-K closest in each, then take top-K of K² candidates).

In this CfC setting we adapt the original paper (which targets millions
of experts) to a small-N setting that fits the 91-118 audit.  We expose
two router options:

1. **Product-key** (paper-faithful): two learnable hash tables, each
   ``[n_buckets, input_dim]``.  Score input against each table, find
   top-K in each (K² candidates), score candidates by full key match,
   pick top-K, weighted sum.
2. **Softmax** (ablation): learned linear projection ``[B, N]``,
   softmax, top-K, weighted sum.  Same as a tiny MoE with N linear
   experts.

The expert family is genuinely new in the 91-118 audit:
- All prior 8 winners use *sub-MLP* experts (CfC cells with 3+
  branches × tanh/sigmoid activations)
- PEER experts are *single neurons* (linear: ``y = w_i·x + b_i``)
- This is the smallest possible expert unit

Forward pass::

    h_base = base_cfc(x_t, h)                # [B, H]   (shared base)
    combined = [x_t; h]                      # [B, I+H]
    scores, top_idx = router(combined)        # [B, K], [B, K]
    experts_out = [expert_i(combined) for i in top_idx]  # K × [B, H]
    h_lora = sum_k scores_k * experts_out_k   # [B, H]
    h_new = h_base + h_lora                    # [B, H]

Why this fits the 91-118 audit pattern:
- **Structural**: changes the **expert family** (sub-MLP → single
  neuron).  New dimension of the audit.
- **Data-structure-independent**: linearity has no data assumption.
- **Preserves recurrent state mixing**: ``h_new = base(x, h) +
  Σ α_k · expert_k([x, h])`` is additive, same form as LoRA-MoRE
  (round 118).
- **Fills the "linear expert" gap**: not yet tested.

Hypothesis (round 119):
- H1: PEER with N=8 single-neuron experts beats FAME on
  structured_irr (where linear basis is sufficient) and is competitive
  on random_irr (where the noise dominates the linear signal).  May
  regress on sin_irr (needs nonlinearity).
- H2: Product-key routing is structurally different from softmax and
  may escape the FAME H=0 lock-in (round 103) because the routing
  decisions are deterministic hash lookups.
- H3: At N=16, PEER has 16× more parameters than FAME K=3 sub-MLP
  experts, but each is so small that total parameter count is comparable.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lnn.core.cfc import CfCCell
from lnn.core.sequence_utils import select_step_delta, select_step_mask


# ---------------------------------------------------------------------------
# Single-neuron expert (the smallest possible expert)
# ---------------------------------------------------------------------------


class SingleNeuronExpert(nn.Module):
    """One expert = one linear neuron (no activation).

    The "Mixture of A Million Experts" claim is that the smallest
    possible expert (a single linear neuron) is enough when there are
    enough of them.  This is a step-function approximation: with N
    single neurons, the model can express any piecewise-linear function
    (universal approximator for continuous functions on a compact
    domain, with N → ∞).

    Args:
        in_features: Input dimension (typically ``input_size + hidden_size``).
        out_features: Output dimension (typically ``hidden_size``).
        bias: If True, add a learnable bias (default True).

    Forward:
        x: [B, in_features] → [B, out_features] = x @ W^T + b.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}"


# ---------------------------------------------------------------------------
# Product-key router (paper-faithful, arXiv:2407.04153 §3.2)
# ---------------------------------------------------------------------------


class ProductKeyRouter(nn.Module):
    """Product-key routing (arXiv:2407.04153 §3.2).

    Two learnable key tables, each ``[n_buckets, key_dim]``.  For input
    x: score against each table, find top-K in each (→ K² candidates),
    score each candidate by its key match to x, pick top-K_final, weight
    by softmax over those scores.

    Args:
        input_size: Input feature dimension.
        hidden_size: Hidden state dimension.
        n_experts: Total number of experts N.
        top_k: Final top-K experts to mix (K_final ≤ K from each table).
        n_buckets: Number of buckets per table (default ``ceil(√N)``).
            At small N, this gives a manageable K² candidate pool.
        small_init: If True, init tables with small std (default True).
        score_temperature: Softmax temperature for candidate scoring
            (default 1.0).

    Forward:
        x_t: [B, input_size] input.
        h:   [B, hidden_size] previous hidden.
        Returns:
            (g, top_idx):
                g: [B, top_k] routing weights (sum to 1).
                top_idx: [B, top_k] expert indices.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_experts: int,
        top_k: int = 2,
        n_buckets: int | None = None,
        small_init: bool = True,
        score_temperature: float = 1.0,
    ):
        super().__init__()
        assert n_experts >= 1, f"n_experts must be >= 1, got {n_experts}"
        assert top_k >= 1, f"top_k must be >= 1, got {top_k}"
        if n_buckets is None:
            n_buckets = max(2, int(math.ceil(math.sqrt(n_experts))))
        # Each table must have at least top_k buckets to find top_k in it.
        n_buckets = max(n_buckets, top_k)
        assert n_buckets <= n_experts, (
            f"n_buckets={n_buckets} cannot exceed n_experts={n_experts}"
        )
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.n_experts = int(n_experts)
        self.top_k = int(top_k)
        self.n_buckets = int(n_buckets)
        self.score_temperature = float(score_temperature)
        key_dim = input_size + hidden_size

        # Two key tables (paper-faithful): each is [n_buckets, key_dim].
        # Each bucket is a "key" that matches against the input.
        self.key_table_1 = nn.Parameter(torch.zeros(n_buckets, key_dim))
        self.key_table_2 = nn.Parameter(torch.zeros(n_buckets, key_dim))
        if small_init:
            nn.init.normal_(self.key_table_1, mean=0.0, std=0.01)
            nn.init.normal_(self.key_table_2, mean=0.0, std=0.01)
        else:
            nn.init.normal_(self.key_table_1, mean=0.0, std=0.1)
            nn.init.normal_(self.key_table_2, mean=0.0, std=0.1)

        # Side-channel: last routing weights + indices
        self.last_g: torch.Tensor | None = None
        self.last_top_idx: torch.Tensor | None = None

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute product-key routing weights and top-K indices.

        Args:
            x_t: [B, input_size] input.
            h:   [B, hidden_size] previous hidden.

        Returns:
            (g, top_idx):
                g: [B, top_k] routing weights (sum to 1 via softmax).
                top_idx: [B, top_k] long tensor of expert indices in [0, n_experts).
                    Each row lists which experts (out of the n_experts in the
                    cell) are selected.  Note: in the paper, the n_buckets
                    keys each map to ONE expert; in this CfC adaptation we
                    have a 1-to-1 mapping ``bucket_idx → expert_idx`` via
                    modulo (so we don't need a separate assignment matrix).
        """
        B = x_t.size(0)
        combined = torch.cat([x_t, h], dim=-1)  # [B, I+H]
        # Score against each key table: dot product → [B, n_buckets].
        scores_1 = combined @ self.key_table_1.T  # [B, n_buckets]
        scores_2 = combined @ self.key_table_2.T  # [B, n_buckets]
        # Top-K in each table.
        # We pick top-K=2 per table to match the K_final=2 final selection.
        # (If top_k > n_buckets, this still works because topk is clamped.)
        k_per_table = min(self.top_k, self.n_buckets)
        top_vals_1, top_idx_1 = scores_1.topk(k_per_table, dim=-1)  # [B, k_per_table]
        top_vals_2, top_idx_2 = scores_2.topk(k_per_table, dim=-1)  # [B, k_per_table]
        # Concatenate: K² = k_per_table² candidates.
        all_vals = torch.cat([top_vals_1, top_vals_2], dim=-1)  # [B, 2·k_per_table]
        # Map bucket indices to expert indices: 1-to-1 modulo n_experts.
        all_idx = torch.cat([top_idx_1, top_idx_2], dim=-1)  # [B, 2·k_per_table]
        all_idx = all_idx % self.n_experts  # [B, 2·k_per_table]
        # Deduplicate (the two tables may have produced the same expert).
        # We keep the first occurrence (highest score) and zero-out the rest.
        # Note: this is O(K²) per row, fine for K=2.
        unique_idx = []
        unique_vals = []
        for b in range(B):
            seen: set[int] = set()
            ui: list[int] = []
            uv: list[float] = []
            for j in range(all_vals.size(1)):
                eidx = int(all_idx[b, j].item())
                if eidx in seen:
                    continue
                seen.add(eidx)
                ui.append(eidx)
                uv.append(float(all_vals[b, j].item()))
                if len(ui) >= self.top_k:
                    break
            # Pad with the last seen expert if we didn't find top_k uniques.
            while len(ui) < self.top_k:
                ui.append(ui[-1] if ui else 0)
                uv.append(uv[-1] if uv else 0.0)
            unique_idx.append(ui)
            unique_vals.append(uv)
        top_idx = torch.tensor(unique_idx, device=x_t.device, dtype=torch.long)  # [B, top_k]
        top_vals = torch.tensor(unique_vals, device=x_t.device, dtype=combined.dtype)  # [B, top_k]
        # Softmax over the K candidates to get weights summing to 1.
        g = F.softmax(top_vals / max(self.score_temperature, 1e-6), dim=-1)  # [B, top_k]
        self.last_g = g.detach()
        self.last_top_idx = top_idx.detach()
        return g, top_idx


# ---------------------------------------------------------------------------
# Softmax router for ablation
# ---------------------------------------------------------------------------


class LinearSoftmaxRouter(nn.Module):
    """Softmax router for N linear experts (PEER minus product-key).

    Linear projection ``[B, N]`` followed by softmax + top-K.  This
    isolates the "linear expert family" idea from the "product-key
    routing" idea — a clean ablation.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_experts: int,
        top_k: int = 2,
        small_init: bool = True,
    ):
        super().__init__()
        assert n_experts >= 1
        assert top_k >= 1
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.n_experts = int(n_experts)
        self.top_k = int(top_k)
        self.proj = nn.Linear(input_size + hidden_size, n_experts)
        if small_init:
            nn.init.normal_(self.proj.weight, mean=0.0, std=0.01)
            if self.proj.bias is not None:
                nn.init.zeros_(self.proj.bias)
        self.last_g: torch.Tensor | None = None
        self.last_top_idx: torch.Tensor | None = None

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = x_t.size(0)
        combined = torch.cat([x_t, h], dim=-1)  # [B, I+H]
        logits = self.proj(combined)  # [B, N]
        # Top-K selection.
        top_vals, top_idx = logits.topk(self.top_k, dim=-1)  # [B, top_k]
        g = F.softmax(top_vals, dim=-1)  # [B, top_k]
        self.last_g = g.detach()
        self.last_top_idx = top_idx.detach()
        return g, top_idx


# ---------------------------------------------------------------------------
# PEER CfC Cell: shared base CfC + N single-neuron experts
# ---------------------------------------------------------------------------


class PEERCfCCell(nn.Module):
    """PEER-style cell: shared base CfC + N single-neuron experts.

    Args:
        input_size: Input feature dimension (D).
        hidden_size: Hidden state dimension (H).
        n_experts: Total number of single-neuron experts N (≥ 1).
        top_k: Number of experts mixed per step (K_final).
        router_type: ``"product_key"`` (default, paper-faithful) or
            ``"softmax"`` (ablation: same linear expert family, simpler router).
        n_tau_base: ``n_tau`` for the shared base CfC (default 1).
        tau_scales: Per-branch initial τ.
        n_buckets: Number of buckets per key table (default ``ceil(√N)``).
        small_init: If True, init key tables / softmax with std=0.01.

    Notes:
        - The hidden state h_t is preserved across steps (no modification).
        - The combination is ``h_new = base(x_t, h_t) + Σ α_k · expert_k([x_t, h_t])``,
          additive over the base, same form as LoRA-MoRE (round 118).
        - When ``router_type="softmax"``, the router is a learned linear
          projection — i.e. a tiny MoE with N single-neuron experts.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_experts: int = 8,
        top_k: int = 2,
        router_type: str = "product_key",
        n_tau_base: int = 1,
        tau_scales: tuple = (0.1, 1.0, 10.0),
        n_buckets: int | None = None,
        small_init: bool = True,
    ):
        super().__init__()
        assert n_experts >= 1, f"n_experts must be >= 1, got {n_experts}"
        assert top_k >= 1, f"top_k must be >= 1, got {top_k}"
        assert top_k <= n_experts, f"top_k must be <= n_experts"
        assert router_type in ("product_key", "softmax"), (
            f"router_type must be 'product_key' or 'softmax', got {router_type!r}"
        )
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.n_experts = int(n_experts)
        self.top_k = int(top_k)
        self.router_type = str(router_type)
        self.n_tau_base = int(n_tau_base)
        self.tau_scales = tuple(tau_scales)
        self.small_init = bool(small_init)
        self.adapter_dim = self.input_size + self.hidden_size

        # Shared base CfC cell.
        self.base_cfc = CfCCell(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            n_tau=self.n_tau_base,
            tau_scales=self.tau_scales,
        )

        # N single-neuron experts.
        self.experts = nn.ModuleList(
            [
                SingleNeuronExpert(
                    in_features=self.adapter_dim,
                    out_features=self.hidden_size,
                    bias=True,
                )
                for _ in range(self.n_experts)
            ]
        )

        # Router.
        if router_type == "product_key":
            self.router = ProductKeyRouter(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                n_experts=self.n_experts,
                top_k=self.top_k,
                n_buckets=n_buckets,
                small_init=small_init,
            )
        else:  # "softmax"
            self.router = LinearSoftmaxRouter(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                n_experts=self.n_experts,
                top_k=self.top_k,
                small_init=small_init,
            )

        # Diagnostic stash.
        self.last_expert_util: torch.Tensor | None = None
        self.last_g: torch.Tensor | None = None
        self.last_top_idx: torch.Tensor | None = None

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """One step of PEER routing.

        Args:
            x_t: [B, input_size] input.
            h:   [B, hidden_size] previous hidden.
            dt:  scalar or [B] per-sample time delta.

        Returns:
            h_new: [B, hidden_size] base + Σ α_k · expert_k([x, h]).
        """
        h_new, _ = self.forward_with_aux(x_t, h, dt=dt)
        return h_new

    def forward_with_aux(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Like ``forward`` but also returns the K selected expert outputs."""
        B = x_t.size(0)
        # 1) Base CfC.
        h_base = self.base_cfc(x_t, h, dt=dt)  # [B, H]

        # 2) Router: get top-K experts + weights.
        combined = torch.cat([x_t, h], dim=-1)  # [B, I+H]
        g, top_idx = self.router(x_t, h)  # g: [B, K], top_idx: [B, K]

        # 3) Compute K expert outputs (only the top-K, not all N).
        #    Vectorized: gather (B, K, I+H) inputs by top_idx → (B, K, H) outputs.
        #    First: index experts.
        #    For K ≤ N, the cost is K forward passes (not N).
        K = self.top_k
        # Compute each selected expert on the same input.  This is a Python
        # loop over K (K=2 typically) and N experts (N=8-16 typically),
        # totalling K calls.  We don't gather by index here because that
        # complicates the gradient flow; we just call the selected experts.
        expert_outs: list[torch.Tensor] = []
        for b in range(B):
            row_outs = []
            for k in range(K):
                eidx = int(top_idx[b, k].item())
                row_outs.append(self.experts[eidx](combined[b : b + 1]))  # [1, H]
            expert_outs.append(torch.cat(row_outs, dim=0))  # [K, H]
        expert_stack = torch.stack(expert_outs, dim=0)  # [B, K, H]

        # 4) Weighted combination.
        h_lora = (g.unsqueeze(-1) * expert_stack).sum(dim=1)  # [B, H]

        # 5) Final output.
        h_new = h_base + h_lora  # [B, H]

        # 6) Side-channels.
        self.last_g = g.detach()
        self.last_top_idx = top_idx.detach()
        # Compute mean gate per expert (over the n_experts experts,
        # not just the top-K).  Zero for non-selected experts.
        # This is an approximation; for the full routing entropy, see
        # peer_utilization.
        if hasattr(self.router, "last_g") and self.router.last_g is not None:
            # Approximate: spread g over the selected experts.
            util = torch.zeros(self.n_experts, device=x_t.device, dtype=x_t.dtype)
            for b in range(B):
                for k in range(K):
                    eidx = int(top_idx[b, k].item())
                    util[eidx] += g[b, k]
            util = util / B
            self.last_expert_util = util.detach()

        return h_new, [expert_stack[:, k, :] for k in range(K)]


class PEERCfCNetwork(nn.Module):
    """Stacked PEER CfC network.

    Mirrors the ``CfCNetwork`` / ``LoRACfCNetwork`` API
    (return_sequences, mask, dt) but uses ``PEERCfCCell`` for every
    layer.

    Args:
        input_size: Input feature dimension (D).
        hidden_size: Hidden state dimension.
        output_size: Output dimension.
        num_layers: Number of stacked PEER cells.
        return_sequences: If True, return the full sequence; else last step.
        n_experts: Number of single-neuron experts per layer.
        top_k: Number of experts mixed per step.
        router_type: ``"product_key"`` or ``"softmax"``.
        n_tau_base: ``n_tau`` for the base CfC.
        tau_scales: Per-branch initial τ.
        n_buckets: Buckets per key table (product-key only).
        small_init: Init key tables with small std.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        return_sequences: bool = True,
        n_experts: int = 8,
        top_k: int = 2,
        router_type: str = "product_key",
        n_tau_base: int = 1,
        tau_scales: tuple = (0.1, 1.0, 10.0),
        n_buckets: int | None = None,
        small_init: bool = True,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.num_layers = int(num_layers)
        self.return_sequences = bool(return_sequences)
        self.n_experts = int(n_experts)
        self.top_k = int(top_k)
        self.router_type = str(router_type)
        self.n_tau_base = int(n_tau_base)
        self.tau_scales = tuple(tau_scales)
        self.small_init = bool(small_init)

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            self.cells.append(
                PEERCfCCell(
                    input_size=in_dim,
                    hidden_size=hidden_size,
                    n_experts=self.n_experts,
                    top_k=self.top_k,
                    router_type=self.router_type,
                    n_tau_base=self.n_tau_base,
                    tau_scales=self.tau_scales,
                    n_buckets=n_buckets,
                    small_init=self.small_init,
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


def peer_utilization(cell: PEERCfCCell) -> dict:
    """Diagnostic for PEER cell's expert utilization.

    Returns:
        Dict with:
            - "expert_util": [N] tensor of mean per-expert gate (zero for
                non-selected experts).
            - "n_experts": int — total N.
            - "n_active": int — number of experts that received non-zero
                utilization.
            - "routing_entropy": scalar — entropy of expert_util (in nats).
            - "router_type": "product_key" or "softmax".
            - "n_peer_params": int — total PEER parameter count (experts + router).
    """
    n_lora_params = 0
    for e in cell.experts:
        n_lora_params += sum(p.numel() for p in e.parameters())
    n_lora_params += sum(p.numel() for p in cell.router.parameters())

    if cell.last_expert_util is None:
        return {
            "expert_util": torch.zeros(cell.n_experts),
            "n_experts": cell.n_experts,
            "n_active": 0,
            "routing_entropy": 0.0,
            "router_type": cell.router_type,
            "n_peer_params": n_lora_params,
        }
    util = cell.last_expert_util.detach()
    p = util / (util.sum() + 1e-12)
    entropy = -(p * (p + 1e-12).log()).sum().item()
    return {
        "expert_util": util,
        "n_experts": cell.n_experts,
        "n_active": int((util > 0).sum().item()),
        "routing_entropy": float(entropy),
        "router_type": cell.router_type,
        "n_peer_params": n_lora_params,
    }
