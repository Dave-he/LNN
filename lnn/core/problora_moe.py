"""ProbLoRA-MoE — Probabilistic Routing + LoRA-rank-r expert deltas (PRD #10-84, round 122, 2026-06-15).

A natural hybrid of two of the eight STRICTLY POSITIVE winners in
the 91-121 audit:

- **ProbMoE (round 121)**: probabilistic routing with marginal
  probability as the per-expert gating signal.  No straight-through
  estimator.
- **LoRA-MoRE (round 118)**: low-rank expert deltas (rank r) added
  to a shared base CfC, with B initialized to zero for warm start.

ProbLoRA-MoE keeps the LoRA-rank-r expert deltas (parameter efficient)
and swaps the FAME top-K router for a ProbMoE-style probabilistic
router.  This isolates the question: **does the routing mechanism
(probabilistic marginals vs softmax top-K) matter when the expert
family is already the parameter-efficient LoRA adapter?**

Key design choices (mirroring round 118 + 121):

* Sub-MLP experts → **LoRA adapters** (consistent with the 8 winners
  in 91-121 audit: 99, 102, 105, 107, 113, 114, 116, 118).
* Probabilistic router (3 modes: exact_k, sample, dynamic_k) —
  same ProbMoERouter API as round 121.
* Shared base CfC, B-initialized-to-zero adapters.
* Cardinality constraint honored (top_k experts per step).

This is the **19th STRUCTURAL** mechanism in the 91-122 audit and
the **2nd PROBABILISTIC ROUTING** mechanism (after round 121).

Forward pass (per step)::

    h_base  = base_cfc(x_t, h, dt)             # shared base
    g, top_idx, probs = prob_router(x_t, h)   # K marginal probs
    combined = [x_t; h]                        # [B, I+H]
    Δ_i = (alpha/r) * (combined @ A_i) @ B_i  # [B, H] per expert
    # gather top-K deltas, weight by g
    h_lora = sum_i g_i * Δ_top_i               # [B, H]
    h_new  = h_base + h_lora
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from lnn.core.cfc import CfCCell
from lnn.core.prob_moe import ProbMoERouter


# ---------------------------------------------------------------------------
# ProbLoRA expert
# ---------------------------------------------------------------------------


class ProbLoRAExpert(nn.Module):
    """Single low-rank adapter for ProbLoRA-MoE.

    Same as LoRAExpert (round 118) but the API is simplified for the
    hybrid here: the ProbMoE router handles selection, so the
    expert itself only computes its own delta.  B is initialized to
    zero (canonical LoRA warm-start) so the model starts identical
    to the base CfC.

    Args:
        in_features: Input dim (typically ``input_size + hidden_size``).
        out_features: Output dim (typically ``hidden_size``).
        rank: LoRA rank r.
        alpha: LoRA scaling alpha (effective scale = alpha / rank).
        dropout: Optional dropout on the intermediate hidden.
        small_init: If True, kaiming_uniform init for A and zeros for B.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        small_init: bool = True,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = alpha / max(rank, 1)
        self.dropout = nn.Dropout(p=float(dropout)) if dropout > 0 else nn.Identity()
        # LoRA A: down-projection, [in_features, rank]
        self.A = nn.Linear(self.in_features, self.rank, bias=False)
        # LoRA B: up-projection, [rank, out_features], zero-initialized
        self.B = nn.Linear(self.rank, self.out_features, bias=False)
        if small_init:
            nn.init.kaiming_uniform_(self.A.weight, a=5 ** 0.5)
            nn.init.zeros_(self.B.weight)
        self.small_init = bool(small_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_features] -> delta: [B, out_features]"""
        return self.scale * self.B(self.dropout(self.A(x)))


# ---------------------------------------------------------------------------
# Cell + Network
# ---------------------------------------------------------------------------


class ProbLoRACfCCell(nn.Module):
    """ProbLoRA-MoE CfC cell.

    Structure:
        base_cfc : shared base CfC (CfCCell)
        experts  : K low-rank LoRA adapters (ProbLoRAExpert)
        router   : ProbMoERouter (3 modes)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_experts: int = 3,
        top_k: int = 2,
        rank: int = 4,
        alpha: float = 1.0,
        temperature: float = 1.0,
        mode: str = "exact_k",
        lora_dropout: float = 0.0,
        small_init: bool = True,
    ):
        super().__init__()
        if top_k > n_experts:
            raise ValueError(f"top_k={top_k} cannot exceed n_experts={n_experts}")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.n_experts = int(n_experts)
        self.top_k = int(top_k)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.mode = mode
        self.adapter_dim = self.input_size + self.hidden_size

        # Shared base CfC
        self.base_cfc = CfCCell(input_size, hidden_size)
        # K low-rank LoRA adapters
        self.experts = nn.ModuleList(
            [
                ProbLoRAExpert(
                    in_features=self.adapter_dim,
                    out_features=self.hidden_size,
                    rank=self.rank,
                    alpha=self.alpha,
                    dropout=lora_dropout,
                    small_init=small_init,
                )
                for _ in range(self.n_experts)
            ]
        )
        # Probabilistic router
        self.router = ProbMoERouter(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            n_experts=self.n_experts,
            top_k=self.top_k,
            temperature=self.temperature,
        )

        # Diagnostic stash
        self.last_g: torch.Tensor | None = None
        self.last_top_idx: torch.Tensor | None = None

    def forward(self, x_t: torch.Tensor, h: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """One step forward.

        Parameters
        ----------
        x_t : [B, I]
        h : [B, H]
        dt : float

        Returns
        -------
        h_new : [B, H]
        """
        B = x_t.size(0)
        # Base CfC
        h_base = self.base_cfc(x_t, h, dt=dt)
        # Routing
        g, top_idx, probs = self.router(x_t, h, mode=self.mode)
        # Compute all K expert deltas
        combined = torch.cat([x_t, h], dim=-1)  # [B, I+H]
        all_deltas = torch.stack(
            [expert(combined) for expert in self.experts],
            dim=1,
        )  # [B, K, H]
        # Gather top-K deltas
        gather_idx = top_idx.unsqueeze(-1).expand(B, self.top_k, self.hidden_size)
        selected_deltas = all_deltas.gather(1, gather_idx)  # [B, k, H]
        # Weight by routing gates
        h_lora = (g.unsqueeze(-1) * selected_deltas).sum(dim=1)  # [B, H]
        h_new = h_base + h_lora
        # Stash for diagnostics
        self.last_g = g.detach()
        self.last_top_idx = top_idx.detach()
        return h_new

    def forward_with_aux(self, x_t: torch.Tensor, h: torch.Tensor, dt: float = 1.0):
        """Forward that also returns the routing distribution (for diagnostics)."""
        B = x_t.size(0)
        h_base = self.base_cfc(x_t, h, dt=dt)
        g, top_idx, probs = self.router(x_t, h, mode=self.mode)
        combined = torch.cat([x_t, h], dim=-1)
        all_deltas = torch.stack(
            [expert(combined) for expert in self.experts],
            dim=1,
        )
        gather_idx = top_idx.unsqueeze(-1).expand(B, self.top_k, self.hidden_size)
        selected_deltas = all_deltas.gather(1, gather_idx)
        h_lora = (g.unsqueeze(-1) * selected_deltas).sum(dim=1)
        h_new = h_base + h_lora
        self.last_g = g.detach()
        self.last_top_idx = top_idx.detach()
        return h_new, {
            "router_probs": probs,
            "router_g": g,
            "router_top_idx": top_idx,
            "all_deltas": all_deltas,
            "h_base": h_base,
        }


class ProbLoRACfCNetwork(nn.Module):
    """Stacked ProbLoRA-MoE CfC network."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        return_sequences: bool = True,
        n_experts: int = 3,
        top_k: int = 2,
        rank: int = 4,
        alpha: float = 1.0,
        temperature: float = 1.0,
        mode: str = "exact_k",
        lora_dropout: float = 0.0,
        small_init: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.return_sequences = return_sequences
        self.cells = nn.ModuleList()
        for layer_idx in range(num_layers):
            layer_in = input_size if layer_idx == 0 else hidden_size
            self.cells.append(
                ProbLoRACfCCell(
                    input_size=layer_in,
                    hidden_size=hidden_size,
                    n_experts=n_experts,
                    top_k=top_k,
                    rank=rank,
                    alpha=alpha,
                    temperature=temperature,
                    mode=mode,
                    lora_dropout=lora_dropout,
                    small_init=small_init,
                )
            )
        # Output head
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """Forward over a sequence.

        Parameters
        ----------
        x : [B, T, I]  input sequence (may contain NaN)
        dt : float

        Returns
        -------
        out : [B, T, O] if return_sequences else [B, O]
        """
        x = torch.nan_to_num(x, nan=0.0)
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(T):
            x_t = x[:, t, :]
            layer_in = x_t
            for cell in self.cells:
                h = cell(layer_in, h, dt=dt)
                layer_in = h
            out_t = self.head(h)
            outputs.append(out_t)
        outputs = torch.stack(outputs, dim=1)
        if self.return_sequences:
            return outputs
        return outputs[:, -1, :]


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def problora_moe_utilization(cell: ProbLoRACfCCell) -> dict:
    """Diagnostic for a ProbLoRA-MoE cell's expert utilization.

    Returns
    -------
    dict with keys:
        n_experts, top_k, rank, mode, temperature
        n_params, n_router_params, n_expert_params, n_base_params
    """
    n_total = sum(p.numel() for p in cell.parameters())
    n_router = sum(p.numel() for p in cell.router.parameters())
    n_experts = sum(p.numel() for p in cell.experts.parameters())
    n_base = sum(p.numel() for p in cell.base_cfc.parameters())
    return {
        "n_experts": cell.n_experts,
        "top_k": cell.top_k,
        "rank": cell.rank,
        "mode": cell.mode,
        "temperature": cell.temperature,
        "n_params": n_total,
        "n_router_params": n_router,
        "n_expert_params": n_experts,
        "n_base_params": n_base,
    }
