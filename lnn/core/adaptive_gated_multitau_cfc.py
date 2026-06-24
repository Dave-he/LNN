"""Adaptive Gated Multi-τ CfC (arXiv:2606.22801 response, round 243).

Reference: arXiv:2606.22801 "Learning Adaptive Dynamical Features via
Multi-$τ$ Liquid-Mamba for All-in-one Image Restoration" (June 2026).
The paper introduces two mechanisms on top of multi-τ liquid cells:

  1. **Adaptive τ (input-conditioned time constants)** — each branch has
     a learnable ``W_tau[i]``, ``b_tau[i]`` that modulate the effective
     τ as a function of the input:
         τ_i(x) = τ_base[i] * sigmoid(W_tau[i] · x + b_tau[i])

  2. **Gated branch fusion** — instead of concat / equal-weight, the
     branch outputs are combined via a softmax gate computed from x:
         gate = softmax(W_gate · x)
         output = sum_i gate[i] * branch_i_output

Round 76 introduced multi-τ CfC with static τ (a learned constant per
branch) and equal-weight fusion. This module ships the input-conditioned
extension so the model can adapt its time-scale response to local input
characteristics (cf. Mamba-Liquid paper's "fast-varying local details +
slowly evolving global structures").

API:
    AdaptiveGatedMultiTauCfCCell(input_size, hidden_size, n_tau=3,
                                tau_base=(0.1, 1.0, 10.0))

    forward(x_t, h) -> h_next
    forward_with_aux(x_t, h, gate_entropy_lambda) -> (h_next, aux_dict)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _extend_scales(scales: tuple, n: int) -> list[float]:
    """Extend ``scales`` to length ``n`` geometrically (×10 each step)."""
    out = list(scales)
    while len(out) < n:
        out.append(out[-1] * 10.0)
    return out[:n]


def gated_fusion_entropy(gate: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Average per-sample Shannon entropy of a softmax gate.

    Returns ``mean( - sum_i gate_i · log(gate_i + eps) )``. The maximum
    entropy of a K-class softmax is ``log K`` — a value close to that
    means the gate is using all branches roughly evenly.
    """
    return (-gate * (gate + eps).log()).sum(dim=-1).mean()


class AdaptiveGatedMultiTauCfCCell(nn.Module):
    """Multi-τ CfC with **input-conditioned** τ and **gated branch fusion**.

    Args:
        input_size: Input feature dimension.
        hidden_size: Hidden state dimension (split across ``n_tau`` branches).
        n_tau: Number of time-scale branches (default 3).
        tau_base: Initial τ per branch (geometric extension if shorter
            than ``n_tau``). Default ``(0.1, 1.0, 10.0)``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_tau: int = 3,
        tau_base: tuple = (0.1, 1.0, 10.0),
    ):
        super().__init__()
        assert n_tau >= 2, f"n_tau must be >= 2 for gating to be meaningful, got {n_tau}"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_tau = int(n_tau)

        scales = _extend_scales(tau_base, self.n_tau)
        self._tau_base = tuple(scales)

        base = hidden_size // self.n_tau
        rem = hidden_size - base * self.n_tau
        self._branch_dims = [
            base + (rem if i == self.n_tau - 1 else 0)
            for i in range(self.n_tau)
        ]

        self.f_gates = nn.ModuleList()
        self.g_branches = nn.ModuleList()
        self.h_branches = nn.ModuleList()
        for i in range(self.n_tau):
            d = self._branch_dims[i]
            self.f_gates.append(nn.Sequential(
                nn.Linear(input_size + hidden_size, d), nn.Sigmoid()))
            self.g_branches.append(nn.Sequential(
                nn.Linear(input_size + hidden_size, d), nn.Tanh()))
            self.h_branches.append(nn.Sequential(
                nn.Linear(input_size + hidden_size, d), nn.Tanh()))

        # Per-branch input-conditioned τ modulation. Use a small non-zero
        # init so the modulation is *testable* from a cold start (otherwise
        # sigmoid(W·x + b) collapses to 0.5 + 0.5·b for any input x).
        self.W_tau = nn.ParameterList([
            nn.Parameter(torch.randn(input_size) * 0.1) for _ in range(self.n_tau)
        ])
        self.b_tau = nn.ParameterList([
            nn.Parameter(torch.zeros(())) for _ in range(self.n_tau)
        ])

        # Branch-fusion gate (softmax over n_tau branches, computed from x).
        self.W_gate = nn.Linear(input_size, self.n_tau, bias=True)

        # Per-branch time scales (log-parameterised so we can multiply by sigmoid mask).
        self.log_tau = nn.ParameterList([
            nn.Parameter(torch.tensor(math.log(scales[i])))
            for i in range(self.n_tau)
        ])

    @property
    def tau_base(self) -> tuple[float, ...]:
        return self._tau_base

    def _per_branch_tau(self, x_t: torch.Tensor) -> torch.Tensor:
        """Return per-branch effective τ for each sample.

        Returns a tensor of shape ``(B, n_tau)`` with::

            τ_i(x) = exp(log_tau[i]) * sigmoid(W_tau[i] · x + b_tau[i])
        """
        # Modulation factors in (0, 1).
        mods = torch.stack([
            torch.sigmoid(x_t @ self.W_tau[i] + self.b_tau[i])
            for i in range(self.n_tau)
        ], dim=-1)  # (B, n_tau)
        # Effective τ.
        taus = torch.stack([
            torch.exp(self.log_tau[i]).expand(x_t.shape[0])
            for i in range(self.n_tau)
        ], dim=-1) * mods
        return taus

    def _fuse(self, branch_outputs: list[torch.Tensor], gate: torch.Tensor
               ) -> torch.Tensor:
        """Concatenate per-branch outputs and re-weight by gate.

        Each branch output has shape ``(B, d_i)``. We concat into a single
        ``(B, H)`` tensor then build a per-element weight by expanding
        ``gate`` along the hidden axis. The result is a *soft mixture* of
        branches (not a hard switch).
        """
        # branch_outputs are tensors of varying last dim; concat them.
        full = torch.cat(branch_outputs, dim=-1)  # (B, H)
        # Build per-element weights from per-branch gate (broadcast).
        # Weight[i] = gate[i] for the slice of dim covered by branch i.
        sizes = self._branch_dims
        elem_gate = torch.cat([
            gate[:, i:i + 1].expand(-1, sizes[i]) for i in range(self.n_tau)
        ], dim=-1)  # (B, H)
        return full * elem_gate

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        combined = torch.cat([x_t, h], dim=-1)
        taus = self._per_branch_tau(x_t)  # (B, n_tau)
        gate = torch.softmax(self.W_gate(x_t), dim=-1)  # (B, n_tau)
        branches = []
        for i in range(self.n_tau):
            f = self.f_gates[i](combined)
            g = self.g_branches[i](combined)
            h_out = self.h_branches[i](combined)
            decay = torch.sigmoid(-f * taus[:, i:i + 1] * dt)
            branches.append(decay * g + (1.0 - decay) * h_out)
        return self._fuse(branches, gate)

    def forward_with_aux(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
        gate_entropy_lambda: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """CfC step + auxiliary diagnostics.

        ``aux_dict`` contains:

        * ``"h_next"`` — output of the step
        * ``"tau_eff"`` — per-branch effective τ, shape ``(B, n_tau)``
        * ``"gate"`` — per-branch fusion gate, shape ``(B, n_tau)``
        * ``"gate_entropy"`` — always present (mean entropy)
        * ``"gate_loss_total"`` — only when ``gate_entropy_lambda > 0``
          (negative entropy = encouraging branch diversity)
        """
        combined = torch.cat([x_t, h], dim=-1)
        taus = self._per_branch_tau(x_t)  # (B, n_tau)
        gate = torch.softmax(self.W_gate(x_t), dim=-1)  # (B, n_tau)
        branches = []
        for i in range(self.n_tau):
            f = self.f_gates[i](combined)
            g = self.g_branches[i](combined)
            h_out = self.h_branches[i](combined)
            decay = torch.sigmoid(-f * taus[:, i:i + 1] * dt)
            branches.append(decay * g + (1.0 - decay) * h_out)
        h_next = self._fuse(branches, gate)
        ent = gated_fusion_entropy(gate)
        aux: dict[str, torch.Tensor] = {
            "h_next": h_next,
            "tau_eff": taus,
            "gate": gate,
            "gate_entropy": ent,
        }
        if gate_entropy_lambda > 0:
            aux["gate_loss_total"] = -gate_entropy_lambda * ent
        return h_next, aux


__all__ = [
    "AdaptiveGatedMultiTauCfCCell",
    "gated_fusion_entropy",
]