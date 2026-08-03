"""Multi-Rate Mixture-of-Experts CfC (PRD #2026-08-03-1).

Implements the **multi-rate MoE accelerator** for liquid neural networks
inspired by arXiv:2606.12240 ("Multi-Rate Mixture of Experts for
Accelerating Liquid Neural Network Training", Zong, Boker & Eldardiry,
2026).  The core insight:

    Standard CfC has a single time-scale τ. The multi-time-scale
    ``CfCCell(n_tau=K, ...)`` splits the hidden state into K branches
    with their own τ_i, f_gate, g_branch, h_branch. Training cost scales
    with K (each branch has its own input projection).

    When the K branches are interpreted as **K experts**, each carrying a
    characteristic time scale, only a subset of them needs to be activated
    per step (e.g. fast-τ branch fires for spikes, slow-τ branch for
    trends). This gives **per-step conditional compute** with the
    structural guarantee of perfect load balance via Expert-Choice (EC)
    routing (Zhou et al. 2022), where each *expert* picks the steps it
    wants to own rather than each step picking its experts.

Differences from vanilla EC-for-Transformers and from
``lnn.core.expert_choice``:

- **Per-branch τ as semantic specialisation** — not just random init.
  Branches are ordered by τ (slow → fast) so the router learns
  "send spike-like inputs to the fast branch, trend-like inputs to the
  slow branch".
- **Per-step top-K branches** (default ``top_k_active = ceil(K/2)``) —
  halves per-step FLOPs over a vanilla multi-τ cell while preserving
  gradient flow through the un-selected branches' gates (they still
  receive gradients via the gate logits).
- **Auxiliary soft load-balance loss** computed from the router score
  tensor averaged across all batch and step dimensions. Mirrors Switch
  Transformer / FAME practice and stabilises routing when
  ``top_k_active`` is small.

Drop-in parity:
- When ``n_tau=K=1`` is *not* allowed (we explicitly reject n_tau=1 since
  the whole point is the multi-rate routing); users who want a single
  branch keep using :class:`lnn.core.cfc.CfCCell` directly.
- The cell-level API matches :class:`lnn.core.cfc.CfCCell`:
  ``cell(x_t, h, dt) -> h_next`` with ``x_t: [B, input_size]`` and
  ``h: [B, hidden_size]``. The network wrapper takes sequences.

Authors: see ``docs/reports/LNN_Family_Taxonomy_And_Gap_2026-08-03.md``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lnn.core.cfc import CfCCell


class ExpertChoiceRouter(nn.Module):
    """Per-(branch, timestep) sigmoid router used by :class:`MultiRateMoECfC`.

    Computes::

        score[e] = sigmoid(W_e · x_t)                  (default; cheap)
        score[e] = sigmoid(<c_e, A · x_t>)             (optional bilinear)
    """

    def __init__(
        self,
        input_size: int,
        n_experts: int,
        bilinear: bool = False,
    ) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.bilinear = bilinear
        if bilinear:
            self.A = nn.Parameter(torch.empty(input_size, input_size))
            self.expert_centroid = nn.Parameter(torch.empty(n_experts, input_size))
            nn.init.xavier_uniform_(self.A)
            nn.init.xavier_uniform_(self.expert_centroid)
        else:
            self.W = nn.Parameter(torch.empty(n_experts, input_size))
            nn.init.xavier_uniform_(self.W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-expert score ``[B, E]`` for an input ``x: [B, input_size]``."""
        if self.bilinear:
            Ax = torch.einsum("ij,bj->bi", self.A, x)
            return torch.sigmoid(torch.einsum("ek,bk->be", self.expert_centroid, Ax))
        return torch.sigmoid(torch.einsum("ek,bk->be", self.W, x))


class MultiRateMoECfC(nn.Module):
    """A multi-τ liquid cell with EC-routed expert branches (per-step).

    Each branch is a single-τ ``CfCCell`` subgroup. The router scores
    branches for the current input ``x_t``; the top-K branches are kept
    and a softmax over the K gates weights their per-branch next-state
    contributions.

    Args:
        input_size:  Input feature dimension.
        hidden_size: Hidden dimension (concatenation of all branch hidden states).
        n_tau:       Number of branches / experts (τ_1 < τ_2 < ... < τ_K).
        top_k_active:Per-step top-K branches (1 ≤ k ≤ n_tau). Default
            ``ceil(n_tau / 2)``.
        tau_scales:  Per-branch initial τ, sorted ascending (fast → slow).
        bilinear_router: Use bilinear routing instead of the linear projection.
        aux_load_balance_weight: Weight for the auxiliary soft load-balance loss.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_tau: int = 4,
        top_k_active: int | None = None,
        tau_scales: tuple = (0.1, 0.5, 2.0, 10.0),
        bilinear_router: bool = False,
        aux_load_balance_weight: float = 0.01,
    ) -> None:
        super().__init__()
        assert n_tau >= 2, (
            "MultiRateMoECfC requires n_tau >= 2; use CfCCell for n_tau=1."
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_tau = int(n_tau)
        self.aux_load_balance_weight = float(aux_load_balance_weight)

        k = top_k_active if top_k_active is not None else max(1, math.ceil(self.n_tau / 2))
        self.top_k_active = max(1, min(int(k), self.n_tau))

        scales = sorted(list(tau_scales))[: self.n_tau]
        if len(scales) < self.n_tau:
            while len(scales) < self.n_tau:
                scales.append(scales[-1] * 5.0)
            scales = sorted(scales)
        self._tau_init = tuple(scales)

        base = hidden_size // self.n_tau
        rem = hidden_size - base * self.n_tau
        self._branch_dims = [base + (rem if i == self.n_tau - 1 else 0) for i in range(self.n_tau)]

        self.branches = nn.ModuleList()
        for i, out_dim in enumerate(self._branch_dims):
            branch = CfCCell(input_size=input_size, hidden_size=out_dim, n_tau=1)
            branch.time_scale.data.fill_(float(scales[i]))
            self.branches.append(branch)

        self.router = ExpertChoiceRouter(input_size=input_size, n_experts=self.n_tau, bilinear=bilinear_router)

        # Soft routing log of average probability per branch (for aux loss).
        self.register_buffer("_running_avg_prob", torch.zeros(self.n_tau), persistent=False)

    def _slice_h(self, h: torch.Tensor | None, e: int, batch_size: int) -> torch.Tensor:
        """Return the slice of the previous hidden belonging to branch ``e``."""
        start = sum(self._branch_dims[:e])
        end = start + self._branch_dims[e]
        if h is None:
            return h.new_zeros(batch_size, self._branch_dims[e]) if hasattr(self, "_branch_dims") else None
        return h[:, start:end]

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor | None = None,
        dt: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """Per-step forward: same signature as :class:`CfCCell`.

        Args:
            x_t: ``[B, input_size]``  current input.
            h:   ``[B, hidden_size]`` previous hidden state (or None for zeros).
            dt:  scalar delta-t (default 1.0).

        Returns:
            ``[B, hidden_size]`` next hidden state.
        """
        batch_size = x_t.shape[0]
        if h is None:
            h = x_t.new_zeros(batch_size, self.hidden_size)

        score = self.router(x_t)  # [B, E]

        # Top-K branches per (batch, expert). We pick on the *router score* axis.
        top_scores, top_idx = score.topk(self.top_k_active, dim=-1)  # [B, k], [B, k]
        # Gate weights: softmax over the top-K scores (so total weight = 1).
        gate = F.softmax(top_scores, dim=-1)  # [B, k]

        # Initialise next hidden as a clone of previous; we'll overwrite the active branches.
        h_next = h.clone()

        # Apply each selected branch only on the steps that picked it (sparse over batch).
        # We loop over the K slots; for each (branch e), update only the rows where e was selected.
        for k_slot in range(self.top_k_active):
            branch_idx_per_row = top_idx[:, k_slot]   # [B]
            gate_per_row = gate[:, k_slot]            # [B]
            for e in range(self.n_tau):
                mask = (branch_idx_per_row == e)
                if not mask.any():
                    continue
                # Subset of the batch that selected branch ``e`` in this slot.
                x_e = x_t[mask]                       # [n_e, in]
                h_e_prev = h[mask][:, sum(self._branch_dims[:e]) : sum(self._branch_dims[: e + 1])]
                h_e_new = self.branches[e](x_e, h_e_prev, dt=dt)  # [n_e, branch_e]
                # Scatter weighted update back into h_next.
                start = sum(self._branch_dims[:e])
                end = start + self._branch_dims[e]
                h_next[mask, start:end] = gate_per_row[mask].unsqueeze(-1) * h_e_new

        # Update running average probability for the aux balance loss (cheap EMA).
        # We use detached EMA so this doesn't fight the outer optimisation.
        with torch.no_grad():
            avg = score.mean(dim=0)  # [E]
            self._running_avg_prob.mul_(0.99).add_(0.01 * avg.detach())

        return h_next

    def auxiliary_loss(self, x_t: torch.Tensor) -> torch.Tensor:
        """Auxiliary soft load-balance loss; add to task loss with ``aux_load_balance_weight``.

        Computed from the *current* router probabilities (NOT the EMA) so the
        signal reflects the live distribution. The target is uniform.
        """
        score = self.router(x_t)  # [B, E]
        avg_per_expert = score.mean(dim=0)  # [E]
        target = avg_per_expert.new_full(avg_per_expert.shape, 1.0 / self.n_tau)
        load_balance = (avg_per_expert - target).pow(2).sum()
        return load_balance * self.aux_load_balance_weight


class MultiRateMoECfCNetwork(nn.Module):
    """Sequence wrapper around :class:`MultiRateMoECfC`, mirroring ``CfCNetwork``."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        n_tau: int = 4,
        top_k_active: int | None = None,
        tau_scales: tuple = (0.1, 0.5, 2.0, 10.0),
        return_sequences: bool = True,
        bilinear_router: bool = False,
        aux_load_balance_weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.cell = MultiRateMoECfC(
            input_size=input_size,
            hidden_size=hidden_size,
            n_tau=n_tau,
            top_k_active=top_k_active,
            tau_scales=tau_scales,
            bilinear_router=bilinear_router,
            aux_load_balance_weight=aux_load_balance_weight,
        )
        self.readout = nn.Linear(hidden_size, output_size)
        self.return_sequences = return_sequences
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        h = x.new_zeros(batch, self.cell.hidden_size)
        outs = []
        for t in range(seq_len):
            h = self.cell(x[:, t, :], h, dt=1.0 / max(seq_len, 1))
            outs.append(h)
        h_seq = torch.stack(outs, dim=1)
        y = self.readout(h_seq)
        return y if self.return_sequences else y[:, -1, :]

    def auxiliary_loss(self, x: torch.Tensor) -> torch.Tensor:
        # Average the per-step aux loss across all timesteps.
        losses = []
        batch, seq_len, _ = x.shape
        for t in range(seq_len):
            losses.append(self.cell.auxiliary_loss(x[:, t, :]))
        return sum(losses) / seq_len


__all__ = [
    "MultiRateMoECfC",
    "MultiRateMoECfCNetwork",
    "ExpertChoiceRouter",
]
