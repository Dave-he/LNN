"""MultiRateTfpCfC — Second-layer synthesis: MR-MoE × TFP retention.

Motivation (loop 2026-08-05, round 282, follow-up to MemoryFusionCfC):
    The cross-paper synthesis ``MemoryFusionCfCCell`` (round 281) closed
    the **N3** gap by exposing TFP-style retention (``exp(-dt/τ)·h_prev + (1-exp)·ĥ``)
    on top of the CfC architecture. The next layer of synthesis combines that
    with the **N2-adjacent** multi-rate MoE structure (arXiv:2606.12240) that
    already lives in :class:`MultiRateMoECfC`.

    Rather than retrofitting the existing ``MultiRateMoECfC`` (which would
    risk ABI breakage for the dozens of existing Pareto / smoke tests), we
    introduce a *standalone* second-layer synthesis:

        MR-TFP-CfC = EC-Router(MultiRateMoECfC routing)
                   × MemoryFusionCfCCell(retention_kind='tfp') experts
                   × per-expert τ bias  →  multi-scale temporal specialisation

    Each of the ``n_tau`` experts is a *TFP-retention* liquid cell (one
    branch of the hidden state, with its own ``τ_proj`` initial bias), and the
    Expert-Choice router selects the top-K branches per timestep.

Cross-paper relation:
    - arXiv 2606.12240 (MR-MoE): multi-rate + EC routing → ``self.router``,
      per-expert hidden slicing → ``self._branch_dims``.
    - arXiv 2607.08283 (TFP): explicit-``dt`` retention → ``MemoryFusionCfCCell``
      expert cells.
    - arXiv 2607.10858 (NSFD): not used here (NSFD is preserved as an option in
      ``MemoryFusionCfCCell`` but unsuitable for symbolic AR(2) data — see the
      Pareto sweep 2026-08-05_mfc_cfc_pareto where MFC-NSFD explodes to
      MSE 160.96 on h=16/sl=64).

Numerical properties:
    - **Per-step top-K routing**: only K of the E experts are evaluated per
      step ⇒ per-step FLOPs ≈ K/E × full-multi-τ cost.
    - **Per-expert τ bias**: experts are ordered by τ (fast → slow) so the
      router learns "send spike-like inputs to fast-τ expert, trend-like
      inputs to slow-τ expert".
    - **Auxiliary load-balance loss**: averaged over batch + step, see
      ``auxiliary_loss()`` — mirrors Switch Transformer / FAME practice.

API parity with :class:`MultiRateMoECfC`:
    ``cell(x_t, h, dt) -> h_next`` and ``network(x_seq) -> y_seq``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lnn.core.memory_fusion_cfc import MemoryFusionCfCCell


class _LinearRouter(nn.Module):
    """Per-expert sigmoid router used by :class:`MultiRateTfpCfC`.

    Computes ``score[e] = sigmoid(<W_e, x_t>)``. Cheaper than bilinear
    routing and sufficient for the n_tau ≤ 8 settings we care about.
    """

    def __init__(self, input_size: int, n_experts: int):
        super().__init__()
        self.n_experts = n_experts
        self.W = nn.Parameter(torch.empty(n_experts, input_size))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(torch.einsum("ek,bk->be", self.W, x))


class MultiRateTfpCfC(nn.Module):
    """EC-routed mixture of TFP-retention experts.

    Args:
        input_size:  Input feature dimension.
        hidden_size: Hidden dimension (concatenation of all expert hidden states).
        n_tau:       Number of experts (≥2).
        top_k_active:Per-step top-K experts (1 ≤ k ≤ n_tau). Default ``ceil(n_tau/2)``.
        tau_scales:  Per-expert initial τ bias (softplus biases set so that
            the average initial retention ``k_t = exp(-dt/τ)`` matches these
            scales for ``dt=1.0``). Sorted ascending (fast → slow).
        aux_load_balance_weight: Weight for the auxiliary soft load-balance loss.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_tau: int = 4,
        top_k_active: int | None = None,
        tau_scales: tuple = (0.1, 0.5, 2.0, 10.0),
        aux_load_balance_weight: float = 0.01,
    ):
        super().__init__()
        if n_tau < 2:
            raise ValueError(
                f"MultiRateTfpCfC requires n_tau >= 2 (got {n_tau}); "
                "for single-expert use MemoryFusionCfCCell directly."
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
        self._branch_dims = [
            base + (rem if i == self.n_tau - 1 else 0) for i in range(self.n_tau)
        ]

        # Build one TFP-retention expert per τ group.
        self.experts = nn.ModuleList()
        for i, out_dim in enumerate(self._branch_dims):
            expert = MemoryFusionCfCCell(
                input_size=input_size,
                hidden_size=out_dim,
                retention_kind="tfp",
                n_tau=1,
            )
            # Set τ_proj bias so that exp(-1/softplus(bias)) gives the target
            # initial τ scale. softplus(bias) ≈ τ  ⇒  bias ≈ log(exp(τ) - 1).
            target_tau = float(scales[i])
            bias_init = math.log(min(math.expm1(target_tau), 1e6) + 1e-3) if target_tau < 20 else float(target_tau)
            # tau_proj is an nn.Sequential: first layer is nn.Linear, second is Softplus.
            tau_lin_seq = expert.tau_proj[0]  # type: ignore[index]
            tau_lin = tau_lin_seq[0]
            with torch.no_grad():
                tau_lin.bias.fill_(bias_init)
            self.experts.append(expert)

        self.router = _LinearRouter(input_size=input_size, n_experts=self.n_tau)
        self.register_buffer("_running_avg_prob", torch.zeros(self.n_tau), persistent=False)

    # ------------------------------------------------------------------ utils

    def _slice_h(self, h: torch.Tensor | None, e: int, idx: torch.Tensor) -> torch.Tensor:
        """Return the previous hidden slice for the selected rows.

        ``h`` is the *full* previous hidden state ``[B, hidden_size]``;
        ``idx`` are the batch indices selected by the router for expert
        ``e``. We slice both the hidden dim (to this expert's branch)
        *and* the batch dim (to the selected rows).
        """
        start = sum(self._branch_dims[:e])
        end = start + self._branch_dims[e]
        if h is None:
            return idx.new_zeros(idx.shape[0], self._branch_dims[e])
        return h.index_select(0, idx)[:, start:end]

    def _full_slice(self, h: torch.Tensor, e: int) -> torch.Tensor:
        """Return the full-batch e-branch slice of ``h`` (used for index_copy_)."""
        start = sum(self._branch_dims[:e])
        end = start + self._branch_dims[e]
        return h[:, start:end]

    def auxiliary_loss(self, x_t: torch.Tensor) -> torch.Tensor:
        """Per-step soft load-balance loss for the router.

        Mirrors the ``MultiRateMoECfC`` formulation: encourage uniform
        routing probability across experts. Averaged over batch.
        """
        with torch.no_grad():
            _ = self.router(x_t)
        score = self.router(x_t)  # [B, E]
        avg_prob = score.mean(dim=0)  # [E]
        # Update running average (for diagnostics, no grad).
        self._running_avg_prob = 0.99 * self._running_avg_prob + 0.01 * avg_prob.detach()
        # Soft load-balance loss: penalise deviation from uniform 1/E.
        target = torch.full_like(avg_prob, 1.0 / self.n_tau)
        return F.mse_loss(avg_prob, target)

    # --------------------------------------------------------------- forward

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor | None = None,
        dt: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        batch_size = x_t.shape[0]
        if h is None:
            h = x_t.new_zeros(batch_size, self.hidden_size)

        score = self.router(x_t)  # [B, E]
        top_scores, top_idx = score.topk(self.top_k_active, dim=-1)  # [B, k], [B, k]
        gate = F.softmax(top_scores, dim=-1)  # [B, k]

        h_next = h.clone()

        for k_slot in range(self.top_k_active):
            branch_idx_per_row = top_idx[:, k_slot]   # [B]
            gate_per_row = gate[:, k_slot]            # [B]

            # Apply each expert only on the rows where it was selected.
            for e in range(self.n_tau):
                mask = branch_idx_per_row == e
                if not mask.any():
                    continue
                # Sub-batch those rows.
                idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
                x_e = x_t.index_select(0, idx)
                h_e = self._slice_h(h, e, idx)
                h_e_next = self.experts[e](x_e, h_e, dt=dt)

                # Weighted write-back into the active rows.
                gate_e = gate_per_row.index_select(0, idx).unsqueeze(-1)
                slice_e = self._full_slice(h_next, e)
                # rows that selected e: blend previous (full-h slice) with expert output
                slice_e.index_copy_(0, idx, gate_e * h_e_next + (1.0 - gate_e) * h_e)
                # The non-selected rows keep their previous hidden slice (already cloned).

        return h_next


class MultiRateTfpCfCNetwork(nn.Module):
    """Sequence wrapper around :class:`MultiRateTfpCfC`, mirroring ``MultiRateMoECfCNetwork``."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        n_tau: int = 4,
        top_k_active: int | None = None,
        tau_scales: tuple = (0.1, 0.5, 2.0, 10.0),
        return_sequences: bool = True,
        aux_load_balance_weight: float = 0.01,
    ):
        super().__init__()
        self.cell = MultiRateTfpCfC(
            input_size=input_size,
            hidden_size=hidden_size,
            n_tau=n_tau,
            top_k_active=top_k_active,
            tau_scales=tau_scales,
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
        losses = []
        batch, seq_len, _ = x.shape
        for t in range(seq_len):
            losses.append(self.cell.auxiliary_loss(x[:, t, :]))
        return sum(losses) / seq_len


__all__ = [
    "MultiRateTfpCfC",
    "MultiRateTfpCfCNetwork",
]
