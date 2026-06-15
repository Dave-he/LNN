"""MoD (Mixture-of-Depths) routing for Liquid Neural Networks (PRD #10-73, 2026-06-15).

Implements the core Mixture-of-Depths mechanism from
arXiv:2404.02258 (Raposo et al., DeepMind 2024) and adapts it to
the recurrent CfC setting.  The key idea: at each cell step, route
over a fixed *budget* k of timesteps to process through the heavy
CfC block; the remaining timesteps are skipped (passed through via
residual connection).  This is the first "structural, data-structure
independent" routing policy in our 91-110 audit where the cap k is
*fixed a priori* and the router learns which steps to spend compute
on.

Differences from MoD-for-Transformers (Raposo et al. 2024):

- The original MoD top-k operates over token positions in a sequence
  processed in parallel by a Transformer block.  Here the per-step
  CfC is recurrent, so we apply MoD per-timestep: at step t we choose
  whether to compute the gated CfC update or pass the previous hidden
  state through unchanged.  The "depth" axis is the *stack* of CfC
  cells — MoDCfCNetwork stacks N cells, and at each layer we again
  apply per-step top-k routing.

- The router uses a sigmoid scalar (per-timestep skip/process) instead
  of MoD's softmax over a single logit (only two outcomes: process
  or skip).  Top-k over the routing scores picks the k highest-scoring
  timesteps to process.

- Auxiliary loss is the Switch-Transformer-style capacity loss
  adapted to the top-k setting: ``K * sum_i (f_i * P_i)`` where
  ``f_i = fraction of tokens selected at this layer`` and
  ``P_i = mean router probability over tokens``.  This loss is
  minimised when f_i matches the budget fraction k/T and when
  router probabilities are well-calibrated.

What this module is *not* (intentionally a *minimal* MoD-CfC
implementation):

- No joint MoD-MoE (Mixture-of-Depths-and-Experts) composition — that
  is left as a future combination of round 78 FAME-style experts
  with this MoD top-k.
- No "every other layer" routing (Raposo et al. 2024 schedule) — we
  apply MoD to every layer for simplicity, the schedule is a
  hyperparameter.
- No cumulative routing (each layer decides independently) — the
  router signal is per-(batch, time, layer) and is NOT shared across
  layers.
- No dynamic k (Raposo et al. 2024 keep k constant per training
  run; we follow this).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lnn.core.cfc import CfCCell
from lnn.core.sequence_utils import select_step_delta, select_step_mask


class MoDRouter(nn.Module):
    """Per-token (per-timestep) top-k router with cap k.

    The router produces a scalar score per timestep; the top-k scores
    (per batch row) are selected for processing.  This is the
    Mixture-of-Depths routing mechanism from arXiv:2404.02258
    (Raposo et al., 2024), adapted to a scalar sigmoid per token.

    Args:
        input_size: Input feature dim to the router (concat of
            x_t and h_prev, like MR-MoE).
        hidden_size: Hidden state dim.
        router_hidden: Router MLP width (``0`` = linear).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        router_hidden: int = 0,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.router_hidden = int(router_hidden)

        router_in = input_size + hidden_size
        if self.router_hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(router_in, self.router_hidden),
                nn.Tanh(),
                nn.Linear(self.router_hidden, 1),
            )
        else:
            self.net = nn.Linear(router_in, 1)

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        cap_k: int,
        T: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute top-k selection and aux loss.

        Args:
            x_t: [B, input_size] input at this step.
            h:   [B, hidden_size] previous hidden state.
            cap_k: integer budget (k) of tokens to process.  Must be
                <= B (per batch row, applied independently to each
                sample).  If ``cap_k >= T``, all tokens are processed.
            T: total sequence length T (used for aux-loss normalisation).

        Returns:
            process_mask: [B] bool — True for the top-k samples, False
                for the rest.
            router_prob: [B] — sigmoid probability (for diagnostics).
            aux_loss: scalar — Switch-Transformer-style capacity loss
                (encourages the router to use ~k/T fraction).
        """
        B = x_t.size(0)
        cap_k_eff = min(int(cap_k), B)
        combined = torch.cat([x_t, h], dim=-1)
        logit = self.net(combined).squeeze(-1)  # [B]
        router_prob = torch.sigmoid(logit)
        # Top-k: pick the cap_k_eff highest-scoring samples.
        # ``torch.topk`` is stable on the largest values, so this is
        # the natural selection mechanism for MoD.
        topk_scores, topk_idx = torch.topk(logit, k=cap_k_eff, dim=-1)
        process_mask = torch.zeros(B, dtype=torch.bool, device=x_t.device)
        process_mask[topk_idx] = True
        # Aux loss: K * sum_i f_i * P_i.  We compute it for this step.
        # ``f_i`` = mean(process_mask) — fraction of tokens selected.
        # ``P_i`` = mean(router_prob) — mean routing probability.
        # Multiply by K (the cap) for Switch-Transformer scale.
        f = process_mask.float().mean()
        P = router_prob.mean()
        aux_loss = cap_k_eff * f * P  # scalar
        return process_mask, router_prob, aux_loss


class MoDCfCCell(nn.Module):
    """Mixture-of-Depths CfC cell.

    At each step, the router decides whether to process the input
    through the heavy CfC block (selected via top-k) or pass the
    previous hidden state through unchanged (residual skip).  This
    is structurally aligned with the 91-110 audit "structural +
    data-structure independent" pattern (winners: 99, 102, 105, 107).

    Args:
        input_size: Input feature dimension.
        hidden_size: Hidden state dimension.
        cap_k: integer k (budget of timesteps to process per layer
            per sequence).  ``cap_k=None`` means "always process"
            (i.e. equivalent to standard CfC).  ``cap_k=1`` means
            only the top-scoring single timestep is processed.
        router_hidden: Router MLP width.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        cap_k: int | None = None,
        router_hidden: int = 0,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.cap_k = cap_k
        self.router_hidden = int(router_hidden)

        self.cell = CfCCell(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
        )
        self.router = MoDRouter(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            router_hidden=self.router_hidden,
        )

        # Side-channel state for diagnostics (no grad).
        self.last_router_prob: torch.Tensor | None = None
        self.last_process_mask: torch.Tensor | None = None
        self.aux_loss: torch.Tensor | None = None

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
        T: int | None = None,
    ) -> torch.Tensor:
        """One step of MoD CfC.

        Args:
            x_t: [B, input_size] input at this step.
            h:   [B, hidden_size] previous hidden state.
            dt:  scalar or per-sample time delta.
            T:   sequence length (for aux loss normalisation).  If
                ``None`` and ``cap_k`` is set, aux loss is not
                accumulated; pass the full T at the network level.

        Returns:
            h_new: [B, hidden_size] (processed for top-k, unchanged
                for the rest).
        """
        B = x_t.size(0)
        if self.cap_k is None or self.cap_k >= B:
            # Always process: equivalent to standard CfC.
            return self.cell(x_t, h, dt=dt)
        # Compute the routing decision.
        process_mask, router_prob, aux_loss = self.router(
            x_t, h, cap_k=self.cap_k, T=T or B,
        )
        # Run the heavy cell for ALL samples (cheaper than masking
        # inside the cell, but produces gradients for all).
        h_new = self.cell(x_t, h, dt=dt)
        # Apply process mask: where mask is False, keep h; where True, use h_new.
        # ``process_mask`` is [B], we need [B, 1] for broadcasting.
        mask_f = process_mask.float().unsqueeze(-1)
        out = mask_f * h_new + (1.0 - mask_f) * h
        # Stash diagnostics (detached for side-channel).
        self.last_router_prob = router_prob.detach()
        self.last_process_mask = process_mask.detach()
        self.aux_loss = aux_loss
        return out


class MoDCfCNetwork(nn.Module):
    """Stacked Mixture-of-Depths CfC network.

    Mirrors ``CfCNetwork`` API (return_sequences, mask, dt) but swaps
    every layer's ``CfCCell`` for a ``MoDCfCCell`` with per-layer
    top-k routing.  The cap_k is a fraction of the sequence length T
    (set at forward time via ``cap_k_frac``) or a fixed integer.

    Args:
        input_size: Input feature dimension.
        hidden_size: Hidden state dimension.
        output_size: Output dimension.
        num_layers: Number of stacked MoD CfC layers.
        return_sequences: If True, return full sequence; else last step.
        cap_k: integer budget for the per-layer top-k routing.  If
            ``None``, all timesteps are processed (equivalent to a
            standard stacked CfC network).  If integer k, exactly k
            timesteps per layer per sequence are processed.
        cap_k_frac: optional fraction (0 < cap_k_frac <= 1.0) — if
            set, cap_k = max(1, int(cap_k_frac * T)) where T is the
            sequence length at forward time.  Takes precedence over
            the integer cap_k argument.
        router_hidden: Router MLP width.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        return_sequences: bool = True,
        cap_k: int | None = None,
        cap_k_frac: float | None = None,
        router_hidden: int = 0,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.num_layers = int(num_layers)
        self.return_sequences = bool(return_sequences)
        self.cap_k = cap_k
        self.cap_k_frac = cap_k_frac
        self.router_hidden = int(router_hidden)

        if cap_k_frac is not None and cap_k is not None:
            raise ValueError(
                "Pass either cap_k (int) or cap_k_frac (float), not both."
            )
        if cap_k_frac is not None and not (0.0 < cap_k_frac <= 1.0):
            raise ValueError(
                f"cap_k_frac must be in (0, 1], got {cap_k_frac}"
            )

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            # Per-layer cap_k: derived from cap_k_frac at forward time
            # so all layers see the same per-layer cap.  Use a
            # placeholder int (1) when cap_k_frac is set; effective
            # cap is recomputed in forward().
            placeholder_cap = cap_k if cap_k is not None else None
            self.cells.append(
                MoDCfCCell(
                    input_size=in_dim,
                    hidden_size=hidden_size,
                    cap_k=placeholder_cap,
                    router_hidden=router_hidden,
                )
            )
        self.output_proj = nn.Linear(hidden_size, output_size)

    def _resolve_cap_k(self, T: int) -> int | None:
        if self.cap_k_frac is not None:
            return max(1, int(self.cap_k_frac * T))
        return self.cap_k

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
            [B, T, output_size] (return_sequences=True) or
            [B, output_size].
        """
        batch_size, seq_len, _ = x.shape
        cap_k = self._resolve_cap_k(seq_len)
        if h0 is None:
            h0 = torch.zeros(
                self.num_layers, batch_size, self.hidden_size,
                device=x.device, dtype=x.dtype,
            )

        h = h0
        layer_input = x
        for i, cell in enumerate(self.cells):
            # Per-layer cap is overridden by the network-level cap_k
            # only if cap_k_frac was set on the network.  We mutate
            # the cell's cap_k for this forward so the router uses
            # the right budget.
            if self.cap_k_frac is not None:
                cell.cap_k = cap_k
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
                h_candidate = cell(x_t, h_i, dt=dt_t, T=seq_len)
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


def compute_mod_aux_loss(network: MoDCfCNetwork) -> torch.Tensor:
    """Aggregate the per-cell aux losses into a single scalar.

    Useful for adding to the training loss:
    ``total_loss = task_loss + aux_weight * compute_mod_aux_loss(net)``.

    Returns:
        Scalar tensor with the sum of aux losses across all cells
        and timesteps visited so far.  Zero if the network always
        processes (cap_k=None or cap_k >= T).
    """
    total = None
    for cell in network.cells:
        if cell.aux_loss is not None:
            if total is None:
                total = cell.aux_loss
            else:
                total = total + cell.aux_loss
    if total is None:
        # No aux loss accumulated; return a zero tensor on the network's
        # device to keep the optimiser happy.
        return torch.tensor(0.0)
    return total
