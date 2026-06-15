"""Expert Choice (EC) routing for Liquid Neural Networks (PRD #10-74, 2026-06-15).

Implements the Expert Choice routing mechanism from
arXiv:2202.09368 (Zhou et al., 2022) and adapts it to the
recurrent CfC setting.  The key idea: at each cell step, instead of
each token/timestep picking its top-K experts (token-choice routing,
FAME / Soft MoE), each **expert** picks its top-K tokens.  This
gives perfect load balancing by construction — every expert
processes the same number of tokens, and the auxiliary load
balancing loss used in Switch Transformer / FAME is unnecessary.

Differences from EC-for-Transformers (Zhou et al. 2022):

- In EC-for-Transformers, the assignment is computed from
  ``softmax(x · W_gate)`` (a token×expert matrix), and each expert
  picks its top-k tokens from this matrix.  In the recurrent CfC
  setting, "tokens" are timesteps in a sequence; we adapt this by
  computing a per-(expert, timestep) assignment score and letting
  each expert pick its top-K timesteps.

- The expert output for a given timestep is the **weighted sum of
  all expert contributions** for that timestep, normalised by the
  sum of weights.  In token-choice routing, this normalisation is
  done by softmax over expert weights; in EC, the normalisation is
  ``Σ_{e picked timestep t} g_e(t) / |{e picked t}|`` — average
  weight per active expert.

- We use a sigmoid (not softmax) for the per-pair score; the
  assignment matrix is ``S[e, t] = sigmoid(W_e · x_t)`` (or a
  bilinear form for more expressivity).  This is in keeping with
  the Switch-Transformer / FAME-style sigmoid gating common in our
  audit stack.

- Per-expert bucket size k is a hyperparameter.  We default
  ``k = ceil(cap_k_frac * T)`` so that on average each timestep
  receives the same number of experts (= K * k / T) — the natural
  balanced operating point of EC.

Why this fits the 91-111 audit pattern:

- Structural: changes the routing mechanism (token-choice →
  expert-choice)
- Data-independent: does not assume data structure, only enforces
  balanced load
- Constructive: removes the need for aux load-balancing loss

This is the natural complement to MoD (round 111):
- MoD = per-timestep compute budget (which timesteps to process)
- EC  = per-expert compute budget (which expert processes which timesteps)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lnn.core.cfc import CfCCell
from lnn.core.sequence_utils import select_step_delta, select_step_mask


class ExpertChoiceRouter(nn.Module):
    """Expert Choice router: each expert picks its top-k tokens.

    Computes a per-(expert, token) assignment score, then for each
    expert picks the top-k scoring tokens.  The output is a sparse
    assignment mask ``[K, T]`` (True where expert e processes token
    t) and a per-pair weight matrix ``[K, T]``.

    Args:
        input_size: Input feature dim.
        hidden_size: Hidden state dim.
        n_experts: Number of experts (K).
        router_hidden: Router MLP width (``0`` = linear).
        use_sigmoid: If True, use sigmoid score (per-pair); else use
            a row-softmax (per-token over experts).  Default True
            matches the Switch-Transformer / FAME convention in our
            audit stack.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_experts: int,
        router_hidden: int = 0,
        use_sigmoid: bool = True,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.n_experts = int(n_experts)
        self.router_hidden = int(router_hidden)
        self.use_sigmoid = bool(use_sigmoid)

        # Router: produces K scores per token (concat x_t + h).
        router_in = input_size + hidden_size
        if self.router_hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(router_in, self.router_hidden),
                nn.Tanh(),
                nn.Linear(self.router_hidden, self.n_experts),
            )
        else:
            self.net = nn.Linear(router_in, self.n_experts)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        cap_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute EC assignment.

        Args:
            x: [B, T, input_size] input sequence for the layer.
            h: [B, T, hidden_size] previous hidden states.
            cap_k: bucket size k — each expert processes exactly
                ``cap_k`` tokens per sequence (so total work is
                ``K * cap_k`` per forward).

        Returns:
            assign_mask: [B, K, T] bool — True where expert e
                processes token t.
            assign_w: [B, K, T] float — per-pair weight (sigmoid
                probability for sigmoid mode, softmax weight for
                softmax mode).
        """
        B, T, _ = x.shape
        cap_k_eff = min(int(cap_k), T)
        # [B, T, K]
        combined = torch.cat([x, h], dim=-1)
        scores = self.net(combined)
        if self.use_sigmoid:
            assign_w = torch.sigmoid(scores)
        else:
            assign_w = F.softmax(scores, dim=-1)
        # ``assign_w`` is [B, T, K].  For each (B, K) we want to
        # pick the top-k scoring tokens along the T dim.  Transpose
        # to [B, K, T] and take top-k along the last dim.
        assign_w_T = assign_w.transpose(-1, -2)  # [B, K, T]
        topk_w, topk_idx = assign_w_T.topk(cap_k_eff, dim=-1)  # [B, K, k]
        # Build sparse mask.
        assign_mask = torch.zeros(
            B, self.n_experts, T, dtype=torch.bool, device=x.device,
        )
        assign_mask.scatter_(-1, topk_idx, True)
        return assign_mask, assign_w_T


class ExpertChoiceCfCCell(nn.Module):
    """Expert Choice CfC cell.

    At each step, the router computes a (K, T) assignment and each
    expert processes the K timesteps it selected.  The cell output
    at each timestep is the **weighted sum of all expert
    contributions** for that timestep (averaged over the active
    experts for that timestep), giving a balanced representation.

    Args:
        input_size: Input feature dim.
        hidden_size: Hidden state dim.
        n_experts: Number of CfC experts (K).
        cap_k: bucket size (tokens per expert).  ``cap_k=None``
            means "all tokens" (equivalent to dense).  ``cap_k=1``
            means each expert processes only the single highest-
            scoring token.
        router_hidden: Router MLP width.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_experts: int = 3,
        cap_k: int | None = None,
        router_hidden: int = 0,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.n_experts = int(n_experts)
        self.cap_k = cap_k
        self.router_hidden = int(router_hidden)

        self.experts = nn.ModuleList([
            CfCCell(input_size=self.input_size, hidden_size=self.hidden_size)
            for _ in range(self.n_experts)
        ])
        self.router = ExpertChoiceRouter(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            n_experts=self.n_experts,
            router_hidden=self.router_hidden,
        )

        # Side-channel diagnostics.
        self.last_assign_mask: torch.Tensor | None = None
        self.last_assign_w: torch.Tensor | None = None
        # Per-expert load (mean #tokens per expert in last call) — should be ~cap_k.
        self.last_load_per_expert: torch.Tensor | None = None

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
        T: int | None = None,
        assign_mask_K: torch.Tensor | None = None,
        assign_w_K: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One step of EC CfC.

        Args:
            x_t: [B, input_size] input at this step.
            h:   [B, hidden_size] previous hidden state.
            dt:  scalar or per-sample time delta.
            T:   sequence length (used for cap_k resolution at the
                network level).
            assign_mask_K: optional pre-computed assignment mask of
                shape [B, K] for this timestep.  If passed, the
                router is bypassed (useful for the per-timestep
                loop in the network wrapper that pre-computes the
                assignment once per sequence).
            assign_w_K: optional pre-computed assignment weights of
                shape [B, K] for this timestep.

        Returns:
            h_new: [B, hidden_size] — the EC-mixed hidden state at
                this timestep.
        """
        B = x_t.size(0)
        if assign_mask_K is not None and assign_w_K is not None:
            # Use the pre-computed assignment (network-level).
            mask_K = assign_mask_K  # [B, K]
            w_K = assign_w_K  # [B, K]
        else:
            # Single-step fallback: route over a single token (less
            # useful, but keeps the contract valid).
            # Treat the batch as T=B tokens, K experts, k=1.
            scores = self.router.net(torch.cat([x_t, h], dim=-1))  # [B, K]
            if self.router.use_sigmoid:
                w_K = torch.sigmoid(scores)
            else:
                w_K = F.softmax(scores, dim=-1)
            # top-1 per expert.
            topk_w, topk_idx = w_K.topk(1, dim=-1)  # [B, K, 1]
            mask_K = torch.zeros(B, self.n_experts, dtype=torch.bool, device=x_t.device)
            mask_K.scatter_(-1, topk_idx.squeeze(-1).unsqueeze(-1), True)

        # Run all K experts.
        outs = [expert(x_t, h, dt=dt) for expert in self.experts]  # K × [B, H]
        stacked = torch.stack(outs, dim=1)  # [B, K, H]
        # EC mix: per-batch, weight by the assignment weight where
        # the mask is True; sum over active experts; divide by the
        # count of active experts.
        # ``mask_K`` and ``w_K`` are [B, K].
        w_used = mask_K.float() * w_K  # [B, K], zero where not active
        active_count = mask_K.float().sum(dim=-1, keepdim=True).clamp(min=1.0)  # [B, 1]
        # Normalise by the number of active experts (so the
        # representation is on a similar scale regardless of how
        # many experts picked this token).
        norm_w = w_used / active_count  # [B, K]
        out = (norm_w.unsqueeze(-1) * stacked).sum(dim=1)  # [B, H]
        # Stash side-channel diagnostics (detached for safety).
        # We accumulate the per-step assignment as a (B, K) slice
        # at this timestep into the full (B, K, T) buffer.
        if not hasattr(self, "_accum_assign_mask") or self._accum_assign_mask is None:
            # Allocate when we see the first batch size / T.
            self._accum_assign_mask = []
            self._accum_assign_w = []
        # We need the (B, K, T) shape; the network wrapper stashes
        # the full thing.  Here, since the network calls the cell
        # one step at a time, we let the network do the diagnostic
        # stashing.  The cell only stores the latest step's slice
        # for ``expert_choice_load`` consumers that want per-step.
        self.last_assign_mask = mask_K.detach()  # [B, K]
        self.last_assign_w = w_K.detach()  # [B, K]
        return out


class ExpertChoiceCfCNetwork(nn.Module):
    """Stacked Expert Choice CfC network.

    Mirrors ``CfCNetwork`` API (return_sequences, mask, dt) but
    replaces every layer's ``CfCCell`` with an
    ``ExpertChoiceCfCCell``.  The EC assignment is computed once
    per sequence at each layer (using the full sequence), and
    applied per-step inside the recurrent loop.  This avoids
    re-running the router at every step.

    Args:
        input_size: Input feature dim.
        hidden_size: Hidden state dim.
        output_size: Output dim.
        num_layers: Number of stacked EC CfC layers.
        return_sequences: If True, return full sequence; else last step.
        n_experts: Number of CfC experts (K).
        cap_k: integer bucket size.  ``cap_k=None`` means
            ``cap_k=T`` (dense).  ``cap_k=1`` means each expert
            processes 1 timestep per sequence.
        cap_k_frac: optional fraction of T (0 < frac <= 1).  At
            forward time, ``cap_k = max(1, int(frac * T))``.  Takes
            precedence over ``cap_k``.
        router_hidden: Router MLP width.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        return_sequences: bool = True,
        n_experts: int = 3,
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
        self.n_experts = int(n_experts)
        self.cap_k = cap_k
        self.cap_k_frac = cap_k_frac
        self.router_hidden = int(router_hidden)

        if cap_k_frac is not None and cap_k is not None:
            raise ValueError("Pass either cap_k or cap_k_frac, not both.")
        if cap_k_frac is not None and not (0.0 < cap_k_frac <= 1.0):
            raise ValueError(f"cap_k_frac must be in (0, 1], got {cap_k_frac}")

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            placeholder_cap = cap_k if cap_k is not None else None
            self.cells.append(
                ExpertChoiceCfCCell(
                    input_size=in_dim,
                    hidden_size=hidden_size,
                    n_experts=self.n_experts,
                    cap_k=placeholder_cap,
                    router_hidden=router_hidden,
                )
            )
        self.output_proj = nn.Linear(hidden_size, output_size)

    def _resolve_cap_k(self, T: int) -> int:
        if self.cap_k_frac is not None:
            return max(1, int(self.cap_k_frac * T))
        if self.cap_k is not None:
            return min(self.cap_k, T)
        return T  # dense: each expert processes all tokens

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
            if self.cap_k_frac is not None:
                cell.cap_k = cap_k
            # Pre-compute the EC assignment once per sequence at this
            # layer.  We use a dummy hidden state (zeros) for the
            # router; a more elaborate scheme could use the running
            # hidden state, but pre-computing keeps the design clean
            # and matches the static-graph property of EC.
            h0_for_router = h0[i].unsqueeze(1).expand(-1, seq_len, -1)
            assign_mask, assign_w = cell.router(
                torch.nan_to_num(layer_input), h0_for_router, cap_k=cap_k,
            )  # [B, K, T], [B, K, T]
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
                # Slice the pre-computed assignment for this timestep.
                # ``assign_mask`` is [B, K, T], we want [B, K] at index t.
                assign_mask_t = assign_mask[:, :, t]
                assign_w_t = assign_w[:, :, t]
                h_candidate = cell(
                    x_t, h_i, dt=dt_t, T=seq_len,
                    assign_mask_K=assign_mask_t, assign_w_K=assign_w_t,
                )
                h_i = h_candidate if update_mask is None else update_mask * h_candidate + (1.0 - update_mask) * h_i
                outputs.append(h_i)
            layer_input = torch.stack(outputs, dim=1)
            h = torch.cat(
                [h_i.unsqueeze(0) if j == i else h[j].unsqueeze(0) for j in range(self.num_layers)],
                dim=0,
            )
            # Stash full-sequence diagnostics AFTER the per-step
            # loop so the [B, K, T] tensor is preserved (the cell's
            # own forward overwrites last_assign_mask with a [B, K]
            # slice at every step).
            cell.last_assign_mask = assign_mask.detach()
            cell.last_assign_w = assign_w.detach()

        if self.return_sequences:
            return self.output_proj(layer_input)
        return self.output_proj(layer_input[:, -1, :])


def expert_choice_load(cell: ExpertChoiceCfCCell) -> torch.Tensor:
    """Return the per-expert load (# of tokens selected) for the
    most recent forward pass.  Should be approximately ``cap_k`` for
    every expert (perfect load balance by construction).

    Returns:
        [K] tensor of per-expert token counts.  Zeros if no
        forward pass has been recorded.
    """
    if cell.last_assign_mask is None:
        return torch.zeros(cell.n_experts)
    # Sum over batch and T.
    return cell.last_assign_mask.float().sum(dim=(0, -1))
