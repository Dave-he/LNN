"""Round 303 — STE-ParallelCfC: discrete routing compensates anchor error.

Combines two ideas from prior rounds:

  - **r301 ParallelCfCCell** (`lnn/core/parallel_cfc.py`): PLAN-inspired
    vectorised W-step CfC. Within a window of W steps, h_0 is used as a
    constant anchor (instead of sequentially threading h_{t-1} → h_t).
    Empirically r301 shows that on smooth periodic targets the anchor
    approximation costs < 7% MSE while cutting inference latency by 60%.

  - **r265 STENeuronWiseCfCCell** (`lnn/core/ste_neuron_wise_cfc.py`):
    per-neuron routing with a **straight-through estimator (STE)** mask.
    Forward uses a hard binary top-k mask; backward uses a soft sigmoid
    mask so gradients still flow to the underlying logits. This was
    extended by r267 with **soft-mask entropy regularisation** to keep
    the discrete routing concentrated.

Hypothesis (r303): the anchor approximation error in ParallelCfC is
**per-neuron** — some hidden units really do need the recurrent h_t
to be threaded (the "anchor-sensitive" units), while others are
dominated by the input x_t and the anchor is fine (the "anchor-safe"
units).  STE gives us a *learned, differentiable* way to identify the
two groups and route accordingly.

Cell design (the only non-trivial change vs r301):

    h_parallel   = parallel_anchor_h  (B, hidden)     # r301 forward
    h_sequential = sequential_h_step_at_t             # one vanilla CfC step
    mask_hard    = top_k(route_logits, density)       # binary (hidden,)
    mask_soft    = sigmoid(route_logits / τ_ste)       # differentiable
    mask_ste     = (mask_hard - mask_soft).detach() + mask_soft
    h_out        = mask_ste ⊙ h_parallel
                 + (1 - mask_ste) ⊙ h_sequential

So a neuron with mask=1 is "anchor-safe" (use the cheap parallel
result), and a neuron with mask=0 is "anchor-sensitive" (use the
sequential step that respects the actual h_t).

This is *orthogonal* to the r301 axes (window width W, chunking,
τ). It is also *orthogonal* to r265's STE axes (which neurons are
connected to which neighbours). The two STE applications share the
straight-through pattern but operate on entirely different state
objects:

    - r265 STE mask:  (d_h, d_h) — inter-neuron edges
    - r303 STE mask:  (d_h,)      — inter-update-mode (parallel vs seq)

The cell API is intentionally a drop-in for r301's ParallelCfCCell:
``forward(x, h, dt)`` with the same shape rules. When ``window == 1``
the sequential and parallel branches produce the *same* output for any
given neuron (the window has length 1 ⇒ no inter-step anchor), so the
mask is applied to a duplicate and the result is shape-equivalent to
vanilla CfC. We treat the window=1 case as a degenerate sanity path
that exercises the STE machinery on a single (B, hidden) value.

Args:
    input_size:     d_in.
    hidden_size:    d_h.
    window:         W — number of consecutive CfC steps vectorised in
                    parallel (defaults to r301 winner: W=8).
    density:        ρ — target fraction of neurons that use the
                    parallel anchor (1.0 = all, 0.0 = all sequential).
    ste_temperature: τ_ste — sigmoid sharpness in the backward path.
    entropy_lambda: λ — coefficient for soft-mask entropy reg (r267).
    tau_init:       initial time-constant for both branches.

The cell is intentionally a drop-in for r301's ParallelCfCCell in the
sense that ``forward(x, h, dt)`` accepts the same signature, but x may
now be (B, W, d_in).  When ``window == 1`` the cell degenerates to a
per-neuron blend of two numerically-equivalent evaluations, which is
a sanity path.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _topk_binary_mask(logits: torch.Tensor, density: float) -> torch.Tensor:
    """Return a binary 0/1 mask of shape ``logits.shape`` keeping the
    top ``density`` fraction by logit magnitude.

    For density=1.0 returns all-ones. For density=0.0 returns all-zeros.
    Ties are broken by index (lower index wins, matching torch.topk
    semantics when sorted=False).
    """
    n = logits.numel()
    k = max(1, int(round(float(density) * n))) if density > 0 else 0
    if k >= n:
        return torch.ones_like(logits)
    if k <= 0:
        return torch.zeros_like(logits)
    # topk on flat tensor — values, indices.
    _, idx = torch.topk(logits.view(-1), k=k, sorted=False)
    mask = torch.zeros_like(logits.view(-1))
    mask[idx] = 1.0
    return mask.view_as(logits)


class STEParallelCfCCell(nn.Module):
    """ParallelCfC + per-neuron STE mask routing (round 303).

    The cell maintains two branches:

      - **parallel**: r301 PLAN-style W-step vectorised CfC with the
        h_0 anchor (cheap, approximate).
      - **sequential**: a single vanilla CfC step using the true h
        (accurate, expensive at the per-neuron level).

    A per-neuron STE mask ``m ∈ {0, 1}^hidden`` decides which branch
    each neuron uses.  Density ρ = E[m] is a hyperparameter; the
    learned ``route_logits`` parameter converges during training so
    that the right neurons take the right branch.

    Forward uses the **hard** mask (true binary routing — no
    interpolation noise). Backward flows through the **soft** sigmoid
    mask (gradients reach route_logits). The straight-through
    estimator pattern is::

        mask = (hard - soft).detach() + soft

    The entropy regulariser (r267) is a soft-mask concentration
    penalty — it keeps the binary mask from drifting toward 0.5
    during training.

    Args:
        input_size:      d_in.
        hidden_size:     d_h.
        window:          W — number of consecutive CfC steps
                         vectorised in parallel (r301 winner: 8).
        density:         ρ — target fraction of "anchor-safe" neurons
                         using the parallel branch (1.0 = all,
                         0.0 = all sequential).
        ste_temperature: τ_ste — sigmoid sharpness in the backward
                         path. Smaller = sharper gradient.
        entropy_lambda:  λ — soft-mask entropy reg coefficient
                         (r267). 0.0 disables.
        tau_init:        initial time-constant for both branches.
        seed:            random seed for init.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        window: int = 8,
        density: float = 0.5,
        ste_temperature: float = 1.0,
        entropy_lambda: float = 0.01,
        tau_init: float = 1.0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        assert window >= 1, f"window must be >= 1, got {window}"
        assert 0.0 <= density <= 1.0, f"density must be in [0, 1], got {density}"
        assert ste_temperature > 0, f"ste_temperature must be > 0, got {ste_temperature}"
        assert entropy_lambda >= 0, f"entropy_lambda must be >= 0, got {entropy_lambda}"

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.window = int(window)
        self.density = float(density)
        self.ste_temperature = float(ste_temperature)
        self.entropy_lambda = float(entropy_lambda)

        g = torch.Generator().manual_seed(int(seed))

        # Per-branch parameters — *separate* weights for parallel vs
        # sequential so the cell is not forced to learn the same
        # function in two modes.  This adds capacity (the user pays
        # for routing) but cleanly separates the two regimes.
        self.f_gate_p = nn.Linear(input_size + hidden_size, hidden_size)
        self.g_branch_p = nn.Linear(input_size + hidden_size, hidden_size)
        self.h_branch_p = nn.Linear(input_size + hidden_size, hidden_size)
        self.f_gate_s = nn.Linear(input_size + hidden_size, hidden_size)
        self.g_branch_s = nn.Linear(input_size + hidden_size, hidden_size)
        self.h_branch_s = nn.Linear(input_size + hidden_size, hidden_size)
        self.time_scale = nn.Parameter(torch.full((hidden_size,), float(tau_init)))
        # Initialise sequential branch with the standard linear default
        # so window=1 + density=0.0 ≈ vanilla CfC up to init scale.
        with torch.no_grad():
            for layer in (self.f_gate_s, self.g_branch_s, self.h_branch_s):
                layer.weight.mul_(0.5)
                layer.bias.mul_(0.5)
        # Per-neuron routing logits — initialise to 0 so sigmoid(0) = 0.5;
        # the density-controlled hard mask is what enforces the desired
        # fraction at init.
        self.route_logits = nn.Parameter(torch.zeros(hidden_size))

    # ----- branch computations -----
    def _one_step(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        dt: torch.Tensor,
        f_gate: nn.Linear,
        g_branch: nn.Linear,
        h_branch: nn.Linear,
    ) -> torch.Tensor:
        """Vanilla CfC single step (B, d) → (B, hidden)."""
        z = torch.cat([x_t, h], dim=-1)
        f = torch.sigmoid(f_gate(z))
        g = torch.tanh(g_branch(z))
        hp = torch.tanh(h_branch(z))
        if dt.dim() == 0:
            dt_eff = dt
        elif dt.dim() == 1:
            dt_eff = dt.unsqueeze(-1)
        else:
            dt_eff = dt
        decay = torch.sigmoid(-f * self.time_scale * dt_eff)
        return decay * g + (1.0 - decay) * hp

    def _parallel_path(
        self, x: torch.Tensor, h: torch.Tensor, dt: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """r301 parallel path: (B, W, d_in) → (B, hidden) using h_0 anchor."""
        B, W, _ = x.shape
        assert W == self.window
        h_anchor = h.unsqueeze(1).expand(B, W, self.hidden_size)
        z = torch.cat([x, h_anchor], dim=-1)  # (B, W, d_in + hidden)
        f = torch.sigmoid(self.f_gate_p(z))
        g = torch.tanh(self.g_branch_p(z))
        hp = torch.tanh(self.h_branch_p(z))
        if dt is None:
            dt_eff = torch.tensor(1.0, device=x.device, dtype=x.dtype)
        elif dt.dim() == 0:
            dt_eff = dt
        elif dt.dim() == 1:
            dt_eff = dt.view(B, 1, 1)
        else:
            dt_eff = dt.view(B, W, 1)
        decay = torch.sigmoid(-f * self.time_scale * dt_eff)
        h_steps = decay * g + (1.0 - decay) * hp  # (B, W, hidden)
        return h_steps[:, -1, :]

    def _sequential_path(
        self, x: torch.Tensor, h: torch.Tensor, dt: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Sequential path: take the LAST timestep's input and apply one
        vanilla CfC step using the true h.
        """
        if x.dim() == 3:
            x_t = x[:, -1, :]
        else:
            x_t = x
        if dt is None:
            dt_t = torch.tensor(1.0, device=x.device, dtype=x.dtype)
        elif dt.dim() == 0 or (dt.dim() == 1 and dt.shape[0] == x.shape[0]):
            # dt is (B,) or scalar — use as-is for the single step.
            dt_t = dt if dt.dim() > 0 else dt
        else:
            # dt is (B, W, 1) — collapse to (B, 1) for the last step.
            dt_t = dt[:, -1, :]
        return self._one_step(x_t, h, dt_t, self.f_gate_s, self.g_branch_s, self.h_branch_s)

    # ----- STE mask -----
    def get_hard_mask(self) -> torch.Tensor:
        """Hard (binary) per-neuron routing mask of shape (d_h,)."""
        return _topk_binary_mask(self.route_logits, self.density)

    def get_soft_mask(self) -> torch.Tensor:
        """Soft (sigmoid) per-neuron routing mask of shape (d_h,)."""
        return torch.sigmoid(self.route_logits / self.ste_temperature)

    def get_ste_mask(self) -> torch.Tensor:
        """STE mask: forward=hard, backward=soft (straight-through)."""
        hard = self.get_hard_mask()
        soft = self.get_soft_mask()
        return (hard - soft).detach() + soft

    def soft_mask_entropy(self) -> torch.Tensor:
        """Shannon entropy of the soft mask (per-neuron scalar).

        Range: [0, 1] for a single sigmoid variable (the maximum is
        achieved at sigmoid(0) = 0.5, and the min at sigmoid(±∞) = 0/1).
        We compute the per-neuron Bernoulli entropy::

            H_i = -soft_i · log(soft_i) - (1 - soft_i) · log(1 - soft_i)
        """
        soft = self.get_soft_mask()
        eps = 1e-8
        H = -(soft * torch.log(soft + eps) + (1.0 - soft) * torch.log(1.0 - soft + eps))
        return H.mean()

    def extra_loss(self) -> torch.Tensor:
        """Entropy regulariser on the soft mask (r267)."""
        if self.entropy_lambda <= 0:
            return torch.tensor(0.0, device=self.route_logits.device)
        return self.entropy_lambda * self.soft_mask_entropy()

    # ----- forward -----
    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        dt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x:  (B, d_in) for window=1, or (B, W, d_in) for window>1.
            h:  (B, hidden) — initial hidden state for the window.
            dt: optional time-constant (scalar / (B,) / (B, 1) / (B, W, 1)).

        Returns:
            (B, hidden) — routed hidden state at the end of the window.
        """
        if self.window == 1 or x.dim() == 2:
            # Sequential / single-step path — both branches do the
            # same computation, but the mask still gets to learn.
            if dt is None:
                dt_t = torch.tensor(1.0, device=x.device, dtype=x.dtype)
            else:
                dt_t = dt
            h_parallel = self._one_step(
                x, h, dt_t, self.f_gate_p, self.g_branch_p, self.h_branch_p
            )
            h_sequential = self._one_step(
                x, h, dt_t, self.f_gate_s, self.g_branch_s, self.h_branch_s
            )
        else:
            h_parallel = self._parallel_path(x, h, dt)
            h_sequential = self._sequential_path(x, h, dt)
        mask = self.get_ste_mask()  # (hidden,)
        return mask * h_parallel + (1.0 - mask) * h_sequential


class STEParallelCfCNetwork(nn.Module):
    """Multi-layer STEParallelCfC that processes (B, T, d_in) sequences.

    Splits T into chunks of length ``window`` (T must be a multiple of
    ``window`` unless window == 1).  Within each chunk the
    STEParallelCfCCell routes per-neuron between the parallel and
    sequential branches.  The final hidden state of chunk c becomes
    the initial hidden state of chunk c+1.  For ``window == 1`` the
    network degenerates to vanilla sequential (with a learned per-
    neuron blend of two parameter copies, which is still useful for
    diagnostics).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        window: int = 8,
        density: float = 0.5,
        entropy_lambda: float = 0.01,
        return_sequences: bool = False,
    ) -> None:
        super().__init__()
        assert num_layers >= 1
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.num_layers = int(num_layers)
        self.window = int(window)
        self.density = float(density)
        self.return_sequences = return_sequences

        self.cells = nn.ModuleList(
            [
                STEParallelCfCCell(
                    input_size if i == 0 else hidden_size,
                    hidden_size,
                    window=window,
                    density=density,
                    entropy_lambda=entropy_lambda,
                )
                for i in range(num_layers)
            ]
        )
        self.output_proj = nn.Linear(hidden_size, output_size)

    def extra_loss(self) -> torch.Tensor:
        """Sum of per-cell entropy regularisers."""
        return sum(cell.extra_loss() for cell in self.cells)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, T, d_in) — T must be a multiple of ``window`` unless
               ``window == 1``.
        """
        B, T, _ = x.shape
        h_layers = [
            torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
            for _ in range(self.num_layers)
        ]
        outputs: list[torch.Tensor] = []
        if self.window == 1:
            for t in range(T):
                layer_input = torch.nan_to_num(x[:, t, :])
                for i, cell in enumerate(self.cells):
                    h_layers[i] = cell(
                        layer_input, h_layers[i],
                        dt=torch.tensor(1.0, device=x.device, dtype=x.dtype),
                    )
                    layer_input = h_layers[i]
                outputs.append(h_layers[-1])
            seq = torch.stack(outputs, dim=1)
        else:
            assert T % self.window == 0, (
                f"T={T} must be a multiple of window={self.window} in chunked mode"
            )
            num_chunks = T // self.window
            for c in range(num_chunks):
                x_chunk = torch.nan_to_num(
                    x[:, c * self.window : (c + 1) * self.window, :]
                )
                layer_input = x_chunk
                for i, cell in enumerate(self.cells):
                    h_layers[i] = cell(
                        layer_input, h_layers[i],
                        dt=torch.tensor(1.0, device=x.device, dtype=x.dtype),
                    )
                    if i + 1 < self.num_layers:
                        layer_input = h_layers[i].unsqueeze(1).expand(
                            B, self.window, self.hidden_size
                        )
                outputs.append(
                    h_layers[-1].unsqueeze(1).expand(B, self.window, self.hidden_size)
                )
            seq = torch.cat(outputs, dim=1)
        if self.return_sequences:
            return self.output_proj(seq)
        return self.output_proj(seq[:, -1, :])


__all__ = ["STEParallelCfCCell", "STEParallelCfCNetwork", "_topk_binary_mask"]
