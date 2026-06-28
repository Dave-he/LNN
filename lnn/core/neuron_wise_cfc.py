"""NeuronWiseCfCCell (round 263).

A CfC variant that implements **per-neuron dynamics** inspired by
arXiv:2606.21295 (Topological Neural Dynamics, Cai & Zhao 2026).

Each neuron ``i`` has its own:

  1. **Learnable neighborhood** N(i) — enforced via top-k sparsification
     of a learnable adjacency ``neighbor_logits`` so each neuron has a
     bounded in-degree.
  2. **Per-neuron time constant** ``τ_i ∈ [τ_min, τ_max]`` parameterised
     in log-space and sigmoid-bounded. This breaks the shared-τ
     assumption of the original CfC and lets neurons evolve at
     heterogeneous rates (TND's central claim).
  3. **Per-neuron self-feedback** ``α_i ∈ [-0.5, 0.5]`` (clamped) for
     stable local recurrence.
  4. **Per-neuron bias** ``b_i``.

The forward pass is a CfC-style closed-form leaky integrator applied
per-neuron (not per-layer):

  s_i^t = (∑_{j∈N(i)} M_{ij} W_{ij}) v_j^t + W_i^in x_i^t + b_i + α_i h_i^t
  h_i^{t+1} = (1 - τ̃_i) h_i^t + τ̃_i tanh(s_i^t)
  v_i^t = h_i^t

where ``τ̃_i = sigmoid(tau_per_neuron[i]) ∈ (0, 1)`` is the CfC-style
normalised time constant, and ``M`` is the sparse adjacency.

Why this closes the TND gap identified in r257's bridge document:
  r257-262 operate on the **basin axis** (between per-basin centers).
  r263 operates on the **neuron axis** (between individual neurons in
  the same basin). After r263, the LNN+MoE stack has coverage on both.

Hypotheses (PRD #10-100):

  H1: NeuronWiseCfCCell beats plain CfC on toy_sin and structured.
  H2: The learned neighborhood mask becomes ASYMMETRIC (avg off-diag
      density > 0.3) after training.
  H3: Per-neuron τ values span a wide range (std(τ) > 0.3 × mean(τ)).
  H4: With density=1.0 and uniform τ, the cell degenerates to a
      recurrent network equivalent to CfC's gate × tanh form.

API::

    NeuronWiseCfCCell(input_size, hidden_size, density=0.3,
                      base_tau=0.5, tau_min=0.05, tau_max=0.95,
                      alpha_max=0.5, seed=42)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def sparse_topk_mask(
    logits: torch.Tensor,
    density: float,
) -> torch.Tensor:
    """Convert a real-valued logit matrix to a binary sparse mask.

    For each *row* (target neuron), keep the top-k columns by logit
    value where ``k = max(1, round(density * num_columns))``. The
    diagonal is **always** kept; if it isn't in the top-k, the
    lowest-ranked top-k entry is replaced. This guarantees the
    row sum equals exactly ``k``.

    Args:
        logits: (n, n) learnable per-edge logits.
        density: Fraction of edges to keep per row (∈ (0, 1]).

    Returns:
        (n, n) binary mask (float), where ``mask[i, j] = 1`` means
        neuron j is in neuron i's in-neighborhood.
    """
    n = logits.shape[0]
    k = max(1, int(round(density * n)))
    k = min(k, n)

    # Top-k indices per row.
    topk_idx = torch.topk(logits, k=k, dim=-1).indices  # (n, k)
    mask = torch.zeros_like(logits)
    mask.scatter_(1, topk_idx, 1.0)

    # Ensure self-edge is in the mask. If diagonal was NOT in top-k,
    # the lowest-ranked top-k entry is replaced with the diagonal.
    diag_idx = torch.arange(n, device=logits.device)
    diag_in_mask = mask[diag_idx, diag_idx] > 0.5  # (n,) bool

    if not bool(diag_in_mask.all()):
        # Replace the last column (lowest logit among top-k) with diagonal
        # for rows where diagonal wasn't selected.
        rows_to_fix = ~diag_in_mask
        # For those rows, find the column with the smallest top-k logit.
        # Equivalent: argmin over topk of logits.
        topk_logits = torch.gather(logits, 1, topk_idx)  # (n, k)
        worst = topk_logits.argmin(dim=-1)  # (n,)
        # Build new topk: same as before, except position 'worst' becomes diag_idx.
        new_idx = topk_idx.clone()
        new_idx[rows_to_fix, worst[rows_to_fix]] = diag_idx[rows_to_fix]
        mask = torch.zeros_like(logits)
        mask.scatter_(1, new_idx, 1.0)

    return mask


class NeuronWiseCfCCell(nn.Module):
    """Per-neuron dynamics CfC variant.

    The cell extends CfC with:
      * per-neuron τ (sigmoid-bounded in (0, 1))
      * per-neuron α (clamped to [-alpha_max, alpha_max])
      * learnable sparse neighborhood mask (top-k per row)
      * per-neuron input projection strength

    Args:
        input_size: Input feature dimension (d_in).
        hidden_size: Hidden state dimension (d_h) — also the number of
            neurons (each neuron has a scalar hidden state).
        density: Fraction of edges to keep per row in the learned
            neighborhood mask. 1.0 = fully connected (degenerates to
            dense recurrent net); 0.3 = sparse (TND-inspired).
        base_tau: Initial τ for all neurons (before per-neuron
            perturbation). Should be in (0, 1) for CfC-style dynamics.
        tau_min: Lower bound of the learned τ (post-sigmoid).
        tau_max: Upper bound of the learned τ (post-sigmoid).
        alpha_max: Absolute value clamp for per-neuron α.
        init_rec_scale: Scale of the recurrent weight init.
        input_strength_init: Initial per-neuron input projection strength.
        seed: Random seed for init.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        density: float = 0.3,
        base_tau: float = 0.5,
        tau_min: float = 0.05,
        tau_max: float = 0.95,
        alpha_max: float = 0.5,
        init_rec_scale: float | None = None,
        input_strength_init: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        if not (0.0 < density <= 1.0):
            raise ValueError(f"density must be in (0, 1], got {density}")
        if hidden_size < 2:
            raise ValueError("hidden_size must be >= 2 (need neighbor structure)")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.density = float(density)
        self.base_tau = float(base_tau)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.alpha_max = float(alpha_max)

        # --- recurrent weight: (d_h, d_h) shared, masked by sparse M ---
        rec_scale = init_rec_scale if init_rec_scale is not None else 1.0 / math.sqrt(hidden_size)
        gen = torch.Generator().manual_seed(seed)
        W_rec_init = torch.randn(hidden_size, hidden_size, generator=gen) * rec_scale
        self.W_rec = nn.Parameter(W_rec_init)

        # --- neighborhood logits: (d_h, d_h) ---
        neighbor_init = torch.randn(hidden_size, hidden_size, generator=gen) * 0.1
        # Encourage initial sparsity to roughly match density target.
        # We do NOT enforce sparsity through the loss — we only use
        # the top-k operator at forward time. The logits can grow in
        # magnitude during training.
        self.neighbor_logits = nn.Parameter(neighbor_init)

        # --- per-neuron time constant (logit, post-sigmoid) ---
        # Inverse sigmoid of base_tau gives an init logit near 0.
        init_tau_logit = math.log(base_tau / (1 - base_tau)) if 0 < base_tau < 1 else 0.0
        self.tau_per_neuron = nn.Parameter(torch.full((hidden_size,), init_tau_logit))

        # --- per-neuron self-feedback (clamped at forward) ---
        self.alpha_per_neuron = nn.Parameter(torch.zeros(hidden_size))

        # --- per-neuron bias ---
        self.bias_per_neuron = nn.Parameter(torch.zeros(hidden_size))

        # --- per-neuron input projection strength ---
        self.input_strength_per_neuron = nn.Parameter(
            torch.full((hidden_size,), input_strength_init)
        )

        # --- input projection: (d_in, d_h) ---
        self.W_in = nn.Linear(input_size, hidden_size, bias=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def get_tau(self) -> torch.Tensor:
        """Per-neuron τ bounded to [tau_min, tau_max].

        Returns:
            (d_h,) tensor of τ values.
        """
        raw = torch.sigmoid(self.tau_per_neuron)  # (d_h,) in (0, 1)
        return self.tau_min + (self.tau_max - self.tau_min) * raw

    def get_alpha(self) -> torch.Tensor:
        """Per-neuron α clamped to [-alpha_max, alpha_max].

        Returns:
            (d_h,) tensor of α values.
        """
        return torch.clamp(self.alpha_per_neuron, -self.alpha_max, self.alpha_max)

    def get_neighborhood_mask(self) -> torch.Tensor:
        """Sparse top-k per-row mask.

        Returns:
            (d_h, d_h) binary mask.
        """
        return sparse_topk_mask(self.neighbor_logits, self.density)

    def neighborhood_density(self, mask: torch.Tensor | None = None) -> float:
        """Fraction of edges kept (off-diagonal)."""
        if mask is None:
            mask = self.get_neighborhood_mask()
        assert mask is not None
        n = mask.shape[0]
        # Subtract self-edge contribution.
        off_diag = mask.clone()
        off_diag.fill_diagonal_(0.0)
        return float(off_diag.sum().item()) / float(n * (n - 1))

    def neighborhood_asymmetry(self, mask: torch.Tensor | None = None) -> float:
        """Mean |M[i,j] - M[j,i]| off-diagonal.

        A value of 0 = symmetric; a value of 1 = fully anti-symmetric.
        """
        if mask is None:
            mask = self.get_neighborhood_mask()
        assert mask is not None
        diff = (mask - mask.t()).abs()
        diff.fill_diagonal_(0.0)
        n = mask.shape[0]
        return float(diff.sum().item()) / float(n * (n - 1))

    def per_neuron_tau_summary(self) -> dict:
        """Summary statistics of the learned τ distribution."""
        tau = self.get_tau().detach()
        return {
            "mean": float(tau.mean().item()),
            "std": float(tau.std().item()),
            "min": float(tau.min().item()),
            "max": float(tau.max().item()),
            "cv": float(tau.std().item() / max(tau.mean().item(), 1e-8)),
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict]:
        """Run the cell on a sequence.

        Args:
            x: (B, T, d_in) input sequence.
            h0: (B, d_h) initial hidden state. Defaults to zeros.
            return_aux: If True, also return a dict of diagnostics.

        Returns:
            outputs: (B, T, d_h) hidden states at each step.
            h_final: (B, d_h) final hidden state.
            aux (optional): dict with final mask, tau, alpha, etc.
        """
        B, T, _ = x.shape
        device = x.device
        dtype = x.dtype

        if h0 is None:
            h = torch.zeros(B, self.hidden_size, device=device, dtype=dtype)
        else:
            h = h0

        # Cache the effective recurrent operator and parameters once
        # per forward pass (the mask is fixed at forward time).
        mask = self.get_neighborhood_mask()
        W_eff = mask * self.W_rec  # (d_h, d_h) — sparse recurrent weights
        tau = self.get_tau()  # (d_h,)
        alpha = self.get_alpha()  # (d_h,)

        outputs = []
        for t in range(T):
            x_t = x[:, t, :]  # (B, d_in)
            # Per-neuron signal s_i (B, d_h):
            #   s = h @ W_eff.T + (input_strength_per_neuron * (W_in @ x_t)) + bias + alpha * h
            rec = h @ W_eff.T  # (B, d_h)
            in_proj = self.W_in(x_t)  # (B, d_h)
            in_proj = self.input_strength_per_neuron.unsqueeze(0) * in_proj
            s = rec + in_proj + self.bias_per_neuron + alpha.unsqueeze(0) * h
            # CfC-style per-neuron leaky update.
            h = (1.0 - tau).unsqueeze(0) * h + tau.unsqueeze(0) * torch.tanh(s)
            outputs.append(h)

        out = torch.stack(outputs, dim=1)
        if not return_aux:
            return out, h

        aux = {
            "mask": mask.detach(),
            "neighborhood_density": self.neighborhood_density(mask),
            "neighborhood_asymmetry": self.neighborhood_asymmetry(mask),
            "tau_summary": self.per_neuron_tau_summary(),
            "alpha_mean": float(alpha.mean().item()),
            "alpha_std": float(alpha.std().item()),
        }
        return out, h, aux


__all__ = ["NeuronWiseCfCCell", "sparse_topk_mask"]
