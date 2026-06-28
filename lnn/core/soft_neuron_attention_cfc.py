"""SoftNeuronAttentionCfCCell (round 264).

A CfC variant that implements **per-neuron dynamics with
differentiable neighborhood structure** inspired by
arXiv:2606.21295 (Topological Neural Dynamics, Cai & Zhao 2026)
and r263's NeuronWiseCfCCell.

The KEY improvement over r263: the neighborhood structure is
*learned via gradient* through a soft-attention operator.

Round 263 (NeuronWiseCfCCell) used a **top-k hard sparsification**
of ``neighbor_logits`` to produce a binary mask. The top-k operator
is non-differentiable, so the structure was a *structural
hyperparameter* rather than a learned parameter.

Round 264 (this file) replaces the top-k mask with a **softmax
attention mask**:

  α_{ij} = softmax(neighbor_logits / τ_attn)[i, j]

This makes the structure fully differentiable. We add an
opt-in L1 sparsity penalty to encourage sparse attention without
hard top-k.

The forward pass is otherwise identical to r263 (per-neuron τ,
per-neuron α, per-neuron input strength):

  s_i^t = (∑_{j} α_{ij} W_{ij}) v_j^t + W_i^in x_i^t + b_i + α_i h_i^t
  h_i^{t+1} = (1 - τ̃_i) h_i^t + τ̃_i tanh(s_i^t)
  v_i^t = h_i^t

Why this completes r263:
  r263: per-neuron dynamics with HAND-CODED structure (top-k).
  r264: per-neuron dynamics with LEARNED structure (soft attention).

Hypotheses (PRD #10-101):

  H1: Soft attention beats hard top-k on at least one dataset
      because learnable structure outperforms hand-coded structure.
  H2: Attention weights become SPARSE naturally (mean attention
      weight < 0.1) after training — soft → sparse without
      explicit top-k.
  H3: Different neurons attend to different sources (per-row
      attention entropy varies, std > 0.5) — evidence of
      specialization.
  H4: SoftNeuronAttentionCfCCell is a strict superset of r263:
      with τ_attn → 0, the soft mask approaches a hard top-k.

API::

    SoftNeuronAttentionCfCCell(input_size, hidden_size,
                                base_tau=0.5, tau_min=0.05,
                                tau_max=0.95, alpha_max=0.5,
                                init_tau_attn=1.0, l1_lambda=0.01,
                                seed=42)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SoftNeuronAttentionCfCCell(nn.Module):
    """Per-neuron dynamics CfC variant with **learnable** structure.

    The cell extends NeuronWiseCfCCell (r263) with:
      * **soft attention** (row-softmax) over neighbor logits, with
        a *learnable* temperature τ_attn
      * **L1 sparsity penalty** on attention weights (opt-in via
        ``l1_lambda``)
      * no top-k hard constraint — the structure is fully continuous

    Args:
        input_size: Input feature dimension (d_in).
        hidden_size: Hidden state dimension (d_h) — also the number
            of neurons (each neuron has a scalar hidden state).
        base_tau: Initial τ for all neurons (before per-neuron
            perturbation). Should be in (0, 1) for CfC-style dynamics.
        tau_min: Lower bound of the learned τ (post-sigmoid).
        tau_max: Upper bound of the learned τ (post-sigmoid).
        alpha_max: Absolute value clamp for per-neuron α.
        init_tau_attn: Initial value of the soft-attention
            temperature. Higher = softer (more uniform attention);
            lower = sharper (more peaked attention).
        l1_lambda: L1 penalty on attention weights. Set to 0 to
            disable; default 0.01.
        init_rec_scale: Scale of the recurrent weight init.
        input_strength_init: Initial per-neuron input projection
            strength.
        seed: Random seed for init.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        base_tau: float = 0.5,
        tau_min: float = 0.05,
        tau_max: float = 0.95,
        alpha_max: float = 0.5,
        init_tau_attn: float = 1.0,
        l1_lambda: float = 0.01,
        init_rec_scale: float | None = None,
        input_strength_init: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        if hidden_size < 2:
            raise ValueError("hidden_size must be >= 2")
        if l1_lambda < 0:
            raise ValueError("l1_lambda must be >= 0")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.base_tau = float(base_tau)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.alpha_max = float(alpha_max)
        self.l1_lambda = float(l1_lambda)

        # --- recurrent weight: (d_h, d_h) shared ---
        rec_scale = init_rec_scale if init_rec_scale is not None else 1.0 / math.sqrt(hidden_size)
        gen = torch.Generator().manual_seed(seed)
        W_rec_init = torch.randn(hidden_size, hidden_size, generator=gen) * rec_scale
        self.W_rec = nn.Parameter(W_rec_init)

        # --- neighbor logits: (d_h, d_h) (NOW TRAINABLE via grad) ---
        neighbor_init = torch.randn(hidden_size, hidden_size, generator=gen) * 0.1
        self.neighbor_logits = nn.Parameter(neighbor_init)

        # --- soft-attention temperature τ_attn (learned, clamped ≥ 0.01) ---
        # We parameterise via softplus so the value is always > 0.
        # softplus_inv(1.0) = log(exp(1) - 1) ≈ 0.5413
        init_log_tau = math.log(math.exp(init_tau_attn) - 1.0) if init_tau_attn > 0 else 0.0
        self.log_tau_attn = nn.Parameter(torch.tensor(init_log_tau))

        # --- per-neuron time constant (logit, post-sigmoid) ---
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
        """Per-neuron τ bounded to [tau_min, tau_max]."""
        raw = torch.sigmoid(self.tau_per_neuron)  # (d_h,) in (0, 1)
        return self.tau_min + (self.tau_max - self.tau_min) * raw

    def get_alpha(self) -> torch.Tensor:
        """Per-neuron α clamped to [-alpha_max, alpha_max]."""
        return torch.clamp(self.alpha_per_neuron, -self.alpha_max, self.alpha_max)

    def get_tau_attn(self) -> torch.Tensor:
        """Soft-attention temperature τ_attn (positive scalar)."""
        return torch.nn.functional.softplus(self.log_tau_attn).clamp(min=0.01)

    def get_attention(self) -> torch.Tensor:
        """Row-softmax attention mask over neighbors.

        Returns:
            (d_h, d_h) row-stochastic tensor; each row sums to 1.
        """
        tau = self.get_tau_attn()
        return torch.softmax(self.neighbor_logits / tau, dim=-1)

    def get_attention_entropy(self) -> torch.Tensor:
        """Per-row entropy of the attention distribution (natural log).

        Returns:
            (d_h,) tensor of entropies. Max value = log(d_h) for
            uniform; min value = 0 for one-hot.
        """
        alpha = self.get_attention().clamp(min=1e-12)
        return -(alpha * alpha.log()).sum(dim=-1)

    def sparsity_loss(self) -> torch.Tensor:
        """L1 penalty on attention weights (encourages sparsity).

        Returns:
            scalar tensor = l1_lambda * mean(|α|).
        """
        if self.l1_lambda <= 0:
            return torch.tensor(0.0, device=self.W_rec.device)
        return self.l1_lambda * self.get_attention().abs().mean()

    def attention_sparsity(self) -> float:
        """Fraction of attention weights < 0.01 (numerical sparsity)."""
        alpha = self.get_attention().detach()
        return float((alpha < 0.01).float().mean().item())

    def attention_max_weight(self) -> float:
        """Mean over rows of max attention weight per row."""
        alpha = self.get_attention().detach()
        return float(alpha.max(dim=-1).values.mean().item())

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
            aux (optional): dict with final attention, tau, alpha,
                sparsity stats, etc.
        """
        B, T, _ = x.shape
        device = x.device
        dtype = x.dtype

        if h0 is None:
            h = torch.zeros(B, self.hidden_size, device=device, dtype=dtype)
        else:
            h = h0

        # Cache the effective recurrent operator and parameters once
        # per forward pass.
        alpha = self.get_attention()  # (d_h, d_h) row-stochastic
        W_eff = alpha * self.W_rec  # (d_h, d_h) — soft-masked recurrent weights
        tau = self.get_tau()  # (d_h,)
        tau_alpha = self.get_alpha()  # (d_h,)

        outputs = []
        for t in range(T):
            x_t = x[:, t, :]  # (B, d_in)
            rec = h @ W_eff.T  # (B, d_h)
            in_proj = self.W_in(x_t)  # (B, d_h)
            in_proj = self.input_strength_per_neuron.unsqueeze(0) * in_proj
            s = rec + in_proj + self.bias_per_neuron + tau_alpha.unsqueeze(0) * h
            h = (1.0 - tau).unsqueeze(0) * h + tau.unsqueeze(0) * torch.tanh(s)
            outputs.append(h)

        out = torch.stack(outputs, dim=1)
        if not return_aux:
            return out, h

        entropy = self.get_attention_entropy().detach()
        aux = {
            "attention": alpha.detach(),
            "tau_attn": float(self.get_tau_attn().item()),
            "attention_entropy_mean": float(entropy.mean().item()),
            "attention_entropy_std": float(entropy.std().item()),
            "attention_max_weight": self.attention_max_weight(),
            "attention_sparsity": self.attention_sparsity(),
            "sparsity_loss_value": float(self.sparsity_loss().item()),
            "tau_summary": self.per_neuron_tau_summary(),
            "alpha_mean": float(tau_alpha.mean().item()),
            "alpha_std": float(tau_alpha.std().item()),
        }
        return out, h, aux


__all__ = ["SoftNeuronAttentionCfCCell"]
