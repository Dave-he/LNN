"""Round 102 — QuITE Query-Based Irregular TS Embedding (PRD #10-64).

Implements the QuITE plug-and-play embedding for Irregular Multivariate
Time Series (IMTS), based on arXiv:2605.28166 (Lim, ICML 2026) —
*QuITE: Query-Based Irregular Time-Series Embedding*.

The paper identifies that **the bottleneck in IMTS modeling is the
conventional embedding layer** (which assumes uniform sampling), not
the backbone. QuITE provides a plug-and-play replacement that:

1. Takes (irregular_time, irregular_value, mask) triples as input.
2. Embeds values and times into a shared ``d_model`` space.
3. Uses ``n_queries`` learnable query tokens to aggregate the irregular
   observations via a **single masked self-attention layer**.
4. Outputs ``(n_queries, d_model)`` latent tokens that are compatible
   with any standard MTS backbone (CfC, LSTM, MLP, etc.).

Key features:
- **Mask-aware**: handles missing observations (NaN values) via masking.
- **Single attention layer**: O(n_queries * T * d_model) — fast even for
  T=200 observations per sequence.
- **Plug-and-play**: outputs are flat ``(n_queries, d_model)`` and can
  be flattened to feed any backbone.
- **Time-aware**: separate time embedding captures sampling intervals.

Functions and classes:
- ``QueryIrregularEmbedding`` (nn.Module) — main module
- ``apply_quite_embedding(observations, times, mask, module)`` — forward
- ``quite_baseline_modes(observations, times, mask, mode)`` — baseline
  embeddings ('mean', 'concat', 'add') for ablation comparison
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinusoidal_time_embedding(
    times: torch.Tensor,
    d_model: int,
) -> torch.Tensor:
    """Sinusoidal positional encoding for irregular time stamps.

    Args:
        times: (B, T) tensor of irregular time stamps (in arbitrary units).
        d_model: embedding dimension.

    Returns:
        (B, T, d_model) time embeddings.
    """
    half = d_model // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(0, half, device=times.device, dtype=times.dtype)
        / max(half - 1, 1),
    )
    # (B, T, half)
    args = times.unsqueeze(-1) * freqs.view(1, 1, -1)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < d_model:
        emb = F.pad(emb, (0, d_model - emb.shape[-1]))
    return emb


def _build_attn_mask(mask: torch.Tensor) -> torch.Tensor:
    """Convert a (B, T) value mask into a (B, 1, 1, T) attention mask.

    Args:
        mask: (B, T) bool/float mask — True/1 = valid, False/0 = pad.

    Returns:
        (B, 1, 1, T) attention mask — True = keep, False = mask out.
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()
    return mask.unsqueeze(1).unsqueeze(1)


def _build_attn_mask(mask: torch.Tensor) -> torch.Tensor:
    """Convert a (B, T) value mask into a (B, 1, 1, T) attention mask.

    Args:
        mask: (B, T) bool/float mask — True/1 = valid, False/0 = pad.

    Returns:
        (B, 1, 1, T) attention mask — True = keep, False = mask out.
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()
    return mask.unsqueeze(1).unsqueeze(1)


class QueryIrregularEmbedding(nn.Module):
    """QuITE-style query-based irregular time series embedding.

    Uses ``n_queries`` learnable query tokens to aggregate irregular
    observations via a single masked multi-head self-attention layer.

    Args:
        d_input: number of input features (D).
        n_queries: number of learnable query tokens (output dimension).
        d_model: embedding dimension (must match backbone input dim).
        n_heads: number of attention heads. Must divide d_model.
        dropout: attention dropout probability.
    """

    def __init__(
        self,
        d_input: int = 1,
        n_queries: int = 8,
        d_model: int = 16,
        n_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})",
            )
        if n_queries < 1:
            raise ValueError(f"n_queries must be >= 1, got {n_queries}")
        self.d_input = d_input
        self.n_queries = n_queries
        self.d_model = d_model
        self.n_heads = n_heads
        # Value projection: (D,) → (d_model,)
        self.value_proj = nn.Linear(d_input, d_model)
        # Learnable query tokens: (n_queries, d_model)
        self.queries = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)
        # Self-attention layer
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Output norm
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        observations: torch.Tensor,
        times: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Aggregate irregular observations via learnable query tokens.

        Args:
            observations: (B, T, D) input values. D = n_features.
                NaN values are treated as missing (masked).
            times: (B, T) input time stamps. Should be normalized to [0, 1]
                or similar range.
            mask: (B, T) bool/float mask — True/1 = valid, False/0 = pad.
                If None, all positions are valid. NaN observations are
                always treated as invalid regardless of mask.

        Returns:
            (B, n_queries, d_model) latent tokens — feed to any backbone.
        """
        B, T, D = observations.shape
        if D != self.d_input:
            raise ValueError(
                f"Expected D={self.d_input} features, got {D}",
            )
        # Build mask: missing if NaN OR mask==0
        obs_mask = torch.isfinite(observations).all(dim=-1)  # (B, T)
        if mask is None:
            mask = torch.ones(B, T, device=observations.device, dtype=torch.bool)
        elif mask.dtype != torch.bool:
            mask = mask.bool()
        mask = mask & obs_mask  # (B, T)
        # Replace NaN observations with 0 to avoid NaN propagation in value_proj
        clean_obs = torch.where(
            obs_mask.unsqueeze(-1), observations, torch.zeros_like(observations),
        )
        # Embed values: (B, T, D) → (B, T, d_model)
        value_emb = self.value_proj(clean_obs)  # (B, T, d_model)
        # Time embedding
        time_emb = _sinusoidal_time_embedding(times, self.d_model)  # (B, T, d_model)
        # Combine value + time
        kv = value_emb + time_emb  # (B, T, d_model)
        # Build queries: (B, n_queries, d_model)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        # MultiheadAttention expects key_padding_mask of shape (B, T)
        # where True = IGNORED (pad). We invert our mask.
        key_padding_mask = ~mask  # True = pad
        # Self-attention with queries as Q, observations as K/V
        out, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask)
        # Residual + norm
        out = self.norm(q + out)
        return out


def apply_quite_embedding(
    observations: torch.Tensor,
    times: torch.Tensor,
    mask: torch.Tensor | None,
    module: QueryIrregularEmbedding,
) -> torch.Tensor:
    """Forward pass wrapper for ``QueryIrregularEmbedding``.

    Args:
        observations: (B, T, D) input values.
        times: (B, T) input time stamps.
        mask: (B, T) bool/float mask (True = valid) or None.
        module: the QuITE module.

    Returns:
        (B, n_queries, d_model) latent tokens.
    """
    return module(observations, times, mask)


def quite_baseline_modes(
    observations: torch.Tensor,
    times: torch.Tensor,
    mask: torch.Tensor | None,
    mode: str = "mean",
) -> torch.Tensor:
    """Baseline embeddings for ablation: 'mean', 'concat', 'add'.

    These are simple time/value aggregation strategies that DON'T use
    learnable queries — for comparison against QuITE.

    Args:
        observations: (B, T, D) input values. NaN treated as missing.
        times: (B, T) input time stamps.
        mask: (B, T) bool/float mask or None.
        mode: one of 'mean', 'concat', 'add'.
            - 'mean': average over time, weighted by mask → (B, D)
            - 'concat': concatenate last value with time → (B, D+1)
            - 'add': value + time embedding (sinusoidal) → (B, D+d_model)

    Returns:
        Aggregated tensor (shape depends on mode).
    """
    if mode not in ("mean", "concat", "add"):
        raise ValueError(f"mode must be one of mean/concat/add, got {mode}")
    B, T, D = observations.shape
    # Build combined mask
    obs_mask = torch.isfinite(observations).all(dim=-1)  # (B, T)
    if mask is None:
        mask = torch.ones(B, T, device=observations.device, dtype=torch.bool)
    elif mask.dtype != torch.bool:
        mask = mask.bool()
    full_mask = mask & obs_mask  # (B, T)
    if mode == "mean":
        # Weighted mean over time
        # Replace NaN with 0 for the mean
        clean_obs = torch.where(obs_mask.unsqueeze(-1), observations, torch.zeros_like(observations))
        weights = full_mask.float().unsqueeze(-1)  # (B, T, 1)
        summed = (clean_obs * weights).sum(dim=1)  # (B, D)
        count = weights.sum(dim=1).clamp(min=1.0)  # (B, 1)
        return summed / count  # (B, D)
    if mode == "concat":
        # Take last valid observation + last time
        # If no valid, use 0
        clean_obs = torch.where(
            obs_mask.unsqueeze(-1), observations, torch.zeros_like(observations),
        )
        # Find last valid index per batch element
        idx = (full_mask.long() * torch.arange(T, device=observations.device).unsqueeze(0)).argmax(dim=1)
        idx_exp = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, D)
        last_obs = clean_obs.gather(1, idx_exp).squeeze(1)  # (B, D)
        last_time = times.gather(1, idx.unsqueeze(-1)).squeeze(-1)  # (B,)
        return torch.cat([last_obs, last_time.unsqueeze(-1)], dim=-1)  # (B, D+1)
    # mode == "add"
    # Add sinusoidal time embedding to the value
    d_model = D
    time_emb = _sinusoidal_time_embedding(times, d_model)  # (B, T, d_model)
    # For each timestep, weight by mask and mean
    clean_obs = torch.where(obs_mask.unsqueeze(-1), observations, torch.zeros_like(observations))
    weights = full_mask.float().unsqueeze(-1)  # (B, T, 1)
    summed_obs = (clean_obs * weights).sum(dim=1)  # (B, D)
    summed_time = (time_emb * weights).sum(dim=1)  # (B, d_model)
    count = weights.sum(dim=1).clamp(min=1.0)  # (B, 1)
    out = (summed_obs + summed_time) / count  # (B, D)
    return out


__all__ = [
    "QueryIrregularEmbedding",
    "apply_quite_embedding",
    "quite_baseline_modes",
]
