"""Time-Domain Self-Attention CfC (TDSA-CfC) (PRD #10-114, Round 152, 2026-06-15).

Implements a parallel time-domain self-attention stream, projected
to input_size, then concatenated with x as input to CfC. Inspired
by the Transformer (Vaswani 2017) and tests whether attention
outperforms conv (MSDC 151) for parallel context on time series.

The key idea: self-attention over the time axis lets the model
attend to whichever timestep is most relevant, unconstrained by
local windows (conv) or no receptive field (linear)::

    # Self-attention over time axis
    q = Linear_q(x)  # [B, T, attn_dim]
    k = Linear_k(x)  # [B, T, attn_dim]
    v = Linear_v(x)  # [B, T, attn_dim]
    attn = softmax(q @ k.transpose(-1, -2) / sqrt(attn_dim))  # [B, T, T]
    c = attn @ v  # [B, T, attn_dim]
    # Project to input_size
    c = Linear_o(c)  # [B, T, D]
    # Concat with x
    aug_x = concat([x, c], dim=-1)  # [B, T, 2D]
    # Standard CfC with augmented input
    h = CfCCell(aug_x, h)

This is structurally different from:
- **MSDC 151 (strictly positive 14th)**: MSDC uses multi-scale conv
  (local windows, receptive fields 1/3/5). TDSA uses self-attention
  (full sequence).
- **TCC 149 (target-dep 8th)**: TCC uses single-K conv. TDSA uses
  attention (no receptive field constraint).
- **LiNo 150 (target-dep 9th)**: LiNo uses linear projection (no
  receptive field, no per-timestep weighting). TDSA uses attention
  (learns per-timestep weights).
- **Transformer (Vaswani 2017)**: Pure attention REPLACES recurrence.
  TDSA ADDS attention as parallel context, KEEPS the recurrent step.

Risks:
- O(T²) memory and compute (manageable for T=32).
- May overfit on T=32 with only 30 epochs.
- Attention output is dense (every timestep gets non-zero contribution
  from all previous).

Audit context (91-151):
- 14 strictly positive (preserves recurrent step + adds structure)
- 9 target-dep (input-side processing, bidi, SCRN, Time-Decay, TCC,
  LiNo)
- 20 negatives (per-step mods, alternatives, regularizers,
  bottlenecks, redundant info, replacements, long-α SCRN, Clockwork)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from lnn.core.cfc import CfCCell


class TimeDomainSelfAttentionCfCCell(nn.Module):
    """TDSA-CfC cell: parallel time-domain self-attention + CfC.

    Single-head (or multi-head) self-attention over the time axis,
    projected to input_size, then concatenated with x as input to CfC.

    Args:
        input_size: input feature dimension D.
        hidden_size: hidden state dimension.
        num_heads: number of attention heads (default 1).
        attn_dim: attention dimension (default = input_size).
        causal: if True, mask future positions (default True).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_heads: int = 1,
        attn_dim: int | None = None,
        causal: bool = True,
    ):
        super().__init__()
        if attn_dim is None:
            attn_dim = input_size
        if attn_dim % num_heads != 0:
            raise ValueError(
                f"attn_dim ({attn_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attn_dim = attn_dim
        self.head_dim = attn_dim // num_heads
        self.causal = causal

        # Linear projections for Q, K, V.
        self.q_proj = nn.Linear(input_size, attn_dim)
        self.k_proj = nn.Linear(input_size, attn_dim)
        self.v_proj = nn.Linear(input_size, attn_dim)
        # Output projection: from attn_dim back to input_size.
        self.o_proj = nn.Linear(attn_dim, input_size)

        # CfC cell takes aug_x = concat([x, c]) → 2 * input_size.
        self.cfc = CfCCell(2 * input_size, hidden_size, n_tau=1)

        # Causal mask (registered as buffer).
        mask = torch.triu(torch.ones(1, 1, 1, 1), diagonal=1)  # placeholder
        # Will be initialized in forward; but we register a max-size buffer
        # to avoid recomputing.

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        """Compute self-attention output.

        Args:
            x: input of shape [B, T, input_size].
        Returns:
            Attention output of shape [B, T, input_size].
        """
        B, T, D = x.shape
        H = self.num_heads
        d_k = self.head_dim

        # Project to Q, K, V.
        q = self.q_proj(x).view(B, T, H, d_k).transpose(1, 2)  # [B, H, T, d_k]
        k = self.k_proj(x).view(B, T, H, d_k).transpose(1, 2)  # [B, H, T, d_k]
        v = self.v_proj(x).view(B, T, H, d_k).transpose(1, 2)  # [B, H, T, d_k]

        # Attention scores.
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)  # [B, H, T, T]

        # Causal mask.
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1,
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))

        # Softmax.
        attn = torch.softmax(scores, dim=-1)  # [B, H, T, T]

        # Apply attention to V.
        c = torch.matmul(attn, v)  # [B, H, T, d_k]
        c = c.transpose(1, 2).contiguous().view(B, T, self.attn_dim)  # [B, T, attn_dim]

        # Project to input_size.
        c = self.o_proj(c)  # [B, T, input_size]
        return c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass over the full sequence.

        Args:
            x: input of shape [B, T, input_size].
        Returns:
            Hidden states of shape [B, T, hidden_size].
        """
        B, T, D = x.shape
        device, dtype = x.device, x.dtype

        # NaN handling: zero-fill input.
        x_clean = torch.nan_to_num(x, nan=0.0)

        # Self-attention.
        c = self._attention(x_clean)  # [B, T, D]

        # Concat with x.
        aug_x = torch.cat([x_clean, c], dim=-1)  # [B, T, 2D]

        # Standard CfC with augmented input.
        h = torch.zeros(B, self.cfc.hidden_size, device=device, dtype=dtype)
        outputs = []
        for t in range(T):
            aug_x_t = aug_x[:, t, :]
            h = self.cfc(aug_x_t, h)
            outputs.append(h)
        return torch.stack(outputs, dim=1)


class TimeDomainSelfAttentionCfCStackedNetwork(nn.Module):
    """Stacked TDSA-CfC network (PRD #10-114).

    Args:
        input_size: input feature dimension.
        hidden_size: hidden dimension (per layer).
        output_size: output feature dimension.
        num_layers: number of stacked TDSA-CfC cells.
        num_heads: number of attention heads per cell (default 1).
        attn_dim: attention dimension (default = input_size).
        causal: if True, mask future positions (default True).
        return_sequences: if True, return per-timestep outputs.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        num_heads: int = 1,
        attn_dim: int | None = None,
        causal: bool = True,
        return_sequences: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.return_sequences = return_sequences

        # Each layer's input size: layer 0 takes raw input, layer i>0
        # takes the previous layer's CfC output (hidden_size).
        # But the TDSA cell needs to compute attention over its input,
        # so we need attn_dim flexibility. We use the same attn_dim
        # across layers (defaults to input_size at layer 0, then
        # hidden_size for deeper).
        # Simpler: each cell uses attn_dim=cell_input_size, num_heads=1.
        layer_in_sizes = [input_size] + [hidden_size] * (num_layers - 1)

        self.cells = nn.ModuleList()
        for li in range(num_layers):
            self.cells.append(
                TimeDomainSelfAttentionCfCCell(
                    input_size=layer_in_sizes[li],
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    attn_dim=None,  # default to cell's input_size
                    causal=causal,
                )
            )

        # Final head.
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: input of shape [B, T, input_size].
        Returns:
            Output of shape [B, T, output_size] if return_sequences else
            [B, output_size].
        """
        layer_input = x
        for cell in self.cells:
            layer_input = cell(layer_input)
        out = self.head(layer_input)
        if self.return_sequences:
            return out
        return out[:, -1, :]
