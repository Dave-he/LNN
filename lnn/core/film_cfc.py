"""FiLM-CfC (Feature-wise Linear Modulation) (PRD #10-115, Round 153, 2026-06-15).

Implements a context-driven multiplicative + additive modulation
of the CfC's hidden state::

    out = gamma * h + beta

where gamma and beta are computed from a context (either the
input itself or a global summary). Inspired by Perez et al. 2018
("FiLM: Visual Reasoning with a General Conditioning Layer") for
visual reasoning. The key insight: a small conditioning network
can produce powerful modulation parameters that change the
behavior of the main network.

The key idea::

    # Context (sequence-level summary or per-step)
    if ctx_mode == 'global':
        ctx = x.mean(dim=1, keepdim=True)  # [B, 1, D]
    elif ctx_mode == 'self':
        ctx = x  # [B, T, D]

    # Modulation parameters
    gamma = Linear_gamma(ctx)  # [B, T, hidden_size]
    beta = Linear_beta(ctx)    # [B, T, hidden_size]

    # Standard CfC
    h_t = CfCCell(x_t, h_{t-1})  # [B, T, hidden_size]

    # Modulated output
    out = gamma * h + beta  # [B, T, hidden_size]

This is structurally different from:
- **TCC 149 / MSDC 151 / TDSA 152**: ADD context to input via concat.
- **LiNo 150**: ADD context via sum.
- **FiLM 153 (this round)**: MODULATE hidden state via
  multiplicative + additive (first mechanism to use multiplicative
  interaction with hidden state in the audit).

Risks:
- Multiplicative modulation is powerful but can be unstable.
- gamma, beta from x_t (self) is essentially a per-step
  modification (loses pattern in audit).
- gamma, beta from global mean is constant across timesteps
  (may not have enough expressivity).

Audit context (91-152):
- 14 strictly positive (preserves recurrent step + adds structure)
- 9 target-dep (input-side processing, bidi, SCRN, Time-Decay, TCC,
  LiNo)
- 21 negatives (per-step mods, alternatives, regularizers,
  bottlenecks, redundant info, replacements, long-α SCRN, Clockwork,
  self-attention)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.cfc import CfCCell


class FiLMCfCCell(nn.Module):
    """FiLM-CfC cell: context-driven modulation of CfC hidden state.

    Args:
        input_size: input feature dimension D.
        hidden_size: hidden state dimension.
        ctx_mode: 'self' (per-step from x_t), 'global' (from global
            mean of x), or 'concat' (no modulation, just concat with
            global mean as input to CfC — control).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        ctx_mode: str = "global",
    ):
        super().__init__()
        assert ctx_mode in ("self", "global", "concat"), f"ctx_mode must be 'self', 'global', or 'concat', got {ctx_mode}"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ctx_mode = ctx_mode

        if ctx_mode in ("self", "global"):
            # Two modulation projections: gamma and beta.
            self.gamma_proj = nn.Linear(input_size, hidden_size)
            self.beta_proj = nn.Linear(input_size, hidden_size)
            # Standard CfC with raw input.
            self.cfc = CfCCell(input_size, hidden_size, n_tau=1)
        else:  # concat
            # CfC takes augmented input: x + global_mean (concatenated).
            self.cfc = CfCCell(2 * input_size, hidden_size, n_tau=1)

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

        if self.ctx_mode in ("self", "global"):
            # Context.
            if self.ctx_mode == "self":
                ctx = x_clean  # [B, T, D]
            else:  # global
                ctx = x_clean.mean(dim=1, keepdim=True)  # [B, 1, D]
                ctx = ctx.expand(-1, T, -1)  # [B, T, D]

            # Modulation parameters.
            gamma = self.gamma_proj(ctx)  # [B, T, hidden_size]
            beta = self.beta_proj(ctx)    # [B, T, hidden_size]

            # Standard CfC.
            h = torch.zeros(B, self.cfc.hidden_size, device=device, dtype=dtype)
            outputs = []
            for t in range(T):
                x_t = x_clean[:, t, :]
                h = self.cfc(x_t, h)
                outputs.append(h)
            h_seq = torch.stack(outputs, dim=1)  # [B, T, hidden_size]

            # Modulated output.
            out = gamma * h_seq + beta  # [B, T, hidden_size]
        else:  # concat
            # Concat x with global mean.
            ctx = x_clean.mean(dim=1, keepdim=True).expand(-1, T, -1)  # [B, T, D]
            aug_x = torch.cat([x_clean, ctx], dim=-1)  # [B, T, 2D]
            h = torch.zeros(B, self.cfc.hidden_size, device=device, dtype=dtype)
            outputs = []
            for t in range(T):
                aug_x_t = aug_x[:, t, :]
                h = self.cfc(aug_x_t, h)
                outputs.append(h)
            out = torch.stack(outputs, dim=1)  # [B, T, hidden_size]

        return out


class FiLMCfCStackedNetwork(nn.Module):
    """Stacked FiLM-CfC network (PRD #10-115).

    Args:
        input_size: input feature dimension.
        hidden_size: hidden dimension (per layer).
        output_size: output feature dimension.
        num_layers: number of stacked FiLM-CfC cells.
        ctx_mode: 'self', 'global', or 'concat' (per layer).
        return_sequences: if True, return per-timestep outputs.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        ctx_mode: str = "global",
        return_sequences: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.ctx_mode = ctx_mode
        self.return_sequences = return_sequences

        # All layers take the same input size (CfC input is x or
        # the previous layer's output, but we keep it as input_size
        # for the modulation projection — the modulation can operate
        # on hidden_size but uses input_size as the conditioning
        # signal). Wait, the modulation projects from input_size to
        # hidden_size. For layer i > 0, the input is the previous
        # layer's output (hidden_size). So we need to track this.

        # Simpler: each cell's input_size is its actual input size.
        layer_in_sizes = [input_size] + [hidden_size] * (num_layers - 1)

        self.cells = nn.ModuleList()
        for li in range(num_layers):
            self.cells.append(
                FiLMCfCCell(
                    input_size=layer_in_sizes[li],
                    hidden_size=hidden_size,
                    ctx_mode=ctx_mode,
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
