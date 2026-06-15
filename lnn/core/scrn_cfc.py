"""Slow Context RNN CfC (PRD #10-108, Round 146, 2026-06-15).

Implements the Slow Recurrent Neural Network idea from Mikolov et al.
2015 ("LSTM with Working Memory" / "Slow Recurrent Neural Network",
SCRN) applied to CfC.

The original SCRN (Mikolov 2015) augments an RNN with a parallel
slow context unit that low-pass-filters the input stream::

    s_t = α * s_{t-1} + (1 - α) * (W_s x_t)   (slow context, EMA of input)
    h_t = RNN(x_t, h_{t-1})                    (fast hidden, unchanged)
    h_combined_t = [h_t, s_t]                  (concat hidden + slow)

This is a STRUCTURAL ADDITION (not a per-step modification) and
PRESERVES both x and h, per the 91-145 audit pattern:
- 5 of 5 input-side winners (LN 135, conv 137, GLU+skip 139,
  decoupled/IndRNN 143, bidi_concat 144) preserve x.
- 16 negatives include all input-side REPLACEMENTS (diff_only 145,
  multiplicative integration 142, etc.).
- Per-step modifications to the recurrent step (ATC 141, MI 142,
  zoneout 136, etc.) all lose.

Risks:
- 2x hidden dim downstream (concat means second-layer input grows).
- α optimization requires care: we use logit-alpha parameterization
  to avoid sigmoid saturation. Init at 0.95 with logit_alpha ≈ 2.94.
- NaN handling: zero-fill input before slow context (s_t stays bounded).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from lnn.core.cfc import CfCCell, CfCNetwork


class SlowContextEncoder(nn.Module):
    """EMA-based slow context unit.

    Computes::

        s_t = α * s_{t-1} + (1 - α) * (W_s x_t)

    where α is a learnable scalar (sigmoid-parameterized).

    Args:
        input_size: input feature dimension D.
        slow_size: output dimension of slow context (= hidden_size for concat).
        alpha_init: initial value of α (in [0, 1)).
    """

    def __init__(self, input_size: int, slow_size: int, alpha_init: float = 0.95):
        super().__init__()
        assert 0.0 <= alpha_init < 1.0, f"alpha_init must be in [0, 1), got {alpha_init}"
        self.input_size = input_size
        self.slow_size = slow_size
        self.proj = nn.Linear(input_size, slow_size)
        # logit-alpha for unconstrained optimization.
        # logit(α) = log(α / (1 - α))
        init_logit = math.log(alpha_init / (1.0 - alpha_init))
        self.logit_alpha = nn.Parameter(torch.tensor(init_logit))

    @property
    def alpha(self) -> torch.Tensor:
        """Current α value (sigmoid of logit_alpha)."""
        return torch.sigmoid(self.logit_alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute slow context stream.

        Args:
            x: input of shape [B, T, D]. NaNs are zero-filled.
        Returns:
            Slow context of shape [B, T, slow_size].
        """
        B, T, _ = x.shape
        x_clean = torch.nan_to_num(x, nan=0.0)
        s = torch.zeros(B, self.slow_size, device=x.device, dtype=x.dtype)
        outputs = []
        alpha = self.alpha
        for t in range(T):
            x_proj = self.proj(x_clean[:, t, :])
            s = alpha * s + (1.0 - alpha) * x_proj
            outputs.append(s)
        return torch.stack(outputs, dim=1)  # [B, T, slow_size]


class SCRNCfCCell(nn.Module):
    """CfC cell augmented with Slow Context (SCRN).

    Combines a CfC cell with a SlowContextEncoder via concatenation::

        h_t = CfCCell(x_t, h_{t-1})              # hidden (unchanged)
        s_t = α * s_{t-1} + (1-α) * (W_s x_t)    # slow context
        h_combined_t = [h_t, s_t]                # concat

    Output dim is hidden_size + slow_size (default: 2 × hidden).

    Args:
        input_size: input feature dimension D.
        hidden_size: CfC hidden dimension.
        slow_size: slow context dimension (default: same as hidden_size).
        alpha_init: initial value of α.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        slow_size: int | None = None,
        alpha_init: float = 0.95,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.slow_size = slow_size if slow_size is not None else hidden_size
        self.output_size = hidden_size + self.slow_size

        self.cfc_cell = CfCCell(input_size, hidden_size, n_tau=1)
        self.slow_encoder = SlowContextEncoder(
            input_size=input_size, slow_size=self.slow_size, alpha_init=alpha_init,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass over the full sequence.

        Args:
            x: input of shape [B, T, input_size].
        Returns:
            Combined [hidden, slow] of shape [B, T, hidden_size + slow_size].
        """
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
        h_outputs = []
        for t in range(T):
            x_t = torch.nan_to_num(x[:, t, :], nan=0.0)
            h = self.cfc_cell(x_t, h)
            h_outputs.append(h)
        h_stack = torch.stack(h_outputs, dim=1)  # [B, T, hidden]
        s_stack = self.slow_encoder(x)  # [B, T, slow]
        return torch.cat([h_stack, s_stack], dim=-1)


class SCRNCfCStackedNetwork(nn.Module):
    """Stacked SCRN-CfC network (PRD #10-108).

    Args:
        input_size: input feature dimension.
        hidden_size: CfC hidden dimension.
        output_size: output feature dimension.
        slow_size: slow context dimension (default: same as hidden_size).
        num_layers: number of stacked SCRN cells.
        alpha_init: initial value of α.
        return_sequences: if True, return per-timestep outputs.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        slow_size: int | None = None,
        num_layers: int = 2,
        alpha_init: float = 0.95,
        return_sequences: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.slow_size = slow_size if slow_size is not None else hidden_size
        self.num_layers = num_layers
        self.alpha_init = alpha_init
        self.return_sequences = return_sequences

        self.cells = nn.ModuleList()
        for li in range(num_layers):
            if li == 0:
                in_size = input_size
            else:
                # Previous layer's output is hidden + slow.
                in_size = hidden_size + self.slow_size
            self.cells.append(
                SCRNCfCCell(
                    input_size=in_size,
                    hidden_size=hidden_size,
                    slow_size=self.slow_size,
                    alpha_init=alpha_init,
                )
            )

        # Final head.
        self.head = nn.Linear(hidden_size + self.slow_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: input of shape [B, T, input_size].
        Returns:
            Output of shape [B, T, output_size] if return_sequences else [B, output_size].
        """
        layer_input = x
        for cell in self.cells:
            layer_input = cell(layer_input)
        out = self.head(layer_input)
        if self.return_sequences:
            return out
        return out[:, -1, :]
