"""Bidirectional CfC (PRD #10-106, 2026-06-15).

Implements the Bidirectional Recurrent Neural Network idea from
Schuster & Paliwal (1997, IEEE Transactions on Signal Processing)
applied to CfC.

Standard (unidirectional) CfC processes the input sequence in one
direction (forward). At timestep t, the hidden state h_t depends
only on x[0..t] and h[0..t-1].

Bidirectional CfC runs two separate recurrent passes (forward and
backward), then combines their hidden states::

    # Forward pass
    h_fwd[0] = 0
    h_fwd[t+1] = forward_cell(x_t, h_fwd[t])    for t in [0..T-1]

    # Backward pass
    h_bwd[T] = 0
    h_bwd[t] = backward_cell(x_t, h_bwd[t+1])   for t in [T-1..0]

    # Combined hidden state at each t
    h_combined[t] = concat(h_fwd[t+1], h_bwd[t])

The forward pass sees ``x[0..t]`` (past context); the backward
pass sees ``x[T..t]`` (future context). The combined hidden
state at each t has access to the FULL sequence.

This is a STRUCTURAL ADDITION (not a per-step modification) and
follows the audit pattern where structural additions have higher
POSITIVE probability than per-step modifications.

This module contains:

- **BidirectionalCfCCell**: forward + backward CfC cells,
  concatenation output.
- **BidirectionalWeightedCfCCell**: forward + backward with
  learned per-timestep weighting.
- **BidirectionalCfCStackedNetwork**: stack of bidirectional CfC.

Risks:

- 2x parameter count (forward + backward cells).
- May overfit on noisy data (model has access to future noise).
- NaN handling must be careful in both passes.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.cfc import CfCCell


class BidirectionalCfCCell(nn.Module):
    """Bidirectional CfC cell: forward + backward, concat outputs.

    Args:
        input_size: input feature dimension.
        hidden_size: hidden state dimension (per direction).
        merge_mode: "concat" (concatenate forward and backward)
            or "sum" (sum forward and backward).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        merge_mode: str = "concat",
    ):
        super().__init__()
        assert merge_mode in ("concat", "sum"), f"merge_mode must be in (concat, sum), got {merge_mode}"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.merge_mode = merge_mode

        self.forward_cell = CfCCell(input_size, hidden_size, n_tau=1)
        self.backward_cell = CfCCell(input_size, hidden_size, n_tau=1)

        if merge_mode == "concat":
            self.output_size = 2 * hidden_size
        else:
            self.output_size = hidden_size

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass over the full sequence.

        Args:
            x: input of shape [B, T, input_size].
        Returns:
            Combined hidden states of shape [B, T, output_size].
        """
        B, T, _ = x.shape
        # Forward pass.
        h_fwd = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
        fwd_outputs = []
        for t in range(T):
            x_t = torch.nan_to_num(x[:, t, :], nan=0.0)
            h_fwd = self.forward_cell(x_t, h_fwd)
            fwd_outputs.append(h_fwd)

        # Backward pass.
        h_bwd = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
        bwd_outputs = [None] * T
        for t in range(T - 1, -1, -1):
            x_t = torch.nan_to_num(x[:, t, :], nan=0.0)
            h_bwd = self.backward_cell(x_t, h_bwd)
            bwd_outputs[t] = h_bwd

        # Stack and merge.
        fwd_stack = torch.stack(fwd_outputs, dim=1)  # [B, T, hidden]
        bwd_stack = torch.stack(bwd_outputs, dim=1)  # [B, T, hidden]
        if self.merge_mode == "concat":
            return torch.cat([fwd_stack, bwd_stack], dim=-1)
        return fwd_stack + bwd_stack


class BidirectionalWeightedCfCCell(nn.Module):
    """Bidirectional CfC with learned per-timestep weighting.

    Computes a per-timestep weight α_t that combines forward and
    backward: ``h_combined[t] = α_t * h_fwd[t] + (1 - α_t) * h_bwd[t]``
    where ``α_t = σ(W_α [h_fwd[t], h_bwd[t]])``.

    This lets the model dynamically decide how much to use forward
    vs backward context at each timestep.

    Args:
        input_size: input feature dimension.
        hidden_size: hidden state dimension.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.forward_cell = CfCCell(input_size, hidden_size, n_tau=1)
        self.backward_cell = CfCCell(input_size, hidden_size, n_tau=1)

        # Learned weighting: α = σ(W_α [h_fwd, h_bwd]).
        self.alpha_proj = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass over the full sequence.

        Args:
            x: input of shape [B, T, input_size].
        Returns:
            Combined hidden states of shape [B, T, hidden_size].
        """
        B, T, _ = x.shape
        # Forward pass.
        h_fwd = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
        fwd_outputs = []
        for t in range(T):
            x_t = torch.nan_to_num(x[:, t, :], nan=0.0)
            h_fwd = self.forward_cell(x_t, h_fwd)
            fwd_outputs.append(h_fwd)

        # Backward pass.
        h_bwd = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
        bwd_outputs = [None] * T
        for t in range(T - 1, -1, -1):
            x_t = torch.nan_to_num(x[:, t, :], nan=0.0)
            h_bwd = self.backward_cell(x_t, h_bwd)
            bwd_outputs[t] = h_bwd

        fwd_stack = torch.stack(fwd_outputs, dim=1)
        bwd_stack = torch.stack(bwd_outputs, dim=1)
        # Per-timestep alpha.
        alpha = torch.sigmoid(self.alpha_proj(torch.cat([fwd_stack, bwd_stack], dim=-1)))
        # Weighted combination.
        return alpha * fwd_stack + (1.0 - alpha) * bwd_stack


class BidirectionalCfCStackedNetwork(nn.Module):
    """Stacked Bidirectional CfC network (PRD #10-106).

    Args:
        input_size: input feature dimension.
        hidden_size: hidden state dimension (per direction).
        output_size: output feature dimension.
        num_layers: number of stacked bidirectional cells.
        merge_mode: "concat" or "sum" or "weighted".
        return_sequences: if True, return outputs at every
            timestep; else return only the last.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        merge_mode: str = "concat",
        return_sequences: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        assert merge_mode in ("concat", "sum", "weighted"), f"merge_mode must be in (concat, sum, weighted), got {merge_mode}"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.merge_mode = merge_mode
        self.return_sequences = return_sequences

        # Each layer's bidirectional cell.
        self.cells = nn.ModuleList()
        for li in range(num_layers):
            if li == 0:
                in_size = input_size
            else:
                # Input dim is determined by previous layer's output.
                if merge_mode == "concat":
                    in_size = 2 * hidden_size
                elif merge_mode == "sum":
                    in_size = hidden_size
                else:  # weighted
                    in_size = hidden_size

            if merge_mode == "weighted":
                self.cells.append(
                    BidirectionalWeightedCfCCell(
                        input_size=in_size,
                        hidden_size=hidden_size,
                    )
                )
            else:
                self.cells.append(
                    BidirectionalCfCCell(
                        input_size=in_size,
                        hidden_size=hidden_size,
                        merge_mode=merge_mode,
                    )
                )

        # Final head projects from cell output to output_size.
        if merge_mode == "concat":
            head_in = 2 * hidden_size
        else:
            head_in = hidden_size
        self.head = nn.Linear(head_in, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: input of shape [B, T, input_size].
        Returns:
            Output of shape [B, T, output_size] if
            ``return_sequences=True`` else [B, output_size].
        """
        layer_input = x
        for cell in self.cells:
            layer_input = cell(layer_input)  # [B, T, cell_output_size]

        out = self.head(layer_input)
        if self.return_sequences:
            return out
        return out[:, -1, :]
