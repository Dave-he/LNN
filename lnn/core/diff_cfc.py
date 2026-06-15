"""Difference Features CfC (PRD #10-107, Round 145, 2026-06-15).

Augments CfC input with finite differences (Box-Jenkins 1976, Hamilton 1994).

The classical ARIMA approach to time series uses finite differences to make
non-stationary data stationary. We apply the same idea as input feature
augmentation:

  Δx_t   = x_t - x_{t-1}        (1st order, local slope)
  Δ²x_t  = x_t - 2·x_{t-1} + x_{t-2}    (2nd order, local curvature)

These are passed to the CfC cell as ADDITIONAL input channels, alongside
the original x_t. This is a STRUCTURAL ADDITION (input-side processing)
per the 91-144 audit pattern: 5 of 5 target-dep winners are input-side
processing (LN 135, conv 137, GLU+skip 139, decoupled/IndRNN 143,
bidi_concat 144).

Variants:
  - DiffCfCConcat1: input = [x, Δx]                          (2 × D dim)
  - DiffCfCConcat2: input = [x, Δx, Δ²x]                     (3 × D dim)
  - DiffCfCOnly1:   input = [Δx]                             (1 × D dim)
  - DiffCfCOnly2:   input = [Δx, Δ²x]                        (2 × D dim)

Risks:
  - No new learnable params in the difference encoder itself.
  - Input dim grows by 1-3x; the CfC's input projection (Linear) does
    grow with input dim, so total params grow ~10-20% for hidden=16.
  - Difference of NaN is NaN. We zero-fill x BEFORE computing diffs, so
    difference features are well-defined.
  - On noisy data (random_irr), difference features amplify noise.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.cfc import CfCCell, CfCNetwork


class DifferenceInputEncoder(nn.Module):
    """Pre-computes finite differences and concatenates with x.

    Args:
        input_size: original input dimension D.
        n_diff: number of finite differences (0=none, 1=Δx, 2=Δx+Δ²x).
        mode: "concat" ([x, Δx, ...]) or "diff_only" ([Δx, Δ²x]).

    Output dim:
        - mode="concat" + n_diff=k  ->  (1+k) × D
        - mode="diff_only" + n_diff=k  ->  k × D
    """

    def __init__(self, input_size: int, n_diff: int = 1, mode: str = "concat"):
        super().__init__()
        assert n_diff >= 0, f"n_diff must be >= 0, got {n_diff}"
        assert mode in ("concat", "diff_only"), f"mode must be in (concat, diff_only), got {mode}"
        if n_diff == 0 and mode == "diff_only":
            raise ValueError("n_diff=0 with diff_only has no output")
        self.input_size = input_size
        self.n_diff = n_diff
        self.mode = mode
        if mode == "concat":
            self.output_size = input_size * (1 + n_diff)
        else:
            self.output_size = input_size * n_diff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute difference features.

        Args:
            x: input of shape [B, T, D]. NaNs are zero-filled first.
        Returns:
            Encoded features of shape [B, T, output_size].
        """
        # Zero-fill NaN BEFORE computing differences so diff is well-defined.
        x_clean = torch.nan_to_num(x, nan=0.0)
        # Always compute all n_diff differences.
        diffs = []
        prev = x_clean
        for _ in range(self.n_diff):
            dx = torch.zeros_like(prev)
            dx[:, 1:, :] = prev[:, 1:, :] - prev[:, :-1, :]
            diffs.append(dx)
            prev = dx
        if self.mode == "concat":
            return torch.cat([x_clean] + diffs, dim=-1)
        return torch.cat(diffs, dim=-1)


class DiffCfCNetwork(nn.Module):
    """CfC network with finite-difference input augmentation.

    Args:
        input_size: original input dimension D.
        hidden_size: CfC hidden dimension.
        output_size: output dimension.
        n_diff: number of finite differences (0=none, 1=Δx, 2=Δx+Δ²x).
        mode: "concat" ([x, Δx, ...]) or "diff_only" ([Δx, Δ²x]).
        num_layers: number of stacked CfC cells.
        return_sequences: if True, return per-timestep outputs.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        n_diff: int = 1,
        mode: str = "concat",
        num_layers: int = 2,
        return_sequences: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_diff = n_diff
        self.mode = mode
        self.num_layers = num_layers
        self.return_sequences = return_sequences

        self.encoder = DifferenceInputEncoder(input_size, n_diff=n_diff, mode=mode)
        encoded_dim = self.encoder.output_size

        self.cfc = CfCNetwork(
            input_size=encoded_dim,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
            return_sequences=return_sequences,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: input of shape [B, T, input_size].
        Returns:
            Output of shape [B, T, output_size] if return_sequences else [B, output_size].
        """
        encoded = self.encoder(x)
        return self.cfc(encoded)
