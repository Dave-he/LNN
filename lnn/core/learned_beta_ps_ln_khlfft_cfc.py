"""LearnedBetaPS+LN+Khl+FFT-CfC (Per-Scale Learnable β + LayerNorm + Kh Ladder + FFT Input Features) (PRD #10-149, Round 187, 2026-06-16).

Hybrid of:
- Round 180: Kh ladder for multi-time-scale features
- Round 186: FFT magnitude as additional input features

Hypothesis:
- H1 (positive): FFT helps sin (periodic) + Kh ladder
  handles structured (regime changes) → SP on both
- H2 (negative): FFT + Kh ladder don't compose
- H3 (mixed): helps one but hurts the other

Audit context (91-186): 45 strictly positive + 19 target-dep +
46 negatives = 110 mechanism classes.

Mechanism::

    For each timestep t:
        # FFT input features (round 186):
        x_clean = nan_to_num(x, nan=0)
        x_fft = abs(rfft(x_clean, dim=1))
        x_aug = cat([x, x_fft_pad], dim=-1)  # [B, T, 2D]
        # Per-scale EMAs (round 171) on x_aug:
        ema_x_k = β_x_k * ema_x_k + (1 - β_x_k) * x_aug_t
        ema_h_k = β_h_k * ema_h_k + (1 - β_h_k) * h_{t-1}
        # Augmented input + LN (round 179):
        z = cat([aug_x, aug_h])
        z_norm = LayerNorm(z)
        # CfC closed-form:
        h_t = σ(-f·τ)·g + (1-σ(-f·τ))·h_branch
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.learned_beta_ps_ln_fft_cfc import FFTInputEncoder
from lnn.core.learned_beta_ps_ln_khl_cfc import LearnedBetaPSLNKhlCfCStackedNetwork


class LearnedBetaPSLNKhlFftCfCStackedNetwork(nn.Module):
    """Stacked LearnedBetaPS+LN+Khl+FFT-CfC: FFT input + Kh ladder."""

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=3,
        Kh_ladder=None,
        Kx=5,
        mode_x="diff",
        mode_h="diff",
        beta_x_init=0.75,
        beta_h_init=0.75,
        return_sequences=True,
        n_fft=None,
    ):
        super().__init__()
        self.input_size = input_size
        self.augmented_input_size = 2 * input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.Kx = Kx
        self.Kh_ladder = list(Kh_ladder) if Kh_ladder is not None else [3] * num_layers
        assert len(self.Kh_ladder) == num_layers
        self.return_sequences = return_sequences

        # FFT input encoder
        self.fft_encoder = FFTInputEncoder(n_fft=n_fft)

        # CfC stacked network on augmented input with Kh ladder
        self.cfc_net = LearnedBetaPSLNKhlCfCStackedNetwork(
            input_size=self.augmented_input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
            Kh_ladder=Kh_ladder,
            Kx=Kx,
            mode_x=mode_x,
            mode_h=mode_h,
            beta_x_init=beta_x_init,
            beta_h_init=beta_h_init,
            return_sequences=return_sequences,
        )

    def forward(self, x):
        x_aug = self.fft_encoder(x)
        return self.cfc_net(x_aug)


def make_lbps_lnkhlfft_2_5_2(input_size, hidden_size, output_size, num_layers=3):
    """Kh=[2,5,2] + LN + FFT (sin-friendly Kh ladder)."""
    return LearnedBetaPSLNKhlFftCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kh_ladder=[2, 5, 2], Kx=5,
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


def make_lbps_lnkhlfft_5_3_2(input_size, hidden_size, output_size, num_layers=3):
    """Kh=[5,3,2] + LN + FFT (structured-friendly Kh ladder)."""
    return LearnedBetaPSLNKhlFftCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kh_ladder=[5, 3, 2], Kx=5,
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


__all__ = [
    "LearnedBetaPSLNKhlFftCfCStackedNetwork",
    "make_lbps_lnkhlfft_2_5_2",
    "make_lbps_lnkhlfft_5_3_2",
]
