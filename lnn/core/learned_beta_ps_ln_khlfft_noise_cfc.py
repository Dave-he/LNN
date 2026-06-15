"""LearnedBetaPS+LN+Khl+FFT+Noise-CfC (PRD #10-154, Round 192, 2026-06-16).

Wraps round 187's winner (lbps_lnkhlfft_5_3_2) with **input
Gaussian noise augmentation** as a regularizer.

After 3 distributional loss failures (rounds 189-191: BW/SWD/ED
all NEGATIVE), pivot to a classic regularizer:
**additive Gaussian noise on input during training**,
different from dropout (multiplicative).

Motivation:
- Dropout zeros out features (round 92 showed dropout
  hurts CfC)
- Additive noise keeps signal but adds jitter
- Forces model to be robust to small input perturbations
- Different mechanism class from distributional losses

Mechanism (TRAINING ONLY, not eval):
    x_clean = nan_to_num(x, nan=0)  # fill NaN
    noise = randn_like(x_clean) * sigma
    x_noisy = x_clean + noise
    x_noisy[is_nan] = nan  # restore NaN
    x_aug = fft_encoder(x_noisy)
    h = cfc_stack(x_aug, Kh_ladder=[5,3,2])
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.learned_beta_ps_ln_khlfft_cfc import LearnedBetaPSLNKhlFftCfCStackedNetwork


class LearnedBetaPSLNKhlFftNoiseCfCStackedNetwork(nn.Module):
    """Round 187 winner + input Gaussian noise augmentation.

    Args:
        input_size, hidden_size, output_size, num_layers
        noise_sigma: stddev of additive Gaussian noise (0=disabled)
        Kh_ladder: list of Kh values per layer
        Kx: number of EMA scales for input
        mode_x, mode_h: 'diff' or 'abs'
        beta_x_init, beta_h_init: initial EMA decay
        return_sequences: True for [B, T, D_out]
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=3,
        noise_sigma=0.05,
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
        self.noise_sigma = noise_sigma
        # Reuse round 187 stack (FFT + Kh ladder + LN)
        self.cfc_net = LearnedBetaPSLNKhlFftCfCStackedNetwork(
            input_size=input_size,
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
            n_fft=n_fft,
        )

    def forward(self, x):
        if self.training and self.noise_sigma > 0:
            # Save NaN mask
            nan_mask = torch.isnan(x)
            # Fill NaN with 0 for noise generation
            x_clean = torch.nan_to_num(x, nan=0.0)
            # Generate noise only on non-NaN positions
            noise = torch.randn_like(x_clean) * self.noise_sigma
            x_noisy = x_clean + noise
            # Restore NaN where originally NaN
            x_noisy = torch.where(nan_mask, x, x_noisy)
            x = x_noisy
        return self.cfc_net(x)


def make_lbps_lnkhlfft_noise_5_3_2(input_size, hidden_size, output_size, num_layers=3, noise_sigma=0.05):
    """Kh=[5,3,2] + LN + FFT + input Gaussian noise."""
    return LearnedBetaPSLNKhlFftNoiseCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kh_ladder=[5, 3, 2], Kx=5,
        noise_sigma=noise_sigma,
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


__all__ = [
    "LearnedBetaPSLNKhlFftNoiseCfCStackedNetwork",
    "make_lbps_lnkhlfft_noise_5_3_2",
]
