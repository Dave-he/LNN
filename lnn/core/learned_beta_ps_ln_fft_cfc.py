"""LearnedBetaPS+LN+FFT-CfC (Per-Scale Learnable β + LayerNorm + FFT Input Features) (PRD #10-148, Round 186, 2026-06-16).

Variant of round 179's LearnedBetaPS+LN-CfC with **FFT magnitude
as additional input features**. The FFT magnitude captures
frequency-domain information that the time-domain EMA features
miss.

Hypothesis:
- H1 (positive): FFT magnitude captures frequency patterns
  that EMAs miss → improvement
- H2 (negative): FFT adds noise (magnitude can be dominated
  by noise) → no help
- H3 (mixed): helps periodic (sin) but not non-periodic
  (random)

Audit context (91-185): 45 strictly positive + 18 target-dep +
46 negatives = 109 mechanism classes.

Mechanism::

    For each timestep t:
        # FFT input features (NEW round 186):
        x_clean = nan_to_num(x, nan=0)  # [B, T, D]
        x_fft = abs(rfft(x_clean, dim=1))  # [B, T//2+1, D]
        x_fft_pad = pad(x_fft, [0, T - T//2 - 1])  # [B, T, D]
        x_aug = cat([x, x_fft_pad], dim=-1)  # [B, T, 2D]
        # Per-scale EMAs (round 171) on x_aug:
        ema_x_k = β_x_k * ema_x_k + (1 - β_x_k) * x_aug_t
        ema_h_k = β_h_k * ema_h_k + (1 - β_h_k) * h_{t-1}
        # Augmented input + LN (round 179):
        z = cat([aug_x, aug_h])  # [B, (Kx+1)*2D + (Kh+1)*H]
        z_norm = LayerNorm(z)
        # CfC closed-form (unchanged):
        h_t = σ(-f·τ)·g + (1-σ(-f·τ))·h_branch
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lnn.core.learned_beta_ps_ln_khl_cfc import LearnedBetaPSLNKhlCfCStackedNetwork


def _logit(p):
    return float(torch.log(torch.tensor(p / (1.0 - p))).item())


class FFTInputEncoder(nn.Module):
    """Compute FFT magnitude of input and concatenate with original.

    Input: [B, T, D] (with possible NaN)
    Output: [B, T, 2D] (concat of original + FFT magnitude, padded to T)
    """

    def __init__(self, n_fft=None):
        super().__init__()
        self.n_fft = n_fft  # if None, use T (full FFT)

    def forward(self, x):
        # x: [B, T, D]
        T = x.shape[1]
        x_clean = torch.nan_to_num(x, nan=0.0)
        # FFT along time dim
        if self.n_fft is None:
            x_fft = torch.fft.rfft(x_clean, dim=1)  # [B, T//2+1, D]
        else:
            x_fft = torch.fft.rfft(x_clean, n=self.n_fft, dim=1)
        x_fft_mag = torch.abs(x_fft)  # [B, T//2+1, D]
        # Pad T//2+1 to T (zero pad on the right of T dim)
        n_fft_bins = x_fft_mag.shape[1]
        if n_fft_bins < T:
            # F.pad order: (last_dim_left, last_dim_right, ..., T_left, T_right)
            pad_amount = T - n_fft_bins
            x_fft_pad = F.pad(x_fft_mag, (0, 0, 0, pad_amount))  # [B, T, D]
        else:
            x_fft_pad = x_fft_mag[:, :T, :]
        # Concat with original (keep NaN as-is for the model to handle)
        return torch.cat([x, x_fft_pad], dim=-1)  # [B, T, 2D]


class LearnedBetaPSLNFftCfCStackedNetwork(nn.Module):
    """Stacked LearnedBetaPS+LN-CfC with FFT input features."""

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
        self.input_size = input_size  # raw input size (D)
        self.augmented_input_size = 2 * input_size  # after FFT concat
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.Kx = Kx
        self.Kh_ladder = list(Kh_ladder) if Kh_ladder is not None else [3] * num_layers
        assert len(self.Kh_ladder) == num_layers
        self.return_sequences = return_sequences

        # FFT input encoder (input_size → 2*input_size)
        self.fft_encoder = FFTInputEncoder(n_fft=n_fft)

        # CfC stacked network on augmented input
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
        # x: [B, T, D] — raw input (may have NaN)
        x_aug = self.fft_encoder(x)  # [B, T, 2D]
        return self.cfc_net(x_aug)


def make_lbps_lnfft_h3_75(input_size, hidden_size, output_size, num_layers=3):
    """Kh=3 + LN + FFT input features (control)."""
    return LearnedBetaPSLNFftCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kh_ladder=[3] * num_layers, Kx=5,
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


def make_lbps_lnfft_h2_75(input_size, hidden_size, output_size, num_layers=3):
    """Kh=2 + LN + FFT input features."""
    return LearnedBetaPSLNFftCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kh_ladder=[2] * num_layers, Kx=5,
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


def make_lbps_lnfft_h5_75(input_size, hidden_size, output_size, num_layers=3):
    """Kh=5 + LN + FFT input features."""
    return LearnedBetaPSLNFftCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kh_ladder=[5] * num_layers, Kx=5,
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


__all__ = [
    "FFTInputEncoder",
    "LearnedBetaPSLNFftCfCStackedNetwork",
    "make_lbps_lnfft_h3_75",
    "make_lbps_lnfft_h2_75",
    "make_lbps_lnfft_h5_75",
]
