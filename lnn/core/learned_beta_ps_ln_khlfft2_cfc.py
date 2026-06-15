"""LearnedBetaPS+LN+Khl+FFT2-CfC (Per-Scale Learnable β + LayerNorm + Kh Ladder + FFT Magnitude+Phase Input Features) (PRD #10-150, Round 188, 2026-06-16).

Variant of round 187's LearnedBetaPS+LN+Khl+FFT-CfC with **BOTH
magnitude AND phase** of FFT as input features.

Rationale:
- Magnitude captures the AMPLITUDE of frequency components
  (what frequencies are present).
- Phase captures the TIMING of those components (where the
  peaks are). At a regime boundary, the phase jumps.
- Sin has fixed phase (one tone) — phase feature won't
  disrupt it.
- Structured has regime changes — phase may help preserve
  the regime info that pure magnitude loses.

Hypothesis:
- H1 (positive): phase captures regime timing → fixes
  structured regression from round 187 while preserving
  sin benefit → SP
- H2 (negative): phase adds noise (high variance) → no
  help or hurts
- H3 (mixed): helps structured but doesn't help sin as
  much

Audit context (91-187): 46 strictly positive + 19 target-dep
+ 46 negatives = 111 mechanism classes.

Mechanism::

    For each timestep t:
        # FFT2 input features (NEW round 188):
        x_clean = nan_to_num(x, nan=0)
        x_fft = rfft(x_clean, dim=1)  # complex
        x_fft_mag = abs(x_fft)        # amplitude
        x_fft_phase = angle(x_fft)    # phase
        # Pad to T
        x_aug = cat([x, x_fft_mag_pad, x_fft_phase_pad], dim=-1)  # [B, T, 3D]
        # Per-scale EMAs (round 171) on x_aug:
        ema_x_k = β_x_k * ema_x_k + (1 - β_x_k) * x_aug_t
        ema_h_k = β_h_k * ema_h_k + (1 - β_h_k) * h_{t-1}
        # Augmented input + LN (round 179):
        z = cat([aug_x, aug_h])
        z_norm = LayerNorm(z)
        # CfC closed-form (unchanged):
        h_t = σ(-f·τ)·g + (1-σ(-f·τ))·h_branch
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FFT2InputEncoder(nn.Module):
    """Compute FFT magnitude AND phase of input, concatenate with original.

    Input: [B, T, D] (with possible NaN)
    Output: [B, T, 3D] (concat of original + FFT mag + FFT phase, padded to T)
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
            x_fft = torch.fft.rfft(x_clean, dim=1)  # [B, T//2+1, D] complex
        else:
            x_fft = torch.fft.rfft(x_clean, n=self.n_fft, dim=1)
        x_fft_mag = torch.abs(x_fft)        # [B, T//2+1, D]
        x_fft_phase = torch.angle(x_fft)    # [B, T//2+1, D]
        # Pad T//2+1 to T (zero pad on the right of T dim)
        n_fft_bins = x_fft_mag.shape[1]
        if n_fft_bins < T:
            pad_amount = T - n_fft_bins
            x_fft_mag_pad = F.pad(x_fft_mag, (0, 0, 0, pad_amount))      # [B, T, D]
            x_fft_phase_pad = F.pad(x_fft_phase, (0, 0, 0, pad_amount))  # [B, T, D]
        else:
            x_fft_mag_pad = x_fft_mag[:, :T, :]
            x_fft_phase_pad = x_fft_phase[:, :T, :]
        # Concat: original (with NaN) + mag + phase
        return torch.cat([x, x_fft_mag_pad, x_fft_phase_pad], dim=-1)  # [B, T, 3D]


class LearnedBetaPSLNKhlFft2CfCStackedNetwork(nn.Module):
    """Stacked LearnedBetaPS+LN+Khl+FFT2-CfC: FFT2 input + Kh ladder."""

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
        self.augmented_input_size = 3 * input_size  # original + mag + phase
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.Kx = Kx
        self.Kh_ladder = list(Kh_ladder) if Kh_ladder is not None else [3] * num_layers
        assert len(self.Kh_ladder) == num_layers
        self.return_sequences = return_sequences

        # FFT2 input encoder
        self.fft2_encoder = FFT2InputEncoder(n_fft=n_fft)

        # CfC stacked network on augmented input with Kh ladder
        from lnn.core.learned_beta_ps_ln_khl_cfc import LearnedBetaPSLNKhlCfCStackedNetwork
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
        x_aug = self.fft2_encoder(x)
        return self.cfc_net(x_aug)


def make_lbps_lnkhlfft2_2_5_2(input_size, hidden_size, output_size, num_layers=3):
    """Kh=[2,5,2] + LN + FFT2 (sin-friendly Kh ladder)."""
    return LearnedBetaPSLNKhlFft2CfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kh_ladder=[2, 5, 2], Kx=5,
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


def make_lbps_lnkhlfft2_5_3_2(input_size, hidden_size, output_size, num_layers=3):
    """Kh=[5,3,2] + LN + FFT2 (round 187 winner config)."""
    return LearnedBetaPSLNKhlFft2CfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kh_ladder=[5, 3, 2], Kx=5,
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


__all__ = [
    "FFT2InputEncoder",
    "LearnedBetaPSLNKhlFft2CfCStackedNetwork",
    "make_lbps_lnkhlfft2_2_5_2",
    "make_lbps_lnkhlfft2_5_3_2",
]
