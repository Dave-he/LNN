"""LearnedBetaPS+LN+Khl+FFT+XHNoise-CfC (PRD #10-156, Round 194, 2026-06-16).

Combines round 192 (input Gaussian noise) + round 193 (hidden
state Gaussian noise) — best of both worlds?

Round 192 (input noise σ=0.05): sin -16% struct +6% random -26%
Round 193 (hidden noise σ=0.05): sin -20% struct -16% random +21%

Both help sin. Each wins on 1 dataset, loses on the other.
Test if combining both noises:
- Preserves sin improvement (both contribute -16%/-20%)
- Gets best of both on structured (one wins, one ties)
- Gets best of both on random (one wins, one loses)
- Or regresses because too much noise

Mechanism (TRAINING ONLY, not eval):
    x_noisy = x + sigma_in * randn  # round 192 (input)
    For each cell call:
        h = cell(x_noisy_t, h)
        h = h + sigma_h * randn  # round 193 (hidden)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.learned_beta_ps_ln_khlfft_cfc import LearnedBetaPSLNKhlFftCfCStackedNetwork
from lnn.core.learned_beta_ps_ln_khlfft_noise_cfc import (
    LearnedBetaPSLNKhlFftNoiseCfCStackedNetwork,
)


class LearnedBetaPSLNKhlFftXHNoiseCfCStackedNetwork(nn.Module):
    """Round 187 winner + input AND hidden state Gaussian noise.

    Args:
        input_size, hidden_size, output_size, num_layers
        noise_sigma: stddev of input Gaussian noise (0=disabled)
        hnoise_sigma: stddev of hidden state Gaussian noise (0=disabled)
        Kh_ladder, Kx, etc.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=3,
        noise_sigma=0.05,
        hnoise_sigma=0.05,
        Kh_ladder=None,
        Kx=5,
        mode_x="diff",
        mode_h="diff",
        beta_x_init=0.75,
        beta_h_init=0.75,
        return_sequences=True,
        n_fft=None,
        noise_layers="all",
    ):
        super().__init__()
        self.noise_sigma = noise_sigma
        self.hnoise_sigma = hnoise_sigma
        self.noise_layers = noise_layers
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

    def _should_noise(self, layer_idx):
        if self.noise_layers == "all":
            return True
        if isinstance(self.noise_layers, (list, tuple)):
            return layer_idx in self.noise_layers
        return False

    def forward(self, x):
        # Step 1: input noise (round 192) — train mode only
        if self.training and self.noise_sigma > 0:
            nan_mask = torch.isnan(x)
            x_clean = torch.nan_to_num(x, nan=0.0)
            x_noisy = x_clean + torch.randn_like(x_clean) * self.noise_sigma
            x_noisy = torch.where(nan_mask, x, x_noisy)
            x = x_noisy
        # Step 2: hidden state noise (round 193) — train mode only
        if not (self.training and self.hnoise_sigma > 0):
            return self.cfc_net(x)
        # Manual forward with hidden state noise
        x_aug = self.cfc_net.fft_encoder(x)
        inner = self.cfc_net.cfc_net
        B, T, _ = x_aug.shape
        device = x_aug.device
        hs = [torch.zeros(B, inner.hidden_size, device=device) for _ in range(inner.num_layers)]
        emas_x = [
            [torch.zeros(B, inner.cells[l].input_size, device=device) for _ in range(inner.Kx)]
            for l in range(inner.num_layers)
        ]
        emas_h = [
            [torch.zeros(B, inner.hidden_size, device=device) for _ in range(inner.Kh_ladder[l])]
            for l in range(inner.num_layers)
        ]
        outputs = []
        for t in range(T):
            inp = x_aug[:, t, :]
            for l, cell in enumerate(inner.cells):
                hs[l], emas_x[l], emas_h[l] = cell(
                    inp, hs[l], emas_x[l], emas_h[l],
                )
                if self._should_noise(l):
                    hs[l] = hs[l] + torch.randn_like(hs[l]) * self.hnoise_sigma
                inp = hs[l]
            outputs.append(inner.head(hs[-1]))
        outputs = torch.stack(outputs, dim=1)
        if inner.return_sequences:
            return outputs
        return outputs[:, -1, :]


def make_lbps_lnkhlfft_xhnoise_5_3_2(input_size, hidden_size, output_size, num_layers=3, noise_sigma=0.05, hnoise_sigma=0.05):
    """Kh=[5,3,2] + LN + FFT + input AND hidden state Gaussian noise."""
    return LearnedBetaPSLNKhlFftXHNoiseCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kh_ladder=[5, 3, 2], Kx=5,
        noise_sigma=noise_sigma, hnoise_sigma=hnoise_sigma,
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


__all__ = [
    "LearnedBetaPSLNKhlFftXHNoiseCfCStackedNetwork",
    "make_lbps_lnkhlfft_xhnoise_5_3_2",
]
