"""LearnedBetaPS+LN+Khl+FFT+HNoise-CfC (PRD #10-155, Round 193, 2026-06-16).

Wraps round 187's winner (lbps_lnkhlfft_5_3_2) with **hidden state
Gaussian noise augmentation** as a regularizer.

After round 192 (input Gaussian noise, σ=0.05) was STRICTLY
POSITIVE (-24% mean), test the **orthogonal dimension**:
additive Gaussian noise on the HIDDEN STATE h after each
cell call. This is a different mechanism than input noise:
- Input noise: perturbs what the model sees
- Hidden noise: perturbs what the model "remembers"

Hidden noise is a form of "weight noise" / "recurrent noise"
from Graves 2011 ("Practical Variational Inference for
Neural Networks"), where noise on internal state acts as
a Bayesian-style regularizer.

Mechanism (TRAINING ONLY, not eval):
    After each cell call:
        h_t_noisy = h_t + sigma * randn_like(h_t)
    Forward: cell(x_t, h_{t-1}) -> h_t
             h_t_noisy = h_t + sigma * randn
             h_t_noisy is passed as h_t to next layer
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.learned_beta_ps_ln_khlfft_cfc import LearnedBetaPSLNKhlFftCfCStackedNetwork
from lnn.core.learned_beta_ps_ln_khl_cfc import LearnedBetaPSLNKhlCfCStackedNetwork


class LearnedBetaPSLNKhlFftHNoiseCfCStackedNetwork(nn.Module):
    """Round 187 winner + hidden state Gaussian noise augmentation.

    Args:
        input_size, hidden_size, output_size, num_layers
        hnoise_sigma: stddev of additive Gaussian noise on h (0=disabled)
        Kh_ladder: list of Kh values per layer
        Kx: number of EMA scales for input
        mode_x, mode_h: 'diff' or 'abs'
        beta_x_init, beta_h_init: initial EMA decay
        return_sequences: True for [B, T, D_out]
        noise_layers: which layers to add noise to ('all' or list of int)
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=3,
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
        if not (self.training and self.hnoise_sigma > 0):
            return self.cfc_net(x)
        # Manual forward with hidden state noise
        x_aug = self.cfc_net.fft_encoder(x)
        # Reach into the inner cfc_net (LearnedBetaPSLNKhlCfCStackedNetwork)
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


def make_lbps_lnkhlfft_hnoise_5_3_2(input_size, hidden_size, output_size, num_layers=3, hnoise_sigma=0.05):
    """Kh=[5,3,2] + LN + FFT + hidden state Gaussian noise."""
    return LearnedBetaPSLNKhlFftHNoiseCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kh_ladder=[5, 3, 2], Kx=5,
        hnoise_sigma=hnoise_sigma,
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


__all__ = [
    "LearnedBetaPSLNKhlFftHNoiseCfCStackedNetwork",
    "make_lbps_lnkhlfft_hnoise_5_3_2",
]
