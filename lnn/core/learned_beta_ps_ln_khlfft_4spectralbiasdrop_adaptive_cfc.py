"""LearnedBetaPS+LN+Khl+FFT+4ScaleSpectralBiasDropout+Adaptive-CfC (PRD #10-181, Round 219, 2026-06-16).

Like r216 (4-scale + bias + dropout) but with **adaptive scale
weights** (vs simple average). 4 scale weights are learned via
softmax from the z vector — like r211's 3-scale adaptive but
extended to 4 scales with bias and dropout.

Mechanism:
    z = cat([aug_x, aug_h]) → LN
    f, h_branch from z
    H1..H4 = FFT(h_t) at 4 scales
    For each scale:
        mag = |H| + bias
        mask = sigmoid(linear(mag))
        if training: mask = F.dropout(mask, p=0.2)
        g = IFFT(H * mask, n=hidden)
    # Adaptive weights (vs simple avg in r216)
    weights = softmax(linear(z))  # [B, 4]
    g_combined = sum(weights_i * g_i)
    h_new = τ_eff * g + (1-τ_eff) * h_branch
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lnn.core.learned_beta_ps_ln_fft_cfc import FFTInputEncoder


class FourScaleSpectralBiasDropAdaptiveCfCCell(nn.Module):
    """CfC cell with 4-scale spectral + bias + dropout + adaptive weights."""

    def __init__(
        self,
        input_size,
        hidden_size,
        Kx,
        Kh,
        mode_x="diff",
        mode_h="diff",
        beta_x_init=0.75,
        beta_h_init=0.75,
        ln_eps=1e-5,
        dropout_p=0.2,
    ):
        super().__init__()
        assert mode_x in ("diff", "concat")
        assert mode_h in ("diff", "concat")
        assert Kx >= 1
        assert Kh >= 1
        assert hidden_size >= 8
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Kx = Kx
        self.Kh = Kh
        self.mode_x = mode_x
        self.mode_h = mode_h
        self.dropout_p = dropout_p

        aug_input_size = (Kx + 1) * input_size
        aug_hidden_size = (Kh + 1) * hidden_size
        aug_total = aug_input_size + aug_hidden_size

        self.beta_x_raw = nn.Parameter(
            torch.full((Kx,), float(torch.log(torch.tensor(beta_x_init / (1.0 - beta_x_init))).item()))
        )
        self.beta_h_raw = nn.Parameter(
            torch.full((Kh,), float(torch.log(torch.tensor(beta_h_init / (1.0 - beta_h_init))).item()))
        )

        self.layer_norm = nn.LayerNorm(aug_total, eps=ln_eps)

        self.f_gate = nn.Sequential(nn.Linear(aug_total, hidden_size), nn.Sigmoid())
        self.h_branch = nn.Sequential(nn.Linear(aug_total, hidden_size), nn.Tanh())
        self.time_scale = nn.Parameter(torch.ones(hidden_size))

        # 4 scales
        n_freq_full = hidden_size // 2 + 1
        n_freq_half = hidden_size // 4 + 1
        n_freq_quarter = hidden_size // 8 + 1
        n_freq_eighth = max(hidden_size // 16 + 1, 1)
        self.spec_mask1 = nn.Linear(n_freq_full, n_freq_full)
        self.spec_mask2 = nn.Linear(n_freq_half, n_freq_half)
        self.spec_mask3 = nn.Linear(n_freq_quarter, n_freq_quarter)
        self.spec_mask4 = nn.Linear(n_freq_eighth, n_freq_eighth)
        self.spec_bias1 = nn.Parameter(torch.zeros(n_freq_full))
        self.spec_bias2 = nn.Parameter(torch.zeros(n_freq_half))
        self.spec_bias3 = nn.Parameter(torch.zeros(n_freq_quarter))
        self.spec_bias4 = nn.Parameter(torch.zeros(n_freq_eighth))

        # Adaptive scale weights (NEW vs r216)
        self.scale_weight = nn.Linear(aug_total, 4)

    @property
    def beta_x(self):
        return torch.sigmoid(self.beta_x_raw)

    @property
    def beta_h(self):
        return torch.sigmoid(self.beta_h_raw)

    def _4scale_spectral(self, h, weights):
        H1 = torch.fft.rfft(h, dim=-1)
        # Scale 1: full
        mag1 = torch.abs(H1) + self.spec_bias1
        mask1 = torch.sigmoid(self.spec_mask1(mag1))
        if self.training and self.dropout_p > 0:
            mask1 = F.dropout(mask1, p=self.dropout_p, training=True)
        g1 = torch.fft.irfft(H1 * mask1, n=self.hidden_size, dim=-1)
        # Scale 2: half
        H2 = H1[:, :self.hidden_size // 4 + 1]
        mag2 = torch.abs(H2) + self.spec_bias2
        mask2 = torch.sigmoid(self.spec_mask2(mag2))
        if self.training and self.dropout_p > 0:
            mask2 = F.dropout(mask2, p=self.dropout_p, training=True)
        H2_full = torch.zeros_like(H1)
        H2_full[:, :H2.shape[1]] = H2 * mask2
        g2 = torch.fft.irfft(H2_full, n=self.hidden_size, dim=-1)
        # Scale 3: quarter
        H3 = H1[:, :self.hidden_size // 8 + 1]
        mag3 = torch.abs(H3) + self.spec_bias3
        mask3 = torch.sigmoid(self.spec_mask3(mag3))
        if self.training and self.dropout_p > 0:
            mask3 = F.dropout(mask3, p=self.dropout_p, training=True)
        H3_full = torch.zeros_like(H1)
        H3_full[:, :H3.shape[1]] = H3 * mask3
        g3 = torch.fft.irfft(H3_full, n=self.hidden_size, dim=-1)
        # Scale 4: eighth
        H4 = H1[:, :self.hidden_size // 16 + 1]
        mag4 = torch.abs(H4) + self.spec_bias4
        mask4 = torch.sigmoid(self.spec_mask4(mag4))
        if self.training and self.dropout_p > 0:
            mask4 = F.dropout(mask4, p=self.dropout_p, training=True)
        H4_full = torch.zeros_like(H1)
        H4_full[:, :H4.shape[1]] = H4 * mask4
        g4 = torch.fft.irfft(H4_full, n=self.hidden_size, dim=-1)
        # Adaptive combination
        return (weights[:, 0:1] * g1 + weights[:, 1:2] * g2 +
                weights[:, 2:3] * g3 + weights[:, 3:4] * g4)

    def forward(self, x_t, h_t, emas_x, emas_h, dt=1.0):
        x_t = torch.nan_to_num(x_t, nan=0.0)
        h_t = torch.nan_to_num(h_t, nan=0.0)
        emas_x = [torch.nan_to_num(e, nan=0.0) for e in emas_x]
        emas_h = [torch.nan_to_num(e, nan=0.0) for e in emas_h]

        beta_x = self.beta_x
        emas_x_new = [
            beta_x[k] * emas_x[k] + (1.0 - beta_x[k]) * x_t
            for k in range(self.Kx)
        ]
        beta_h = self.beta_h
        emas_h_new = [
            beta_h[k] * emas_h[k] + (1.0 - beta_h[k]) * h_t
            for k in range(self.Kh)
        ]

        if self.mode_x == "concat":
            aug_x = torch.cat([x_t] + emas_x_new, dim=-1)
        else:
            aug_x = torch.cat([x_t] + [e - x_t for e in emas_x_new], dim=-1)

        if self.mode_h == "concat":
            aug_h = torch.cat([h_t] + emas_h_new, dim=-1)
        else:
            aug_h = torch.cat([h_t] + [e - h_t for e in emas_h_new], dim=-1)

        z = torch.cat([aug_x, aug_h], dim=-1)
        z = self.layer_norm(z)

        f = self.f_gate(z)
        h_branch = self.h_branch(z)
        weights = F.softmax(self.scale_weight(z), dim=-1)
        g = self._4scale_spectral(h_t, weights)

        if isinstance(dt, torch.Tensor):
            dt_b = dt
            if dt_b.dim() < 2:
                dt_b = dt_b.unsqueeze(-1)
            tau_eff = torch.exp(-f * dt_b / torch.abs(self.time_scale))
        else:
            tau_eff = torch.exp(-f * float(dt) / torch.abs(self.time_scale))
        h_new = tau_eff * g + (1.0 - tau_eff) * h_branch

        return h_new, emas_x_new, emas_h_new


class FourScaleSpectralBiasDropAdaptiveCfCStackedNetwork(nn.Module):
    """Stacked 4-scale spectral + bias + dropout + adaptive weights CfC."""

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
        dropout_p=0.2,
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

        self.fft_encoder = FFTInputEncoder(n_fft=n_fft)

        self.cells = nn.ModuleList()
        for l in range(num_layers):
            in_size = self.augmented_input_size if l == 0 else hidden_size
            self.cells.append(
                FourScaleSpectralBiasDropAdaptiveCfCCell(
                    in_size, hidden_size, Kx, self.Kh_ladder[l],
                    mode_x=mode_x, mode_h=mode_h,
                    beta_x_init=beta_x_init,
                    beta_h_init=beta_h_init,
                    dropout_p=dropout_p,
                ),
            )

        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        B, T, _ = x.shape
        device = x.device
        x_aug = self.fft_encoder(x)
        hs = [torch.zeros(B, self.hidden_size, device=device) for _ in range(self.num_layers)]
        emas_x = [
            [torch.zeros(B, self.cells[l].input_size, device=device) for _ in range(self.Kx)]
            for l in range(self.num_layers)
        ]
        emas_h = [
            [torch.zeros(B, self.hidden_size, device=device) for _ in range(self.Kh_ladder[l])]
            for l in range(self.num_layers)
        ]
        outputs = []
        for t in range(T):
            inp = x_aug[:, t, :]
            for l, cell in enumerate(self.cells):
                hs[l], emas_x[l], emas_h[l] = cell(
                    inp, hs[l], emas_x[l], emas_h[l],
                )
                inp = hs[l]
            outputs.append(self.head(hs[-1]))
        outputs = torch.stack(outputs, dim=1)
        if self.return_sequences:
            return outputs
        return outputs[:, -1, :]


def make_lbps_lnkhlfft_4spectralbiasdrop_adaptive_5_3_2(input_size, hidden_size, output_size, num_layers=3, dropout_p=0.2):
    """Kh=[5,3,2] + LN + FFT + 4-Scale Spectral + bias + dropout + adaptive scale weights."""
    return FourScaleSpectralBiasDropAdaptiveCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kh_ladder=[5, 3, 2], Kx=5,
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
        dropout_p=dropout_p,
    )


__all__ = [
    "FourScaleSpectralBiasDropAdaptiveCfCCell",
    "FourScaleSpectralBiasDropAdaptiveCfCStackedNetwork",
    "make_lbps_lnkhlfft_4spectralbiasdrop_adaptive_5_3_2",
]
