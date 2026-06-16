"""LearnedBetaPS+LN+Khl+FFT+MultiResSpectral-CfC (PRD #10-171, Round 209, 2026-06-16).

Multi-resolution spectral gating (Sonnet 2026 style).
Apply r200's spectral gating at TWO resolutions (full and half)
and combine. Captures both fine and coarse spectral structure.

Mechanism:
    For each timestep t:
        # Same FFT + Kh + LN as r187
        z = cat([aug_x, aug_h])
        z = LayerNorm(z)
        f = σ(linear(z))
        h_branch = tanh(linear(z))
        # NEW: multi-resolution spectral gating
        # Res 1: full FFT
        H1 = FFT(h_t, dim=-1)
        mask1 = sigmoid(linear(|H1|))
        g1 = IFFT(H1 * mask1, n=hidden_size)
        # Res 2: half FFT (coarser)
        H2 = FFT(h_t, dim=-1)[:, :hidden_size//2]  # truncate
        mask2 = sigmoid(linear(|H2|))
        # Pad back to full size
        H2_full = pad(H2, (0, hidden_size//2))
        mask2_full = pad(mask2, (0, hidden_size//2))
        g2 = IFFT(H2_full * mask2_full, n=hidden_size)
        g_combined = (g1 + g2) / 2  # simple average
        tau_eff = exp(-f * dt / |time_scale|)
        h_t = tau_eff * g_combined + (1 - tau_eff) * h_branch
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.learned_beta_ps_ln_fft_cfc import FFTInputEncoder


class MultiResSpectralGatedCfCCell(nn.Module):
    """CfC cell with multi-resolution spectral gating."""

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
    ):
        super().__init__()
        assert mode_x in ("diff", "concat")
        assert mode_h in ("diff", "concat")
        assert Kx >= 1
        assert Kh >= 1
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Kx = Kx
        self.Kh = Kh
        self.mode_x = mode_x
        self.mode_h = mode_h

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

        # Res 1: full FFT
        n_freq_full = hidden_size // 2 + 1
        self.spec_mask1 = nn.Linear(n_freq_full, n_freq_full)
        # Res 2: half FFT (coarser)
        n_freq_half = (hidden_size // 2) // 2 + 1
        self.spec_mask2 = nn.Linear(n_freq_half, n_freq_half)

    @property
    def beta_x(self):
        return torch.sigmoid(self.beta_x_raw)

    @property
    def beta_h(self):
        return torch.sigmoid(self.beta_h_raw)

    def _multires_spectral(self, h):
        """Multi-resolution spectral gating."""
        # Res 1: full FFT
        H1 = torch.fft.rfft(h, dim=-1)
        mag1 = torch.abs(H1)
        mask1 = torch.sigmoid(self.spec_mask1(mag1))
        g1 = torch.fft.irfft(H1 * mask1, n=self.hidden_size, dim=-1)

        # Res 2: half FFT (truncate to lower half of freqs)
        H2 = H1[:, :self.hidden_size // 4 + 1]  # half of full rfft size
        mag2 = torch.abs(H2)
        mask2 = torch.sigmoid(self.spec_mask2(mag2))
        # Apply mask and pad back
        H2_filtered = H2 * mask2
        # Pad to full rfft size for inverse
        H2_full = torch.zeros_like(H1)
        H2_full[:, :H2_filtered.shape[1]] = H2_filtered
        g2 = torch.fft.irfft(H2_full, n=self.hidden_size, dim=-1)

        return (g1 + g2) / 2.0

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
        g = self._multires_spectral(h_t)

        if isinstance(dt, torch.Tensor):
            dt_b = dt
            if dt_b.dim() < 2:
                dt_b = dt_b.unsqueeze(-1)
            tau_eff = torch.exp(-f * dt_b / torch.abs(self.time_scale))
        else:
            tau_eff = torch.exp(-f * float(dt) / torch.abs(self.time_scale))
        h_new = tau_eff * g + (1.0 - tau_eff) * h_branch

        return h_new, emas_x_new, emas_h_new


class MultiResSpectralGatedCfCStackedNetwork(nn.Module):
    """Stacked MultiResSpectralGated-CfC with Kh ladder."""

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

        self.fft_encoder = FFTInputEncoder(n_fft=n_fft)

        self.cells = nn.ModuleList()
        for l in range(num_layers):
            in_size = self.augmented_input_size if l == 0 else hidden_size
            self.cells.append(
                MultiResSpectralGatedCfCCell(
                    in_size, hidden_size, Kx, self.Kh_ladder[l],
                    mode_x=mode_x, mode_h=mode_h,
                    beta_x_init=beta_x_init,
                    beta_h_init=beta_h_init,
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


def make_lbps_lnkhlfft_multiresgated_5_3_2(input_size, hidden_size, output_size, num_layers=3):
    """Kh=[5,3,2] + LN + FFT + Multi-Resolution Spectral Gating (2 scales)."""
    return MultiResSpectralGatedCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kh_ladder=[5, 3, 2], Kx=5,
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


__all__ = [
    "MultiResSpectralGatedCfCCell",
    "MultiResSpectralGatedCfCStackedNetwork",
    "make_lbps_lnkhlfft_multiresgated_5_3_2",
]
