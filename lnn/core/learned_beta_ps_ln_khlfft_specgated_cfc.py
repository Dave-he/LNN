"""LearnedBetaPS+LN+Khl+FFT+SpectralGated-CfC (PRD #10-162, Round 200, 2026-06-16).

Wraps round 187's winner (lbps_lnkhlfft_5_3_2) with
**spectral gating on the hidden state** (FNO-style).

After 8 rounds of NEG/TD (r193-r199), pivot to a
fundamentally different mechanism. Inspired by **Fourier
Neural Operator (FNO)** (Li et al 2021) which parameterizes
the integral kernel in Fourier space.

Mechanism:
    For each timestep t:
        # Same FFT + Kh + LN as r187 for the input
        x_aug = fft_encode(x)
        ema_x_k = β_x_k * ema_x_k + (1 - β_x_k) * x_aug_t
        ema_h_k = β_h_k * ema_h_k + (1 - β_h_k) * h_{t-1}
        z = cat([aug_x, aug_h])
        z = LayerNorm(z)
        # f_gate and h_branch as usual
        f = σ(linear(z))
        h_branch = tanh(linear(z))
        # NEW (round 200): spectral gating on h_t
        H = FFT(h_t)  # complex [B, H/2+1]
        magnitude = |H|  # [B, H/2+1]
        mask = sigmoid(linear(magnitude))  # [B, H/2+1]
        g = IFFT(H * mask)  # complex → real, [B, H]
        # CfC closed-form
        tau_eff = exp(-f * dt / |time_scale|)
        h_t = tau_eff * g + (1 - tau_eff) * h_branch

Hypothesis:
- H1 (positive): spectral gating helps structured (2 regimes
  have different dominant frequencies)
- H2 (negative): spectral gating overfits (too many params)
- H3 (mixed): helps smooth (sin) hurts noise (random)
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn

from lnn.core.learned_beta_ps_ln_fft_cfc import FFTInputEncoder


class SpectralGatedCfCCell(nn.Module):
    """CfC cell with FNO-style spectral gating on the hidden state.

    The g_branch is replaced by a learned spectral filter applied
    to the previous hidden state. This is fundamentally different
    from r187's linear g_branch — it operates in the frequency
    domain, allowing the model to selectively keep/discard
    frequencies at each timestep.
    """

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

        # Per-scale learned β.
        self.beta_x_raw = nn.Parameter(
            torch.full((Kx,), float(torch.log(torch.tensor(beta_x_init / (1.0 - beta_x_init))).item()))
        )
        self.beta_h_raw = nn.Parameter(
            torch.full((Kh,), float(torch.log(torch.tensor(beta_h_init / (1.0 - beta_h_init))).item()))
        )

        # LayerNorm.
        self.layer_norm = nn.LayerNorm(aug_total, eps=ln_eps)

        # CfC components.
        self.f_gate = nn.Sequential(nn.Linear(aug_total, hidden_size), nn.Sigmoid())
        self.h_branch = nn.Sequential(nn.Linear(aug_total, hidden_size), nn.Tanh())
        self.time_scale = nn.Parameter(torch.ones(hidden_size))

        # NEW (round 200): spectral gating on h.
        # FFT size = hidden_size, produces (hidden_size//2 + 1) freq bins.
        n_freq = hidden_size // 2 + 1
        self.spec_mask = nn.Linear(n_freq, n_freq)

    @property
    def beta_x(self):
        return torch.sigmoid(self.beta_x_raw)

    @property
    def beta_h(self):
        return torch.sigmoid(self.beta_h_raw)

    def _spectral_gating(self, h):
        """Apply spectral gating to h.

        h: [B, hidden_size] real
        Returns: [B, hidden_size] real (after IFFT)
        """
        # rFFT: real → complex [B, n_freq]
        H = torch.fft.rfft(h, dim=-1)
        magnitude = torch.abs(H)
        # Learn mask from magnitude (real-valued)
        mask = torch.sigmoid(self.spec_mask(magnitude))
        # Apply mask
        H_filtered = H * mask
        # IRFFT: complex → real
        g = torch.fft.irfft(H_filtered, n=self.hidden_size, dim=-1)
        return g

    def forward(self, x_t, h_t, emas_x, emas_h, dt=1.0):
        x_t = torch.nan_to_num(x_t, nan=0.0)
        h_t = torch.nan_to_num(h_t, nan=0.0)
        emas_x = [torch.nan_to_num(e, nan=0.0) for e in emas_x]
        emas_h = [torch.nan_to_num(e, nan=0.0) for e in emas_h]

        # Per-scale EMA updates.
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

        # NEW: spectral gating replaces g_branch
        g = self._spectral_gating(h_t)

        if isinstance(dt, torch.Tensor):
            dt_b = dt
            if dt_b.dim() < 2:
                dt_b = dt_b.unsqueeze(-1)
            tau_eff = torch.exp(-f * dt_b / torch.abs(self.time_scale))
        else:
            tau_eff = torch.exp(-f * float(dt) / torch.abs(self.time_scale))
        h_new = tau_eff * g + (1.0 - tau_eff) * h_branch

        return h_new, emas_x_new, emas_h_new


class SpectralGatedCfCStackedNetwork(nn.Module):
    """Stacked SpectralGated-CfC with Kh ladder."""

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

        # SpectralGated cells
        self.cells = nn.ModuleList()
        for l in range(num_layers):
            in_size = self.augmented_input_size if l == 0 else hidden_size
            self.cells.append(
                SpectralGatedCfCCell(
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


def make_lbps_lnkhlfft_specgated_5_3_2(input_size, hidden_size, output_size, num_layers=3):
    """Kh=[5,3,2] + LN + FFT + Spectral Gating (FNO-style)."""
    return SpectralGatedCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kh_ladder=[5, 3, 2], Kx=5,
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


__all__ = [
    "SpectralGatedCfCCell",
    "SpectralGatedCfCStackedNetwork",
    "make_lbps_lnkhlfft_specgated_5_3_2",
]
