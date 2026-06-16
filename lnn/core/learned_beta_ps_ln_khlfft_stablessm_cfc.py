"""LearnedBetaPS+LN+Khl+FFT+StableSSM-CfC (PRD #10-170, Round 208, 2026-06-16).

Fixes r207's divergence by bounding the SSM decay A to [0,1]
via sigmoid: A = sigmoid(linear_A(z)).

A diagonal SSM with bounded decay is guaranteed stable:
    |h_ssm_t| <= |h_ssm_{t-1}| + |B * x_t|

Mechanism:
    For each timestep t:
        # Same FFT + Kh + LN as r187
        z = cat([aug_x, aug_h])
        z = LayerNorm(z)
        f = σ(linear(z))
        h_branch = tanh(linear(z))
        # NEW: stable diagonal SSM
        A = sigmoid(linear_A(z))         # [B, H], bounded [0,1]
        B = linear_B(x_t)                # [B, H]
        C = linear_C(z)                  # [B, H]
        h_ssm = A * h_ssm + B             # [B, H]
        g_ssm = C * h_ssm                 # [B, H]
        g_combined = h_branch * g_ssm     # element-wise
        tau_eff = exp(-f * dt / |time_scale|)
        h_t = tau_eff * g_combined + (1 - tau_eff) * h_branch
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.learned_beta_ps_ln_fft_cfc import FFTInputEncoder


class StableSSMCfCCell(nn.Module):
    """CfC cell with stable diagonal SSM (sigmoid-bounded A)."""

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

        # Stable SSM: A bounded to [0,1] via sigmoid
        self.ssm_A = nn.Sequential(nn.Linear(aug_total, hidden_size), nn.Sigmoid())
        self.ssm_B = nn.Linear(input_size, hidden_size)
        self.ssm_C = nn.Linear(aug_total, hidden_size)

    @property
    def beta_x(self):
        return torch.sigmoid(self.beta_x_raw)

    @property
    def beta_h(self):
        return torch.sigmoid(self.beta_h_raw)

    def _ssm_step(self, x_t, h_ssm, z):
        """One stable diagonal SSM step."""
        A = self.ssm_A(z)                                # [B, H], in [0,1]
        B = self.ssm_B(x_t)                              # [B, H]
        C = self.ssm_C(z)                                # [B, H]
        h_ssm_new = A * h_ssm + B                         # [B, H]
        g_ssm = C * h_ssm_new                            # [B, H]
        return h_ssm_new, g_ssm

    def forward(self, x_t, h_t, h_ssm, emas_x, emas_h, dt=1.0):
        x_t = torch.nan_to_num(x_t, nan=0.0)
        h_t = torch.nan_to_num(h_t, nan=0.0)
        h_ssm = torch.nan_to_num(h_ssm, nan=0.0)
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
        h_ssm_new, g_ssm = self._ssm_step(x_t, h_ssm, z)

        g_combined = h_branch * g_ssm

        if isinstance(dt, torch.Tensor):
            dt_b = dt
            if dt_b.dim() < 2:
                dt_b = dt_b.unsqueeze(-1)
            tau_eff = torch.exp(-f * dt_b / torch.abs(self.time_scale))
        else:
            tau_eff = torch.exp(-f * float(dt) / torch.abs(self.time_scale))
        h_new = tau_eff * g_combined + (1.0 - tau_eff) * h_branch

        return h_new, h_ssm_new, emas_x_new, emas_h_new


class StableSSMCfCStackedNetwork(nn.Module):
    """Stacked StableSSM-CfC with Kh ladder."""

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
                StableSSMCfCCell(
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
        h_ssms = [torch.zeros(B, self.hidden_size, device=device) for _ in range(self.num_layers)]
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
                hs[l], h_ssms[l], emas_x[l], emas_h[l] = cell(
                    inp, hs[l], h_ssms[l], emas_x[l], emas_h[l],
                )
                inp = hs[l]
            outputs.append(self.head(hs[-1]))
        outputs = torch.stack(outputs, dim=1)
        if self.return_sequences:
            return outputs
        return outputs[:, -1, :]


def make_lbps_lnkhlfft_stablessm_5_3_2(input_size, hidden_size, output_size, num_layers=3):
    """Kh=[5,3,2] + LN + FFT + Stable Diagonal SSM (sigmoid-A)."""
    return StableSSMCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kh_ladder=[5, 3, 2], Kx=5,
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


__all__ = [
    "StableSSMCfCCell",
    "StableSSMCfCStackedNetwork",
    "make_lbps_lnkhlfft_stablessm_5_3_2",
]
