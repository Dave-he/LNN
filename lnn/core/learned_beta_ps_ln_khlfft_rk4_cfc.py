"""LearnedBetaPS+LN+Khl+FFT+RK4-CfC (PRD #10-160, Round 198, 2026-06-16).

After 5 regularization rounds (r192-r197) all NEGATIVE or
TARGET-DEPENDENT, pivot to a fundamentally different paradigm:
**higher-order ODE integration**.

Default CfC uses closed-form solution (forward Euler / exact
exponential decay) — fast but only 1st order accurate. The
GLNN paper (arXiv 2025) integrates Runge-Kutta DOPRI5 into
LNN and extends to non-sequence tasks.

Here we use **RK4 (4th order Runge-Kutta)** for the CfC step:
    k1 = cf_delta(h, x)
    k2 = cf_delta(h + 0.5*dt*k1, x)
    k3 = cf_delta(h + 0.5*dt*k2, x)
    k4 = cf_delta(h + dt*k3, x)
    h_new = h + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

The cell uses the same f_gate, g_branch, h_branch as r187 —
the only change is the integration scheme.

Hypothesis:
- H1 (positive): RK4 captures long-term structure better
  than closed-form (less numerical drift on smooth data)
- H2 (negative): RK4 overfits, slow to converge
- H3 (mixed): helps smooth (sin), hurts multi-regime
  (structured)

Mechanism::
    For each timestep t:
        # Same FFT + Kh + LN as r187
        x_aug = fft_encode(x)
        ema_x_k = β_x_k * ema_x_k + (1 - β_x_k) * x_aug_t
        ema_h_k = β_h_k * ema_h_k + (1 - β_h_k) * h_{t-1}
        z = cat([aug_x, aug_h])  # [B, (Kx+1)D + (Kh+1)H]
        z = LayerNorm(z)
        f = σ(linear(z))
        g = tanh(linear(z))
        h_branch = tanh(linear(z))
        # RK4 over the closed-form delta:
        delta = cf_delta(h, z, f, g, h_branch)
        k1 = delta
        k2 = cf_delta(h + 0.5*dt*k1, z, f, g, h_branch)
        k3 = cf_delta(h + 0.5*dt*k2, z, f, g, h_branch)
        k4 = cf_delta(h + dt*k3, z, f, g, h_branch)
        h_t = h + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.learned_beta_ps_ln_fft_cfc import FFTInputEncoder
from lnn.core.learned_beta_ps_ln_khl_cfc import LearnedBetaPSLNKhlCfCStackedNetwork


class RK4CfCCell(nn.Module):
    """Single CfC cell with RK4 integration over the closed-form ODE.

    Same f_gate, g_branch, h_branch, time_scale as r187, but
    the integration is RK4 instead of closed-form.
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

        # Per-scale learned β (r171).
        self.beta_x_raw = nn.Parameter(
            torch.full((Kx,), float(torch.log(torch.tensor(beta_x_init / (1.0 - beta_x_init))).item()))
        )
        self.beta_h_raw = nn.Parameter(
            torch.full((Kh,), float(torch.log(torch.tensor(beta_h_init / (1.0 - beta_h_init))).item()))
        )

        # LayerNorm (r179).
        self.layer_norm = nn.LayerNorm(aug_total, eps=ln_eps)

        # CfC closed-form components.
        self.f_gate = nn.Sequential(nn.Linear(aug_total, hidden_size), nn.Sigmoid())
        self.g_branch = nn.Sequential(nn.Linear(aug_total, hidden_size), nn.Tanh())
        self.h_branch = nn.Sequential(nn.Linear(aug_total, hidden_size), nn.Tanh())
        self.time_scale = nn.Parameter(torch.ones(hidden_size))

    @property
    def beta_x(self):
        return torch.sigmoid(self.beta_x_raw)

    @property
    def beta_h(self):
        return torch.sigmoid(self.beta_h_raw)

    def _broadcast_dt(self, dt):
        """Normalize dt to shape [B, 1] for broadcasting with [B, H] tensors."""
        if isinstance(dt, torch.Tensor):
            if dt.dim() == 0:
                dt = dt.unsqueeze(0).unsqueeze(-1)
            elif dt.dim() == 1:
                dt = dt.unsqueeze(-1)
            elif dt.dim() >= 2:
                # Already [B, ...], squeeze trailing dims
                while dt.dim() > 2:
                    dt = dt.squeeze(-1)
            return dt
        # Scalar → broadcast
        return dt

    def _cf_delta(self, h, f, g, h_branch, dt):
        """Compute the CfC step delta (h_new - h) under closed-form.

        h_new = tau_eff * g + (1 - tau_eff) * h_branch
        delta = h_new - h
        """
        dt_b = self._broadcast_dt(dt)
        if isinstance(dt_b, torch.Tensor):
            tau_eff = torch.exp(-f * dt_b / torch.abs(self.time_scale))
        else:
            tau_eff = torch.exp(-f * float(dt_b) / torch.abs(self.time_scale))
        h_new = tau_eff * g + (1.0 - tau_eff) * h_branch
        return h_new - h

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

        # Compute f, g, h_branch once
        f = self.f_gate(z)
        g = self.g_branch(z)
        h_branch = self.h_branch(z)

        # RK4 step on the closed-form ODE
        dt_b = self._broadcast_dt(dt)
        k1 = self._cf_delta(h_t, f, g, h_branch, dt)
        h2 = h_t + 0.5 * dt_b * k1
        k2 = self._cf_delta(h2, f, g, h_branch, dt)
        h3 = h_t + 0.5 * dt_b * k2
        k3 = self._cf_delta(h3, f, g, h_branch, dt)
        h4 = h_t + dt_b * k3
        k4 = self._cf_delta(h4, f, g, h_branch, dt)

        h_new = h_t + (dt_b / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        return h_new, emas_x_new, emas_h_new


class RK4CfCStackedNetwork(nn.Module):
    """Stacked RK4-CfC with Kh ladder (replaces r187 with RK4)."""

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

        # RK4 cells
        self.cells = nn.ModuleList()
        for l in range(num_layers):
            in_size = self.augmented_input_size if l == 0 else hidden_size
            self.cells.append(
                RK4CfCCell(
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
        # Apply FFT encoder to add frequency features
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


def make_lbps_lnkhlfft_rk4_5_3_2(input_size, hidden_size, output_size, num_layers=3):
    """Kh=[5,3,2] + LN + FFT + RK4 (replaces r187's closed-form with RK4)."""
    return RK4CfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kh_ladder=[5, 3, 2], Kx=5,
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


__all__ = [
    "RK4CfCCell",
    "RK4CfCStackedNetwork",
    "make_lbps_lnkhlfft_rk4_5_3_2",
]
