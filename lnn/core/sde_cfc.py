"""SDE-CfC — Stochastic Differential Equation generalization of CfC (r306).

Reference: arXiv:2608.28702 (Stochastic Liquid Deformation Fields, 2026-08-27).
The paper notes that the CfC closed-form solution is the deterministic limit of
a noise-driven system and re-introduces a small Gaussian perturbation on the
time gate of every CfC cell:

    σ(-f(x, I; θ_f) · t)  →  σ(-f(x, I; θ_f) · t) + σ_sde · ε,   ε ~ N(0, 1)

This is a minimal-variance SDE extension:
  - noise is added *before* the sigmoid (so the gate stays in ℝ),
  - σ_sde is a single scalar hyper-parameter (default 0.05, scaled like
    the r192 input-noise σ that gave toy_sin -24% mean);
  - noise is **training-only**; at eval time σ_sde is forced to 0 so the
    model degrades to the deterministic CfC exact limit (paper §3.2);
  - the noise is per-element (per-neuron, per-sample, per-step), so it
    acts as a structured regularizer on the temporal gate itself rather
    than as input corruption.

Why a separate module rather than a CfCCell subclass:
  - the existing CfCCell forward is invoked by ~150 variants; wrapping every
    call site is risky and would duplicate the multi-τ branch logic;
  - a small wrapper cell is cleaner and matches the r192 / r193 noise-axis
    conventions used elsewhere in the repo.

Why this is novel within this repo:
  - r192 (input noise), r193 (hidden noise), r194 (xh noise) all add noise
    to the **input/hidden stream**. None of them touches the *gate* itself,
    which is what the SDE view argues is the natural home for stochasticity
    because the gate IS the time-continuous relaxation mechanism.
  - r265 / r267 (STE) add **discrete** stochasticity via a STE mask on
    neurons, not a Gaussian perturbation on the time constant.

The hypothesis we test on toy_sin / structured_irr / random_irr:
  - small σ_sde helps toy_sin (r192 input noise did: -24% mean);
  - we expect it to be near-neutral or harmful on random_irr (r193 hidden
    noise: +21%, the gate is sensitive to noise when the signal is weak).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.cfc import CfCCell


class SDECfCCell(nn.Module):
    """CfC cell with an additive Gaussian perturbation on the time gate.

    Args:
        input_size: Input feature dimension.
        hidden_size: Hidden state dimension.
        n_tau: Number of independent time-scale groups (≥1).
        tau_scales: Per-branch initial time constants.
        sigma_sde: Initial std of the Gaussian perturbation. The noise is
            scaled by this value during training and dropped at eval time.
        learnable_sigma: If True, ``sigma_sde`` becomes a non-negative
            parameter (``softplus``-parameterized) learned jointly with the
            rest of the network. If False, it stays fixed.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_tau: int = 1,
        tau_scales: tuple = (0.1, 1.0, 10.0),
        sigma_sde: float = 0.05,
        learnable_sigma: bool = False,
    ):
        super().__init__()
        # Delegate all the actual mechanics to a stock CfCCell — we only
        # wrap forward() to inject the noise term.  This way multi-τ
        # behaviour stays consistent with the rest of the repo.
        self.cell = CfCCell(
            input_size=input_size,
            hidden_size=hidden_size,
            n_tau=n_tau,
            tau_scales=tau_scales,
        )
        if learnable_sigma:
            # softplus(s_raw) keeps sigma_sde strictly > 0
            self._sigma_raw = nn.Parameter(torch.tensor(float(sigma_sde)))
        else:
            self.register_buffer("_sigma_fixed", torch.tensor(float(sigma_sde)))
        self.learnable_sigma = bool(learnable_sigma)

    @property
    def hidden_size(self) -> int:
        return self.cell.hidden_size

    @property
    def input_size(self) -> int:
        return self.cell.input_size

    def current_sigma(self) -> float:
        """Return the *effective* σ_sde right now (always ≥ 0)."""
        if self.learnable_sigma:
            # softplus to keep > 0; matches r192 sweep convention
            return float(torch.nn.functional.softplus(self._sigma_raw).item())
        return float(self._sigma_fixed.item())

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        combined = torch.cat([x_t, h], dim=-1)
        if self.cell._multi_tau:
            branch_outputs = []
            for i in range(self.cell.n_tau):
                f = self.cell.f_gates[i](combined)
                g = self.cell.g_branches[i](combined)
                h_out = self.cell.h_branches[i](combined)
                decay_pre = -f * self.cell.time_scales[i] * dt
                decay_pre = self._inject_noise(decay_pre)
                gate = torch.sigmoid(decay_pre)
                branch_outputs.append(gate * g + (1.0 - gate) * h_out)
            return torch.cat(branch_outputs, dim=-1)
        f = self.cell.f_gate(combined)
        g = self.cell.g_branch(combined)
        h_out = self.cell.h_branch(combined)
        decay_pre = -f * self.cell.time_scale * dt
        decay_pre = self._inject_noise(decay_pre)
        return torch.sigmoid(decay_pre) * g + (1.0 - torch.sigmoid(decay_pre)) * h_out

    def _inject_noise(self, decay_pre: torch.Tensor) -> torch.Tensor:
        """Add a Gaussian perturbation on the pre-sigmoid gate.

        Training mode: noise enabled, scaled by current σ_sde.
        Eval mode: identity (σ_sde=0 effectively) — matches paper §3.2.
        """
        if not self.training or self.current_sigma() <= 0.0:
            return decay_pre
        noise = torch.randn_like(decay_pre) * self.current_sigma()
        return decay_pre + noise


class SDECfCNetwork(nn.Module):
    """Stacked SDE-CfC network with a vanilla ``CfCNetwork``-shaped API."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        return_sequences: bool = True,
        n_tau: int = 1,
        tau_scales: tuple = (0.1, 1.0, 10.0),
        sigma_sde: float = 0.05,
        learnable_sigma: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.return_sequences = return_sequences

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            self.cells.append(
                SDECfCCell(
                    in_dim,
                    hidden_size,
                    n_tau=n_tau,
                    tau_scales=tau_scales,
                    sigma_sde=sigma_sde,
                    learnable_sigma=learnable_sigma,
                )
            )
        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
        dt: float | torch.Tensor | None = None,
        mask: torch.Tensor | None = None,  # noqa: ARG002 — API parity with CfCNetwork
    ) -> torch.Tensor:
        B, T, _ = x.shape
        L = self.num_layers
        H = self.hidden_size
        device = x.device

        if h0 is None:
            h = torch.zeros(L, B, H, device=device, dtype=x.dtype)
        elif h0.dim() == 2:
            h = h0.unsqueeze(0).expand(L, -1, -1).contiguous()
        else:
            h = h0

        outputs: list[torch.Tensor] = []
        h_state = h.clone() if isinstance(h, torch.Tensor) else h

        outputs: list[torch.Tensor] = []
        h_state = h.clone() if isinstance(h, torch.Tensor) else h
        # Per-layer per-timestep output buffer for inter-layer routing.
        layer_buf: list[torch.Tensor] = [
            torch.zeros(B, H, device=device, dtype=x.dtype) for _ in range(L)
        ]

        for t in range(T):
            x_t = x[:, t, :]
            for layer_idx, cell in enumerate(self.cells):
                h_layer = h_state[layer_idx]
                if layer_idx == 0:
                    inp = x_t
                else:
                    inp = layer_buf[layer_idx - 1]
                new_h = cell(inp, h_layer, dt=dt if dt is not None else 1.0)
                layer_buf[layer_idx] = new_h
                h_state[layer_idx] = new_h

            outputs.append(layer_buf[-1])

        stacked = torch.stack(outputs, dim=1)  # [B, T, H]
        y = self.output_proj(stacked)
        return y if self.return_sequences else y[:, -1, :]


__all__ = ["SDECfCCell", "SDECfCNetwork"]
