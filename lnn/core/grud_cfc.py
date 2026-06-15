"""Time-Decay CfC (GRU-D / CT-RNN style) (PRD #10-110, Round 148, 2026-06-15).

Implements the GRU-D / CT-RNN / ODE-RNN idea (Che et al. 2018,
Jia & Benson 2019, Lechner & Hasani 2020) applied to CfC.

The key idea: between observations, the hidden state decays based
on the elapsed time Δt and a learnable per-feature decay rate γ:

    decay_t = exp(-γ * Δt)        # in (0, 1]
    h_t     = h_{t-1} * decay_t   # time-aware decay
    h_t     = CfCCell(x_t, h_t)   # standard CfC update

This is structurally different from:
- **Clockwork 147 (NEGATIVE)**: binary carry-forward (h stays the
  same for K-1 steps). Discontinuous.
- **SCRN 146 (target-dep α=0.5)**: parallel slow context with FIXED
  α, separate from main h.
- **GRU-D / CT-RNN (this round)**: continuous time-aware decay
  applied to MAIN h with LEARNABLE per-feature γ.

Risks:
- Time decay applied to h is closer to Clockwork (h modification)
  than to SCRN (parallel context). But the decay is CONTINUOUS, not
  binary, so it should be less harmful.
- Per-feature γ adds capacity but also the risk of overfitting.
- If γ is initialized too high, the model loses memory immediately.

Audit context (91-147):
- 13 strictly positive (preserves recurrent step + adds structure)
- 6 target-dep (input-side processing that preserves x OR bidi
  structural addition)
- 20 negatives (per-step modifications, alternatives, regularizers,
  bottlenecks, redundant info, replacements, long-α SCRN,
  Clockwork partition)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lnn.core.cfc import CfCCell


class TimeDecayCfCCell(nn.Module):
    """GRU-D / CT-RNN time-decay CfC cell.

    Args:
        input_size: input feature dimension D.
        hidden_size: hidden state dimension.
        gamma_init: initial value for log(gamma + eps). Default -3.0
            gives γ ≈ 0.05 (very light decay).
        use_time_input: if True, expects dt as an extra input feature
            (B, T, 1). If False, uses dt=1.0 for all steps (regular TS).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        gamma_init: float = -3.0,
        use_time_input: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_time_input = use_time_input

        # Per-feature learnable decay rate (in log-space for stability).
        # gamma_param initialized so softplus(gamma_param) ≈ gamma_init_value
        # For gamma_init=-3.0, softplus(-3.0) ≈ 0.05.
        self.gamma_param = nn.Parameter(torch.full((hidden_size,), gamma_init))

        # The CfC cell.
        if use_time_input:
            # Cell input includes dt as an extra feature.
            cell_input_size = input_size
        else:
            cell_input_size = input_size
        self.cfc = CfCCell(cell_input_size, hidden_size, n_tau=1)

    def get_gamma(self) -> torch.Tensor:
        """Return γ = softplus(gamma_param), shape (hidden_size,)."""
        return F.softplus(self.gamma_param)

    def forward(
        self,
        x: torch.Tensor,
        dt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass over the full sequence.

        Args:
            x: input of shape [B, T, input_size].
            dt: time deltas of shape [B, T, 1]. If None, uses dt=1.0
                for all steps.
        Returns:
            Hidden states of shape [B, T, hidden_size].
        """
        B, T, _ = x.shape
        device, dtype = x.device, x.dtype

        if dt is None:
            dt = torch.ones(B, T, 1, device=device, dtype=dtype)
        # Zero out NaN time deltas (no decay when time is missing).
        dt = torch.nan_to_num(dt, nan=0.0)
        # Clamp dt to non-negative (negative dt would invert decay).
        dt = torch.clamp(dt, min=0.0)

        gamma = self.get_gamma()  # (hidden_size,)
        # decay factor per step: exp(-γ * dt), shape (B, T, hidden_size)
        # We broadcast gamma over (B, T) and multiply by dt (B, T, 1).
        # decay = exp(-gamma * dt)  (gamma broadcasts over batch)
        decay = torch.exp(-gamma.view(1, 1, -1) * dt)  # (B, T, hidden)

        h = torch.zeros(B, self.hidden_size, device=device, dtype=dtype)
        outputs = []
        for t in range(T):
            # Time-aware decay of h.
            h = h * decay[:, t, :]
            # NaN-aware input.
            x_t = torch.nan_to_num(x[:, t, :], nan=0.0)
            # Standard CfC update.
            h = self.cfc(x_t, h)
            outputs.append(h)
        return torch.stack(outputs, dim=1)  # [B, T, hidden_size]


class TimeDecayCfCStackedNetwork(nn.Module):
    """Stacked Time-Decay CfC network (PRD #10-110).

    Args:
        input_size: input feature dimension.
        hidden_size: hidden dimension (per layer).
        output_size: output feature dimension.
        num_layers: number of stacked Time-Decay CfC cells.
        gamma_init: initial log(gamma) value.
        return_sequences: if True, return per-timestep outputs.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        gamma_init: float = -3.0,
        return_sequences: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.return_sequences = return_sequences

        self.cells = nn.ModuleList()
        for li in range(num_layers):
            if li == 0:
                in_size = input_size
            else:
                in_size = hidden_size  # output of previous layer
            self.cells.append(
                TimeDecayCfCCell(
                    input_size=in_size,
                    hidden_size=hidden_size,
                    gamma_init=gamma_init,
                )
            )

        # Final head.
        self.head = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x: torch.Tensor,
        dt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: input of shape [B, T, input_size].
            dt: time deltas of shape [B, T, 1]. If None, uses dt=1.0.
        Returns:
            Output of shape [B, T, output_size] if return_sequences else
            [B, output_size].
        """
        layer_input = x
        for cell in self.cells:
            layer_input = cell(layer_input, dt=dt)
        out = self.head(layer_input)
        if self.return_sequences:
            return out
        return out[:, -1, :]
