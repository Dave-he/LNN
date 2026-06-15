"""Decoupled CfC + IndRNN-CfC (PRD #10-105, 2026-06-15).

Implements the natural control experiment for round 142
(Multiplicative Integration CfC, which was CATASTROPHICALLY
NEGATIVE). Round 142 replaced the standard additive integration
`W_x x + W_h h` with the element-wise product `W_x x ⊙ W_h h`.

This round tests the additive analog with two variants:

1. **Decoupled CfC**: separate W_x and W_h matrices, additive
   combination. `inter = W_x x + W_h h`
2. **IndRNN-CfC**: element-wise recurrent weights (Li et al. 2018,
   CVPR 2018). `inter = W_x x + u ⊙ h` where u is a d-vector.

Both use the standard 3-branch CfC on `inter`:

```
inter = W_x x + W_h h       # Decoupled
# or
inter = W_x x + u ⊙ h       # IndRNN
f = sigma(W_f inter + b_f)
g = tanh(W_g inter + b_g)
h_out = tanh(W_h_out inter + b_h_out)
h_t = sigma(-f * tau) * g + (1 - sigma(-f * tau)) * h_out
```

This isolates whether the catastrophic failure of MI-CfC (round
142) was due to:
- (a) the **element-wise product** (multiplicative amplifies noise)
- (b) the **decoupling itself** (separate W_x, W_h projections)

Risks:

- Decoupled CfC might be equivalent to standard CfC (the linear
  layer in W[x, h] can learn the same function as W_x x + W_h h).
- IndRNN's element-wise recurrent weights may be too restrictive.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DecoupledCfCCell(nn.Module):
    """CfC cell with decoupled projections and additive combination.

    The standard CfC uses ``concat([x, h]) -> linear``. The
    decoupled variant uses separate ``W_x`` and ``W_h`` projections
    with additive combination.

    Args:
        input_size: input feature dimension.
        hidden_size: hidden state dimension.
        time_scale_init: initial value of the time constant τ.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        time_scale_init: float = 1.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Decoupled projections.
        self.x_proj = nn.Linear(input_size, hidden_size)
        self.h_proj = nn.Linear(hidden_size, hidden_size)

        # 3-branch CfC on (x_proj + h_proj).
        self.f_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
        )
        self.g_branch = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.h_branch = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        # Time scale.
        self.time_scale = nn.Parameter(torch.ones(hidden_size) * time_scale_init)

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """One recurrent step with decoupled projections.

        Args:
            x_t: input at current timestep, [B, input_size].
            h: hidden state, [B, hidden_size].
            dt: scalar time delta.
        Returns:
            New hidden state, [B, hidden_size].
        """
        x_proj = self.x_proj(x_t)
        h_proj = self.h_proj(h)
        # ADDITIVE combination (natural control for round 142 MI).
        inter = x_proj + h_proj

        f = self.f_gate(inter)
        g = self.g_branch(inter)
        h_out = self.h_branch(inter)
        decay = torch.sigmoid(-f * self.time_scale * dt)
        return decay * g + (1.0 - decay) * h_out


class IndRNNCfCCell(nn.Module):
    """CfC cell with IndRNN element-wise recurrent weights.

    Replaces the d×d recurrent weight matrix with a d-vector
    (element-wise multiplication per Li et al. 2018).

    Args:
        input_size: input feature dimension.
        hidden_size: hidden state dimension.
        time_scale_init: initial value of the time constant τ.
        u_init: initial value of the element-wise recurrent weights.
            Per the paper, |u| < 1 helps with gradient stability.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        time_scale_init: float = 1.0,
        u_init: float = 0.5,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input projection.
        self.x_proj = nn.Linear(input_size, hidden_size)
        # Element-wise recurrent weights (d-vector, NOT d×d matrix).
        self.u = nn.Parameter(torch.ones(hidden_size) * u_init)

        # 3-branch CfC on (x_proj + u * h).
        self.f_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
        )
        self.g_branch = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.h_branch = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        # Time scale.
        self.time_scale = nn.Parameter(torch.ones(hidden_size) * time_scale_init)

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        x_proj = self.x_proj(x_t)
        # Element-wise recurrent projection (d-vector, not matrix).
        h_proj = self.u * h
        # ADDITIVE combination.
        inter = x_proj + h_proj

        f = self.f_gate(inter)
        g = self.g_branch(inter)
        h_out = self.h_branch(inter)
        decay = torch.sigmoid(-f * self.time_scale * dt)
        return decay * g + (1.0 - decay) * h_out


class DecoupledCfCStackedNetwork(nn.Module):
    """Stacked Decoupled CfC cells (PRD #10-105).

    Args:
        input_size: input feature dimension.
        hidden_size: hidden state dimension.
        output_size: output feature dimension.
        num_layers: number of stacked cells.
        variant: "decoupled" or "indrnn".
        time_scale_init: initial time constant.
        u_init: initial IndRNN recurrent weight.
        return_sequences: if True, return outputs at every
            timestep; else return only the last.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        variant: str = "decoupled",
        time_scale_init: float = 1.0,
        u_init: float = 0.5,
        return_sequences: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        assert variant in ("decoupled", "indrnn"), f"variant must be in (decoupled, indrnn), got {variant}"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.variant = variant
        self.return_sequences = return_sequences

        cell_cls = DecoupledCfCCell if variant == "decoupled" else IndRNNCfCCell
        extra_kwargs = {"u_init": u_init} if variant == "indrnn" else {}
        self.cells = nn.ModuleList()
        for li in range(num_layers):
            in_size = input_size if li == 0 else hidden_size
            self.cells.append(
                cell_cls(
                    input_size=in_size,
                    hidden_size=hidden_size,
                    time_scale_init=time_scale_init,
                    **extra_kwargs,
                )
            )
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: input of shape ``[B, T, input_size]``.
        Returns:
            Output of shape ``[B, T, output_size]`` if
            ``return_sequences=True`` else ``[B, output_size]``.
        """
        B, T, _ = x.shape
        h = [
            torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
            for _ in range(self.num_layers)
        ]
        outputs = []
        for t in range(T):
            inp = torch.nan_to_num(x[:, t, :], nan=0.0)
            for li, cell in enumerate(self.cells):
                h_new = cell(inp, h[li])
                h[li] = h_new
                inp = h[li]
            outputs.append(self.head(h[-1]))
        outputs = torch.stack(outputs, dim=1)
        if self.return_sequences:
            return outputs
        return outputs[:, -1, :]
