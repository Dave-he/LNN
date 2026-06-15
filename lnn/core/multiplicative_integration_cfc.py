"""Multiplicative Integration CfC (MI-CfC, PRD #10-104, 2026-06-15).

Implements the Multiplicative Integration idea from Wu et al. 2016
(NIPS 2016, "On Multiplicative Integration with Recurrent Neural
Networks") applied to CfC.

Standard CfC uses additive integration::

    combined = W_x x_t + W_h h_{t-1}    # [B, hidden_size]
    f = sigma(W_f combined + b_f)
    g = tanh(W_g combined + b_g)
    h_out = tanh(W_h combined + b_h)
    h_t = sigma(-f * tau) * g + (1 - sigma(-f * tau)) * h_out

Multiplicative Integration replaces the additive combination with
element-wise product of separately projected x and h::

    x_proj = W_x x_t                    # [B, hidden_size]
    h_proj = W_h h_{t-1}                # [B, hidden_size]
    inter = x_proj * h_proj             # [B, hidden_size]  <-- multiplicative
    f = sigma(W_f inter + b_f)
    g = tanh(W_g inter + b_g)
    h_out = tanh(W_h inter + b_h)
    h_t = sigma(-f * tau) * g + (1 - sigma(-f * tau)) * h_out

The element-wise product creates a *feature-wise modulation* where
each hidden dimension receives a multiplicative gate from the
corresponding input dimension. This is structurally different
from additive (which sums contributions) and is the standard
form proposed in Wu et al. 2016.

This module contains:

- **MultiplicativeIntegrationCfCCell**: standard 3-branch CfC cell
  with pure multiplicative integration.
- **MultiplicativeIntegrationCfCStackedNetwork**: stack of MI-CfC
  cells.
- **MultiplicativeIntegrationXResidualCfCCell**: MI variant with
  additive x residual ``inter = x_proj * h_proj + x_proj`` to
  handle the h=0 chicken-and-egg problem.

Risks:

- **h=0 chicken-and-egg**: when h=0 at t=0, ``inter = 0`` and
  h_new is determined entirely by gate biases. We initialize
  gate biases to non-zero defaults to break symmetry.
- **Multiplicative may amplify noise**: the product is more
  sensitive to noise than additive. The x_residual variant
  helps mitigate this.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MultiplicativeIntegrationCfCCell(nn.Module):
    """CfC cell with multiplicative integration (Wu et al. 2016).

    Replaces the standard additive ``W_x x + W_h h`` with the
    element-wise product ``W_x x ⊙ W_h h``. The rest of the
    3-branch CfC structure is preserved.

    Args:
        input_size: input feature dimension.
        hidden_size: hidden state dimension.
        time_scale_init: initial value of the time constant τ.
        f_bias_init: initial bias for the f_gate (default 1.0 to
            break the h=0 chicken-and-egg symmetry).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        time_scale_init: float = 1.0,
        f_bias_init: float = 1.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Multiplicative projections: separate W for x and h.
        self.x_proj = nn.Linear(input_size, hidden_size)
        self.h_proj = nn.Linear(hidden_size, hidden_size)

        # 3-branch CfC on the multiplicative interaction.
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

        # Time scale (per hidden neuron).
        self.time_scale = nn.Parameter(torch.ones(hidden_size) * time_scale_init)

        # Initialize biases to break h=0 symmetry.
        # When inter = 0 at t=0, gates output is determined by biases.
        # f_bias > 0 → more weight on h_out (recurrence dominates).
        with torch.no_grad():
            self.f_gate[0].bias.fill_(float(f_bias_init))
            self.g_branch[0].bias.fill_(0.5)
            self.h_branch[0].bias.fill_(0.5)

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """One recurrent step with multiplicative integration.

        Args:
            x_t: input at current timestep, [B, input_size].
            h: hidden state, [B, hidden_size].
            dt: scalar time delta.
        Returns:
            New hidden state, [B, hidden_size].
        """
        # Multiplicative interaction.
        x_proj = self.x_proj(x_t)            # [B, hidden]
        h_proj = self.h_proj(h)              # [B, hidden]
        inter = x_proj * h_proj              # [B, hidden] element-wise

        # Standard 3-branch CfC on inter.
        f = self.f_gate(inter)
        g = self.g_branch(inter)
        h_out = self.h_branch(inter)
        decay = torch.sigmoid(-f * self.time_scale * dt)
        return decay * g + (1.0 - decay) * h_out


class MultiplicativeIntegrationXResidualCfCCell(nn.Module):
    """MI-CfC with additive x residual to handle h=0 at init.

    ``inter = x_proj ⊙ h_proj + x_proj``

    When h=0, the multiplicative product is 0 but the additive
    x_proj still flows, so the gates can produce non-trivial
    outputs and h can evolve. This is the safer variant.

    Args:
        input_size: input feature dimension.
        hidden_size: hidden state dimension.
        time_scale_init: initial value of the time constant τ.
        f_bias_init: initial bias for the f_gate.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        time_scale_init: float = 1.0,
        f_bias_init: float = 1.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Multiplicative projections.
        self.x_proj = nn.Linear(input_size, hidden_size)
        self.h_proj = nn.Linear(hidden_size, hidden_size)

        # 3-branch CfC on (x_proj * h_proj + x_proj).
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

        # Initialize biases to non-zero defaults.
        with torch.no_grad():
            self.f_gate[0].bias.fill_(float(f_bias_init))
            self.g_branch[0].bias.fill_(0.5)
            self.h_branch[0].bias.fill_(0.5)

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        x_proj = self.x_proj(x_t)
        h_proj = self.h_proj(h)
        # Multiplicative + additive x residual.
        inter = x_proj * h_proj + x_proj

        f = self.f_gate(inter)
        g = self.g_branch(inter)
        h_out = self.h_branch(inter)
        decay = torch.sigmoid(-f * self.time_scale * dt)
        return decay * g + (1.0 - decay) * h_out


class MultiplicativeIntegrationCfCStackedNetwork(nn.Module):
    """Stacked MI-CfC cells (PRD #10-104).

    Each layer is a ``MultiplicativeIntegrationCfCCell`` (or its
    x_residual variant). The MI is applied per layer. The output
    is the head projection of the last layer's hidden state.

    Args:
        input_size: input feature dimension.
        hidden_size: hidden state dimension.
        output_size: output feature dimension.
        num_layers: number of stacked cells.
        variant: "pure" or "x_residual".
        time_scale_init: initial time constant.
        f_bias_init: initial f_gate bias.
        return_sequences: if True, return outputs at every
            timestep; else return only the last.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        variant: str = "x_residual",
        time_scale_init: float = 1.0,
        f_bias_init: float = 1.0,
        return_sequences: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        assert variant in ("pure", "x_residual"), f"variant must be in (pure, x_residual), got {variant}"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.variant = variant
        self.return_sequences = return_sequences

        cell_cls = (
            MultiplicativeIntegrationCfCCell
            if variant == "pure"
            else MultiplicativeIntegrationXResidualCfCCell
        )
        self.cells = nn.ModuleList()
        for li in range(num_layers):
            in_size = input_size if li == 0 else hidden_size
            self.cells.append(
                cell_cls(
                    input_size=in_size,
                    hidden_size=hidden_size,
                    time_scale_init=time_scale_init,
                    f_bias_init=f_bias_init,
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
                # For subsequent layers, the input is the previous layer's hidden.
                inp = h[li]
            outputs.append(self.head(h[-1]))
        outputs = torch.stack(outputs, dim=1)
        if self.return_sequences:
            return outputs
        return outputs[:, -1, :]
