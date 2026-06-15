"""DELTA-CfC (Hidden State Delta Augmentation) (PRD #10-117, Round 155, 2026-06-15).

Augments the CfC's hidden state output with the temporal
derivative Δh_t = h_t - h_{t-1}. The temporal derivative carries
information about regime switches, noise level, and stability
of h_t that the closed-form solution does not explicitly expose.

Mechanism::

    h_t       = CfC(x_t, h_{t-1})            # standard
    delta_t   = h_t - h_{t-1}                # temporal derivative
    h_aug_t   = concat([h_t, delta_t])       # 2*hidden_size output

This is structurally different from:
- **DiffCfC (round 145)**: input-side deltas Δx_t, Δ²x_t.
- **TDSA 152 / MSDC 151 / TCC 149**: parallel context, concat
  with x.
- **FiLM 153**: γ, β modulation.
- **SCRN 146 / Time-Decay 148 / Clockwork 147**: alternative
  memory structures.

Variants:
- delta_concat: h_aug = concat([h, Δh]), 2*hidden_size output.
- delta_proj: Linear(2*hidden_size, hidden_size) projection.
- delta_gated: h_out = (1-α) * h + α * Δh, learned α.
- delta_concat_input: pass Δh as additional input to next layer.

Audit context (91-154): 14 strictly positive + 10 target-dep +
23 negatives = 47 mechanism classes.

Risks:
- Δh has 0 mean over time (h_t is a stable process), may be
  uninformative.
- Concat doubles hidden dim → more params in next layer.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.sequence_utils import select_step_delta, select_step_mask


class DeltaCfCCell(nn.Module):
    """Delta-CfC cell: augments h with temporal derivative Δh.

    Args:
        input_size: input feature dimension D.
        hidden_size: hidden state dimension.
        delta_mode: 'concat' (2H output), 'proj' (H output, projected),
            'gated' (H output, gated), 'concat_input' (H output, passes
            Δh to next layer as additional input).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        delta_mode: str = "concat",
    ):
        super().__init__()
        assert delta_mode in ("concat", "proj", "gated", "concat_input"), \
            f"delta_mode must be one of 'concat', 'proj', 'gated', 'concat_input', got {delta_mode}"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.delta_mode = delta_mode

        # Standard CfC.
        self.f_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid(),
        )
        self.g_branch = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.h_branch = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.time_scale = nn.Parameter(torch.ones(hidden_size))

        if delta_mode == "proj":
            # Project concat([h, delta]) back to hidden_size.
            self.delta_proj = nn.Linear(2 * hidden_size, hidden_size)
        elif delta_mode == "gated":
            # Learned per-dim scalar gate.
            self.delta_gate = nn.Parameter(torch.zeros(hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """One step of DELTA-CfC. Returns the AUGMENTED output.

        The next step's h_i is derived by the stacked network from
        the augmented output (the h_new portion, which is the first
        hidden_size dims of the concat).

        Args:
            x: input at this step [B, input_size].
            h: previous hidden state [B, hidden_size].
            dt: time delta.

        Returns:
            - For 'concat': augmented hidden [B, 2*hidden_size].
            - For 'proj', 'gated', 'concat_input': hidden [B, hidden_size].
        """
        x = torch.nan_to_num(x, nan=0.0)
        z = torch.cat([x, h], dim=-1)

        # Closed-form CfC solution.
        f = self.f_gate(z)
        g = self.g_branch(z)
        h_branch = self.h_branch(z)

        if isinstance(dt, torch.Tensor):
            dt_b = dt.unsqueeze(-1) if dt.dim() < 1 else dt
            if dt_b.dim() == 1:
                dt_b = dt_b.unsqueeze(-1)
            tau_eff = torch.exp(-f * dt_b / torch.abs(self.time_scale))
        else:
            tau_eff = torch.exp(-f * float(dt) / torch.abs(self.time_scale))

        h_new = tau_eff * g + (1.0 - tau_eff) * h_branch

        # Compute delta and apply mode.
        delta = h_new - h
        if self.delta_mode == "concat":
            # 2*hidden_size output.
            out = torch.cat([h_new, delta], dim=-1)
        elif self.delta_mode == "proj":
            aug = torch.cat([h_new, delta], dim=-1)
            out = self.delta_proj(aug)
        elif self.delta_mode == "gated":
            alpha = torch.sigmoid(self.delta_gate)
            out = (1.0 - alpha) * h_new + alpha * delta
        else:  # concat_input
            out = h_new

        return out

    def delta_for_next_layer(self, h_new: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """Return Δh to be passed to the next layer (for concat_input mode)."""
        return h_new - h_prev

    @staticmethod
    def extract_h(out: torch.Tensor, hidden_size: int, delta_mode: str) -> torch.Tensor:
        """Extract the h_new (hidden_size) portion from the cell's output.

        For 'concat': the first hidden_size dims are h_new.
        For others: the entire output is h_new.
        """
        if delta_mode == "concat":
            return out[:, :hidden_size]
        return out


class DeltaCfCStackedNetwork(nn.Module):
    """Stacked Delta-CfC network (PRD #10-117).

    Args:
        input_size: input feature dimension.
        hidden_size: hidden dimension (per layer).
        output_size: output feature dimension.
        num_layers: number of stacked Delta-CfC cells.
        delta_mode: 'concat', 'proj', 'gated', 'concat_input'.
        return_sequences: if True, return per-timestep outputs.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        delta_mode: str = "concat",
        return_sequences: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.delta_mode = delta_mode
        self.return_sequences = return_sequences

        # For 'concat' mode, the cell output is 2*hidden_size.
        # For 'proj'/'gated'/'concat_input', the cell output is hidden_size.
        if delta_mode == "concat":
            layer_out = [2 * hidden_size] * num_layers
        else:
            layer_out = [hidden_size] * num_layers

        # For 'concat_input', the cell's input is augmented with the
        # previous layer's delta (hidden_size extra).
        # For 'concat', the next layer's input is 2*hidden_size.
        if delta_mode == "concat_input":
            layer_in = [input_size] + [2 * hidden_size] * (num_layers - 1)
        elif delta_mode == "concat":
            layer_in = [input_size] + [2 * hidden_size] * (num_layers - 1)
        else:
            layer_in = [input_size] + [hidden_size] * (num_layers - 1)

        self.cells = nn.ModuleList()
        for li in range(num_layers):
            self.cells.append(DeltaCfCCell(layer_in[li], hidden_size, delta_mode=delta_mode))

        # Final head: input dim is layer_out[-1].
        self.output_proj = nn.Linear(layer_out[-1], output_size)

    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
        dt: float | torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass over the sequence.

        Args:
            x: input [B, T, input_size].
            h0: optional initial hidden state.
            dt: optional per-step time deltas.
            mask: optional observed-feature mask.

        Returns:
            Output [B, T, output_size] if return_sequences else [B, output_size].
        """
        batch_size, seq_len, _ = x.shape
        if h0 is None:
            h0 = torch.zeros(
                self.num_layers, batch_size, self.hidden_size,
                device=x.device, dtype=x.dtype,
            )

        h = h0
        layer_input = x
        for i, cell in enumerate(self.cells):
            outputs = []
            deltas = []
            h_i = h[i]
            for t in range(seq_len):
                dt_t = select_step_delta(dt, t, batch_size, seq_len, x.device, x.dtype)
                input_mask, update_mask = select_step_mask(
                    mask, t, batch_size, seq_len, self.input_size, x.device, x.dtype
                )
                x_t = torch.nan_to_num(layer_input[:, t, :])
                if i == 0 and input_mask is not None:
                    x_t = x_t * input_mask
                h_prev = h_i.clone()
                out = cell(x_t, h_i, dt=dt_t)
                # Extract the h_new portion for the next step's h_i.
                h_new = DeltaCfCCell.extract_h(out, self.hidden_size, self.delta_mode)
                h_i = h_new if update_mask is None else update_mask * h_new + (1.0 - update_mask) * h_i
                outputs.append(out)
                if self.delta_mode == "concat_input":
                    deltas.append(h_new - h_prev)
            layer_input = torch.stack(outputs, dim=1)
            if self.delta_mode == "concat_input" and i < self.num_layers - 1:
                # Pass Δh as additional input to next layer.
                delta_seq = torch.stack(deltas, dim=1)  # [B, T, H]
                layer_input = torch.cat([layer_input, delta_seq], dim=-1)
            h = torch.cat(
                [h_i.unsqueeze(0) if j == i else h[j].unsqueeze(0) for j in range(self.num_layers)],
                dim=0,
            )

        if self.return_sequences:
            return self.output_proj(layer_input)
        return self.output_proj(layer_input[:, -1, :])
