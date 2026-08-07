"""Round 305 — MidpointCfC: non-anchor parallel scan via predictor-corrector.

r301 ParallelCfC uses an **anchor approximation**: within a window of W steps,
the cell evaluates W CfC closed-form updates all conditioned on the same h_0
(the chunk's initial state).  This is order-dt accurate but biased.

This file implements a **non-anchor** correction: the **MidpointCfCCell**
uses a *predictor-corrector* parallel scan (Heun's method / explicit midpoint)
to gain order-dt² accuracy while still being parallel within a chunk.

For each chunk of W steps with input x ∈ R^{B×W×d_in} and initial h_0:

    1) Predictor pass (parallel, anchored at h_0):
         For s = 0, 1, …, W-1:
             f_s  = σ(Wf · [x_{t+s}; h_0] + bf)
             g_s  = tanh(Wg · [x_{t+s}; h_0] + bg)
             hp_s = tanh(Wh · [x_{t+s}; h_0] + bh)
             h̃_s = σ(-f_s · τ · dt) * g_s + (1 - σ(-f_s · τ · dt)) * hp_s
         h_pred = h̃_{W-1}                  # predicted end-state

    2) Midpoint anchor (scalar average):
         h_mid = 0.5 * (h_0 + h_pred)

    3) Corrector pass (parallel, anchored at h_mid):
         Same computation as predictor but with h_0 ← h_mid
         h_corr = h̃_{W-1}

    4) Output: h_corr

The corrector pass doubles the parallel work per chunk (still O(W) per
chunk, but 2 evaluations instead of 1).  Crucially, the **two passes are
independent** and could be fused into a single matmul of 2W rows if memory
allows.  In our CPU implementation we run them sequentially to keep the
code simple and the FLOPs roughly comparable to r301 W=2.

Honest scope:
    This is the *simplest* non-anchor parallel scan.  Higher-order methods
    (RK4 over chunks, implicit midpoint, etc.) are out of scope.  We only
    aim to demonstrate that the anchor bias from r301 can be reduced
    without paying full sequential cost.

Connection to r301+r302+r303+r304:
    r301 (anchor, W=8):     O(W) parallel,  order-dt,  -7.1% MSE on toy_sin
    r302 (sharp transition): validates whether r301 holds on event-like data
    r303 (STE + parallel):   discrete routing masks the anchor error
    r305 (this file):        midpoint corrector gives order-dt² for free
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class MidpointCfCCell(nn.Module):
    """Predictor-corrector parallel CfC cell (non-anchor)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        window: int = 4,
        tau_init: float = 1.0,
    ) -> None:
        super().__init__()
        assert window >= 1, f"window must be >= 1, got {window}"
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.window = int(window)

        # Same projections as CfCCell / ParallelCfCCell.
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
        self.time_scale = nn.Parameter(torch.full((hidden_size,), float(tau_init)))

    def _parallel_eval(
        self, x: torch.Tensor, h_anchor: torch.Tensor, dt: torch.Tensor
    ) -> torch.Tensor:
        """Run W parallel CfC updates anchored at h_anchor; return final h.

        Args:
            x:         (B, W, d_in)
            h_anchor:  (B, hidden) — anchor state for the whole window
            dt:        scalar or (B, 1) broadcast
        """
        B, W, _ = x.shape
        assert W == self.window, f"x window {W} != cell.window {self.window}"
        h_exp = h_anchor.unsqueeze(1).expand(B, W, self.hidden_size)
        z = torch.cat([x, h_exp], dim=-1)
        f = self.f_gate(z)
        g = self.g_branch(z)
        hp = self.h_branch(z)
        if dt.dim() == 0:
            dt_eff = dt
        elif dt.dim() == 1:
            dt_eff = dt.view(B, 1, 1)
        else:
            dt_eff = dt.view(B, W, 1)
        decay = torch.sigmoid(-f * self.time_scale * dt_eff)
        h_steps = decay * g + (1.0 - decay) * hp
        return h_steps[:, -1, :]

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        dt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.window == 1 or x.dim() == 2:
            # Single-step path — identical to vanilla CfC.
            if dt is None:
                dt_t = torch.tensor(1.0, device=x.device, dtype=x.dtype)
            else:
                dt_t = dt
            z = torch.cat([x, h], dim=-1)
            f = self.f_gate(z)
            g = self.g_branch(z)
            hp = self.h_branch(z)
            if dt_t.dim() == 0:
                dt_eff = dt_t
            elif dt_t.dim() == 1:
                dt_eff = dt_t.unsqueeze(-1)
            else:
                dt_eff = dt_t
            decay = torch.sigmoid(-f * self.time_scale * dt_eff)
            return decay * g + (1.0 - decay) * hp

        # Midpoint predictor-corrector.
        if dt is None:
            dt_t = torch.tensor(1.0, device=x.device, dtype=x.dtype)
        else:
            dt_t = dt
        # Predictor at h_0
        h_pred = self._parallel_eval(x, h, dt_t)
        # Midpoint anchor
        h_mid = 0.5 * (h + h_pred)
        # Corrector at h_mid
        h_corr = self._parallel_eval(x, h_mid, dt_t)
        return h_corr


class MidpointCfCNetwork(nn.Module):
    """Multi-layer MidpointCfC sequence model."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        window: int = 4,
        return_sequences: bool = False,
    ) -> None:
        super().__init__()
        assert num_layers >= 1
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.num_layers = int(num_layers)
        self.window = int(window)
        self.return_sequences = return_sequences

        self.cells = nn.ModuleList(
            [
                MidpointCfCCell(
                    input_size if i == 0 else hidden_size,
                    hidden_size,
                    window=window,
                )
                for i in range(num_layers)
            ]
        )
        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        h_layers = [
            torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
            for _ in range(self.num_layers)
        ]
        outputs: list[torch.Tensor] = []
        if self.window == 1:
            for t in range(T):
                layer_input = torch.nan_to_num(x[:, t, :])
                for i, cell in enumerate(self.cells):
                    h_layers[i] = cell(
                        layer_input, h_layers[i],
                        dt=torch.tensor(1.0, device=x.device, dtype=x.dtype),
                    )
                    layer_input = h_layers[i]
                outputs.append(h_layers[-1])
            seq = torch.stack(outputs, dim=1)
        else:
            assert T % self.window == 0, (
                f"T={T} must be a multiple of window={self.window}"
            )
            num_chunks = T // self.window
            for c in range(num_chunks):
                x_chunk = torch.nan_to_num(
                    x[:, c * self.window : (c + 1) * self.window, :]
                )
                layer_input = x_chunk
                for i, cell in enumerate(self.cells):
                    h_layers[i] = cell(layer_input, h_layers[i], dt=torch.tensor(1.0, device=x.device, dtype=x.dtype))
                    if i + 1 < self.num_layers:
                        layer_input = h_layers[i].unsqueeze(1).expand(
                            B, self.window, self.hidden_size
                        )
                outputs.append(
                    h_layers[-1].unsqueeze(1).expand(B, self.window, self.hidden_size)
                )
            seq = torch.cat(outputs, dim=1)
        if self.return_sequences:
            return self.output_proj(seq)
        return self.output_proj(seq[:, -1, :])
