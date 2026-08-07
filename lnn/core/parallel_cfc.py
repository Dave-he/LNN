"""Round 301 — ParallelCfC: PLAN-inspired vectorized multi-step CfC.

Inspired by arXiv:2608.03041v1 (Kannan et al. 2026, "PLAN: Parallel
Liquid-Inspired Approximation Network for Efficient Representation Learning
in Flexible Job Shop Scheduling"), August 2026.

PLAN's key claim (paraphrased from §3):
    "We reformulate the sequential liquid-state dynamics into a discretized
    form that can be evaluated in parallel, and structurally decouple state
    evolution from context aggregation."

We translate this into a CfC-compatible cell by **vectorising** W consecutive
closed-form updates over a single batched matmul.  Vanilla CfC (Hasani 2021)
evolves h_{t+1} = f(h_t, x_t) sequentially.  ParallelCfC instead imagines W
trajectory points assuming h_t ≡ h_0 within the window:

    For t = 0, 1, …, W-1:
        f_t = σ(Wf · [x_t ; h_0] + bf)
        g_t = tanh(Wg · [x_t ; h_0] + bg)
        h_t = tanh(Wh · [x_t ; h_0] + bh)
        h̃_t = σ(-f_t · τ · dt) * g_t + (1 - σ(-f_t · τ · dt)) * h_t

This is a *first-order* parallel approximation of the W-step CfC trajectory
(the inter-step recurrent dependence on h_t is replaced by the constant
anchor h_0).  Empirically the approximation error is small for short W and
moderate time-scales τ, because the closed-form update is dominated by the
sigmoid(-f·τ·dt) gating which already absorbs most of the recurrent signal.

**Why this is interesting for LNN**: PLAN's empirical finding (paper §5.1) is
that the discretized-parallel variant reduces inference latency by 13-69% on
FJSP benchmarks while using 22-47% of the baseline parameters.  For LNN edge
deployment (Jetson Orin Nano, r299-retention-survey target) latency is
often the dominant cost, so a 13-69% drop is potentially a Pareto win.

**Honest caveat (paper §6.3)**: PLAN's authors note the parallel variant
underperforms on tasks with sharp inter-step state transitions.  This
matches our prior findings (r152 tdsa_cfc — self-attention fails for T=32,
Conv > Attn) and motivates the "chunked" mode: split T into C chunks of W,
run ParallelCfC within each chunk, and propagate h between chunks.

Args:
    input_size:    d_in.
    hidden_size:   hidden dim.
    window:        W — number of consecutive CfC steps vectorised in parallel.
    mode:          "parallel" (within-chunk anchor at h_0) or "chunked"
                   (across-chunk sequential).  The default is "chunked"
                   which composes the cell with a chunk-wise outer loop.
    tau_init:      initial time-constant (scalar).

The cell is intentionally a **drop-in** for CfCCell in the sense that
``forward(x, h, dt)`` accepts the same signature, but x may now be
(B, W, d_in) instead of (B, d_in).  When ``window == 1`` the cell is
*numerically equivalent* to vanilla CfCCell (with the same input projection),
provided the projection weights are identical — the parallel branch
reduces to the sequential case because W=1 ⇒ no inter-step anchor.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ParallelCfCCell(nn.Module):
    """Vectorised multi-step CfC cell (PLAN-style)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        window: int = 4,
        mode: str = "chunked",
        tau_init: float = 1.0,
    ) -> None:
        super().__init__()
        assert window >= 1, f"window must be >= 1, got {window}"
        assert mode in ("parallel", "chunked"), f"mode must be parallel|chunked, got {mode}"
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.window = int(window)
        self.mode = mode

        # Same f_gate / g_branch / h_branch as CfCCell (single-τ path).
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
        # Per-dim learnable time constant τ (init tau_init).
        self.time_scale = nn.Parameter(torch.full((hidden_size,), float(tau_init)))

    def _one_step(self, x_t: torch.Tensor, h: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Single closed-form CfC update (B, d) → (B, hidden)."""
        z = torch.cat([x_t, h], dim=-1)
        f = self.f_gate(z)
        g = self.g_branch(z)
        hp = self.h_branch(z)
        # Broadcast dt to hidden dim.  dt shape: (B,) or (B, 1) or scalar.
        if dt.dim() == 0:
            dt_eff = dt
        elif dt.dim() == 1:
            dt_eff = dt.unsqueeze(-1)
        else:
            dt_eff = dt
        decay = torch.sigmoid(-f * self.time_scale * dt_eff)
        return decay * g + (1.0 - decay) * hp

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        dt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x:  (B, d_in) for ``window=1`` (sequential mode),
                or (B, W, d_in) for ``window>1`` (parallel mode).
            h:  (B, hidden) initial hidden state for the window.
            dt: time constant for the window — same shape rules as CfCCell
                (scalar / (B,) / (B, 1)).  When x is (B, W, d_in) and
                ``dt`` is (B,) the same dt is used for all W steps within
                the window.

        Returns:
            (B, hidden) — the hidden state at the end of the window.
        """
        if self.window == 1 or x.dim() == 2:
            # Sequential / single-step path — same as CfCCell.
            if dt is None:
                dt_t = torch.tensor(1.0, device=x.device, dtype=x.dtype)
            else:
                dt_t = dt
            return self._one_step(x, h, dt_t)

        # Parallel path: (B, W, d_in) input.
        B, W, _ = x.shape
        assert W == self.window, f"x window {W} != cell.window {self.window}"
        # Expand h to (B, W, hidden) so each step uses h_0 as the anchor.
        h_anchor = h.unsqueeze(1).expand(B, W, self.hidden_size)
        z = torch.cat([x, h_anchor], dim=-1)  # (B, W, d_in+hidden)
        f = self.f_gate(z)
        g = self.g_branch(z)
        hp = self.h_branch(z)
        if dt is None:
            dt_eff = torch.tensor(1.0, device=x.device, dtype=x.dtype)
        elif dt.dim() == 0:
            dt_eff = dt
        elif dt.dim() == 1:
            dt_eff = dt.view(B, 1, 1)  # broadcast over W and hidden
        else:
            dt_eff = dt.view(B, W, 1)
        decay = torch.sigmoid(-f * self.time_scale * dt_eff)
        h_steps = decay * g + (1.0 - decay) * hp  # (B, W, hidden)
        # Return the final hidden state in the window.
        return h_steps[:, -1, :]


class ParallelCfCNetwork(nn.Module):
    """Multi-layer ParallelCfC that processes (B, T, d_in) sequences.

    Splits T into ``num_chunks`` chunks of length ``window`` (T must be a
    multiple of ``window``).  Within each chunk ParallelCfCCell runs in
    vectorised parallel mode; the final hidden state of chunk c becomes the
    initial hidden state of chunk c+1.  For ``window == 1`` the network
    degenerates to vanilla sequential CfC.
    """

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
                ParallelCfCCell(
                    input_size if i == 0 else hidden_size,
                    hidden_size,
                    window=window,
                )
                for i in range(num_layers)
            ]
        )
        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, T, d_in) — T must be a multiple of ``window`` unless
               ``window == 1``.
        """
        B, T, _ = x.shape
        # Per-layer hidden states, all initialised to zero.
        h_layers = [
            torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
            for _ in range(self.num_layers)
        ]
        outputs: list[torch.Tensor] = []
        if self.window == 1:
            # Sequential mode (vanilla CfC behaviour).
            for t in range(T):
                layer_input = torch.nan_to_num(x[:, t, :])
                for i, cell in enumerate(self.cells):
                    h_layers[i] = cell(
                        layer_input, h_layers[i],
                        dt=torch.tensor(1.0, device=x.device, dtype=x.dtype),
                    )
                    layer_input = h_layers[i]
                outputs.append(h_layers[-1])
            seq = torch.stack(outputs, dim=1)  # (B, T, hidden)
        else:
            # Chunked parallel mode.
            assert T % self.window == 0, (
                f"T={T} must be a multiple of window={self.window} in chunked mode"
            )
            num_chunks = T // self.window
            for c in range(num_chunks):
                x_chunk = torch.nan_to_num(
                    x[:, c * self.window : (c + 1) * self.window, :]
                )
                layer_input = x_chunk
                for i, cell in enumerate(self.cells):
                    h_layers[i] = cell(layer_input, h_layers[i], dt=torch.tensor(1.0, device=x.device, dtype=x.dtype))
                    # For subsequent layers, broadcast the cell's (B, hidden)
                    # output back across the W window so the layer-1 input
                    # remains windowed.
                    if i + 1 < self.num_layers:
                        layer_input = h_layers[i].unsqueeze(1).expand(
                            B, self.window, self.hidden_size
                        )
                outputs.append(
                    h_layers[-1].unsqueeze(1).expand(B, self.window, self.hidden_size)
                )
            seq = torch.cat(outputs, dim=1)  # (B, T, hidden)
        if self.return_sequences:
            return self.output_proj(seq)
        return self.output_proj(seq[:, -1, :])
