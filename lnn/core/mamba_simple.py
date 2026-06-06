"""Minimal selective-scan Mamba-1 cell (round 73).

A from-scratch implementation of the Mamba selective state-space layer
(Gu & Dao 2023, arXiv:2312.00752) that is:

* O(T * D) per forward pass via a Python loop over the time axis.
* Pure-PyTorch (no custom CUDA / mamba-ssm / causal-conv1d dep) so it
  runs on any torch installation the rest of the project supports.
* Sized for the canonical reproducible suite used in round 73
  (sMNIST, permuted-MNIST, seq-CIFAR, Mackey-Glass) — sequence
  length <= 1000, hidden size <= 128, batch <= 64. Larger sizes
  should use the official ``mamba_ssm`` package; this is a *teaching
  and benchmarking* implementation, not a production Mamba.

The math follows the Mamba-1 selective scan:

    Given input x in R^{B x T x D}:

    A_log in R^D        — unconstrained; A = -exp(A_log)  (ensures stability)
    B    in R^{B x T x D} — input-dependent selection
    C    in R^{B x T x D} — input-dependent output projection
    D    in R^D          — input-independent skip connection

    h_0 = 0
    h_t = exp(-A * dt) * h_{t-1} + B_t * x_t
    y_t = C_t * h_t + D * x_t

The forward is implemented with a Python ``for`` loop over T. The
loop is O(T*D) — no quadratic T*D*D — because A is per-channel
(diagonal) and B, C are per-timestep per-channel.

Bidirectional support is provided as ``bidirectional=True``, which runs
the scan in reverse on a flipped sequence and then flips the output
back. The reverse scan shares the same A, B, C, D parameters — only
the time direction differs. This is a common variant used in vision
benchmarks (BiMamba).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SelectiveScanMamba(nn.Module):
    """A minimal selective-scan Mamba-1 cell.

    Args:
        input_size: dimension of the input at each timestep.
        hidden_size: dimension of the latent state (D in the math).
        bidirectional: if True, run a second scan in reverse and
            concatenate / average the two directions. We average
            (rather than concatenate) so the output dim stays
            ``hidden_size``.
        dt_init: initial value for the A-log parameter (default -3.0,
            which gives A = exp(-3) ~ 0.05 → fast decay, stable).

    Forward signature:
        x: (B, T, input_size)  — generic features
        → (B, T, hidden_size)  if return_sequences=True
        → (B, hidden_size)     if return_sequences=False
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        bidirectional: bool = False,
        dt_init: float = -3.0,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        # Input projection produces the input-dependent B, C, plus the
        # residual. We project to 3 * hidden_size: [B, C, residual_x].
        self.in_proj = nn.Linear(input_size, 3 * hidden_size)
        # A is per-channel, parameterised in log-space for stability.
        self.A_log = nn.Parameter(torch.full((hidden_size,), dt_init))
        # Skip connection D — per-channel.
        self.D = nn.Parameter(torch.ones(hidden_size))
        # Output projection.
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def _scan(self, x: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """Run the selective scan on (x, B, C).

        All inputs are (B, T, D). Returns (B, T, D).
        """
        B_batch, T, D = x.shape
        # A is per-channel: shape (D,). Negative exp ensures decay.
        A = -torch.exp(self.A_log)  # (D,)
        h = torch.zeros(B_batch, D, device=x.device, dtype=x.dtype)
        ys: list[torch.Tensor] = []
        for t in range(T):
            # h_t = A * h_{t-1} + B_t * x_t
            h = A * h + B[:, t] * x[:, t]
            # y_t = C_t * h + D * x_t
            y_t = C[:, t] * h + self.D * x[:, t]
            ys.append(y_t)
        return torch.stack(ys, dim=1)  # (B, T, D)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_sequences: bool = True,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected (B, T, input_size), got shape {tuple(x.shape)}")
        if x.size(-1) != self.input_size:
            raise ValueError(
                f"input_size mismatch: model expects {self.input_size}, got {x.size(-1)}"
            )

        # Project to [B_proj, C_proj, residual_x].
        proj = self.in_proj(x)
        B_sel = proj[..., : self.hidden_size]
        C_sel = proj[..., self.hidden_size : 2 * self.hidden_size]
        x_res = proj[..., 2 * self.hidden_size :]

        y = self._scan(x_res, B_sel, C_sel)
        if self.bidirectional:
            # Reverse scan: flip time, scan, flip back, then average.
            x_rev = torch.flip(x_res, dims=[1])
            B_rev = torch.flip(B_sel, dims=[1])
            C_rev = torch.flip(C_sel, dims=[1])
            y_rev = self._scan(x_rev, B_rev, C_rev)
            y = 0.5 * (y + torch.flip(y_rev, dims=[1]))

        y = self.out_proj(y)
        if not return_sequences:
            y = y[:, -1, :]
        return y


class SelectiveScanMambaNetwork(nn.Module):
    """Stacked Mamba-1 network with a final linear classifier.

    Args:
        input_size, hidden_size, output_size, num_layers, return_sequences:
            mirror the CfCNetwork API.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        return_sequences: bool = True,
        *,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.return_sequences = return_sequences

        # First layer takes input_size; subsequent take hidden_size.
        layer_input_sizes = [input_size] + [hidden_size] * (num_layers - 1)
        self.cells = nn.ModuleList(
            SelectiveScanMamba(
                layer_input_sizes[i],
                hidden_size,
                bidirectional=bidirectional,
            )
            for i in range(num_layers)
        )
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for cell in self.cells:
            h = cell(h, return_sequences=True)
        if not self.return_sequences:
            h = h[:, -1, :]
        return self.head(h)
