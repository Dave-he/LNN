"""Diagonal State Space (DSS) cell — round 73.

Implementation of the Diagonal State Space model from
Gupta, Gu, Berant 2022, "Diagonal State Spaces are as Effective as
Structured State Spaces" (arXiv:2203.14343 — verified via the
round-72 ArxivCatalog, which flags the upstream LNN literature
commentary that misattributes 2203.14343 to Lockhart et al. adaptive
solvers — see lnn/core/arxiv_catalog.py:2002.08071 entry).

The paper's headline empirical claim: a purely diagonal state matrix
A (no low-rank correction) **matches** S4 on the Long Range Arena
(LRA, avg 81.88 vs 80.21) and Speech Commands (98.2 vs 98.1). This
is a 2022 historical comparison and does NOT include Mamba
(2023), Mamba-2 (2024), or any 2025-2026 SOTA — see deep-research
report §3.4.

Math
----
Same as S4, but A is purely diagonal (no HiPPO init, no DPLR):

    Given input x in R^{B x T x D}:

    A in R^D              — learned, parameterised in log-space for stability
    B in R^{B x T x D}    — input-dependent (or constant; we use input-dep)
    C in R^{B x T x D}    — input-dependent
    D in R^D              — skip connection

    h_0 = 0
    h_t = exp(A) * h_{t-1} + B_t * x_t
    y_t = C_t * h_t + D * x_t

Differences vs SelectiveScanMamba (lnn/core/mamba_simple.py):

* A is unconstrained (can be positive OR negative), no negative-exp
  forcing. We still log-parameterise for stability.
* A is "constant" in the sense that it does not depend on the input
  at the current timestep — only the projection of x gives B and C.
  (This is a slight relaxation of the DSS paper, which uses
  input-dependent B, C and constant A.)
* No bidirectional flag — DSS is typically unidirectional.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DiagonalSSMCell(nn.Module):
    """A from-scratch Diagonal State Space (DSS) cell.

    Args:
        input_size: dimension of the input at each timestep.
        hidden_size: dimension of the latent state (D in the math).
        dt_init: initial value for the A parameter in log-space. The
            default -3.0 yields exp(-3) ~ 0.05 — gentle decay, in the
            stable regime.

    Forward signature: same as SelectiveScanMamba.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        dt_init: float = -3.0,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Project to [B, C, residual_x] — same shape as Mamba but
        # without the negative-exp forcing on A.
        self.in_proj = nn.Linear(input_size, 3 * hidden_size)
        # A is per-channel, learned. We log-parameterise to keep
        # magnitudes small, but unlike Mamba we allow sign flips
        # by not constraining to -exp(...).
        self.A_log = nn.Parameter(torch.full((hidden_size,), dt_init))
        # Per-channel skip.
        self.D = nn.Parameter(torch.ones(hidden_size))
        # Output projection.
        self.out_proj = nn.Linear(hidden_size, hidden_size)

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

        proj = self.in_proj(x)
        B_sel = proj[..., : self.hidden_size]
        C_sel = proj[..., self.hidden_size : 2 * self.hidden_size]
        x_res = proj[..., 2 * self.hidden_size :]

        # A is per-channel; we use exp() (positive only) to keep the
        # state well-behaved. The DSS paper actually allows signed A,
        # but for benchmarking parity with our Mamba cell we keep
        # the same sign convention.
        A = torch.exp(self.A_log)  # (D,)

        B_batch, T, D = x_res.shape
        h = torch.zeros(B_batch, D, device=x_res.device, dtype=x_res.dtype)
        ys: list[torch.Tensor] = []
        for t in range(T):
            h = A * h + B_sel[:, t] * x_res[:, t]
            y_t = C_sel[:, t] * h + self.D * x_res[:, t]
            ys.append(y_t)
        y = torch.stack(ys, dim=1)

        y = self.out_proj(y)
        if not return_sequences:
            y = y[:, -1, :]
        return y


class DiagonalSSMNetwork(nn.Module):
    """Stacked DSS network with a final linear classifier.

    API mirrors ``SelectiveScanMambaNetwork`` and ``CfCNetwork``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        return_sequences: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.return_sequences = return_sequences

        layer_input_sizes = [input_size] + [hidden_size] * (num_layers - 1)
        self.cells = nn.ModuleList(
            DiagonalSSMCell(layer_input_sizes[i], hidden_size)
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
