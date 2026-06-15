"""AntisymmetricRNN for CfC (PRD #10-94, 2026-06-15).

Implements the antisymmetric hidden-to-hidden weight matrix
mechanism from AntisymmetricRNN (Chang, Chen, Haber, Chu, ICLR 2019,
"AntisymmetricRNN: A Dynamical System View on Recurrent Neural
Networks").

The mechanism constrains the recurrent weight matrix M to be
ANTISYMMETRIC (M = -M^T).  All eigenvalues of an antisymmetric
matrix are PURE IMAGINARY (real part = 0), which gives the
recurrent dynamics marginal stability: the hidden state neither
diverges (real part > 0) nor collapses to a fixed point (real
part < 0), but oscillates in a bounded region.

This module ports the core idea to a CfC-friendly setting:

- **AntisymmetricMatrix**: parameterized antisymmetric matrix
  using upper-triangle storage (n*(n-1)/2 parameters,
  reconstructed as M = U - U^T, with the diagonal forced to 0).

- **AntisymmetricCfCCell**: CfC-style update with antisymmetric
  W_h inside the candidate.  The update preserves CfC's
  nonlinearity (W_h @ h_prev) but the W_h matrix is constrained
  to be antisymmetric.

- **AntisymmetricCfCStackedNetwork**: stack of AntisymmetricCfC
  cells.

The mechanism is **structural** (constrains W_h directly) and
**distribution-agnostic** (works on any 1D-ND target).  Per the
91-131 audit, this is the profile of all 12 STRICTLY POSITIVE
winners (preserves the W·h nonlinearity + adds a useful
inductive bias).

Why this should work in 1D:
- The antisymmetric constraint prevents W_h from growing
  eigenvalues with positive real part (no divergence on noisy
  data).
- The rotation tendency in state space can help model regime
  switches in structured_irr (different frequencies of
  oscillation for different regimes).
- It's a STRUCTURAL change to the recurrent step itself, not
  just a routing/gating modification.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AntisymmetricMatrix(nn.Module):
    """Parameterized antisymmetric matrix.

    Stores an upper-triangular matrix ``U`` of shape ``[n, n]``
    and reconstructs ``M = U - U^T`` with the diagonal zeroed
    (so the diagonal of M is forced to 0, which is the standard
    convention for antisymmetric matrices).

    Total parameters: ``n * (n - 1) / 2`` (only the upper
    triangle is stored).

    Args:
        size: dimension of the square matrix.
        init_scale: scale for the upper-triangle init.
    """

    def __init__(self, size: int, init_scale: float = 0.1):
        super().__init__()
        self.size = size
        # Upper-triangle storage. We register a full [n, n] matrix
        # but enforce the lower triangle to be 0 (so M's diagonal is 0).
        self.U = nn.Parameter(torch.zeros(size, size))
        # Initialize upper triangle with small random values.
        with torch.no_grad():
            # In-place triu_ so we modify self.U, not a copy.
            self.U.triu_(diagonal=1).uniform_(-init_scale, init_scale)
            # Diagonal zero (so M's diagonal is 0).
            self.U.diagonal().zero_()

    def forward(self) -> torch.Tensor:
        """Reconstruct the antisymmetric matrix M = U - U^T."""
        # Use triu to keep only upper triangle, then antisymmetrize.
        U_upper = torch.triu(self.U, diagonal=1)
        M = U_upper - U_upper.transpose(0, 1)
        return M

    def extra_repr(self) -> str:
        return f"size={self.size}, n_params={self.size * (self.size - 1) // 2}"


class AntisymmetricCfCCell(nn.Module):
    """AntisymmetricRNN CfC cell (PRD #10-94).

    CfC-style update with the hidden-to-hidden weight matrix
    constrained to be antisymmetric.  The update is::

        candidate = tanh(M @ h + W_x @ x + b)   # M is antisymmetric
        h_new = h + dt * (candidate - h)

    With M antisymmetric, the linearization has eigenvalues
    ``±iω`` (pure imaginary), giving the system a "rotation"
    tendency in state space.

    Args:
        input_size: input feature dimension.
        hidden_size: hidden state dimension.
        dt: step size for the explicit Euler update.
        init_scale: scale for the antisymmetric matrix init.
        init_M: if 'zeros', initialize U to 0 (M=0); if 'small',
            initialize U to small random values.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dt: float = 0.1,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dt = dt

        # Input projection.
        self.W_x = nn.Linear(input_size, hidden_size)
        # Antisymmetric hidden-to-hidden weight matrix.
        self.M = AntisymmetricMatrix(hidden_size, init_scale=init_scale)
        # Bias.
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """One recurrent step.

        Args:
            x_t: input of shape ``[B, input_size]``.
            h: hidden state of shape ``[B, hidden_size]``.
        Returns:
            New hidden state ``[B, hidden_size]``.
        """
        M = self.M()  # [H, H] antisymmetric
        candidate = torch.tanh(F.linear(h, M) + self.W_x(x_t) + self.bias)
        h_new = h + self.dt * (candidate - h)
        return h_new

    def spectral_radius(self) -> float:
        """Largest absolute eigenvalue (≈rotation frequency).

        For an antisymmetric matrix, the spectral radius is
        the maximum absolute imaginary part of the eigenvalues.
        """
        M = self.M().detach()
        eigvals = torch.linalg.eigvals(M)
        return float(eigvals.abs().max().item())

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"dt={self.dt}, spectral_radius={self.spectral_radius():.4f}"
        )


class AntisymmetricCfCStackedNetwork(nn.Module):
    """Stacked AntisymmetricCfC cells (PRD #10-94).

    Each layer is an AntisymmetricCfCCell. The output is the
    head projection of the last layer's hidden state.

    Args:
        input_size: input feature dimension.
        hidden_size: hidden state dimension.
        output_size: output feature dimension.
        num_layers: number of stacked cells.
        dt: step size for the explicit Euler update.
        init_scale: scale for the antisymmetric matrix init.
        return_sequences: if True, return outputs at every
            timestep; else return only the last.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        dt: float = 0.1,
        init_scale: float = 0.1,
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
        for l in range(num_layers):
            in_size = input_size if l == 0 else hidden_size
            self.cells.append(
                AntisymmetricCfCCell(
                    input_size=in_size,
                    hidden_size=hidden_size,
                    dt=dt,
                    init_scale=init_scale,
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
            torch.zeros(B, self.hidden_size, device=x.device)
            for _ in range(self.num_layers)
        ]
        outputs = []
        for t in range(T):
            inp = x[:, t, :]
            for li, cell in enumerate(self.cells):
                if li == 0:
                    h[li] = cell(inp, h[li])
                else:
                    h[li] = cell(h[li - 1], h[li])
            outputs.append(self.head(h[-1]))
        outputs = torch.stack(outputs, dim=1)
        if self.return_sequences:
            return outputs
        return outputs[:, -1, :]
