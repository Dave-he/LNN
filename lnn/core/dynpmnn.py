"""DynPMNN — Dynamical Physics-Modeled Neural Networks.

Implements the FitzHugh-Nagumo (FHN) ODE-based recurrent cell from
arXiv:2605.08176v1 (Felipe-Sosa et al., 2026), §2.2-2.3.

The FHN model is a simplified spiking-neuron ODE:

    dV/dt = V - V^3/3 - W + I
    dW/dt = epsilon * (V + a - b * W)

where (V, W) are the membrane potential and recovery variable per hidden
dimension, I is the input current, and (a, b, epsilon) are learnable
parameters. DynPMNN's contribution: each hidden *layer* is the
integration of this ODE from t=0 to t=T (multiple Euler steps inside
PyTorch's autograd graph), giving an end-to-end-differentiable
physics-informed backbone.

Reference iter#16 deep read:
docs/reports/Physics-Modeled_Neural_Networks_DynPMNN_研读报告.md

This is the *stage A* mini-task of PRD §10 #1.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FHNCell(nn.Module):
    """Single FitzHugh-Nagumo ODE layer, integrated via explicit Euler.

    The state per hidden dimension is (V, W) ∈ R^d. The forward pass
    receives an input tensor and runs ``n_euler_steps`` Euler steps of
    the FHN ODE, using the input as the external current I at every step
    (paper §2.3 — "few forward Euler steps embedded in autograd").

    All learnable parameters are per-dimension:
        a ∈ R^d    (recovery offset, init 0.7)
        b ∈ R^d    (recovery decay, init 0.8)
        epsilon ∈ R^d  (recovery speed, init 0.08)
        W_in ∈ R^(d_in × d)  (input projection)

    The integration uses a fixed step size dt = 1.0 / n_euler_steps so
    the total integrated time T is approximately 1.0 (paper's T is not
    fixed; this is a smoke-scale default).

    Args:
        input_size: dimension of the external current I.
        hidden_size: dimension d of the FHN state (V, W).
        n_euler_steps: number of Euler integration steps per forward
                       call. Paper default 4-8; we default 5 for stability.
        dt: integration step size (defaults to 1.0 / n_euler_steps).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_euler_steps: int = 5,
        dt: float | None = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_euler_steps = n_euler_steps
        self.dt = dt if dt is not None else 1.0 / max(1, n_euler_steps)

        # Learnable input projection: I_t = W_in @ x_t (per dim).
        self.input_proj = nn.Linear(input_size, hidden_size)
        # Learnable FHN parameters per hidden dim.
        # Init values follow typical FHN excitable regime.
        self.a = nn.Parameter(torch.full((hidden_size,), 0.7))
        self.b = nn.Parameter(torch.full((hidden_size,), 0.8))
        self.epsilon = nn.Parameter(torch.full((hidden_size,), 0.08))

    def initial_state(self, batch_size: int, device: torch.device, dtype=torch.float32):
        """Zero-init (V, W) state."""
        V = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        W = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        return V, W

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one forward Euler integration over n_euler_steps.

        Args:
            x: [B, input_size]  external current at this timestep.
            state: optional (V, W) tuple; if None, inits to zero.

        Returns:
            (V_T, W_T, V_seq):
                V_T: [B, hidden_size]  final membrane potential.
                W_T: [B, hidden_size]  final recovery variable.
                V_seq: [B, n_euler_steps + 1, hidden_size]  full V trajectory
                       (the "+1" includes the initial V_0 = state).
        """
        B = x.shape[0]
        device = x.device
        if state is None:
            V, W = self.initial_state(B, device, dtype=x.dtype)
        else:
            V, W = state
        I = self.input_proj(x)  # [B, hidden_size]

        V_seq = [V]
        for _ in range(self.n_euler_steps):
            dV = V - (V ** 3) / 3.0 - W + I
            dW = self.epsilon * (V + self.a - self.b * W)
            V = V + self.dt * dV
            W = W + self.dt * dW
            V_seq.append(V)
        V_seq_t = torch.stack(V_seq, dim=1)  # [B, n+1, hidden]
        return V, W, V_seq_t


class DynPMNNNetwork(nn.Module):
    """Stacked DynPMNN — each layer is a FHNCell integrated over a window.

    Wraps a sequence of FHNCells (one per "ODE layer"). Per the paper,
    each hidden layer is itself an ODE trajectory, so even a 1-layer
    DynPMNNNetwork is non-trivially deeper than a plain RNN cell.

    Args:
        input_size: dimension of input features.
        hidden_size: dimension of FHN state per layer.
        output_size: dimension of final projection.
        num_layers: number of stacked FHN layers (each with its own
                    learnable FHN parameters and input projection).
        n_euler_steps: Euler steps per FHNCell.
        return_sequences: if True, return V sequence from the last layer;
                          else return just the final V.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        n_euler_steps: int = 5,
        return_sequences: bool = True,
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
            self.cells.append(FHNCell(in_dim, hidden_size, n_euler_steps=n_euler_steps))

        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process a batch of sequences.

        Args:
            x: [B, T, input_size]

        Returns:
            y: [B, T, output_size] if return_sequences else [B, output_size].
        """
        B, T, _ = x.shape
        layer_input = x
        for i, cell in enumerate(self.cells):
            V, W = cell.initial_state(B, x.device, dtype=x.dtype)
            outputs = []
            for t in range(T):
                V, W, _ = cell(layer_input[:, t, :], state=(V, W))
                outputs.append(V)
            layer_input = torch.stack(outputs, dim=1)  # [B, T, hidden]

        if self.return_sequences:
            return self.output_proj(layer_input)
        return self.output_proj(layer_input[:, -1, :])
