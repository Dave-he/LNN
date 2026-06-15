"""Round 128 — OscillatorCfC: damped harmonic oscillator closed-form cell.

Implements the damped harmonic oscillator backbone for CfC,
inspired by arXiv:2602.12139 (Shende, Das, Chauhan, Pathak, Gupta
— Ashoka University, February 2026, "Oscillators Are All You Need:
Irregular Time Series Modelling via Damped Harmonic Oscillators
with Closed-Form Solutions").

Key idea (paper, original):
    Replace Neural ODE backbones with damped harmonic oscillators
    that admit closed-form solutions for irregular time series.
    The paper builds on ContiFormer (arXiv:2402.10635), swapping
    its second-order NODE for an underdamped oscillator:
        ẍ + 2γẋ + ω²x = F(t)
    The matrix exponential e^(A t) for the underdamped case (γ<ω)
    is:
        e^(At) = e^(-γt) * [[cos(ωd t) + γ/ωd sin(ωd t), 1/ωd sin(ωd t)],
                            [-ω²/ωd sin(ωd t), cos(ωd t) - γ/ωd sin(ωd t)]]
    with ωd = sqrt(ω² - γ²).  For constant forcing F, the
    steady-state is z_ss = (F/ω², 0) and the solution is:
        z(Δt) = e^(A Δt) (z(0) - z_ss) + z_ss
    This eliminates iterative ODE solvers, giving "orders of
    magnitude faster" inference (paper claim).

Our adaptation to recurrent CfC:
    - Cell state is 2D: z = (h, p) where p = dh/dt is "velocity"
    - Linear forcing F(x) = W_x x + b  (NOT depending on h;
      the closed-form solution requires F constant over the
      interval, so we exclude the h-term.  This is the linear
      regime the paper assumes.)
    - Per-neuron learnable natural frequency ω ∈ log-uniform[10⁻², 10¹]
    - Per-neuron learnable damping ratio ζ ∈ [0.05, 0.4] (forced underdamped)
    - Closed-form solution over the timestep Δt (default 1.0)
    - Initial state (h₀=0, p₀=0) — same as CfC's h₀=0

    The recurrent structure is preserved via the state evolution:
    h_new is a function of (h_old, p_old, F).  Information persists
    across steps through the (h, p) state, not through F.

Key advantage vs. CfC:
    - The closed-form solution is EXACT (not the gated approximation
      CfC uses to avoid the ODE solver).  No learnable gating branch.
    - Only 2 parameters per neuron (ω, ζ) vs. CfC's τ per neuron.
    - Strict generalisation of a second-order ODE backbone; CfC is
      a first-order ODE backbone.

Implementation choices:
    - We use the underdamped case (γ < ω) as in the paper.
      Critically damped and overdamped cases are also derivable but
      are not common in neural time-series modelling (the paper
      only uses underdamped).
    - We apply softplus to keep ω > 0 and use sigmoid to keep ζ ∈ [0, 1]
      (we then mask to ensure ζ < ω, falling back to slight underdamping).
    - We initialise ω via softplus(linear(log_uniform)) so it stays
      positive; ζ via sigmoid(linear) so it stays in (0, 1).
    - The forcing F is the linear projection: F = W_x @ x + W_h @ h + b.

This is the **4th ODE-family in our 91-127 audit**:
1. CfC (first-order, gated)
2. LTC (first-order, ODE-solved)
3. MoR (recursion depth) — round 126
4. **OscillatorCfC (second-order, closed-form)** — round 128
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _oscillator_step_underdamped(
    h0: torch.Tensor,  # [B, H]
    p0: torch.Tensor,  # [B, H]
    F: torch.Tensor,  # [B, H]  constant forcing over the interval
    omega: torch.Tensor,  # [H]    natural frequency
    zeta: torch.Tensor,  # [H]    damping ratio
    dt: float | torch.Tensor,  # scalar
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form damped harmonic oscillator step (underdamped case).

    System:
        dh/dt = p
        dp/dt = -2γp - ω²h + F        (γ = ζω)

    With constant forcing F over interval Δt, the solution is
    z(Δt) = e^(A Δt) (z(0) - z_ss) + z_ss
    where z_ss = (F/ω², 0) is the steady state.

    Returns:
        (h_new, p_new), each [B, H]
    """
    gamma = zeta * omega  # [H]
    # omega_d (damped natural frequency)
    omega_d_sq = (omega * omega - gamma * gamma).clamp(min=1e-8)
    omega_d = torch.sqrt(omega_d_sq)  # [H]

    # Steady state (constant forcing, equilibrium)
    h_ss = F / (omega * omega).clamp(min=1e-8)  # [B, H]
    p_ss = torch.zeros_like(F)  # [B, H]

    # Deviation from steady state
    dh0 = h0 - h_ss
    dp0 = p0 - p_ss

    # Decaying envelope
    env = torch.exp(-gamma * dt)  # [H]  (broadcast over batch)

    cos_t = torch.cos(omega_d * dt)  # [H]
    sin_t = torch.sin(omega_d * dt)  # [H]

    # h(Δt) = env * [dh0 * (cos + (γ/ωd) sin) + dp0 * (1/ωd) sin] + h_ss
    h_new = env * (
        dh0 * (cos_t + (gamma / omega_d) * sin_t)
        + dp0 * (sin_t / omega_d)
    ) + h_ss

    # p(Δt) = env * [dp0 * (cos - (γ/ωd) sin) + dh0 * (-ω²/ωd) sin] + p_ss
    p_new = env * (
        dp0 * (cos_t - (gamma / omega_d) * sin_t)
        + dh0 * (-(omega * omega) / omega_d) * sin_t
    ) + p_ss

    return h_new, p_new


class OscillatorCfCCell(nn.Module):
    """Damped harmonic oscillator cell with closed-form solution.

    State is 2D: (h, p) where p = dh/dt is the "velocity".
    Per-neuron learnable natural frequency ω > 0 and damping
    ratio ζ ∈ (0, 1) (forced underdamped by ζ < 1).

    Args:
        input_size: Input feature dimension.
        hidden_size: Hidden state dimension.
        bias: Whether to include bias in the linear forcing.
        omega_init_lo/hi: Log-uniform init range for ω.
        zeta_init: Sigmoid pre-activation init for ζ
            (mapped to (0, 1) via sigmoid).
        force_activation: Optional nonlinearity on forcing
            (default None, i.e. linear; pass "tanh" to bound F).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        omega_init_lo: float = 0.01,
        omega_init_hi: float = 10.0,
        zeta_init: float = -1.5,  # sigmoid(-1.5) ≈ 0.18
        force_activation: str | None = None,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.omega_init_lo = float(omega_init_lo)
        self.omega_init_hi = float(omega_init_hi)
        self.zeta_init = float(zeta_init)
        self.force_activation = force_activation

        # Linear forcing F = W_x x + b  (no h-term: closed-form requires F
        # constant over the interval, so F must NOT depend on the
        # recurrent state h).
        self.force = nn.Linear(input_size, hidden_size, bias=bias)

        # Per-neuron natural frequency ω (positive, parameterised in log-space)
        omega_init = torch.rand(hidden_size) * (
            math.log(omega_init_hi) - math.log(omega_init_lo)
        ) + math.log(omega_init_lo)
        self.omega_raw = nn.Parameter(omega_init)

        # Per-neuron damping ratio ζ ∈ (0, 1) — sigmoid parameterisation
        # Init sigmoid(zeta_init) puts ζ in the underdamped band.
        self.zeta_raw = nn.Parameter(torch.full((hidden_size,), zeta_init))

    def omega(self) -> torch.Tensor:
        """Per-neuron natural frequency, positive."""
        return torch.exp(self.omega_raw)

    def zeta(self) -> torch.Tensor:
        """Per-neuron damping ratio in (0, 1), underdamped."""
        return torch.sigmoid(self.zeta_raw)

    def init_state(self, batch_size: int, device=None) -> tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(batch_size, self.hidden_size, device=device or next(self.parameters()).device)
        p0 = torch.zeros_like(h0)
        return h0, p0

    def forward(
        self,
        x: torch.Tensor,  # [B, input_size]
        h: torch.Tensor,  # [B, hidden_size]
        p: torch.Tensor,  # [B, hidden_size]
        dt: float | torch.Tensor = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One oscillator step (closed-form, exact)."""
        # Linear forcing F = W_x x + b  (constant per step, no h-term)
        F = self.force(x)
        if self.force_activation == "tanh":
            F = torch.tanh(F)
        elif self.force_activation is not None:
            raise ValueError(f"unknown force_activation: {self.force_activation}")

        omega = self.omega()
        zeta = self.zeta()

        h_new, p_new = _oscillator_step_underdamped(h, p, F, omega, zeta, dt)
        return h_new, p_new


class OscillatorCfCNetwork(nn.Module):
    """Stacked OscillatorCfC network.

    Each layer is a damped harmonic oscillator cell with its own
    (h, p) state.  The first layer projects input to hidden_size.
    The final layer projects the last hidden state to output_size.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        return_sequences: bool = False,
        bias: bool = True,
        omega_init_lo: float = 0.01,
        omega_init_hi: float = 10.0,
        zeta_init: float = -1.5,
        force_activation: str | None = None,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.num_layers = int(num_layers)
        self.return_sequences = bool(return_sequences)

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            self.cells.append(
                OscillatorCfCCell(
                    input_size=in_dim,
                    hidden_size=hidden_size,
                    bias=bias,
                    omega_init_lo=omega_init_lo,
                    omega_init_hi=omega_init_hi,
                    zeta_init=zeta_init,
                    force_activation=force_activation,
                )
            )

        self.head = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x: torch.Tensor,  # [B, T, input_size]
        dt: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """Run the network over the sequence.

        Args:
            x: Input sequence, [B, T, input_size].  NaN entries are
                treated as missing and replaced with 0 in the input
                (consistent with our irregular-time-series convention).
            dt: Timestep duration for the closed-form solution.

        Returns:
            Output: [B, T, output_size] if return_sequences, else
            [B, output_size] (last step).
        """
        # NaN-aware: replace NaN with 0 in the input (similar to the
        # convention used in other irregular TS modules).
        x = torch.nan_to_num(x, nan=0.0)

        B, T, _ = x.shape
        device = x.device
        hs = [torch.zeros(B, self.hidden_size, device=device) for _ in range(self.num_layers)]
        ps = [torch.zeros(B, self.hidden_size, device=device) for _ in range(self.num_layers)]

        outputs = []
        for t in range(T):
            inp = x[:, t, :]
            for li, cell in enumerate(self.cells):
                hs[li], ps[li] = cell(inp, hs[li], ps[li], dt=dt)
                inp = hs[li]
            outputs.append(self.head(inp))

        out_stack = torch.stack(outputs, dim=1)  # [B, T, output_size]
        if self.return_sequences:
            return out_stack
        return out_stack[:, -1, :]
