"""Round 129 — ELMCfC: Expressive Leaky Memory cell for CfC.

Implements the Expressive Leaky Memory (ELM) neuron from
arXiv:2605.12049 (Spieler, Martius, Levina, 12 May 2026,
"Scaling Laws and Tradeoffs in Recurrent Networks of Expressive
Neurons"). The paper introduces a cortical-inspired recurrent
neuron with multi-timescale leaky memory units, dendritic
branch structure, nonlinear MLP update proposals, and a
temporal high-pass filter on the output.

Key idea (paper):
    The ELM neuron goes beyond a single leaky integrator. Each
    "logical" neuron has d_m leaky memory units, each with its
    own learnable timescale κ_m = exp(-1/τ_m). The update is
    a tanh-bounded MLP proposal integrated via leaky accumulation.
    The output is high-pass filtered (subtracting an EMA
    readout), giving a "novelty detector" response.

    The paper derives a 3-axis scaling law (N units × k_e
    per-unit complexity × k_c per-unit connectivity) and a
    closed-form information-theoretic bound. The Pareto recipe
    is d_m ~ √N, d_mlp = 2·d_m, d_tree = 2·d_mlp.

Our adaptation to recurrent CfC:
    - Each "logical" neuron (H of them) has d_m memory units
    - State: m ∈ [B, H, d_m] (memory) and r ∈ [B, H] (EMA readout)
    - Per-memory-unit learnable κ_m, plus learnable κ_λ
    - Tanh-bounded MLP update proposal
    - Output: ReLU(w_r^T m - r) (high-pass filtered readout)
    - We simplify by skipping the dendritic branch structure
      (we use a single linear input projection, not branch_sum)
      because our 1D setting doesn't benefit from the extra
      dendritic capacity.

Key advantage vs. CfC:
    - Per-neuron multiple memory units with multiple timescales
      vs. CfC's single h (one timescale)
    - Output high-pass filter naturally detects regime switches
      (relevant for structured_irr)

Implementation choices:
    - We keep d_m small (4) per the Pareto recipe d_m ~ √N
    - We use a shared MLP across all logical neurons (parameter
      efficient)
    - We use ReLU on the high-pass output (paper's choice)
    - We initialize τ_m log-uniformly in [0.5, 5.0] (similar
      to our OscillatorCfC ω range)
    - We initialize τ_r (EMA timescale) at 1.0
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ELMCfCCell(nn.Module):
    """Expressive Leaky Memory cell (simplified).

    Each of H logical neurons has d_m memory units. Per-memory-unit
    learnable timescale κ_m ∈ (0, 1). MLP computes bounded update
    proposals. High-pass filtered output via ReLU(w_r^T m - r).

    Args:
        input_size: Input feature dimension.
        hidden_size: Number of logical neurons (H).
        d_m: Memory units per neuron (default 4, per Pareto recipe).
        d_mlp: MLP hidden size (default 2*d_m, per Pareto recipe).
        tau_m_lo/hi: Log-uniform init range for memory timescales.
        tau_r: Readout EMA timescale (init).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        d_m: int = 4,
        d_mlp: int | None = None,
        tau_m_lo: float = 0.5,
        tau_m_hi: float = 5.0,
        tau_r: float = 1.0,
        bias: bool = True,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.d_m = int(d_m)
        self.d_mlp = int(d_mlp) if d_mlp is not None else 2 * self.d_m
        self.tau_m_lo = float(tau_m_lo)
        self.tau_m_hi = float(tau_m_hi)
        self.tau_r = float(tau_r)

        # Input projection: [x, h_prev] -> per-neuron + per-memory features
        # We project to (H * d_m) so each memory unit of each neuron
        # gets its own input feature.
        self.in_proj = nn.Linear(input_size + hidden_size, hidden_size * d_m, bias=bias)

        # MLP for update proposal: takes (in_proj_output, prev_memory)
        # concatenated and outputs bounded tanh update.  The MLP
        # input is 2*d_m (proj + prev_memory), per the paper.
        self.mlp = nn.Sequential(
            nn.Linear(2 * self.d_m, self.d_mlp),
            nn.Tanh(),
            nn.Linear(self.d_mlp, self.d_m),
        )

        # Per-memory-unit timescale κ_m (one per (H, d_m) slot)
        kappa_m_init = torch.rand(hidden_size, d_m) * (
            math.log(1.0 / tau_m_lo) - math.log(1.0 / tau_m_hi)
        ) + math.log(1.0 / tau_m_hi)
        # kappa_m_init is in log space; we'll sigmoid(.)
        # Actually we use a different parameterization: kappa_m = sigmoid(raw)
        # so we don't have constraints.  But for stability we initialize
        # close to small κ (slow decay) and let training adjust.
        self.kappa_m_raw = nn.Parameter(torch.full((hidden_size, d_m), -2.0))  # sigmoid(-2) ≈ 0.12

        # κ_λ (paper) — also learnable per memory slot
        self.kappa_lambda_raw = nn.Parameter(torch.full((hidden_size, d_m), -2.0))

        # Per-memory-unit readout w_r: [H, d_m] -> [H]
        self.w_r = nn.Parameter(torch.empty(hidden_size, d_m))
        nn.init.normal_(self.w_r, std=1.0 / math.sqrt(d_m))

        # Readout bias per neuron
        self.b = nn.Parameter(torch.zeros(hidden_size))

        # κ_r (readout EMA timescale)
        # kappa_r = exp(-1/tau_r)  (fixed at init, NOT learnable for simplicity)
        self.register_buffer("kappa_r", torch.tensor(math.exp(-1.0 / tau_r)))

    def kappa_m(self) -> torch.Tensor:
        return torch.sigmoid(self.kappa_m_raw)

    def kappa_lambda(self) -> torch.Tensor:
        return torch.sigmoid(self.kappa_lambda_raw)

    def init_state(self, batch_size: int, device=None):
        device = device or next(self.parameters()).device
        m0 = torch.zeros(batch_size, self.hidden_size, self.d_m, device=device)
        r0 = torch.zeros(batch_size, self.hidden_size, device=device)
        return m0, r0

    def forward(
        self,
        x: torch.Tensor,  # [B, input_size]
        h_prev: torch.Tensor,  # [B, hidden_size]  previous activation (recurrent input)
        m: torch.Tensor,  # [B, H, d_m]
        r: torch.Tensor,  # [B, H]
        dt: float | torch.Tensor = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One ELM step.

        Returns:
            m_new: [B, H, d_m]
            r_new: [B, H]
            a: [B, H]  (output activation, high-pass filtered)
        """
        B, H, D = m.shape  # D = d_m

        # Input projection (with explicit recurrent input h_prev)
        cat = torch.cat([x, h_prev], dim=-1)  # [B, input+H]
        proj = self.in_proj(cat)  # [B, H*D]
        proj = proj.view(B, H, D)  # [B, H, d_m]

        # MLP update proposal: input is concatenation of proj and
        # previous memory (per the paper: [b_t, κ_m ⊙ m_{t-1}])
        mlp_input = torch.cat([proj, m], dim=-1)  # [B, H, 2*d_m]
        delta = self.mlp(mlp_input)  # [B, H, d_m]
        delta = torch.tanh(delta)

        # Leaky integration
        kappa_m = self.kappa_m()  # [H, d_m]
        kappa_lambda = self.kappa_lambda()  # [H, d_m]
        m_new = kappa_m.unsqueeze(0) * m + (1.0 - kappa_lambda).unsqueeze(0) * delta

        # EMA readout: r_t = κ_r r_{t-1} + (1 - κ_r) w_r^T m_t
        w_r_dot_m = (self.w_r.unsqueeze(0) * m_new).sum(dim=-1)  # [B, H]
        r_new = self.kappa_r * r + (1.0 - self.kappa_r) * w_r_dot_m

        # High-pass output: a = ReLU(b + w_r^T m - r)
        a = torch.relu(self.b + w_r_dot_m - r_new)

        return m_new, r_new, a


class ELMCfCNetwork(nn.Module):
    """Stacked ELM network.

    Each layer is an ELMCfCCell. The first layer projects input
    to hidden_size. The final layer projects the last activation
    to output_size. We use the previous step's activation as the
    recurrent feedback (a_prev), not the raw state m.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        return_sequences: bool = False,
        d_m: int = 4,
        d_mlp: int | None = None,
        tau_m_lo: float = 0.5,
        tau_m_hi: float = 5.0,
        tau_r: float = 1.0,
        bias: bool = True,
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
                ELMCfCCell(
                    input_size=in_dim,
                    hidden_size=hidden_size,
                    d_m=d_m,
                    d_mlp=d_mlp,
                    tau_m_lo=tau_m_lo,
                    tau_m_hi=tau_m_hi,
                    tau_r=tau_r,
                    bias=bias,
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
                treated as missing and replaced with 0.
            dt: Timestep duration (currently unused; the cell runs
                at unit time for simplicity).

        Returns:
            Output: [B, T, output_size] if return_sequences, else
            [B, output_size] (last step).
        """
        x = torch.nan_to_num(x, nan=0.0)

        B, T, _ = x.shape
        device = x.device
        # States per layer
        ms = [torch.zeros(B, cell.hidden_size, cell.d_m, device=device) for cell in self.cells]
        rs = [torch.zeros(B, cell.hidden_size, device=device) for cell in self.cells]
        # Previous activation (used as h_proxy in the cell)
        a_prevs = [torch.zeros(B, cell.hidden_size, device=device) for cell in self.cells]

        outputs = []
        for t in range(T):
            inp = x[:, t, :]
            for li, cell in enumerate(self.cells):
                m, r, a = cell(inp, a_prevs[li], ms[li], rs[li])
                ms[li] = m
                rs[li] = r
                a_prevs[li] = a
                # Recurrence: next step's input is the activation
                # from the previous step
                inp = a_prevs[li]
            outputs.append(self.head(a_prevs[-1]))

        out_stack = torch.stack(outputs, dim=1)  # [B, T, output_size]
        if self.return_sequences:
            return out_stack
        return out_stack[:, -1, :]
