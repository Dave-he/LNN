"""MemoryFusionCfCCell — Cross-paper synthesis of CfC × TFP retention × NSFD gain/loss.

Motivation (loop 2026-08-05, follow-up to LNN_Training_Paradigm_2026_Summer_Cross_Section):
    CfC's default retention uses ``σ(-f·τ·dt)`` which is a continuous-time decay
    but does NOT explicitly separate the *elapsed physical interval* ``dt`` from
    the *learned time constant* ``τ`` in a way that exposes them as two independent
    retention controls. Two recent papers independently introduce explicit-``dt``
    retention that better matches physics-aware deployment (irregular sampling,
    variable control rates):

    1. **TFP** (arXiv 2607.08283) — for VLA policy belief:
       ``k_t = exp(-Δt / τ_t)``                            (Eq. 3)
       ``h_t = k_t ⊙ h_{t-1} + (1 - k_t) ⊙ ĥ_t``           (Eq. 4)
       ⇒ Retention is exponential decay parameterized by the *physical* interval.

    2. **NSFD-NODE** (arXiv 2607.10858) — for positivity-preserving dynamics:
       ``x_i^{n+1} = (x_i^n + Δt · G_i) / (1 + Δt · L_i)`` (Eq. 3)
       ⇒ Closed-form gain/loss update with positivity guarantee.

    This module unifies both into a CfC-style cell by exposing three ``retention_kind``:

    - ``'cfc'`` (default, numerically equivalent to :class:`CfCCell` with ``n_tau=1``)
        ``h_new = σ(-f·τ·dt) · g + (1 - σ(-f·τ·dt)) · h_branch``
    - ``'tfp'`` (TFP-style retention with explicit ``dt``)
        ``k     = exp(-dt / softplus(τ_proj))``
        ``h_new = k · h_prev + (1 - k) · h_branch``
    - ``'nsfd'`` (NSFD gain/loss closed-form)
        ``G     = softplus(G_net([x_t, h_prev]))``
        ``L     = softplus(L_net([x_t, h_prev]))``
        ``h_new = (h_prev + dt · G) / (1 + dt · L)``

    All three share the same input projection head (``g_branch``, ``h_branch``), so
    apples-to-apples comparison reduces to *how the previous hidden state is fused
    with the new candidate*.

Cross-paper relation (resolves two gaps from LNN_Family_Taxonomy_And_Gap_2026-08-03):
    - **N3** (TFP → CfC gate, "本周内可落地"): realized as ``retention_kind='tfp'``.
    - **N2** (L-RFM → khlfft_attn_cfc): realized as the closed-form ``retention_kind='nsfd'``,
      which is the same algebraic family as L-RFM's random-feature closed-form surrogate.

Numerical properties:
    - ``'cfc'``: matches :class:`CfCCell` forward within float32 (both use
      ``σ(-f·τ·dt) · g + (1 - σ) · h_branch`` with single τ).
    - ``'tfp'``: ``k ∈ (0, 1]`` ⇒ convex combination ⇒ bounded, gradient-friendly.
      ``τ_proj`` uses softplus + 1e-3 floor so ``k`` is well-defined for ``dt ≥ 0``.
    - ``'nsfd'``: ``L ≥ 0`` ⇒ denominator ``1 + dt·L > 0`` ⇒ no division-by-zero.
      ``h_prev + dt·G ≥ 0`` ⇒ numerator non-negative when ``h_prev ≥ 0`` (positivity
      preserved only if input hidden state is non-negative — caller responsibility).
"""

from __future__ import annotations

import torch
import torch.nn as nn


_VALID_RETENTION = ("cfc", "tfp", "nsfd")


class MemoryFusionCfCCell(nn.Module):
    """CfC-style cell with pluggable retention mechanism.

    Args:
        input_size: Input feature dimension.
        hidden_size: Hidden state dimension.
        retention_kind: One of ``"cfc"``, ``"tfp"``, ``"nsfd"`` (see module docstring).
        n_tau: Number of independent time-scale branches (≥ 1). ``n_tau == 1``
            recovers the single-τ path. Each branch gets ``hidden_size // n_tau``
            hidden dims (last branch absorbs remainder).
        tau_min: Lower bound on τ when using ``retention_kind="cfc"`` (mirrors the
            :class:`CfCCell` convention; not enforced strictly — used only for
            documentation / future clamping).
        tau_max: Upper bound on τ (same comment as ``tau_min``).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        retention_kind: str = "cfc",
        n_tau: int = 1,
        tau_min: float = 0.1,
        tau_max: float = 5.0,
    ):
        super().__init__()
        if retention_kind not in _VALID_RETENTION:
            raise ValueError(
                f"retention_kind must be one of {_VALID_RETENTION}, got {retention_kind!r}"
            )
        if n_tau < 1:
            raise ValueError(f"n_tau must be >= 1, got {n_tau}")

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.retention_kind = retention_kind
        self.n_tau = int(n_tau)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)

        # Branch dims (CfCCell convention: last branch absorbs remainder).
        base = self.hidden_size // self.n_tau
        rem = self.hidden_size - base * self.n_tau
        self._branch_dims = [
            base + (rem if i == self.n_tau - 1 else 0) for i in range(self.n_tau)
        ]

        # Shared input projection head (used by all retention modes).
        in_dim = self.input_size + self.hidden_size
        self.g_branch = nn.ModuleList(
            [nn.Sequential(nn.Linear(in_dim, d), nn.Tanh()) for d in self._branch_dims]
        )
        self.h_branch = nn.ModuleList(
            [nn.Sequential(nn.Linear(in_dim, d), nn.Tanh()) for d in self._branch_dims]
        )

        # Per-mode additional projections.
        if retention_kind == "cfc":
            self.f_gate = nn.ModuleList(
                [nn.Sequential(nn.Linear(in_dim, d), nn.Sigmoid()) for d in self._branch_dims]
            )
            self.time_scale = nn.ParameterList(
                [nn.Parameter(torch.full((d,), 1.0)) for d in self._branch_dims]
            )
            self.tau_proj = self.g_net = self.l_net = None
        elif retention_kind == "tfp":
            self.tau_proj = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(in_dim, d), nn.Softplus())
                    for d in self._branch_dims
                ]
            )
            self.f_gate = self.time_scale = self.g_net = self.l_net = None
        else:  # "nsfd"
            self.g_net = nn.ModuleList(
                [nn.Sequential(nn.Linear(in_dim, d), nn.Softplus()) for d in self._branch_dims]
            )
            self.l_net = nn.ModuleList(
                [nn.Sequential(nn.Linear(in_dim, d), nn.Softplus()) for d in self._branch_dims]
            )
            self.f_gate = self.tau_proj = self.time_scale = None

        self._init_parameters()

    # ------------------------------------------------------------------ utils

    def _init_parameters(self):
        # Mild zero-bias init for the candidate branches so that at init the cell
        # behaves like a small-signal version of CfC (avoid saturating softplus /
        # sigmoid on the first few steps).
        for mlp in (*self.g_branch, *self.h_branch):
            lin = mlp[0]
            nn.init.xavier_uniform_(lin.weight, gain=0.5)
            nn.init.zeros_(lin.bias)
        if self.retention_kind == "cfc" and self.f_gate is not None:
            for mlp in self.f_gate:
                lin = mlp[0]
                nn.init.xavier_uniform_(lin.weight, gain=0.5)
                nn.init.zeros_(lin.bias)
        if self.retention_kind == "tfp" and self.tau_proj is not None:
            for mlp in self.tau_proj:
                lin = mlp[0]
                nn.init.xavier_uniform_(lin.weight, gain=0.5)
                nn.init.constant_(lin.bias, 1.0)  # softplus(1) ≈ 1.31 ≈ "default τ"
        if self.retention_kind == "nsfd":
            for mlp in (*self.g_net, *self.l_net):
                lin = mlp[0]
                nn.init.xavier_uniform_(lin.weight, gain=0.5)
                nn.init.zeros_(lin.bias)

    def _branch_slice(self, h: torch.Tensor, i: int) -> torch.Tensor:
        """Slice ``h`` into branch ``i``'s hidden slice."""
        start = sum(self._branch_dims[:i])
        end = start + self._branch_dims[i]
        return h[..., start:end]

    # --------------------------------------------------------------- forward

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        combined = torch.cat([x_t, h], dim=-1)
        branch_outs = []
        for i in range(self.n_tau):
            g = self.g_branch[i](combined)
            cand = self.h_branch[i](combined)
            h_i = self._branch_slice(h, i)

            if self.retention_kind == "cfc":
                f = self.f_gate[i](combined)  # type: ignore[index]
                decay = torch.sigmoid(-f * self.time_scale[i] * dt)  # type: ignore[index]
                branch_outs.append(decay * g + (1.0 - decay) * cand)
            elif self.retention_kind == "tfp":
                tau = self.tau_proj[i](combined) + 1e-3  # type: ignore[index]
                k = torch.exp(-dt / tau)
                branch_outs.append(k * h_i + (1.0 - k) * cand)
            else:  # "nsfd"
                G = self.g_net[i](combined)  # type: ignore[index]
                L = self.l_net[i](combined)  # type: ignore[index]
                branch_outs.append((h_i + dt * G) / (1.0 + dt * L))
        return torch.cat(branch_outs, dim=-1)


class MemoryFusionCfCNetwork(nn.Module):
    """Sequential wrapper that stacks :class:`MemoryFusionCfCCell` over a sequence.

    Mirrors :class:`CfCNetwork`'s minimal API so the new cell is drop-in
    comparable in :mod:`scripts.jetson_lnn_benchmark` style sweeps.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        retention_kind: str = "cfc",
        n_tau: int = 1,
    ):
        super().__init__()
        self.cell = MemoryFusionCfCCell(
            input_size=input_size,
            hidden_size=hidden_size,
            retention_kind=retention_kind,
            n_tau=n_tau,
        )
        self.head = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x: torch.Tensor,  # (batch, seq_len, input_size)
        dt: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self.cell.hidden_size, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(seq_len):
            h = self.cell(x[:, t, :], h, dt=dt)
            outputs.append(self.head(h))
        return torch.stack(outputs, dim=1)  # (batch, seq_len, output_size)
