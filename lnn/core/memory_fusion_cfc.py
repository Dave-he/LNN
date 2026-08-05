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
    - ``'hybrid'`` (NEW round 283: constructive response to TFP-vs-CfC
      irregular-dt negative result). Convex combination of CfC and TFP
      retentions via a learned per-branch mix weight ``α ∈ [0, 1]``::

          ``k_cfc = σ(-f · τ_cfc · dt)                       ``  (sigmoid path)
          ``k_tfp = exp(-dt / softplus(τ_tfp))              ``  (TFP exponential path)
          ``k     = α · k_cfc + (1 - α) · k_tfp             ``  (learned mix)
          ``h_new = k · h_prev + (1 - k) · h_branch         ``

      The sigmoid path provides **dt-robustness via saturation** (cf.
      benchmark analysis 2026-08-05 — CfC is fully insensitive to
      dt ∈ [0.12, 4.74]); the exponential path provides **explicit
      dt semantics** (TFP paper arXiv 2607.08283). ``α`` is per-branch
      learnable (init 0.5), so the cell can interpolate between the
      two regimes during training.
    - ``'hybrid_gate'`` (NEW round 284: N9 finding was that ``hybrid``'s
      static ``α`` does not provide true *conditional* gating — it's a
      learned scalar that does not depend on the input or dt. This
      variant makes ``α`` an *input-dependent* function ``α(x_t, dt)``
      via a per-branch 2-layer MLP, so different inputs / dt regimes
      can route to different retention paths::

          ``α(x_t, dt) = σ(W_2 · σ(W_1 · [x_t, dt_e] + b_1) + b_2)``
          ``k = α · k_cfc + (1 - α) · k_tfp``

      ``dt_e`` is the per-sample elapsed time broadcast to ``input_size + 1``.
      This is the first retention mode that enables true *conditional*
      gating, as opposed to the static mix in ``'hybrid'``. The cost is
      an extra MLP per branch with ``hidden_dim → hidden_dim`` shape.

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
from torch.nn import ParameterList


_VALID_RETENTION = ("cfc", "tfp", "nsfd", "hybrid", "hybrid_gate")


class MemoryFusionCfCCell(nn.Module):
    """CfC-style cell with pluggable retention mechanism.

    Args:
        input_size: Input feature dimension.
        hidden_size: Hidden state dimension.
        retention_kind: One of ``"cfc"``, ``"tfp"``, ``"nsfd"``, ``"hybrid"`` (see module docstring).
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
        alpha_mlp_depth: int = 1,    # hybrid_gate only: extra hidden layers in alpha MLP (1 = single Linear+Sigmoid)
        alpha_mlp_width: int = 0,    # hybrid_gate only: extra width multiplier (0 = same as branch_dim)
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
        self.alpha_mlp_depth = int(alpha_mlp_depth)
        self.alpha_mlp_width = int(alpha_mlp_width)

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
            self.tau_proj = self.g_net = self.l_net = self.alpha = None
        elif retention_kind == "tfp":
            self.tau_proj = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(in_dim, d), nn.Softplus())
                    for d in self._branch_dims
                ]
            )
            self.f_gate = self.time_scale = self.g_net = self.l_net = self.alpha = None
        elif retention_kind == "nsfd":
            self.g_net = nn.ModuleList(
                [nn.Sequential(nn.Linear(in_dim, d), nn.Softplus()) for d in self._branch_dims]
            )
            self.l_net = nn.ModuleList(
                [nn.Sequential(nn.Linear(in_dim, d), nn.Softplus()) for d in self._branch_dims]
            )
            self.f_gate = self.tau_proj = self.time_scale = self.alpha = None
        elif retention_kind == "hybrid":  # CfC path + TFP path + learned mix alpha per branch
            # CfC components
            self.f_gate = nn.ModuleList(
                [nn.Sequential(nn.Linear(in_dim, d), nn.Sigmoid()) for d in self._branch_dims]
            )
            self.time_scale = nn.ParameterList(
                [nn.Parameter(torch.full((d,), 1.0)) for d in self._branch_dims]
            )
            # TFP components
            self.tau_proj = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(in_dim, d), nn.Softplus())
                    for d in self._branch_dims
                ]
            )
            # Learned per-branch mix weight alpha in [0, 1] via sigmoid parameter
            # 0.5 at init ⇒ equal contribution of both paths.
            self.alpha = ParameterList(  # type: ignore[name-defined]
                [nn.Parameter(torch.zeros(d)) for d in self._branch_dims]
            )
            self.g_net = self.l_net = None
        elif retention_kind == "hybrid_gate":  # input-dependent alpha
            # Same CfC + TFP components as 'hybrid'
            self.f_gate = nn.ModuleList(
                [nn.Sequential(nn.Linear(in_dim, d), nn.Sigmoid()) for d in self._branch_dims]
            )
            self.time_scale = nn.ParameterList(
                [nn.Parameter(torch.full((d,), 1.0)) for d in self._branch_dims]
            )
            self.tau_proj = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(in_dim, d), nn.Softplus())
                    for d in self._branch_dims
                ]
            )
            # Input-dependent α: per-branch MLP taking [x_t, dt_e] → α
            # depth=1: Linear(gate_in_dim, d) -> Sigmoid (single linear projection, original N11)
            # depth=2: Linear -> Sigmoid -> Linear -> Sigmoid (round 284 default)
            # depth=N: N Linear+Sigmoid layers
            # width=k*d: hidden layers have k*d units (0=branch_dim, default)
            gate_in_dim = self.input_size + 1  # x_t + dt
            self.gate_mlps = nn.ModuleList()
            for d in self._branch_dims:
                width = self.alpha_mlp_width if self.alpha_mlp_width > 0 else d
                layers = []
                if self.alpha_mlp_depth == 1:
                    # Single-layer: direct projection
                    layers += [nn.Linear(gate_in_dim, d), nn.Sigmoid()]
                elif self.alpha_mlp_depth == 2:
                    # Two-layer: gate_in -> width -> d
                    layers += [nn.Linear(gate_in_dim, width), nn.Sigmoid(),
                               nn.Linear(width, d), nn.Sigmoid()]
                else:
                    # depth >= 3: gate_in -> width -> width -> ... -> d
                    layers += [nn.Linear(gate_in_dim, width), nn.Sigmoid()]
                    for _ in range(self.alpha_mlp_depth - 2):
                        layers += [nn.Linear(width, width), nn.Sigmoid()]
                    layers += [nn.Linear(width, d), nn.Sigmoid()]
                self.gate_mlps.append(nn.Sequential(*layers))
            self.alpha = None  # distinguish from static hybrid's alpha
            self.g_net = self.l_net = None

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
        if self.retention_kind == "hybrid":
            # bias=0 ⇒ softplus(0)=ln(2) ≈ 0.69 — well-defined default τ
            for mlp in self.tau_proj:  # type: ignore[union-attr]
                lin = mlp[0]
                nn.init.xavier_uniform_(lin.weight, gain=0.5)
                nn.init.zeros_(lin.bias)
            # alpha=0 ⇒ sigmoid(0)=0.5 — equal mix at init
            for p in self.alpha:  # type: ignore[union-attr]
                nn.init.zeros_(p)
        if self.retention_kind == "nsfd":
            for mlp in (*self.g_net, *self.l_net):
                lin = mlp[0]
                nn.init.xavier_uniform_(lin.weight, gain=0.5)
                nn.init.zeros_(lin.bias)
        if self.retention_kind == "hybrid_gate":
            for mlp in self.tau_proj:  # type: ignore[union-attr]
                lin = mlp[0]
                nn.init.xavier_uniform_(lin.weight, gain=0.5)
                nn.init.zeros_(lin.bias)
            # Init gate MLPs to output ~0.5 (sigmoid saturation ensures
            # the second Linear starts at sigmoid(<small>) ≈ 0.5).
            for mlp in self.gate_mlps:  # type: ignore[union-attr]
                for lin in mlp:
                    if hasattr(lin, 'weight'):
                        nn.init.xavier_uniform_(lin.weight, gain=0.1)
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
            elif self.retention_kind == "hybrid":
                # CfC path: sigmoid saturation (dt-robustness from prior benchmark)
                f = self.f_gate[i](combined)  # type: ignore[index]
                k_cfc = torch.sigmoid(-f * self.time_scale[i] * dt)  # type: ignore[index]
                # TFP path: explicit-dt exponential retention
                tau = self.tau_proj[i](combined) + 1e-3  # type: ignore[index]
                k_tfp = torch.exp(-dt / tau)
                # Learned per-element mix in [0, 1]
                a = torch.sigmoid(self.alpha[i])  # type: ignore[index]
                k = a * k_cfc + (1.0 - a) * k_tfp
                branch_outs.append(k * h_i + (1.0 - k) * cand)
            elif self.retention_kind == "nsfd":
                G = self.g_net[i](combined)  # type: ignore[index]
                L = self.l_net[i](combined)  # type: ignore[index]
                branch_outs.append((h_i + dt * G) / (1.0 + dt * L))
            elif self.retention_kind == "hybrid":
                # CfC path: sigmoid saturation (dt-robustness from prior benchmark)
                f = self.f_gate[i](combined)  # type: ignore[index]
                k_cfc = torch.sigmoid(-f * self.time_scale[i] * dt)  # type: ignore[index]
                # TFP path: explicit-dt exponential retention
                tau = self.tau_proj[i](combined) + 1e-3  # type: ignore[index]
                k_tfp = torch.exp(-dt / tau)
                # Learned per-element mix in [0, 1]
                a = torch.sigmoid(self.alpha[i])  # type: ignore[index]
                k = a * k_cfc + (1.0 - a) * k_tfp
                branch_outs.append(k * h_i + (1.0 - k) * cand)
            elif self.retention_kind == "hybrid_gate":
                # CfC + TFP paths (same as hybrid)
                f = self.f_gate[i](combined)  # type: ignore[index]
                k_cfc = torch.sigmoid(-f * self.time_scale[i] * dt)  # type: ignore[index]
                tau = self.tau_proj[i](combined) + 1e-3  # type: ignore[index]
                k_tfp = torch.exp(-dt / tau)
                # Input-dependent α via gate MLP: input = [x_t, dt_e]
                # Normalise dt to a [B, 1] tensor (handle float, 1D, 2D inputs).
                if isinstance(dt, (int, float)):
                    dt_e = torch.full((x_t.shape[0], 1), float(dt), device=x_t.device, dtype=x_t.dtype)
                elif dt.dim() == 1:
                    dt_e = dt.unsqueeze(-1)  # [B, 1]
                elif dt.dim() == 2:
                    dt_e = dt if dt.shape[-1] == 1 else dt[..., -1:]
                else:
                    dt_e = dt.view(dt.shape[0], -1)[:, -1:]
                gate_in = torch.cat([x_t, dt_e], dim=-1)  # [B, input_size + 1]
                a = self.gate_mlps[i](gate_in)  # [B, branch_dim]
                k = a * k_cfc + (1.0 - a) * k_tfp
                branch_outs.append(k * h_i + (1.0 - k) * cand)
            elif self.retention_kind == "nsfd":
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
