"""Riemannian LTC — Liquid Time-Constant networks on curved manifolds.

Implements the tangent-space LTC + Riemannian exp/log map pattern from
arXiv:2601.14115v1 (Lu et al. WWW '26), §3.2-3.3.

The classical LTC ODE is formulated in Euclidean space. RLSTG shows that
the same idea extends naturally to a Riemannian manifold M by:

1. **Local ODE on the tangent space** T_x M (which is a Euclidean
   vector space, so the classical LTC recurrence works unchanged):

       dh/dt = -α ⊙ h + tanh(W_h h + u)             (论文 Eq. 10)

2. **Push forward to the manifold** via the Riemannian exp map:

       x_{t+Δt} = exp_x(Δt · dh/dt)                (论文 Eq. 12)

This module implements a minimal version using a Hyperboloid manifold
(geoopt), the most common choice for hierarchical / tree-like graph
data. The Hyperboloid M = {x ∈ R^{d+1} : ⟨x, x⟩_L = -1, x_0 > 0}
is the canonical Lorentz-model hyperbolic space used by hyperbolic GNNs.

This is the *stage B* of PRD §10 RLSTG复现路线 (iter#36 design).  The
theoretical contribution (stability / universal approximation推广 to
Riemannian domain) is **out of scope** — we ship a usable module, not
a proof.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import geoopt.manifolds
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "RiemannianLTC requires `geoopt`. Install with `pip install geoopt`."
    ) from e


# Default manifold for all classes below. Paper uses Hyperboloid for
# tree-like social/citation graphs.  In geoopt this is the **Lorentz**
# model of the hyperboloid {x ∈ R^{d+1} : ⟨x, x⟩_L = -1, x_0 > 0}.
_DEFAULT_MANIFOLD = "hyperboloid"


def _get_manifold(name: str = _DEFAULT_MANIFOLD):
    if name == "hyperboloid":
        return geoopt.manifolds.Lorentz()
    raise ValueError(f"Unknown manifold: {name!r} (only 'hyperboloid' is supported)")


class TangentSpaceLTC(nn.Module):
    """Classical LTC ODE evaluated in a tangent space (locally Euclidean).

    The state h and the input u live in the same d-dim space (the
    tangent space T_x M is d-dim regardless of ambient dimension d+1
    on the Hyperboloid). The recurrence is

        h_{t+1} = h_t + dt · (-α ⊙ h_t + tanh(W_h h_t + W_u u + b))

    This mirrors the closed-form / sigmoid-gated behavior of
    ``lnn/core/ltc.py::LTCCell`` and exposes the per-dim time constant
    directly so the Riemannian wrapper can read α.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Per-dim decay coefficient α > 0 (论文's `time_scale`).
        self.alpha = nn.Parameter(torch.full((dim,), 1.0))
        # Recurrent + input projection.
        self.W_h = nn.Linear(dim, dim, bias=False)
        self.W_u = nn.Linear(dim, dim, bias=False)
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, h: torch.Tensor, u: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
        """One Euler step: h + dt * (-α ⊙ h + tanh(W_h h + W_u u + b))."""
        return h + dt * (-self.alpha * h + torch.tanh(self.W_h(h) + self.W_u(u) + self.b))


class RiemannianLTC(nn.Module):
    """Tangent-space LTC wrapped with Riemannian exp / log maps.

    The state x lives on the manifold (ambient dim = hidden_size + 1 for
    the Hyperboloid). The input u is in R^{input_size} and is first
    linearly projected to the ambient dimension, then pushed into the
    tangent space T_x M via ``logmap(x, ...)``.  The output is pushed
    back to the manifold via ``expmap(x, ...)``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        manifold_name: str = _DEFAULT_MANIFOLD,
        dt: float = 0.001,
        max_tangent_norm: float = 1.0,
    ):
        super().__init__()
        self.manifold = _get_manifold(manifold_name)
        # Hyperboloid expects features in R^{d+1}.
        self.ambient_size = hidden_size + 1
        self.dt = dt
        self.max_tangent_norm = max_tangent_norm
        self.tangent_ltc = TangentSpaceLTC(self.ambient_size)
        # Input projection from `input_size` to ambient space (R^{d+1}).
        self.input_proj = nn.Linear(input_size, self.ambient_size)
        # Initialize input projection with small std so initial forward doesn't
        # blow up the tangent norm before training kicks in.
        nn.init.normal_(self.input_proj.weight, std=0.1)
        nn.init.zeros_(self.input_proj.bias)

    def init_state(self, batch_size: int, device, dtype=torch.float32) -> torch.Tensor:
        """Return the origin of the manifold as the initial state.

        For Hyperboloid, the origin is (1, 0, 0, ..., 0) (curvature -1).
        """
        x0 = torch.zeros(batch_size, self.ambient_size, device=device, dtype=dtype)
        x0[..., 0] = 1.0  # x_0 = 1 on the Hyperboloid
        return x0

    def forward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        dt: float | None = None,
    ) -> torch.Tensor:
        """One Riemannian step (origin-only expmap for autograd safety).

        Args:
            x: [B, d+1] state on the manifold (treated as ambient for expmap0).
            u: [B, input_size] ambient input.
            dt: integration step size (overrides self.dt if given).

        Returns:
            x_new: [B, d+1] new state on the manifold.

        Note: We use ``expmap0`` / ``logmap0`` (origin-based, closed form
        with autograd support) rather than the full ``expmap`` /
        ``logmap`` at arbitrary ``x``. The full version requires parallel
        transport (which geoopt 0.5.1 does not autograd through). This
        is sufficient for a stage-B smoke implementation; full tangent-
        space logistics is a stage-D concern.
        """
        step = dt if dt is not None else self.dt
        # Project ambient input to ambient space.
        u_amb = self.input_proj(u)                              # [B, d+1]
        # Project to the tangent space at the origin.  On the Hyperboloid
        # model the tangent space at origin is the subspace {v : v[0] = 0};
        # logmap0 on a vector outside this subspace returns NaN (because
        # the closed form requires ⟨v, v⟩_L < -1, which fails when v[0] ≠ 0
        # and ||v[1:]|| is small). We therefore **zero the time-0 coord**
        # before logmap0 — the cleanest fix that stays within geoopt 0.5.1's
        # autograd capabilities.
        u_amb[..., 0] = 0.0
        # Push to tangent space at origin (closed form, autograd-safe).
        u_tan = self.manifold.logmap0(u_amb)                   # [B, d+1]
        # Clamp tangent norm to avoid cosh/sinh overflow downstream.
        u_tan = self._clip_tangent_norm(u_tan, self.max_tangent_norm)
        # Tangent-space LTC step.
        v = self.tangent_ltc(u_tan, u_tan, step)               # [B, d+1]
        v = self._clip_tangent_norm(v, self.max_tangent_norm)
        # Push forward to the manifold at origin.
        return self.manifold.expmap0(v)

    @staticmethod
    def _clip_tangent_norm(v: torch.Tensor, max_norm: float) -> torch.Tensor:
        """Clamp the per-row Euclidean norm of v to ``max_norm``.

        Tangent vectors at the origin satisfy v[0] = 0; their "size" is
        given by the norm of v[1:].  We use the full norm of v here for
        safety (v[0] stays ~0 anyway).
        """
        norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        factor = (max_norm / norm).clamp_max(1.0)
        return v * factor


class RiemannianLTCNetwork(nn.Module):
    """Stacked RiemannianLTC layers + linear readout.

    Mirrors ``CfCNetwork`` / ``LTCNetwork`` / ``DynPMNNNetwork`` API so
    it can drop into existing ablation runners.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        manifold_name: str = _DEFAULT_MANIFOLD,
        return_sequences: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.return_sequences = return_sequences
        # Each layer: input is hidden_size (first: input_size).
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            # RiemannianLTC signature: (input_size, hidden_size, manifold_name)
            self.layers.append(RiemannianLTC(input_size=in_dim, hidden_size=hidden_size,
                                            manifold_name=manifold_name))
        # Readout is a small Euclidean MLP, applied to the (ambient) state.
        self.readout = nn.Linear(hidden_size + 1, output_size)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Process a batch of sequences.

        Args:
            x_seq: [B, T, input_size] ambient input sequence.

        Returns:
            y: [B, T, output_size] if return_sequences else [B, output_size].
        """
        B, T, _ = x_seq.shape
        # Per-layer state on the manifold.
        states = [self.layers[0].init_state(B, x_seq.device, dtype=x_seq.dtype)]
        outputs = []
        for t in range(T):
            x_t = x_seq[:, t, :]
            h_t = states[-1]
            h_new = self.layers[0](h_t, x_t)
            for layer in self.layers[1:]:
                # Subsequent layers: input is the previous layer's output
                # (a tangent vector). We map it back to a manifold point
                # via the next layer's expmap origin (cheap approximation).
                h_new = layer(h_new, h_new)  # self-feeding on the manifold
            states.append(h_new)
            outputs.append(h_new)
        # Stack outputs: [B, T, d+1]
        out_seq = torch.stack(outputs, dim=1)
        if self.return_sequences:
            return self.readout(out_seq)
        return self.readout(out_seq[:, -1, :])
