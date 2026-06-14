import torch
import torch.nn as nn

from lnn.core.sequence_utils import select_step_delta, select_step_mask


class CfCCell(nn.Module):
    """
    Closed-form Continuous-time (CfC) cell.

    Implements the closed-form solution to the LTC ODE:
        x(t) = sigma(-f(x,I;theta_f) * t) * g(x,I;theta_g)
             + [1 - sigma(-f(x,I;theta_f) * t)] * h(x,I;theta_h)

    Key advantage: No ODE solver needed, making it much faster
    than LTC while preserving the continuous-time dynamics.

    Multi-time-scale support (PRD #10-29, 2026-06-14):
    Set ``n_tau > 1`` to split the hidden state into K independent
    time-scale groups. Each group carries its own τ_i, f_gate, g_branch,
    h_branch.  When ``n_tau == 1`` (default) the cell is *numerically
    equivalent* to the original single-τ CfCCell within float32
    precision.  This is the minimum-variance extension that aligns
    with the multi-τ pattern observed in arXiv:2606.12240 (MR-MoE),
    arXiv:2606.11162 (COGENT), arXiv:2606.07670 (Liquid-3DGS), and
    arXiv:2604.18274 (LiquidTAD).

    Args:
        input_size: Input feature dimension.
        hidden_size: Hidden state dimension.
        n_tau: Number of independent time-scale groups (≥1).
            ``n_tau == 1`` reproduces the original cell exactly.
        tau_scales: Per-branch initial time constants, length
            ``n_tau``.  If shorter than ``n_tau`` the last value is
            geometrically extended (×10 each step) to fill the list.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_tau: int = 1,
        tau_scales: tuple = (0.1, 1.0, 10.0),
    ):
        super().__init__()
        assert n_tau >= 1, f"n_tau must be >= 1, got {n_tau}"
        if n_tau > 1:
            assert len(tau_scales) >= 1, "tau_scales must be non-empty when n_tau>1"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_tau = int(n_tau)

        if self.n_tau == 1:
            # Original single-τ path (numerically equivalent to pre-PR behaviour).
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
            self.time_scale = nn.Parameter(torch.ones(hidden_size))
            self._multi_tau = False
        else:
            # Multi-τ path: K independent branches with their own τ and projections.
            # Pad / truncate tau_scales to length n_tau.
            scales = list(tau_scales)
            if len(scales) < self.n_tau:
                while len(scales) < self.n_tau:
                    scales.append(scales[-1] * 10.0)
            scales = scales[: self.n_tau]
            self._tau_init = tuple(scales)

            # Even split of hidden dim, with any remainder absorbed by the last branch.
            base = hidden_size // self.n_tau
            rem = hidden_size - base * self.n_tau

            self.f_gates = nn.ModuleList()
            self.g_branches = nn.ModuleList()
            self.h_branches = nn.ModuleList()
            self.time_scales = nn.ParameterList()
            for i in range(self.n_tau):
                out_dim = base + (rem if i == self.n_tau - 1 else 0)
                self.f_gates.append(
                    nn.Sequential(
                        nn.Linear(input_size + hidden_size, out_dim),
                        nn.Sigmoid(),
                    )
                )
                self.g_branches.append(
                    nn.Sequential(
                        nn.Linear(input_size + hidden_size, out_dim),
                        nn.Tanh(),
                    )
                )
                self.h_branches.append(
                    nn.Sequential(
                        nn.Linear(input_size + hidden_size, out_dim),
                        nn.Tanh(),
                    )
                )
                self.time_scales.append(nn.Parameter(torch.full((out_dim,), float(scales[i]))))
            self._multi_tau = True

    def _branch_dims(self) -> list[int]:
        """Hidden dim per branch (only meaningful when ``n_tau>1``)."""
        base = self.hidden_size // self.n_tau
        rem = self.hidden_size - base * self.n_tau
        return [base + (rem if i == self.n_tau - 1 else 0) for i in range(self.n_tau)]

    def forward(self, x_t: torch.Tensor, h: torch.Tensor, dt: float | torch.Tensor = 1.0) -> torch.Tensor:
        combined = torch.cat([x_t, h], dim=-1)
        if not self._multi_tau:
            f = self.f_gate(combined)
            g = self.g_branch(combined)
            h_out = self.h_branch(combined)
            decay = torch.sigmoid(-f * self.time_scale * dt)
            return decay * g + (1.0 - decay) * h_out

        # Multi-τ path: K branches share the same combined input but evolve at different τ.
        branch_outputs = []
        for i in range(self.n_tau):
            f = self.f_gates[i](combined)
            g = self.g_branches[i](combined)
            h_out = self.h_branches[i](combined)
            decay = torch.sigmoid(-f * self.time_scales[i] * dt)
            branch_outputs.append(decay * g + (1.0 - decay) * h_out)
        return torch.cat(branch_outputs, dim=-1)


class CfCNetwork(nn.Module):
    """
    Full CfC network for sequence processing.

    The CfC network replaces the ODE solver with a closed-form
    approximation, achieving orders of magnitude speedup over LTC
    while maintaining comparable performance.

    Args:
        input_size: Dimension of input features
        hidden_size: Dimension of hidden state
        output_size: Dimension of output
        num_layers: Number of stacked CfC layers
        return_sequences: Whether to return full sequence or last step
        n_tau: Number of independent time-scale groups per cell (>=1).
            Forwarded to every ``CfCCell``.  Default 1 = original behaviour.
        tau_scales: Per-branch initial time constants, forwarded to every
            ``CfCCell``.  See ``CfCCell`` docstring.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        return_sequences: bool = True,
        n_tau: int = 1,
        tau_scales: tuple = (0.1, 1.0, 10.0),
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.return_sequences = return_sequences
        self.n_tau = int(n_tau)
        self.tau_scales = tuple(tau_scales)

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            self.cells.append(CfCCell(in_dim, hidden_size, n_tau=self.n_tau, tau_scales=self.tau_scales))

        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
        dt: float | torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Process a batch of sequences.

        Args:
            x: Input tensor with shape [batch, time, features].
            h0: Optional initial hidden state [layers, batch, hidden].
            dt: Optional per-step time deltas. Supports scalar, [T], [B],
                [B, T], or [B, T, 1] shapes.
            mask: Optional observed-feature or sequence mask. Supports [B, T],
                [B, T, features], [T], or [T, features]. Missing input values
                are zeroed and fully masked steps keep the previous hidden state.
        """
        batch_size, seq_len, _ = x.shape
        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device, dtype=x.dtype)

        h = h0
        layer_input = x
        for i, cell in enumerate(self.cells):
            outputs = []
            h_i = h[i]
            for t in range(seq_len):
                dt_t = select_step_delta(dt, t, batch_size, seq_len, x.device, x.dtype)
                input_mask, update_mask = select_step_mask(
                    mask, t, batch_size, seq_len, self.input_size, x.device, x.dtype
                )
                x_t = torch.nan_to_num(layer_input[:, t, :])
                if i == 0 and input_mask is not None:
                    x_t = x_t * input_mask
                h_candidate = cell(x_t, h_i, dt=dt_t)
                h_i = h_candidate if update_mask is None else update_mask * h_candidate + (1.0 - update_mask) * h_i
                outputs.append(h_i)
            layer_input = torch.stack(outputs, dim=1)
            h = torch.cat(
                [h_i.unsqueeze(0) if j == i else h[j].unsqueeze(0) for j in range(self.num_layers)],
                dim=0,
            )

        if self.return_sequences:
            return self.output_proj(layer_input)
        else:
            return self.output_proj(layer_input[:, -1, :])


class PDNAPulseHead(nn.Module):
    """
    Pulse-Driven Neural Architecture (PDNA) augmentation head.

    Implements the post-hoc gated additive residuals from arXiv:2603.00153v1
    (Paras Sharma, 2026) §3.2-3.3, designed to augment a CfC backbone's hidden
    state sequence with learnable oscillatory dynamics + optional recurrent
    self-attention. The paper shows +4.62 pp on sMNIST multi-gap protocol
    (Cohen's d=0.87, 5/5 seeds) for the pulse variant over a CfC baseline.

    Two gated additive residuals:
        pulse:    h + α · A · sin(ω · t + φ(h))              (Eq. 3-4)
        attend:   h + β · Wself · σ(h)                       (Eq. 5-6)
    where:
        - A ∈ R^d  learnable amplitude (per hidden dim)
        - ω ∈ R^d  learnable frequency, log-uniform init [0.1, 10.0]
        - φ(h) = W_φ h + b_φ  state-dependent phase
        - α, β    scalar gates, init 0.01 (let backbone train first)

    Note: as the paper §8 (iii) acknowledges, this is a *post-hoc* augmentation
    of the full hidden-state tensor rather than a true continuous-time dynamic
    evolving between input steps. A sequential ODE-based architecture would be
    needed for the latter.

    Args:
        hidden_size: d, the CfC backbone's hidden dimension
        use_self_attend: whether to also add the recurrent self-attend module
        omega_low/high: bounds for the log-uniform ω init
        alpha_init: initial value for the pulse scalar gate (paper uses 0.01)
        beta_init:  initial value for the attend scalar gate (paper uses 0.01)
    """

    def __init__(
        self,
        hidden_size: int,
        use_self_attend: bool = True,
        omega_low: float = 0.1,
        omega_high: float = 10.0,
        alpha_init: float = 0.01,
        beta_init: float = 0.01,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_self_attend = use_self_attend

        # Per-dim learnable amplitude A ∈ R^d (init to 1.0, paper leaves default)
        self.amplitude = nn.Parameter(torch.ones(hidden_size))
        # Per-dim learnable frequency ω ∈ R^d, log-uniform init
        log_low, log_high = float(torch.log(torch.tensor(omega_low)).item()), \
                            float(torch.log(torch.tensor(omega_high)).item())
        self.omega = nn.Parameter(torch.empty(hidden_size).uniform_(log_low, log_high).exp())
        # State-dependent phase: φ(h) = W_φ h + b_φ
        self.phase_proj = nn.Linear(hidden_size, hidden_size)
        # Scalar pulse gate α (init 0.01)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        if use_self_attend:
            # Recurrent self-attention: Wself · σ(h)
            self.self_attend_proj = nn.Linear(hidden_size, hidden_size)
            # Scalar attend gate β (init 0.01)
            self.beta = nn.Parameter(torch.tensor(float(beta_init)))

    def forward(self, h: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """
        Apply pulse (+ optional self-attend) augmentation to a hidden state seq.

        Args:
            h: [B, T, d] hidden state sequence (typically CfCNetwork return_sequences output)
            t: optional [T] linear timestep index (defaults to torch.arange(T))

        Returns:
            h_aug: [B, T, d] augmented hidden state sequence (same shape as h)
        """
        B, T, d = h.shape
        assert d == self.hidden_size, f"hidden dim mismatch: got {d}, expected {self.hidden_size}"
        if t is None:
            t = torch.arange(T, device=h.device, dtype=h.dtype)
        # Broadcast t to per-batch: [T] -> [1, T] -> [B, T]
        t_b = t.view(1, T).expand(B, T)

        # Pulse: pulse(t, h) = A · sin(ω · t + φ(h))
        # ω is [d], so ω · t is [B, T, d] via broadcasting
        phase = self.phase_proj(h)                       # [B, T, d]
        angular = self.omega.view(1, 1, d) * t_b.unsqueeze(-1)  # [B, T, d]
        pulse = self.amplitude.view(1, 1, d) * torch.sin(angular + phase)
        h = h + self.alpha * pulse

        if self.use_self_attend:
            attended = self.self_attend_proj(torch.sigmoid(h))
            h = h + self.beta * attended
        return h


# ===================================================================
# SVAF τ-modulated peer-blending (arXiv 2604.03955v1 §7.1 Eq. 19-20)
# ===================================================================
# Implemented as a *standalone* utility — does not require SVAF's full
# Cognitive-Memory-Block + fusion-gate machinery, just the per-neuron
# coupling coefficient. This is the minimum unit that lets a multi-agent
# mesh blend CfC cognitive states with explicit per-neuron τ control.
#
# Reference (Eq. 20 in the paper, iter#17 deep-read):
#     β_i = min(α_eff × K × sim_i / τ_i, 1.0)
# where:
#     sim_i = max(1 - |h_local_i - h_mesh_i| / max(|h_local_i|, |h_mesh_i|), 0)
# Neuron role table (Table 14):
#     τ < 5s     → Fast   — readily coupled (mood, reactive signals)
#     5 ≤ τ ≤ 30 → Medium — moderate
#     τ > 30s    → Slow   — resists coupling (domain expertise)


def similarity_per_dim(h_local: torch.Tensor, h_mesh: torch.Tensor) -> torch.Tensor:
    """Per-neuron similarity in [0, 1], Eq. 19 left half.

    Args:
        h_local: [B, d]  agent's own hidden state.
        h_mesh:  [B, d]  peer's hidden state (or mesh aggregate).

    Returns:
        sim: [B, d] per-neuron similarity.
    """
    diff = (h_local - h_mesh).abs()
    denom = torch.maximum(h_local.abs(), h_mesh.abs()).clamp_min(1e-8)
    sim = (1.0 - diff / denom).clamp_min(0.0)
    return sim


def tau_modulated_blend_coef(
    h_local: torch.Tensor,
    h_mesh: torch.Tensor,
    tau: torch.Tensor,
    alpha_eff: float = 0.40,
    K: float = 30.0,
) -> torch.Tensor:
    """Compute per-neuron blending coefficient β_i per Eq. 20.

    Args:
        h_local:   [B, d] agent's own hidden state.
        h_mesh:    [B, d] peer's hidden state.
        tau:       [d]    per-neuron time constants (in seconds, > 0).
        alpha_eff: scalar peer-level blending strength (0.40 aligned, 0.15 guarded, 0 rejected).
        K:         scalar constant that scales the (sim / τ) term before clipping to 1.0.

    Returns:
        beta: [B, d] per-neuron blending coefficient in [0, 1].
    """
    sim = similarity_per_dim(h_local, h_mesh)        # [B, d]
    beta = alpha_eff * K * sim / tau.clamp_min(1e-8)  # [B, d]
    return beta.clamp_max(1.0)


def tau_modulated_blend_update(
    h_local: torch.Tensor,
    h_mesh: torch.Tensor,
    tau: torch.Tensor,
    alpha_eff: float = 0.40,
    K: float = 30.0,
) -> torch.Tensor:
    """Apply one τ-modulated peer-blend step to h_local (Eq. 20 update form).

    Returns the new hidden state: h_new = (1 - β) ⊙ h_local + β ⊙ h_mesh,
    where β is per-neuron.

    Args:
        h_local:   [B, d] agent's own hidden state.
        h_mesh:    [B, d] peer's hidden state.
        tau:       [d]    per-neuron time constants.
        alpha_eff: scalar (see tau_modulated_blend_coef).
        K:         scalar (see tau_modulated_blend_coef).

    Returns:
        h_new: [B, d] blended hidden state.
    """
    beta = tau_modulated_blend_coef(h_local, h_mesh, tau, alpha_eff, K)
    return (1.0 - beta) * h_local + beta * h_mesh


def default_three_group_tau(d: int) -> torch.Tensor:
    """Convenience helper: split d neurons into Fast / Medium / Slow τ groups.

    Fast:    τ = 1s   (readily coupled)
    Medium:  τ = 10s  (moderate)
    Slow:    τ = 60s  (resists coupling)

    Returns a 1D tensor of length d cycling through the three groups
    in 1/3 + 1/3 + 1/3 ratio (caller may override).
    """
    third = max(1, d // 3)
    groups = torch.cat([
        torch.full((third,), 1.0),                # Fast
        torch.full((third,), 10.0),               # Medium
        torch.full((d - 2 * third,), 60.0),       # Slow (remainder)
    ])
    return groups

