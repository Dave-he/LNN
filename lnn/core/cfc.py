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
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

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

    def forward(self, x_t: torch.Tensor, h: torch.Tensor, dt: float | torch.Tensor = 1.0) -> torch.Tensor:
        combined = torch.cat([x_t, h], dim=-1)
        f = self.f_gate(combined)
        g = self.g_branch(combined)
        h_out = self.h_branch(combined)
        decay = torch.sigmoid(-f * self.time_scale * dt)
        h_new = decay * g + (1.0 - decay) * h_out
        return h_new


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
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
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
            self.cells.append(CfCCell(in_dim, hidden_size))

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
