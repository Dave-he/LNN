"""Round 110 — MoFE-Time Frequency-Domain Experts (PRD #10-72).

Implements MoFE-Time (arXiv:2507.06502 Liu et al. Jul 2025) —
*Mixture of Frequency Domain Experts for Time-Series Forecasting Models*.

The core idea: each expert is a **learnable Fourier reconstructor** with
its own harmonic frequencies and learnable amplitudes. The Fourier
transform is **implicit and learnable** (not pre-computed via FFT) — the
network must discover which frequencies matter.

Per expert k:
  - Has h learnable harmonic frequencies {ω_i} clamped to [0, 2π]
  - Projects input X_t to frequency space via Linear → (B, T, h)
  - Reconstructs in time domain:
        x_n = Σ α_i(t) · cos(ω_i · n) + β_i(t) · sin(ω_i · n)
    where α_i(t), β_i(t) are time-varying amplitudes (from the projection)

The audit pattern (rounds 91-109) "structural > routing-only" predicts
this is **strictly positive on periodic data** and **neutral on random**:
a NEW axis (frequency domain) not yet explored in our 91-109 audit.

Key components:
- ``FrequencyExpertConfig`` — dataclass with K, h_freq, freq_init
- ``FrequencyExpert`` — single learnable Fourier reconstructor
- ``FrequencyExpertPool`` — K frequency experts + top-K routing
- ``TimeFreqMoECfCCell`` — combines time-domain + frequency-domain branches
- ``TimeFreqMoECfCNetwork`` — full network with rolling loop
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Single Frequency Expert
# ----------------------------------------------------------------------------


@dataclass
class FrequencyExpertConfig:
    """Configuration for a single frequency expert.

    Args:
        input_size: Input feature dimension (D).
        hidden_size: Output hidden dimension (H).
        n_freqs: h — number of learnable harmonic frequencies per expert.
        max_omega: Clamp learned frequencies to [0, max_omega] (default 2π).
        use_complex_basis: If True, use cos+sin basis (more expressive);
            if False, use cos only.
    """
    input_size: int = 2
    hidden_size: int = 16
    n_freqs: int = 4
    max_omega: float = 2.0 * math.pi
    use_complex_basis: bool = True


class FrequencyExpert(nn.Module):
    """A learnable Fourier reconstructor.

    Architecture:
        X_t (B, T, D) → Linear → X_f (B, T, h)
        For each freq ω_i:
            basis_cos_i = cos(ω_i · t)              # (T,)
            basis_sin_i = sin(ω_i · sin_basis)      # (T,) (if use_complex)
        Output (B, T, H) = sum over i of X_f[:, :, i] * basis_i projected to H
    """

    def __init__(self, config: FrequencyExpertConfig):
        super().__init__()
        self.config = config

        # Project input to frequency space (h = n_freqs)
        self.to_freq = nn.Linear(config.input_size, config.n_freqs)

        # Project frequency basis activations to output dim
        # If complex basis: output = 2 * n_freqs (cos and sin parts)
        # If real: output = n_freqs (cos only)
        basis_dim = 2 * config.n_freqs if config.use_complex_basis else config.n_freqs
        self.to_hidden = nn.Linear(basis_dim, config.hidden_size)

        # Learnable harmonic frequencies ω_i, clamped to [0, max_omega]
        # Init to small range to avoid initialization issues
        omega_init = torch.linspace(0.1, 1.0, config.n_freqs)
        self.omega_raw = nn.Parameter(omega_init)

        # Pre-compute the time indices 0, 1, 2, ..., T-1 (registered for device)
        # We use a fixed T_max and slice as needed
        self.register_buffer("t_max", torch.tensor(64.0))  # max T

    def _get_omega(self) -> torch.Tensor:
        """Get clamped frequencies in [0, max_omega]."""
        return torch.sigmoid(self.omega_raw) * self.config.max_omega

    def _compute_basis(self, T: int, device: torch.device) -> torch.Tensor:
        """Compute (T, basis_dim) basis functions.

        For complex basis: [cos(ω_0 t), sin(ω_0 t), cos(ω_1 t), sin(ω_1 t), ...]
        For real basis: [cos(ω_0 t), cos(ω_1 t), ...]
        """
        t = torch.arange(T, device=device, dtype=torch.float32)  # (T,)
        omega = self._get_omega()  # (h,)

        if self.config.use_complex_basis:
            # Outer product: (T, h)
            cos_basis = torch.cos(t.unsqueeze(-1) * omega.unsqueeze(0))  # (T, h)
            sin_basis = torch.sin(t.unsqueeze(-1) * omega.unsqueeze(0))  # (T, h)
            basis = torch.stack([cos_basis, sin_basis], dim=-1)  # (T, h, 2)
            basis = basis.reshape(T, -1)  # (T, h*2)
        else:
            basis = torch.cos(t.unsqueeze(-1) * omega.unsqueeze(0))  # (T, h)

        return basis  # (T, basis_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, D) input (may contain NaN).
        Returns:
            (B, T, H) output.
        """
        # NaN-safe input
        x_clean = torch.nan_to_num(x, nan=0.0)
        B, T, D = x_clean.shape

        # Project to frequency space: (B, T, h)
        x_f = self.to_freq(x_clean)

        # Compute basis: (T, basis_dim)
        basis = self._compute_basis(T, x_clean.device)

        # Weighted combination: x_f * basis (element-wise then sum)
        # x_f: (B, T, h), basis: (T, h) or (T, h*2)
        if self.config.use_complex_basis:
            # Pair x_f with cos, sin parts
            # x_f: (B, T, h), split into cos/sin contributions
            x_f_cos = x_f  # use full magnitude for cos
            x_f_sin = x_f  # use full magnitude for sin
            # (B, T, h, 2) → (B, T, h*2)
            x_f_pair = torch.stack([x_f_cos, x_f_sin], dim=-1).reshape(B, T, -1)
        else:
            x_f_pair = x_f  # (B, T, h)

        # Element-wise multiply with basis: (B, T, basis_dim)
        # broadcast basis (T, basis_dim) over batch
        weighted = x_f_pair * basis.unsqueeze(0)  # (B, T, basis_dim)

        # Project to hidden dim
        out = self.to_hidden(weighted)  # (B, T, H)
        return out


# ----------------------------------------------------------------------------
# Frequency Expert Pool (with routing)
# ----------------------------------------------------------------------------


@dataclass
class FrequencyMoEConfig:
    """Top-level config for Frequency MoE."""
    input_size: int = 2
    hidden_size: int = 16
    output_size: int = 1
    n_experts: int = 4
    top_k: int = 2
    n_freqs: int = 4
    max_omega: float = 2.0 * math.pi
    use_complex_basis: bool = True
    use_time_branch: bool = True  # whether to also have a time-domain branch
    aux_loss_coef: float = 0.01  # load balancing loss coefficient


class FrequencyRouter(nn.Module):
    """Top-K router for frequency experts.

    Args:
        input_size: Input feature dim.
        n_experts: K.
        top_k: Top-K for routing.
    """

    def __init__(self, input_size: int, n_experts: int, top_k: int = 2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(input_size, n_experts)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute routing weights.

        Args:
            x: (B, T, D) input.
        Returns:
            full_weights: (B*T, K) full softmax weights
            top_idx: (B*T, top_k) top-K expert indices
            top_w: (B*T, top_k) top-K normalized weights
            aux_loss: scalar auxiliary load-balancing loss
        """
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)  # (B*T, D)
        logit = self.router(x_flat)
        full_weights = F.softmax(logit, dim=-1)  # (B*T, K)

        # Top-K
        top_v, top_idx = full_weights.topk(self.top_k, dim=-1)
        top_w = F.softmax(top_v, dim=-1)

        # Auxiliary load-balancing loss (Switch Transformer style)
        # L_aux = E * sum(f_i * P_i)
        # where f_i = fraction of tokens dispatched to expert i
        # and P_i = average routing probability for expert i
        with torch.no_grad():
            top1 = full_weights.argmax(dim=-1)
        f = torch.zeros(self.n_experts, device=x.device)
        for i in range(self.n_experts):
            f[i] = (top1 == i).float().mean()
        P = full_weights.mean(dim=0)  # (K,)
        aux_loss = self.n_experts * (f * P).sum()

        return full_weights, top_idx, top_w, aux_loss


class TimeFreqMoECfCCell(nn.Module):
    """CfC-style cell with frequency-domain MoE experts.

    Combines:
      - Time-domain branch: optional linear projection
      - Frequency-domain branch: K frequency experts with top-K routing
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = 1,
        config: Optional[FrequencyMoEConfig] = None,
    ):
        super().__init__()
        if config is None:
            config = FrequencyMoEConfig()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.config = config

        # Frequency experts
        self.experts = nn.ModuleList([
            FrequencyExpert(FrequencyExpertConfig(
                input_size=input_size,
                hidden_size=hidden_size,
                n_freqs=config.n_freqs,
                max_omega=config.max_omega,
                use_complex_basis=config.use_complex_basis,
            ))
            for _ in range(config.n_experts)
        ])

        # Router
        self.router = FrequencyRouter(input_size, config.n_experts, config.top_k)

        # Optional time-domain branch
        if config.use_time_branch:
            self.time_branch = nn.Linear(input_size, hidden_size)
        else:
            self.time_branch = None

        # Output projection
        self.output_proj = nn.Linear(hidden_size, output_size)

        # Diagnostics
        self.last_top_idx: torch.Tensor
        self.last_top_w: torch.Tensor
        self.last_aux_loss: torch.Tensor

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (B, T, D) input.
        Returns:
            output: (B, T, O) output projection
            aux_loss: scalar auxiliary load-balancing loss
            info: dict with diagnostics
        """
        x_clean = torch.nan_to_num(x, nan=0.0)
        B, T, D = x_clean.shape

        # Route
        full_weights, top_idx, top_w, aux_loss = self.router(x_clean)
        self.last_top_idx = top_idx.detach()
        self.last_top_w = top_w.detach()
        self.last_aux_loss = aux_loss.detach()

        # Run all experts
        expert_outs = torch.stack(
            [e(x_clean) for e in self.experts],
            dim=0,
        )  # (K, B, T, H)
        expert_outs = expert_outs.permute(1, 2, 0, 3)  # (B, T, K, H)

        # Gather top-K
        # top_idx: (B*T, top_k) → reshape to (B, T, top_k)
        top_idx_3d = top_idx.reshape(B, T, self.config.top_k)
        top_w_3d = top_w.reshape(B, T, self.config.top_k)
        # Gather
        idx_expanded = top_idx_3d.unsqueeze(-1).expand(-1, -1, -1, self.hidden_size)
        top_outs = torch.gather(expert_outs, 2, idx_expanded)  # (B, T, top_k, H)
        # Weighted sum
        mixed = (top_w_3d.unsqueeze(-1) * top_outs).sum(dim=2)  # (B, T, H)

        # Add time branch
        if self.time_branch is not None:
            time_out = self.time_branch(x_clean)
            mixed = mixed + time_out

        # Output
        output = self.output_proj(mixed)

        # Aux loss
        info = {
            "aux_loss": aux_loss,
            "routing_weights_full": full_weights.detach(),
        }
        return output, aux_loss, info


class TimeFreqMoECfCNetwork(nn.Module):
    """Rolling-window network using TimeFreqMoECfCCell.

    The cell is stateless (no recurrent state) so this is essentially
    a wrapper that handles NaN inputs and provides utilization metrics.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = 1,
        config: Optional[FrequencyMoEConfig] = None,
    ):
        super().__init__()
        if config is None:
            config = FrequencyMoEConfig()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.config = config
        self.cell = TimeFreqMoECfCCell(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            config=config,
        )

    def forward(
        self,
        x: torch.Tensor,
        times: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Forward pass on a sequence.

        Args:
            x: (B, T, D) input sequence.
        Returns:
            outputs: (B, T, O) output sequence.
            aux_loss: scalar auxiliary load-balancing loss.
            info: dict with diagnostics.
        """
        output, aux_loss, info = self.cell(x)
        return output, aux_loss, info

    def get_aux_loss(self) -> torch.Tensor:
        """Get the auxiliary load-balancing loss from the last forward pass."""
        return self.cell.last_aux_loss

    def get_utilization(self) -> Dict:
        """Get utilization statistics."""
        weights = self.cell.last_top_w
        idx = self.cell.last_top_idx
        if weights is None or weights.numel() == 0 or idx is None or idx.numel() == 0:
            return {
                "routing_H": 0.0,
                "max_min": 1.0,
                "active_fraction": 0.0,
                "utilization": [0.0] * self.config.n_experts,
            }
        # Per-expert utilization: sum of weights where expert is in top-K
        # top_idx: (B*T, top_k), top_w: (B*T, top_k)
        idx_flat = idx.reshape(-1).cpu().tolist()
        w_flat = weights.reshape(-1).cpu().tolist()
        size = self.config.n_experts
        utilization = [0.0] * size
        for i, w in zip(idx_flat, w_flat):
            utilization[i] += w
        total = sum(utilization) + 1e-8
        probs = [u / total for u in utilization]
        # Entropy
        H = -sum(p * math.log(p + 1e-12) for p in probs)
        H = H / max(math.log(max(size, 2)), 1)
        # Max/min over active
        active = [p for p in probs if p > 1e-8]
        if not active:
            max_min = 1.0
            active_frac = 0.0
        else:
            max_min = max(active) / (min(active) + 1e-8)
            active_frac = len(active) / size
        return {
            "routing_H": H,
            "max_min": max_min,
            "active_fraction": active_frac,
            "utilization": probs,
        }

    def get_omegas(self) -> torch.Tensor:
        """Get the learned frequencies of all experts (K, n_freqs)."""
        return torch.stack([e._get_omega().detach() for e in self.cell.experts], dim=0)


__all__ = [
    "FrequencyExpertConfig",
    "FrequencyExpert",
    "FrequencyMoEConfig",
    "FrequencyRouter",
    "TimeFreqMoECfCCell",
    "TimeFreqMoECfCNetwork",
]
