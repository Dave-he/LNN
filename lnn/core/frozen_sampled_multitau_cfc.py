"""Frozen-Sampled Multi-τ CfC (arXiv:2606.15571 L-RFM response, round 246).

Reference: arXiv:2606.15571 "Liquid Random Feature Methods for Time-
Dependent PDEs" (June 2026). The paper shows that **frozen random
features** with embedded relaxation scales — no learning of temporal
parameters — already capture multi-scale temporal structure and yield
high-accuracy PDE surrogates. The lesson: temporal coverage comes from
**the basis**, not from learned weights.

This module ships a **FrozenSampledMultiTauCfCCell** that applies the
same idea to discrete-time CfC:

  1. Sample ``K`` time-scale values from log-uniform ``[τ_min, τ_max]``
     **once at init**, freeze them as a non-trainable buffer.
  2. Build ``K`` independent CfC branches; force each branch's
     ``time_scale`` to its sampled value.
  3. Mix branch outputs via learned ``softmax(W_mix)``.

The key test: does frozen random τ coverage (L-RFM style) match or
beat hand-picked τ (round 245)?
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from lnn.core.cfc import CfCCell


def sample_log_uniform(
    n: int,
    tau_min: float,
    tau_max: float,
    seed: int = 42,
) -> torch.Tensor:
    """Sample ``n`` τ values uniformly in log-space from
    ``[τ_min, τ_max]`` using a fixed RNG seed.

    Returns a 1-D tensor of length ``n``.
    """
    log_min, log_max = math.log(tau_min), math.log(tau_max)
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(n, generator=g)
    return torch.exp(u * (log_max - log_min) + log_min)


class FrozenSampledMultiTauCfCCell(nn.Module):
    """Multi-branch CfC with **frozen random** τ (L-RFM style).

    Args:
        input_size: Input feature dimension.
        hidden_size: Hidden state dimension per branch.
        n_branches: Number of branches (default 4).
        tau_min: Lower bound of log-uniform τ sampling.
        tau_max: Upper bound of log-uniform τ sampling.
        seed: RNG seed for τ sampling (default 42 for reproducibility).
        learn_mix: If True, mix weights are learned via softmax(W_mix);
            otherwise equal 1/K averaging.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_branches: int = 4,
        tau_min: float = 0.05,
        tau_max: float = 20.0,
        seed: int = 42,
        learn_mix: bool = True,
    ):
        super().__init__()
        assert n_branches >= 2
        assert tau_min > 0 and tau_max > tau_min

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_branches = int(n_branches)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)

        # Sample once, freeze forever.
        taus = sample_log_uniform(self.n_branches, tau_min, tau_max, seed=seed)
        self.register_buffer("tau_frozen", taus)

        # K independent CfC cells. We force each cell's per-neuron
        # time_scale to its sampled value AFTER init.
        self.cells = nn.ModuleList([
            CfCCell(input_size, hidden_size) for _ in range(self.n_branches)
        ])
        for cell, tau in zip(self.cells, taus):
            with torch.no_grad():
                cell.time_scale.fill_(float(tau))

        self.learn_mix = bool(learn_mix)
        if learn_mix:
            # Equal init → uniform softmax.
            self.mix_param = nn.Parameter(torch.zeros(self.n_branches))
        else:
            self.register_buffer("_mix_const",
                                 torch.ones(self.n_branches) / self.n_branches)

    @property
    def tau_values(self) -> torch.Tensor:
        """The frozen τ values, in increasing order."""
        return self.tau_frozen

    @property
    def alpha(self) -> torch.Tensor:
        """Current mix weights (softmax of ``mix_param``)."""
        if self.learn_mix:
            return torch.softmax(self.mix_param, dim=0)
        return self._mix_const

    def log_coverage(self) -> float:
        """Log-space coverage of the frozen τ (in decades).

        ``log10(tau_max / tau_min)`` is the maximum possible; this
        returns the actual coverage of the sampled values.
        """
        if self.n_branches < 2:
            return 0.0
        ratio = self.tau_frozen.max() / self.tau_frozen.min()
        return float(torch.log10(ratio).item())

    def init_state(self, batch_size: int, device: torch.device | None = None
                    ) -> list[torch.Tensor]:
        d = device or torch.device("cpu")
        return [torch.zeros(batch_size, self.hidden_size, device=d)
                for _ in range(self.n_branches)]

    def forward(
        self,
        x_t: torch.Tensor,
        h_list: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """One step.

        Args:
            x_t: Input, shape ``(B, input_size)``.
            h_list: List of K hidden states, each ``(B, hidden_size)``.

        Returns:
            ``(h_next, h_list_next)``.
        """
        outs = [cell(x_t, h_k) for cell, h_k in zip(self.cells, h_list)]
        h_next = sum(a * o for a, o in zip(self.alpha, outs))
        return h_next, outs

    def forward_with_aux(
        self,
        x_t: torch.Tensor,
        h_list: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
        """Step + diagnostics."""
        outs = [cell(x_t, h_k) for cell, h_k in zip(self.cells, h_list)]
        alpha = self.alpha
        h_next = sum(a * o for a, o in zip(alpha, outs))
        eps = 1e-8
        ent = (-alpha * (alpha + eps).log()).sum().item()
        aux = {
            "h_next": h_next,
            "alpha": alpha.detach(),
            "alpha_entropy": torch.tensor(ent),
            "tau_frozen": self.tau_frozen.detach(),
            "log_coverage": torch.tensor(self.log_coverage()),
        }
        return h_next, outs, aux


__all__ = [
    "FrozenSampledMultiTauCfCCell",
    "sample_log_uniform",
]