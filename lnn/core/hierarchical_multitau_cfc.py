"""Hierarchical Multi-τ CfC (arXiv:2606.19579 FlowFake response, round 245).

Reference: arXiv:2606.19579 "FlowFake: Liquid Networks for Audio
Deepfake Detection" (June 2026). The paper uses a multi-timescale LTC
where different time constants serve different roles:

  - Fast band (~10 ms)   — spectral features, frame-level cues
  - Slow band (~2 s)     — prosodic features, trajectory-level cues

This module ships a **Hierarchical Multi-τ CfC** with two
**non-geometric** time-scale bands and a learned mixing weight:

    h_fast = CfCCell(tau_fast)      # local detail branch
    h_slow = CfCCell(tau_slow)      # global structure branch
    alpha  = sigmoid(self.mix)      # learned mixing ∈ (0, 1)
    h_next = alpha * h_fast + (1 - alpha) * h_slow

Round 76 introduced geometric multi-τ (τ = 0.1, 1.0, 10.0) with equal
fusion. Round 243 introduced input-conditioned τ + softmax gate (and
regressed on task). Round 245 tests the hypothesis that **non-geometric
τ separation** (fast vs slow band) with a **learned scalar mix** is
the right multi-τ pattern, with simpler wiring than round 243.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.cfc import CfCCell


class HierarchicalMultiTauCfCCell(nn.Module):
    """Two-band hierarchical multi-τ CfC.

    Args:
        input_size: Input feature dimension.
        hidden_size: Hidden state dimension (each band has its own
            ``hidden_size``-dim state).
        tau_fast: Time constant for the fast band.
        tau_slow: Time constant for the slow band.
        learn_mix: If True, the mixing weight is a learned scalar;
            if False, mix=0.5 is fixed (equal weights).
        mix_init: Initial value of the (pre-sigmoid) mixing weight.
            ``0.0`` → α=0.5 (equal), positive → fast-leaning, negative →
            slow-leaning.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        tau_fast: float = 0.1,
        tau_slow: float = 5.0,
        learn_mix: bool = True,
        mix_init: float = 0.0,
    ):
        super().__init__()
        assert tau_fast > 0 and tau_slow > 0
        assert tau_fast < tau_slow, (
            f"tau_fast ({tau_fast}) should be < tau_slow ({tau_slow})"
        )

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau_fast = float(tau_fast)
        self.tau_slow = float(tau_slow)

        # Two independent CfC cells, one per band.
        self.fast_cell = CfCCell(input_size, hidden_size)
        self.slow_cell = CfCCell(input_size, hidden_size)

        # Override each cell's per-neuron time_scale with a constant
        # for "band-uniform" τ (geometric patterns test the multi-band
        # structure rather than per-neuron τ).
        with torch.no_grad():
            self.fast_cell.time_scale.fill_(tau_fast)
            self.slow_cell.time_scale.fill_(tau_slow)

        self.learn_mix = bool(learn_mix)
        if learn_mix:
            self.mix_param = nn.Parameter(torch.tensor(float(mix_init)))
        else:
            self.register_buffer("_mix_const", torch.tensor(0.5))

    @property
    def alpha(self) -> torch.Tensor:
        """Current mixing weight (in (0, 1))."""
        if self.learn_mix:
            return torch.sigmoid(self.mix_param)
        return self._mix_const

    def forward(
        self,
        x_t: torch.Tensor,
        h_fast: torch.Tensor,
        h_slow: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single step.

        Args:
            x_t: Input at this step, shape ``(B, input_size)``.
            h_fast: Fast-band state, shape ``(B, hidden_size)``.
            h_slow: Slow-band state, shape ``(B, hidden_size)``.

        Returns:
            ``(h_next, h_fast_next, h_slow_next)``.
        """
        h_fast_next = self.fast_cell(x_t, h_fast)
        h_slow_next = self.slow_cell(x_t, h_slow)
        a = self.alpha
        h_next = a * h_fast_next + (1.0 - a) * h_slow_next
        return h_next, h_fast_next, h_slow_next

    def init_state(self, batch_size: int, device: torch.device | None = None
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        d = device or torch.device("cpu")
        z = torch.zeros(batch_size, self.hidden_size, device=d)
        return z, z

    def forward_with_aux(
        self,
        x_t: torch.Tensor,
        h_fast: torch.Tensor,
        h_slow: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Step + diagnostics.

        ``aux_dict`` contains:

        * ``"h_next"``     — fused state
        * ``"h_fast_next"``— fast-band state
        * ``"h_slow_next"``— slow-band state
        * ``"alpha"``      — current mixing weight
        """
        h_fast_next = self.fast_cell(x_t, h_fast)
        h_slow_next = self.slow_cell(x_t, h_slow)
        a = self.alpha
        h_next = a * h_fast_next + (1.0 - a) * h_slow_next
        aux = {
            "h_next": h_next,
            "h_fast_next": h_fast_next,
            "h_slow_next": h_slow_next,
            "alpha": a.detach(),
        }
        return h_next, h_fast_next, h_slow_next, aux


__all__ = ["HierarchicalMultiTauCfCCell"]