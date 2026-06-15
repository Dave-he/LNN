"""Gumbel-Softmax Routing MoE for CfC (PRD #10-79, round 117, 2026-06-15).

Response to Jang et al. 2017 (arXiv:1611.01144, ICLR 2017) — *Categorical
Reparameterization with Gumbel-Softmax* — combined with Switch
Transformer's top-1 + stochastic routing (arXiv:2101.03961). The 5th
major router family in the 91-116 audit and the 1st **stochastic**
router (after softmax, sigmoid, ReLU, cosine which are all deterministic).

Why this fits the 91-116 audit pattern:
- Structural: changes the routing topology (deterministic → stochastic),
  but experts/CfCCell structure is unchanged.
- Data-structure-independent: noise is per-sample, no data-dependent bias.
- Preserves recurrent state mixing: ``h_new = sum_i g_i * expert_i(x_t, h_t)``
  has the same form as FAME/ReMoE/Sigmoid.
- Fills a real gap: 91-116 has 4 deterministic router families and 0
  stochastic families. Round 103 found FAME H=0 lock-in; Gumbel noise
  is one natural way to address it.

Three properties of Gumbel-Softmax routing:
1. **Stochastic at training time, deterministic at inference** —
   ``torch.no_grad()`` removes the noise.
2. **Temperature annealing** — natural curriculum from exploration
   to exploitation. T starts high (1.0), anneals to low (0.1).
3. **Gumbel-Softmax is differentiable** — gradient flows through the
   soft mixture even though the decision is stochastic.

Forward pass::

    z = W x                              # [B, K] router logits
    g = -log(-log(U))                    # [B, K] Gumbel noise, U ~ Uniform(0, 1)
    z' = (z + g) / T                     # add noise + scale by temperature
    g_routing = softmax(z', dim=-1)      # [B, K] stochastic mixture weights
    h_new = sum_i g_routing_i * expert_i(x_t, h_t)  # [B, H]
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from lnn.core.cfc import CfCCell
from lnn.core.sequence_utils import select_step_delta, select_step_mask


def _sample_gumbel(shape: tuple, device: torch.device, dtype: torch.dtype,
                   eps: float = 1e-9) -> torch.Tensor:
    """Sample Gumbel(0, 1) noise.

    Args:
        shape: Output shape.
        device: Torch device.
        dtype: Torch dtype.
        eps: Small constant to avoid log(0).

    Returns:
        Gumbel noise tensor of the given shape, device, dtype.
    """
    U = torch.rand(shape, device=device, dtype=dtype)
    return -torch.log(-torch.log(U + eps) + eps)


class GumbelRouter(nn.Module):
    """Gumbel-Softmax router: stochastic categorical sampling.

    Args:
        input_size: Input feature dimension (D).
        hidden_size: Hidden state dimension (H).
        n_experts: Number of experts K.
        temperature: Initial temperature T (annealed externally).
        router_hidden: Optional 2-layer router MLP width.
        small_init: If True, init W ~ N(0, 0.01) to avoid early saturation.
        anneal_rate: Temperature decay per epoch (e.g., 0.95).
        min_temperature: Minimum temperature (T cannot go below this).

    Forward:
        x_t: [B, input_size] input.
        h:   [B, hidden_size] previous hidden.
        epoch: Current training epoch (for annealing).
        Returns: g: [B, n_experts] Gumbel-Softmax mixture weights.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_experts: int,
        temperature: float = 1.0,
        router_hidden: int = 0,
        small_init: bool = True,
        anneal_rate: float = 0.95,
        min_temperature: float = 0.1,
    ):
        super().__init__()
        assert n_experts >= 1, f"n_experts must be >= 1, got {n_experts}"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_experts = int(n_experts)
        self.temperature_init = float(temperature)
        self.temperature = float(temperature)
        self.anneal_rate = float(anneal_rate)
        self.min_temperature = float(min_temperature)
        self.router_hidden = int(router_hidden)
        self.small_init = bool(small_init)

        if router_hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(input_size + hidden_size, router_hidden),
                nn.Tanh(),
                nn.Linear(router_hidden, n_experts),
            )
        else:
            self.net = nn.Linear(input_size + hidden_size, n_experts)

        if self.small_init:
            last_layer = self.net[-1] if router_hidden > 0 else self.net
            nn.init.normal_(last_layer.weight, mean=0.0, std=0.01)
            if last_layer.bias is not None:
                nn.init.zeros_(last_layer.bias)

        # Side-channel
        self.last_g: torch.Tensor | None = None
        self.last_logits: torch.Tensor | None = None

    def set_temperature(self, t: float) -> None:
        """Override current temperature (e.g., from external scheduler)."""
        self.temperature = max(t, self.min_temperature)

    def anneal_step(self) -> None:
        """Apply one anneal step: T <- max(T * anneal_rate, min_temperature)."""
        self.temperature = max(
            self.temperature * self.anneal_rate, self.min_temperature,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        training: bool = True,
    ) -> torch.Tensor:
        """Compute Gumbel-Softmax routing weights.

        Args:
            x_t: [B, input_size] input.
            h:   [B, hidden_size] previous hidden.
            training: If True, add Gumbel noise; else deterministic softmax.

        Returns:
            g: [B, n_experts] mixture weights (sum to 1 per row).
        """
        combined = torch.cat([x_t, h], dim=-1)
        logits = self.net(combined)
        self.last_logits = logits

        if training and self.temperature > 0.0:
            g_noise = _sample_gumbel(
                logits.shape, logits.device, logits.dtype,
            )
            g_routing = F.softmax((logits + g_noise) / self.temperature, dim=-1)
        else:
            # Inference: deterministic softmax
            g_routing = F.softmax(logits, dim=-1)

        self.last_g = g_routing.detach()
        return g_routing


class GumbelMoECfCCell(nn.Module):
    """Gumbel-Softmax MoE cell: K experts, stochastic mixture.

    Args:
        input_size: Input feature dimension (D).
        hidden_size: Hidden state dimension (H).
        n_experts: Number of experts K.
        temperature: Initial temperature T (annealed via ``anneal_step``).
        anneal_rate: Temperature decay per epoch.
        min_temperature: Minimum temperature.
        n_tau_per_expert: Per-expert ``n_tau``.
        tau_scales: Per-branch τ init, forwarded to every expert.
        router_hidden: Router MLP width.
        small_init: If True, init router W ~ N(0, 0.01).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_experts: int = 3,
        temperature: float = 1.0,
        anneal_rate: float = 0.95,
        min_temperature: float = 0.1,
        n_tau_per_expert: int = 1,
        tau_scales: tuple = (0.1, 1.0, 10.0),
        router_hidden: int = 0,
        small_init: bool = True,
    ):
        super().__init__()
        assert n_experts >= 1, f"n_experts must be >= 1, got {n_experts}"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_experts = int(n_experts)
        self.temperature = float(temperature)
        self.anneal_rate = float(anneal_rate)
        self.min_temperature = float(min_temperature)
        self.n_tau_per_expert = int(n_tau_per_expert)
        self.tau_scales = tuple(tau_scales)
        self.router_hidden = int(router_hidden)
        self.small_init = bool(small_init)

        self.experts = nn.ModuleList(
            [
                CfCCell(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    n_tau=n_tau_per_expert,
                    tau_scales=tau_scales,
                )
                for _ in range(self.n_experts)
            ]
        )
        self.router = GumbelRouter(
            input_size=input_size,
            hidden_size=hidden_size,
            n_experts=self.n_experts,
            temperature=self.temperature,
            router_hidden=router_hidden,
            small_init=small_init,
            anneal_rate=anneal_rate,
            min_temperature=min_temperature,
        )

        # Diagnostic stash.
        self.last_expert_util: torch.Tensor | None = None

    def anneal_step(self) -> None:
        """Apply one anneal step to the router's temperature."""
        self.router.anneal_step()

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """One step of Gumbel-Softmax MoE.

        Args:
            x_t: [B, input_size] input at this step.
            h:   [B, hidden_size] previous hidden state.
            dt:  scalar or [B] per-sample time delta.

        Returns:
            h_new: [B, hidden_size] mixed expert output.
        """
        B = x_t.size(0)

        # 1) Compute Gumbel-Softmax routing weights.
        g = self.router(x_t, h, training=self.training)  # [B, K]

        # 2) Run all K experts.
        outs = [expert(x_t, h, dt=dt) for expert in self.experts]  # K × [B, H]
        stacked = torch.stack(outs, dim=1)  # [B, K, H]

        # 3) Weighted combination (same form as FAME/ReMoE/Sigmoid).
        h_new = (g.unsqueeze(-1) * stacked).sum(dim=1)  # [B, H]

        # 4) Track expert utilization.
        if g.dim() == 2 and g.size(0) == B:
            self.last_expert_util = g.mean(dim=0).detach()  # [K] mean per-expert gate

        return h_new

    def forward_with_aux(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Same as forward but also returns per-expert outputs (for diagnostics)."""
        g = self.router(x_t, h, training=self.training)
        outs = [expert(x_t, h, dt=dt) for expert in self.experts]
        stacked = torch.stack(outs, dim=1)
        h_new = (g.unsqueeze(-1) * stacked).sum(dim=1)
        if g.dim() == 2 and g.size(0) == x_t.size(0):
            self.last_expert_util = g.mean(dim=0).detach()
        return h_new, outs


class GumbelMoECfCNetwork(nn.Module):
    """Stacked Gumbel-Softmax MoE CfC network.

    Mirrors the ``CfCNetwork`` / ``FAMECfCNetwork`` API (return_sequences,
    mask, dt) but uses ``GumbelMoECfCCell`` for every layer. The caller
    is responsible for calling ``anneal_step()`` between epochs.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        return_sequences: bool = True,
        n_experts: int = 3,
        temperature: float = 1.0,
        anneal_rate: float = 0.95,
        min_temperature: float = 0.1,
        n_tau_per_expert: int = 1,
        tau_scales: tuple = (0.1, 1.0, 10.0),
        router_hidden: int = 0,
        small_init: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.return_sequences = return_sequences
        self.n_experts = int(n_experts)
        self.temperature = float(temperature)
        self.anneal_rate = float(anneal_rate)
        self.min_temperature = float(min_temperature)
        self.n_tau_per_expert = int(n_tau_per_expert)
        self.tau_scales = tuple(tau_scales)
        self.router_hidden = int(router_hidden)
        self.small_init = bool(small_init)

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            self.cells.append(
                GumbelMoECfCCell(
                    input_size=in_dim,
                    hidden_size=hidden_size,
                    n_experts=self.n_experts,
                    temperature=self.temperature,
                    anneal_rate=self.anneal_rate,
                    min_temperature=self.min_temperature,
                    n_tau_per_expert=self.n_tau_per_expert,
                    tau_scales=self.tau_scales,
                    router_hidden=self.router_hidden,
                    small_init=self.small_init,
                )
            )
        self.output_proj = nn.Linear(hidden_size, output_size)

    def anneal_step(self) -> None:
        """Anneal temperature on all cells' routers."""
        for cell in self.cells:
            cell.anneal_step()

    def get_temperature(self) -> float:
        """Return current temperature of the first cell's router."""
        return self.cells[0].router.temperature

    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
        dt: float | torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Process a batch of sequences.

        Args:
            x: [B, T, F] input sequence.
            h0: Optional [num_layers, B, H] initial hidden state.
            dt: Same per-step time-delta shapes as ``CfCNetwork``.
            mask: Same mask shapes as ``CfCNetwork``.

        Returns:
            [B, T, output_size] (return_sequences=True) or [B, output_size].
        """
        batch_size, seq_len, _ = x.shape
        if h0 is None:
            h0 = torch.zeros(
                self.num_layers, batch_size, self.hidden_size,
                device=x.device, dtype=x.dtype,
            )

        h = h0
        layer_input = x
        for i, cell in enumerate(self.cells):
            outputs = []
            h_i = h[i]
            for t in range(seq_len):
                dt_t = select_step_delta(dt, t, batch_size, seq_len, x.device, x.dtype)
                input_mask, update_mask = select_step_mask(
                    mask, t, batch_size, seq_len, self.input_size, x.device, x.dtype,
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
        return self.output_proj(layer_input[:, -1, :])


def gumbel_moe_utilization(cell: GumbelMoECfCCell) -> dict:
    """Diagnostic for Gumbel MoE cell's expert utilization.

    Returns:
        Dict with:
            - "expert_util": [K] tensor of mean Gumbel-Softmax gate per expert.
            - "expert_count": [K] rough count (util * 100).
            - "routing_entropy": scalar — entropy of expert_util (in nats).
            - "temperature": current router temperature.
    """
    if cell.last_expert_util is None:
        return {
            "expert_util": torch.zeros(cell.n_experts),
            "expert_count": torch.zeros(cell.n_experts),
            "routing_entropy": torch.tensor(0.0),
            "temperature": cell.router.temperature,
        }
    util = cell.last_expert_util.cpu()
    eps = 1e-8
    # Normalize to a probability distribution for entropy.
    util_p = util / (util.sum() + eps)
    entropy = -(util_p * torch.log(util_p + eps)).sum().cpu()
    return {
        "expert_util": util,
        "expert_count": (util * 100).long(),
        "routing_entropy": entropy,
        "temperature": cell.router.temperature,
    }
