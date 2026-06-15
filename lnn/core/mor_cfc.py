"""Round 126 — Mixture-of-Recursions (MoR) for CfC.

Implements the Mixture-of-Recursions mechanism from
arXiv:2507.10524 (Bae et al., Google DeepMind August 2025) and
adapts it to the recurrent CfC setting.

Key idea (MoR paper, original): instead of stacking N distinct
Transformer layers, share parameters across N_r recursive blocks.
A per-token router predicts how many times to apply the shared
block for that token (1, 2, ..., N_r). Easy tokens get depth 1,
hard tokens get depth N_r.

Our adaptation to recurrent CfC:
- The shared "recursion block" is a single CfC cell.
- At each time step, the router predicts recursion depth via
  softmax over {1, 2, ..., max_depth}.
- We apply the cell depth times: h_1, h_2, ..., h_max_depth.
- For training stability we use a continuous relaxation:
    h_new = sum_d w_d * h_d
  where w_d = softmax(router_logits)[d-1] is the router weight.
  This is a strict generalisation of the depth-1 baseline.
- At inference the model can either:
    a) use the continuous relaxation (fast, smooth), or
    b) hard-select the argmax depth (true MoR).

The 5th orthogonal dimension in the 91-125 audit:
1. Expert family (round 118)
2. Aggregation (round 120)
3. Shared pathway (round 113)
4. Shared multiplicity (round 125)
5. **Recursion depth (round 126) — variable compute per timestep**

Hypotheses to test:
- H1: MoR matches baseline at depth=1 (warm start)
- H2: MoR with max_depth=3 helps on structured_irr (regime switch)
- H3: MoR with max_depth=2 is the sweet spot
- H4: MoR composes with the triple hybrid (LoRA-DAG-Shared)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lnn.core.cfc import CfCCell
from lnn.core.sequence_utils import select_step_delta


class MoRRouter(nn.Module):
    """Per-timestep router predicting recursion depth distribution.

    Returns softmax weights over {1, 2, ..., max_depth} depths.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        max_depth: int = 3,
        router_hidden: int = 0,
    ):
        super().__init__()
        assert max_depth >= 1, f"max_depth must be >= 1, got {max_depth}"
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.max_depth = int(max_depth)
        in_dim = input_size + hidden_size
        if router_hidden > 0:
            self.body = nn.Sequential(
                nn.Linear(in_dim, router_hidden),
                nn.Tanh(),
            )
            self.out_proj = nn.Linear(router_hidden, max_depth)
        else:
            self.body = nn.Identity()
            self.out_proj = nn.Linear(in_dim, max_depth)
        # Init: bias toward shallow depth (warm start at depth 1)
        # softmax of [-2, -4, -6, ...] strongly favours d=1
        with torch.no_grad():
            self.out_proj.bias.data = torch.tensor(
                [-2.0 * d for d in range(max_depth)], dtype=torch.float32
            )
            if router_hidden > 0:
                self.body[0].weight.data *= 0.1
                self.out_proj.weight.data *= 0.1
            else:
                self.out_proj.weight.data *= 0.1
        self.last_weights = None  # [B, max_depth]

    def forward(self, x_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x_t, h], dim=-1)
        features = self.body(combined)
        logits = self.out_proj(features)  # [B, max_depth]
        weights = F.softmax(logits, dim=-1)  # [B, max_depth]
        self.last_weights = weights.detach()
        return weights


class MoRCfCCell(nn.Module):
    """Mixture-of-Recursions CfC cell.

    At each time step the router predicts recursion depth weights.
    We compute h_1, h_2, ..., h_max_depth by applying the CfC cell
    recursively, then mix them with the router weights:

        h_new = sum_d w_d * h_d
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        max_depth: int = 3,
        router_hidden: int = 0,
        n_tau: int = 1,
        tau_scales: tuple = (0.1, 1.0, 10.0),
    ):
        super().__init__()
        assert max_depth >= 1, f"max_depth must be >= 1, got {max_depth}"
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.max_depth = int(max_depth)
        self.cell = CfCCell(
            input_size=input_size,
            hidden_size=hidden_size,
            n_tau=n_tau,
            tau_scales=tau_scales,
        )
        self.router = MoRRouter(
            input_size=input_size,
            hidden_size=hidden_size,
            max_depth=max_depth,
            router_hidden=router_hidden,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        h_new, _ = self.forward_with_aux(x_t, h, dt=dt)
        return h_new

    def forward_with_aux(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ):
        B = x_t.size(0)
        # Router predicts depth weights
        weights = self.router(x_t, h)  # [B, max_depth]
        # Compute h_1, h_2, ..., h_max_depth
        h_states = []
        h_curr = h
        for d in range(self.max_depth):
            h_curr = self.cell(x_t, h_curr, dt=dt)
            h_states.append(h_curr)
        h_stack = torch.stack(h_states, dim=1)  # [B, max_depth, H]
        h_new = (weights.unsqueeze(-1) * h_stack).sum(dim=1)  # [B, H]
        return h_new, {
            "weights": weights,
            "h_states": h_stack,
            "h_new": h_new,
        }


class MoRCfCNetwork(nn.Module):
    """Stacked Mixture-of-Recursions CfC network.

    Each layer has its own MoRCfCCell (with its own router and
    recursion depth). Stacking is conventional (not recursive).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        return_sequences: bool = True,
        max_depth: int = 3,
        router_hidden: int = 0,
        n_tau: int = 1,
        tau_scales: tuple = (0.1, 1.0, 10.0),
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.num_layers = int(num_layers)
        self.return_sequences = bool(return_sequences)
        self.max_depth = int(max_depth)
        self.cells = nn.ModuleList()
        for layer_idx in range(num_layers):
            layer_in = input_size if layer_idx == 0 else hidden_size
            self.cells.append(
                MoRCfCCell(
                    input_size=layer_in,
                    hidden_size=hidden_size,
                    max_depth=max_depth,
                    router_hidden=router_hidden,
                    n_tau=n_tau,
                    tau_scales=tau_scales,
                )
            )
        self.head = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x: torch.Tensor,
        dt: float | torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0)
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
        layer_input = x
        for cell in self.cells:
            outputs = []
            h_i = h
            for t in range(T):
                dt_t = select_step_delta(dt, t, B, T, x.device, x.dtype)
                x_t = layer_input[:, t, :]
                h_i = cell(x_t, h_i, dt=dt_t)
                outputs.append(h_i)
            layer_input = torch.stack(outputs, dim=1)
            h = h_i
        out = self.head(layer_input)
        if self.return_sequences:
            return out
        return out[:, -1, :]


def mor_router_weights(cell: MoRCfCCell) -> torch.Tensor:
    """Return the last router weights [B, max_depth]."""
    if cell.router.last_weights is None:
        return None
    return cell.router.last_weights.detach()


def mor_router_summary(cell: MoRCfCCell) -> dict:
    """Summarise the MoR router's depth distribution."""
    w = mor_router_weights(cell)
    if w is None:
        return {"mean_depth_weights": [0.0] * cell.max_depth, "max_depth_frac": 0.0}
    mean_w = w.mean(dim=0).cpu().tolist()
    argmax_depth = int(w.argmax(dim=-1).mode().values.item())
    max_frac = float(w.argmax(dim=-1).eq(argmax_depth).float().mean().item())
    return {
        "mean_depth_weights": mean_w,
        "argmax_depth": argmax_depth,
        "argmax_depth_frac": max_frac,
    }
