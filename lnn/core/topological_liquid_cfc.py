"""TopologicalLiquidCfCCell — neuron-wise topological dynamics (round 297).

Paper grounding (/loop 2026-07-12, pivot from r284-r296):
    arXiv:2606.21295 "Topological Neural Dynamics: A Neuron-wise
    Framework for Sequence Modeling" (Cai & Zhao, 2026-06) argues that
    existing sequence models (RNNs, LSTMs, continuous-time networks,
    Transformers) share a common structural principle: *layer-wise*
    dynamics, where all neurons in the same layer co-evolve through a
    shared parameterized operator. This means individual neurons have
    no freedom to specialise.

    The paper proposes *neuron-wise* dynamics where each neuron has its
    own update function on a learnable directed graph. Beats CfC / S4 /
    Transformer on Pong BC.

    This round implements a simplified version: each neuron has
    ``n_incoming`` random source connections (instead of d_h dense
    connections). The adjacency is fixed at init per neuron (random
    sparse), not learned. The output is a graph-structured hidden state
    update where each neuron only "listens to" a small subset of other
    neurons.

    The cell combines with the r295 decorrelation default.

Mechanism::

    For each neuron i ∈ [0, d_h):
        src_i = random sample of n_incoming neuron indices (fixed at init)
        rec_i = sum_k W_per_neuron[i, k] · h[src_i[k]]
        in_i  = input_W[i] · x_t
        s_i   = rec_i + in_i + bias_i + α_i · h_i + decor_loss
        h_i(t+1) = (1 - τ_i) · h_i + τ_i · tanh(s_i)

    Where τ_i is gated by the r280 blend gate (max vel/accel).

Hypotheses (PRD #10-138):

    H1 (topology helps): TopologicalLiquidCfCCell on Henry Hub
       improves overall test MSE vs blend_gated (r280/r295 default).
    H2 (sparsity is enough): random sparse topology beats dense
       recurrent weights at the same parameter count (or same compute).
    H3 (graph structure): the per-neuron adjacency pattern matters
       (random vs uniform vs structured).

API::

    TopologicalLiquidCfCCell(input_size, hidden_size, density=0.3,
        n_incoming=8, gate_mode='blend', decorr_lambda=1e-5, ...)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.decorrelation_loss import state_decorrelation_loss


class TopologicalLiquidCfCCell(nn.Module):
    """Liquid-τ cell with per-neuron sparse (graph-structured)
    recurrent connections.

    Each neuron i has a small set of ``n_incoming`` source neurons
    (``src_i``) sampled at init. The recurrent update only uses these
    n_incoming connections (not all d_h), giving a graph-structured
    topology rather than a layer-wise dense operator.

    Args:
        input_size: Input feature dimension (d_in).
        hidden_size: Hidden state dimension (d_h).
        n_incoming: number of source neurons per target neuron (the
            graph out-degree). Default 8 (≈ d_h * 0.06 for d_h=128).
        density: alternative to n_incoming; if set, n_incoming = int(d_h
            * density). Used when n_incoming is None.
        base_tau: initial τ bias.
        tau_min, tau_max: τ clamp.
        gate_mode: 'blend' / 'velocity' / 'acceleration' (r280).
        decorr_lambda: r295 decorrelation default.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_incoming: int = 8,
        density: float | None = None,
        base_tau: float = 0.5,
        tau_min: float = 0.05,
        tau_max: float = 0.95,
        pred_gate_beta: float = 4.0,
        ema_gamma: float = 0.5,
        gate_mode: str = "blend",
        decorr_lambda: float = 1e-5,
        seed: int = 42,
    ):
        super().__init__()
        if density is not None and n_incoming is None:
            n_incoming = max(1, int(hidden_size * density))
        if n_incoming is None:
            n_incoming = max(1, int(hidden_size * 0.06))
        if n_incoming > hidden_size:
            n_incoming = hidden_size
        if gate_mode not in ("blend", "velocity", "acceleration"):
            raise ValueError("gate_mode must be 'blend', 'velocity', 'acceleration'")
        if decorr_lambda < 0:
            raise ValueError("decorr_lambda must be ≥ 0")

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.n_incoming = int(n_incoming)
        self.base_tau = float(base_tau)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.pred_gate_beta = float(pred_gate_beta)
        self.ema_gamma = float(ema_gamma)
        self.gate_mode = gate_mode
        self.decorr_lambda = float(decorr_lambda)

        # Per-neuron sparse recurrent weights (d_h, n_incoming).
        gen = torch.Generator().manual_seed(seed)
        # init: small random
        self.W_rec_sparse = nn.Parameter(
            torch.randn(self.hidden_size, self.n_incoming,
                         generator=gen) * (1.0 / max(self.n_incoming ** 0.5, 1)))

        # Source neuron indices per target neuron: shape (d_h, n_incoming).
        # Each row contains the indices of neurons that neuron i reads from.
        src_per_neuron = torch.zeros(self.hidden_size, self.n_incoming,
                                       dtype=torch.long)
        for i in range(self.hidden_size):
            # Sample n_incoming distinct neurons from [0, d_h), allow self.
            perm = torch.randperm(self.hidden_size, generator=gen)
            src_per_neuron[i] = perm[:self.n_incoming]
        # Register as buffer (not trainable).
        self.register_buffer("src_indices", src_per_neuron)

        # Per-neuron bias.
        self.bias_per_neuron = nn.Parameter(
            torch.zeros(self.hidden_size))
        # Per-neuron input projection.
        self.W_in = nn.Linear(self.input_size, self.hidden_size,
                               bias=False)
        # Per-neuron tau bias.
        self.tau_bias = nn.Parameter(
            torch.full((self.hidden_size,), float(base_tau)))
        # Per-neuron alpha (self-recurrence).
        self.alpha = nn.Parameter(
            torch.zeros(self.hidden_size))

        # r295 decorrelation cache.
        self._last_outputs: torch.Tensor | None = None

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        """Run the topological liquid-τ cell.

        Args:
            x: (B, T, d_in) input sequence.
            h0: (B, d_h) initial hidden state. Defaults to zeros.
            return_aux: If True, also return a dict of diagnostics.

        Returns:
            outputs: (B, T, d_h) hidden states at each step.
            h_final: (B, d_h) final hidden state.
            aux (optional): dict of diagnostics.
        """
        B, T, _ = x.shape
        device = x.device
        dtype = x.dtype

        if h0 is None:
            h = torch.zeros(B, self.hidden_size, device=device, dtype=dtype)
        else:
            h = h0

        # Precompute per-step sparsity: gather source neurons into a
        # (B, T, d_h, n_incoming) tensor of h values.
        # h has shape (B, T, d_h); src_indices is (d_h, n_incoming).
        # h[:, :, src_indices] -> (B, T, d_h, n_incoming).
        # We do this inside the loop for memory efficiency.

        vol1 = torch.zeros(B, device=device, dtype=dtype)
        vol2 = torch.zeros(B, device=device, dtype=dtype)
        prev_x = None
        prev_prev_x = None

        outputs = []
        tau_steps = [] if return_aux else None
        gate_steps = [] if return_aux else None
        for t in range(T):
            x_t = x[:, t, :]
            # Gate computation (r280 blend).
            if prev_x is not None:
                d1 = (x_t - prev_x).abs().mean(dim=-1)
                vol1 = self.ema_gamma * vol1 + (1.0 - self.ema_gamma) * d1
            if prev_x is not None and prev_prev_x is not None:
                d2 = (x_t - 2.0 * prev_x + prev_prev_x).abs().mean(dim=-1)
                vol2 = self.ema_gamma * vol2 + (1.0 - self.ema_gamma) * d2
            g_vel = torch.exp(-self.pred_gate_beta * vol1)
            g_acc = torch.exp(-self.pred_gate_beta * vol2)
            if self.gate_mode == "velocity":
                gate = g_vel
            elif self.gate_mode == "acceleration":
                gate = g_acc
            else:
                gate = torch.max(g_vel, g_acc)
            gate = gate.unsqueeze(-1)  # (B, 1)
            prev_prev_x = prev_x
            prev_x = x_t

            # Per-neuron tau (gated by predictability).
            tau = torch.sigmoid(self.tau_bias + gate * h).clamp(
                self.tau_min, self.tau_max)

            # Per-neuron sparse recurrent update.
            # h[:, src_indices[i]] for each i: (B, d_h, n_incoming).
            h_gathered = h[:, self.src_indices]  # (B, d_h, n_incoming)
            rec = (h_gathered * self.W_rec_sparse.unsqueeze(0)).sum(dim=-1)
            in_proj = self.W_in(x_t)
            s = rec + in_proj + self.bias_per_neuron + self.alpha * h
            h = (1.0 - tau) * h + tau * torch.tanh(s)
            outputs.append(h)
            if return_aux:
                tau_steps.append(tau.detach())
                gate_steps.append(gate.detach())

        out = torch.stack(outputs, dim=1)
        self._last_outputs = out  # gradient flows back
        if not return_aux:
            return out, h

        tau_stack = torch.stack(tau_steps, dim=1)
        gate_stack = torch.stack(gate_steps, dim=1)
        aux = {
            "n_incoming": self.n_incoming,
            "tau_summary": {
                "mean": float(self.tau_bias.mean().item()),
                "std": float(self.tau_bias.std().item()),
            },
            "tau_temporal_std": float(tau_stack.std(dim=1).mean().item()),
            "tau_dynamic_mean": float(tau_stack.mean().item()),
            "gate_mean": float(gate_stack.mean().item()),
            "gate_min": float(gate_stack.min().item()),
            "gate_max": float(gate_stack.max().item()),
            "gate_mode": self.gate_mode,
            "pred_gate_beta": self.pred_gate_beta,
            "ema_gamma": self.ema_gamma,
            "decorr_lambda": self.decorr_lambda,
            "n_params_recurrent": self.W_rec_sparse.numel(),
        }
        return out, h, aux

    def extra_loss(self) -> torch.Tensor:
        """r295 default: entropy + decorrelation (no entropy_lambda here
        since the topology mask is fixed at init)."""
        if self.decorr_lambda == 0.0 or self._last_outputs is None:
            return torch.tensor(0.0, device=self._src_indices_buf_device())
        return state_decorrelation_loss(
            self._last_outputs, lambda_coeff=self.decorr_lambda)

    def _src_indices_buf_device(self):
        return self.src_indices.device


__all__ = ["TopologicalLiquidCfCCell"]