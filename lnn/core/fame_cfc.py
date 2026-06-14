"""FAME-style top-K sparse MoE wrapper around K CfCCell experts (PRD #10-36, 2026-06-14).

Wraps ``K`` independent ``CfCCell`` experts behind a
``ForecastabilityRouter`` (FAME, arXiv:2606.08896).  The cell output
is ``Σ_k g_k · expert_k(x_t, h_prev)`` where ``g`` has at most
``top_k`` non-zero entries.

Reuses the round 77 ``CfCCell(n_tau)`` interface and the round 76
multi-time-scale machinery; the only change vs ``MRMoECfCCell`` is
the router (sparse top-K instead of dense softmax).

This module is intentionally a *cell-level* FAME implementation:
- No production-data replay simulator (FAME §4.2).
- No cost-aware router training (FAME §3.4 mines expert-suitability
  targets from validation; we just use the router logits directly).
- No multi-modal fingerprinting (FAME §3.2 uses a 6-d fingerprint;
  we use ``[x_t; h_prev]`` as a proxy).
- No load-balancing auxiliary loss.

The follow-up PRD #10-37 (orthogonality constraint) and #10-38
(K×n_tau×top_K sweep) extend this base.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.cfc import CfCCell
from lnn.core.cosine_router import CosineRouter
from lnn.core.forecastability_router import ForecastabilityRouter
from lnn.core.phi_balancing import PhiBalancer
from lnn.core.sequence_utils import select_step_delta, select_step_mask


class FAMECfCCell(nn.Module):
    """FAME-style top-K sparse MoE wrapper around ``K`` CfCCell experts.

    Args:
        input_size: Input feature dimension.
        hidden_size: Hidden state dimension.
        n_experts: Number of experts (K ≥ 1).
        top_k: Number of experts activated per step (K' ∈ [1, K]).
            Default 2 matches the FAME paper's empirical choice
            (1.92 experts/series on the production vending dataset).
        n_tau_per_expert: Per-expert ``n_tau`` (round 76 compatibility).
        tau_scales: Per-branch initial time constants, forwarded to every expert.
        router_hidden: Width of the optional 2-layer router MLP (``0`` = linear).
        phi_balance: If True, attach a ``PhiBalancer`` (PRD #10-40) to
            the router and update it on every training step.  Default
            ``False`` (back-compat with round 80).
        ema_alpha: φ-balancing EMA decay (forwarded to ``PhiBalancer``).
        phi_step_size: φ-balancing mirror-descent step size η.
        router_type: ``"learned"`` (default, back-compat) uses
            ``ForecastabilityRouter``; ``"cosine"`` uses the
            parameter-free ``CosineRouter`` (PRD #10-41, arXiv:2605.12476).
            ``phi_balance`` is ignored when ``router_type="cosine"``
            (no learned logits to bias).
        ecology_gated_balancing: If True (default False), automatically
            enable φ-balancing when the live MoE ecology E drops below
            ``ecology_E_min`` (PRD #10-43, round 84).  Zero effect on
            behaviour when False.
        ecology_E_min: Threshold for ecology-gated intervention.  Default
            0.5 (the paper's claim that E ≥ 0.5 alone is sufficient).
        ecology_warmup_steps: Don't auto-enable φ in the first N steps
            even if E < threshold (router needs time to settle).
        ecology_gated_orth: If True (default False), automatically
            rescale the orth loss weight λ down to ``ecology_orth_lambda_safe``
            when E drops below threshold (PRD #10-44, round 85).  Zero
            effect on behaviour when False.  Use ``compute_orth_loss()``
            instead of ``orthogonality_loss()`` to get the rescaling.
        ecology_orth_lambda_safe: Target effective λ when orth gate fires.
            Default 0.001 (round 80 default, validated in round 83 B).
        ecology_combined: If True (default False), attach BOTH the φ gate
            AND the orth gate co-actively (PRD #10-48, round 86).  This
            is a strict superset of ``ecology_gated_balancing=True`` and
            ``ecology_gated_orth=True`` — those flags are turned on
            automatically.  Use this for the full 2-axis adaptive policy.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_experts: int = 3,
        top_k: int = 2,
        n_tau_per_expert: int = 1,
        tau_scales: tuple = (0.1, 1.0, 10.0),
        router_hidden: int = 0,
        phi_balance: bool = False,
        ema_alpha: float = 0.01,
        phi_step_size: float = 0.01,
        router_type: str = "learned",
        ecology_gated_balancing: bool = False,
        ecology_E_min: float = 0.5,
        ecology_warmup_steps: int = 0,
        ecology_gated_orth: bool = False,
        ecology_orth_lambda_safe: float = 0.001,
        ecology_combined: bool = False,
    ):
        super().__init__()
        assert n_experts >= 1
        assert 1 <= top_k <= n_experts
        assert router_type in ("learned", "cosine"), (
            f"router_type must be 'learned' or 'cosine', got {router_type!r}"
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_experts = int(n_experts)
        self.top_k = int(top_k)
        self.n_tau_per_expert = int(n_tau_per_expert)
        self.tau_scales = tuple(tau_scales)
        self.router_hidden = int(router_hidden)
        self.phi_balance = bool(phi_balance)
        self.ema_alpha = float(ema_alpha)
        self.phi_step_size = float(phi_step_size)
        self.router_type = router_type
        self.ecology_E_min = float(ecology_E_min)
        self.ecology_warmup_steps = int(ecology_warmup_steps)
        # Step counter for ecology-gated balancing (incremented in forward).
        self._step_idx: int = 0

        # Ecology-gated balancer (PRD #10-43, round 84).  Only attached
        # when the user opts in via ``ecology_gated_balancing=True`` or
        # via the round 86 combined gate.
        if ecology_gated_balancing or ecology_combined:
            from lnn.core.ecology_gated_balancing import EcologyGatedBalancer
            self.ecology_gate = EcologyGatedBalancer(
                E_min=self.ecology_E_min,
                warmup_steps=self.ecology_warmup_steps,
            )
        else:
            self.ecology_gate = None
        # Ecology-gated orth rescaling (PRD #10-44, round 85).  Only
        # attached when ``ecology_gated_orth=True`` or via the round 86
        # combined gate.  Rescales user's orth λ down to
        # ``ecology_orth_lambda_safe`` when E<threshold.
        if ecology_gated_orth or ecology_combined:
            from lnn.core.ecology_gated_balancing import EcologyGatedOrth
            self.orth_gate = EcologyGatedOrth(
                E_min=self.ecology_E_min,
                lambda_safe=ecology_orth_lambda_safe,
                warmup_steps=self.ecology_warmup_steps,
            )
        else:
            self.orth_gate = None
        # Combined ecology gate (PRD #10-48, round 86).  When
        # ``ecology_combined=True``, also attach a unified orchestrator
        # that runs both sub-gates in parallel and reports a combined
        # state in the diagnostic.  We pass the SAME sub-gate instances
        # to the orchestrator so state stays consistent across the
        # cell and the orchestrator.
        if ecology_combined:
            from lnn.core.ecology_gated_balancing import CombinedEcologyGate
            self.combined_gate = CombinedEcologyGate(
                E_min=self.ecology_E_min,
                lambda_safe=ecology_orth_lambda_safe,
                eta=self.phi_step_size,
                warmup_steps=self.ecology_warmup_steps,
                phi_gate=self.ecology_gate,
                orth_subgate=self.orth_gate,
            )
        else:
            self.combined_gate = None

        self.experts = nn.ModuleList(
            [
                CfCCell(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    n_tau=self.n_tau_per_expert,
                    tau_scales=self.tau_scales,
                )
                for _ in range(self.n_experts)
            ]
        )
        # φ-balancing (PRD #10-40): per-layer balancer instance shared
        # between the router (for bias add) and the cell (for EMA update).
        # We expose it as a submodule so .to(device) etc. move the buffers.
        # Note: φ-balancing only makes sense with the learned router.
        if self.phi_balance and router_type == "learned":
            self.balancer = PhiBalancer(
                n_experts=self.n_experts,
                ema_alpha=self.ema_alpha,
                step_size=self.phi_step_size,
            )
        else:
            self.balancer = None
        if router_type == "learned":
            self.router = ForecastabilityRouter(
                input_size=input_size,
                hidden_size=hidden_size,
                n_experts=self.n_experts,
                top_k=self.top_k,
                router_hidden=self.router_hidden,
                balancer=self.balancer,  # may be None — back-compat
            )
        else:  # "cosine" — parameter-free
            self.router = CosineRouter(
                input_size=input_size,
                hidden_size=hidden_size,
                n_experts=self.n_experts,
                top_k=self.top_k,
                ema_alpha=self.ema_alpha,
            )

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """One step of top-K sparse FAME routing over CfC experts.

        Args:
            x_t: [B, input_size] input at this step.
            h:   [B, hidden_size] previous hidden state.
            dt:  scalar or [B] per-sample time delta.

        Returns:
            h_new: [B, hidden_size] mixed expert output.
        """
        h_new, expert_outs = self.forward_with_aux(x_t, h, dt=dt)
        del expert_outs  # forward() doesn't expose them; use forward_with_aux
        return h_new

    def moe_ecology_diagnostic(self, B: float = 0.0, T: float = 1.0, O: float = 0.0) -> dict:
        """Return current MoE ecology diagnostic (PRD #10-42, Zhang 2026).

        Computes E = T·H/(O+B) and dead-expert count from the most
        recent ``last_g``.  Useful for live-monitoring whether the
        cell is in a healthy ecology regime (E ≥ 0.5 in the paper's
        setting) or has dead experts.

        If ``ecology_gated_balancing=True`` was set on construction,
        this method also runs the gate: when E < ``ecology_E_min``,
        it attaches a ``PhiBalancer`` to the cell (if not already
        present) and returns the gate state in the ``ecology_gate``
        key of the returned dict (PRD #10-43, round 84).

        Args:
            B: Balance weight — pass ``lambda_coeff`` (orth) or
                ``phi_step_size`` (φ) or 0 (plain).
            T: Routing temperature.  Default 1.0 (no scaling in FAME).
            O: Oracle weight.  Default 0.0.

        Returns:
            Dict with ``E`` (float), ``dead_experts`` (int),
            ``utilization`` (list of K floats), and (if ecology-gated
            balancing is on) ``ecology_gate`` (gate state dict).
        """
        from lnn.core.moe_ecology import moe_ecology_number
        if not hasattr(self, "last_g") or self.last_g is None:
            return {"E": float("nan"), "dead_experts": -1, "utilization": []}
        E = moe_ecology_number(
            router_logits=self.last_g, last_g=self.last_g,
            T=T, H=None, O=O, B=B,
        )
        util = self.last_g.mean(dim=0)
        dead = int((util < 0.01).sum().item())
        out = {
            "E": float(E.item()),
            "dead_experts": dead,
            "utilization": util.tolist(),
        }
        # Ecology-gated balancing (PRD #10-43): run the gate, then
        # auto-attach a PhiBalancer if the gate fires.  The attach is
        # gated on `self.training` so eval-mode diagnostics don't
        # mutate the cell.
        if self.ecology_gate is not None:
            gate_info = self.ecology_gate.step(
                E=float(E.item()), B_active=B, step_idx=self._step_idx,
            )
            out["ecology_gate"] = gate_info
            if (
                gate_info["intervened"]
                and self.balancer is None
                and self.training
            ):
                # Auto-attach a PhiBalancer to the learned router.
                from lnn.core.phi_balancing import PhiBalancer
                self.balancer = PhiBalancer(
                    n_experts=self.n_experts,
                    ema_alpha=self.ema_alpha,
                    step_size=self.phi_step_size,
                )
                if self.router_type == "learned" and hasattr(self.router, "set_balancer"):
                    self.router.set_balancer(self.balancer)
        # Ecology-gated orth rescaling (PRD #10-44): run the orth gate
        # and stash its decision in the diagnostic for ``compute_orth_loss``
        # to use.  Eval mode runs the gate but does not mutate the cell.
        if self.orth_gate is not None:
            orth_gate_info = self.orth_gate.step(
                E=float(E.item()), lambda_coeff=B, step_idx=self._step_idx,
            )
            out["ecology_gate_orth"] = orth_gate_info
        # Combined ecology gate (PRD #10-48, round 86): when on, run the
        # orchestrator and stash a unified summary.  The orchestrator
        # composes the same φ + orth sub-gates, so the per-gate keys
        # above are still populated.
        if self.combined_gate is not None:
            combined_info = self.combined_gate.step(
                E=float(E.item()), lambda_coeff=B, step_idx=self._step_idx,
            )
            out["ecology_gate_combined"] = combined_info
        return out

    def compute_orth_loss(
        self,
        outs: list[torch.Tensor],
        user_lambda: float = 0.0,
    ) -> torch.Tensor:
        """Compute orth loss with ecology-gated rescaling applied (PRD #10-44).

        If ``ecology_gated_orth=True`` and the gate has fired, scales
        ``user_lambda`` down to ``ecology_orth_lambda_safe``.  Otherwise
        returns the standard ``orthogonality_loss(outs, user_lambda)``.

        Callers should use this instead of ``orthogonality_loss()``
        directly when they want the gate to apply transparently.

        Args:
            outs: List of K [B, hidden_size] per-expert hidden states
                (typically the output of ``forward_with_aux``).
            user_lambda: The user's original orth loss weight (e.g.,
                1.0 or 10.0).  Pass 0 to skip the orth loss entirely.

        Returns:
            Scalar orth loss tensor (0 if user_lambda ≤ 0).
        """
        if user_lambda <= 0.0:
            return torch.tensor(0.0, device=outs[0].device if outs else "cpu")
        effective_lambda = user_lambda
        if self.orth_gate is not None and self.training:
            # Run the gate (uses last_g from prior forward).
            diag = self.moe_ecology_diagnostic(B=user_lambda)
            gate_info = diag.get("ecology_gate_orth", {})
            effective_lambda = gate_info.get("effective_lambda", user_lambda)
        from lnn.core.orthogonality import orthogonality_loss
        return orthogonality_loss(outs, lambda_coeff=effective_lambda)

    def forward_with_aux(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Same as ``forward`` but also returns the per-expert outputs.

        Used by ``FAMECfCNetwork.forward_with_aux`` to compute the
        geometric orthogonality constraint (PRD #10-37, AnchorMoE
        arXiv:2606.03631) over the K expert hidden states.

        Returns:
            (h_new, outs):
                h_new: [B, hidden_size] mixed expert output.
                outs:  K-element list of [B, hidden_size] per-expert outputs,
                       in the same order as ``self.experts``.  Useful for
                       the orthogonality loss.
        """
        g = self.router(x_t, h)  # [B, K] with K' nonzeros
        # Bump step counter for ecology-gated balancing (PRD #10-43).
        self._step_idx += 1
        # Diagnostics side-channel: mixture weights and top-K indices.
        self.last_g = g.detach()
        self.last_top_idx = self.router.last_top_idx.detach()
        # φ-balancing (PRD #10-40): update the EMA of per-expert assignment
        # from the hard top-K indices.  Skipped in eval mode (the bias is
        # frozen, matching the paper's protocol).  Note: the bias was
        # already added inside router.forward() via the same instance.
        if self.balancer is not None and self.training:
            self.balancer.update(self.last_top_idx)
        # CosineRouter (PRD #10-41): update per-expert running hidden-state
        # mean from the just-routed [x_t; h].  Same train/eval gate.
        if self.router_type == "cosine" and self.training:
            combined = torch.cat([x_t, h], dim=-1)
            self.router.update(combined, self.last_top_idx)
        # Run all K experts but only the top-K contribute via g.
        # (Masking rather than skipping the non-top-K forward keeps
        # autograd simple and ensures gradient flows only to activated experts.)
        outs = [expert(x_t, h, dt=dt) for expert in self.experts]  # K × [B, H]
        stacked = torch.stack(outs, dim=1)  # [B, K, H]
        h_new = (g.unsqueeze(-1) * stacked).sum(dim=1)  # [B, H]
        return h_new, outs


class FAMECfCNetwork(nn.Module):
    """Stacked FAME-style top-K sparse MoE CfC network.

    Mirrors the ``CfCNetwork`` / ``MRMoECfCNetwork`` API
    (return_sequences, mask, dt) but swaps every layer's ``CfCCell``
    for a ``FAMECfCCell``.

    Args:
        input_size: Input feature dimension.
        hidden_size: Hidden state dimension.
        output_size: Output dimension.
        num_layers: Number of stacked FAME CfC layers.
        return_sequences: If True, return the full sequence; else last step.
        n_experts: Number of experts per layer (K).
        top_k: Number of experts activated per step (K').
        n_tau_per_expert: Per-expert ``n_tau`` (round 76 compatibility).
        tau_scales: Per-branch τ init, forwarded to every expert.
        router_hidden: Router MLP width (``0`` = linear).
        phi_balance: Forward to every layer's ``FAMECfCCell``.
        ema_alpha: Forward to every layer's balancer / CosineRouter.
        phi_step_size: Forward to every layer's balancer.
        router_type: ``"learned"`` (default) or ``"cosine"`` (PRD #10-41).
        ecology_gated_balancing: If True, each layer auto-enables φ when
            E drops below ``ecology_E_min`` (PRD #10-43, round 84).
        ecology_E_min: E threshold for ecology-gated intervention.
        ecology_warmup_steps: Don't auto-enable in the first N steps.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        return_sequences: bool = True,
        n_experts: int = 3,
        top_k: int = 2,
        n_tau_per_expert: int = 1,
        tau_scales: tuple = (0.1, 1.0, 10.0),
        router_hidden: int = 0,
        phi_balance: bool = False,
        ema_alpha: float = 0.01,
        phi_step_size: float = 0.01,
        router_type: str = "learned",
        ecology_gated_balancing: bool = False,
        ecology_E_min: float = 0.5,
        ecology_warmup_steps: int = 0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.return_sequences = return_sequences
        self.n_experts = int(n_experts)
        self.top_k = int(top_k)
        self.n_tau_per_expert = int(n_tau_per_expert)
        self.tau_scales = tuple(tau_scales)
        self.router_hidden = int(router_hidden)
        self.phi_balance = bool(phi_balance)
        self.ema_alpha = float(ema_alpha)
        self.phi_step_size = float(phi_step_size)
        self.router_type = router_type
        self.ecology_E_min = float(ecology_E_min)
        self.ecology_warmup_steps = int(ecology_warmup_steps)

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            self.cells.append(
                FAMECfCCell(
                    input_size=in_dim,
                    hidden_size=hidden_size,
                    n_experts=self.n_experts,
                    top_k=self.top_k,
                    n_tau_per_expert=self.n_tau_per_expert,
                    tau_scales=self.tau_scales,
                    router_hidden=self.router_hidden,
                    phi_balance=self.phi_balance,
                    ema_alpha=self.ema_alpha,
                    phi_step_size=self.phi_step_size,
                    router_type=self.router_type,
                    ecology_gated_balancing=ecology_gated_balancing,
                    ecology_E_min=self.ecology_E_min,
                    ecology_warmup_steps=self.ecology_warmup_steps,
                )
            )
        self.output_proj = nn.Linear(hidden_size, output_size)

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

    def forward_with_aux(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
        dt: float | torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[list[list[torch.Tensor]]]]:
        """Like ``forward`` but also returns per-step, per-layer, per-expert outputs.

        The shape of the returned nested list is
        ``[num_layers][T][K]`` of ``[B, hidden_size]`` tensors.  The
        final-step expert outputs (T-1) per layer are the most useful
        for the orthogonality loss, but we keep the full trace so
        callers can choose.

        Returns:
            (y_pred, expert_outputs):
                y_pred: same shape as ``forward`` would return.
                expert_outputs: nested list ``[num_layers][T][K]`` of
                    ``[B, hidden_size]`` tensors.
        """
        batch_size, seq_len, _ = x.shape
        if h0 is None:
            h0 = torch.zeros(
                self.num_layers, batch_size, self.hidden_size,
                device=x.device, dtype=x.dtype,
            )

        h = h0
        layer_input = x
        # expert_outputs[layer_idx][t_idx] = list of K [B, H] tensors
        expert_outputs: list[list[list[torch.Tensor]]] = [[] for _ in range(self.num_layers)]
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
                h_candidate, outs_t = cell.forward_with_aux(x_t, h_i, dt=dt_t)
                expert_outputs[i].append(outs_t)
                h_i = h_candidate if update_mask is None else update_mask * h_candidate + (1.0 - update_mask) * h_i
                outputs.append(h_i)
            layer_input = torch.stack(outputs, dim=1)
            h = torch.cat(
                [h_i.unsqueeze(0) if j == i else h[j].unsqueeze(0) for j in range(self.num_layers)],
                dim=0,
            )

        if self.return_sequences:
            y_pred = self.output_proj(layer_input)
        else:
            y_pred = self.output_proj(layer_input[:, -1, :])
        return y_pred, expert_outputs
