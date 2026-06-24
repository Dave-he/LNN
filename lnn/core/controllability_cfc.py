"""Controllability-Regularized CfC (arXiv:2606.08431 response, round 241).

Reference: arXiv:2606.08431 "Control-Theoretic View of Neural ODEs:
Empirical Controllability and Observability" (June 2026). The paper uses
the Linear Time-Varying (LTV) controllability Gramian to quantify how
much an input trajectory can move the hidden state.

For discrete CfC steps we cannot integrate the Gramian analytically, so
this module ships a **practical proxy**:

    c_t = || cell(x_t, h) - cell(0, h) || / || cell(x_t, h) ||

* ``c_t = 1`` → cell output is entirely driven by the input (max controllable)
* ``c_t = 0`` → cell output is invariant to the input (uncontrollable / dead cell)

The controllability loss is

    ctrl_loss = relu(margin - mean(c_t))

which is zero whenever the average input-sensitivity exceeds ``margin``.

We also provide an optional **linearization-based Gramian proxy** for
diagnostics only (not used in the training loss): the row-wise L2 norm of
``dh_next / dx_t`` (the input Jacobian), summed across time.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.cfc import CfCCell


def input_sensitivity(
    x_t: torch.Tensor,
    h: torch.Tensor,
    cell: CfCCell,
    dt: float | torch.Tensor = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute the **relative input sensitivity** of a CfC step.

    Returns a scalar tensor ``c_t`` per sample in the batch:

        c_t = ||cell(x_t, h) - cell(0, h)||_2 / (||cell(x_t, h)||_2 + eps)

    A value close to 1 means the cell output is dominated by the input;
    a value close to 0 means the cell ignores the input.
    """
    h_with = cell(x_t, h, dt=dt)
    h_without = cell(torch.zeros_like(x_t), h, dt=dt)
    diff = (h_with - h_without).norm(dim=-1)
    denom = h_with.norm(dim=-1) + eps
    return diff / denom


def input_jacobian_norm(
    x_t: torch.Tensor,
    h: torch.Tensor,
    cell: CfCCell,
    dt: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Sum of row-wise L2 norms of ``dh_next / dx_t`` (a diagnostic).

    Returns a scalar tensor per sample. Use this to *audit* whether the
    Jacobian has collapsed to zero (which would indicate the model has
    saturated into a region where inputs have no effect).
    """
    if not x_t.requires_grad:
        x_t = x_t.detach().requires_grad_(True)
    h_next = cell(x_t, h, dt=dt)
    # Compute |dh_i / dx_t| for each output dim i, then L2 row sum.
    d_h = h_next.shape[-1]
    grads = []
    for i in range(d_h):
        gi = torch.autograd.grad(
            h_next[:, i].sum(), x_t, retain_graph=True, create_graph=False
        )[0]
        grads.append(gi.norm(dim=-1))
    return torch.stack(grads, dim=-1).sum(dim=-1)


def controllability_loss(
    x_t: torch.Tensor,
    h: torch.Tensor,
    cell: CfCCell,
    margin: float = 0.05,
    dt: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Margin-based controllability loss.

    Returns ``mean( relu(margin - c_t) )`` over the batch. The loss is 0
    whenever the average relative input-sensitivity exceeds ``margin``.
    """
    c_t = input_sensitivity(x_t, h, cell, dt=dt)
    return torch.clamp(margin - c_t, min=0.0).mean()


class ControllabilityCfCCell(nn.Module):
    """CfC cell wrapper that ships a controllability loss.

    The wrapper stores the same hyperparameters as ``CfCCell`` and exposes
    two helpers:

    * ``forward_with_aux(x_t, h, ctrl_lambda, margin)`` — returns
      ``(h_next, aux_dict)`` with the controllability loss ready to be
      added to the task loss.
    * ``input_sensitivity(x_t, h)`` — public proxy for diagnostics.
    * ``input_jacobian_norm(x_t, h)`` — public diagnostic (no_grad safe).

    Args:
        input_size, hidden_size, n_tau, tau_scales: forwarded to CfCCell.
        margin: minimum acceptable input sensitivity (default 0.05).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_tau: int = 1,
        tau_scales: tuple = (0.1, 1.0, 10.0),
        margin: float = 0.05,
    ):
        super().__init__()
        self.cell = CfCCell(input_size, hidden_size, n_tau=n_tau, tau_scales=tau_scales)
        self.margin = float(margin)

    @property
    def hidden_size(self) -> int:
        return self.cell.hidden_size

    def forward(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        dt: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        return self.cell(x_t, h, dt=dt)

    def forward_with_aux(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        ctrl_lambda: float = 0.0,
        dt: float | torch.Tensor = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """CfC step + auxiliary controllability loss.

        ``aux_dict`` contains:

        * ``"h_next"`` — output of the step
        * ``"c_t"`` — per-sample relative input-sensitivity
        * ``"ctrl_loss"`` — always present (raw loss)
        * ``"ctrl_loss_total"`` — only present when ``ctrl_lambda > 0``
        * ``"jacobian_norm"`` — diagnostic, computed when ``ctrl_lambda > 0``
        """
        h_next = self.cell(x_t, h, dt=dt)
        c_t = input_sensitivity(x_t, h, self.cell, dt=dt)
        loss = torch.clamp(self.margin - c_t, min=0.0).mean()
        aux: dict[str, torch.Tensor] = {
            "h_next": h_next,
            "c_t": c_t,
            "ctrl_loss": loss,
        }
        if ctrl_lambda > 0:
            aux["ctrl_loss_total"] = ctrl_lambda * loss
            aux["jacobian_norm"] = input_jacobian_norm(x_t, h, self.cell, dt=dt).mean().detach()
        return h_next, aux

    def input_sensitivity(self, x_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return input_sensitivity(x_t, h, self.cell)

    def input_jacobian_norm(self, x_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return input_jacobian_norm(x_t, h, self.cell)


__all__ = [
    "input_sensitivity",
    "input_jacobian_norm",
    "controllability_loss",
    "ControllabilityCfCCell",
]