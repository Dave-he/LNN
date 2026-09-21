"""MDN-CfC — Closed-form Continuous-time network with a Mixture Density head (r307).

Reference: arXiv:2603.27058 (Correll, 2026-03-28) "Liquid Networks with
Mixture Density Heads for Efficient Imitation Learning". The paper claims
LNN+MDN gives 2.4× lower offline prediction error, 1.8× faster inference
and ~half the parameters of diffusion-policy heads.

Our toy setting: replace the standard ``Linear(hidden → output)`` head of a
CfC network with an ``MDNHead`` and switch the loss from MSE to
``mdn_negative_log_likelihood`` (Bishop 1994 mixture-density NLL). At eval
time we use ``mdn_mean`` so the prediction is deterministic and comparable
to a baseline MSE head.

Why this is novel within this repo:
  - r34 / r259 / r262 / r266 / r267 explored *regularization on CfC* but
    none touched the output head;
  - r304 studied *Parallel-CfC* at the backbone level; MDN-CfC keeps the
    vanilla CfC backbone and changes only the head;
  - the repo already had ``MDNHead`` (r-stage predates 2026, see
    ``lnn/core/mdn.py``) but no network bound it to a recurrent backbone.

Hypothesis (toy benchmarks):
  - ``toy_sin`` (single-mode sin): MDN ≈ MSE (one mixture should suffice).
  - ``structured_irr`` (multi-frequency sin*cos, well-shaped): MDN may
    beat MSE because the target has structure an MSE loss can't capture
    via a single point estimate (multi-modal local minima).
  - ``random_irr`` (Gaussian noise target): MDN ≈ MSE, no structure to
    exploit but also no penalty for over-parameterisation.
  - ``params``: MDN adds K×O + K×O + K linear projections in place of one
    Linear(H,O). For H=16,O=1,K=5: 80+80+16 = 176 vs 17 → ~10× more head
    params. We measure total params to make the trade-off explicit.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.cfc import CfCNetwork
from lnn.core.mdn import MDNHead, mdn_mean, mdn_negative_log_likelihood


class MDNCfCNetwork(nn.Module):
    """CfC backbone with a Mixture Density Network head on top.

    Forward returns a dict of MDN parameters ``{"logits","loc","log_scale"}``
    of shapes ``[B,T,K]``, ``[B,T,K,O]``, ``[B,T,K,O]``. Use
    :func:`mdn_mean` for a deterministic point prediction.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        num_mixtures: int = 5,
        return_sequences: bool = True,
        n_tau: int = 1,
        tau_scales: tuple = (0.1, 1.0, 10.0),
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_mixtures = num_mixtures

        # CfC backbone — identical to the baseline ``CfCNetwork`` up to the
        # final projection.
        self.backbone = CfCNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=hidden_size,  # keep hidden state, head reads it
            num_layers=num_layers,
            return_sequences=return_sequences,
            n_tau=n_tau,
            tau_scales=tau_scales,
        )
        # Drop the default Linear projection; we replace with MDNHead.
        self.backbone.output_proj = nn.Identity()  # type: ignore[assignment]

        self.head = MDNHead(
            input_size=hidden_size,
            output_size=output_size,
            num_mixtures=num_mixtures,
        )

    def forward(self, x: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        """Run CfC backbone then MDN head.

        Returns a dict with keys ``logits`` ([B,T,K]), ``loc`` ([B,T,K,O]),
        ``log_scale`` ([B,T,K,O]).
        """
        features = self.backbone(x, **kwargs)  # [B,T,H]
        return self.head(features)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Deterministic mean prediction (mixture-averaged)."""
        params = self.forward(x, **kwargs)
        return mdn_mean(params)


def mdn_cfc_loss(
    model: MDNCfCNetwork,
    x: torch.Tensor,
    y: torch.Tensor,
    kl_warmup: float = 1.0,
) -> torch.Tensor:
    """Compute NLL loss for an MDN-CfC network on a (x,y) batch."""
    params = model(x)
    return mdn_negative_log_likelihood(params, y) * kl_warmup


__all__ = ["MDNCfCNetwork", "mdn_cfc_loss"]