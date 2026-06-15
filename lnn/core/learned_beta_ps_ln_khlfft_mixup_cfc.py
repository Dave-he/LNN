"""LearnedBetaPS+LN+Khl+FFT+Mixup-CfC (PRD #10-159, Round 197, 2026-06-16).

Wraps round 187's winner (lbps_lnkhlfft_5_3_2) with **Mixup**
data augmentation (Zhang et al 2018).

After 5 noise/regularization rounds (r192 input, r193 hidden,
r194 combined, r195 σ sweep, r196 dropconnect), pivot to a
**sample-level** augmentation: Mixup interpolates between
two random samples and their targets.

Mixup formula (Zhang et al 2018):
    x_mix = λ * x_i + (1-λ) * x_j
    t_mix = λ * t_i + (1-λ) * t_j
where λ ~ Beta(α, α)

Sonnet (Shu & Lampos 2026, AAAI Oral) uses multi-resolution
spectral features for forecasting. Different paradigm.
Mixup is **inter-sample** not intra-sample noise — provides
diversity in the dataset itself.

Mechanism (TRAINING ONLY, not eval):
    batch_x: [B, T, D], batch_t: [B, T, 1]
    # Permute batch
    idx = randperm(B)
    # Sample λ from Beta(α, α)
    λ = sample from Beta(α, α)
    # Mix
    x_mixed = λ * batch_x + (1-λ) * batch_x[idx]
    t_mixed = λ * batch_t + (1-λ) * batch_t[idx]
    # Forward
    y = cfc_net(x_mixed)
    loss = λ * MSE(y, t) + (1-λ) * MSE(y, t[idx])
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lnn.core.learned_beta_ps_ln_khlfft_cfc import LearnedBetaPSLNKhlFftCfCStackedNetwork


def sample_mixup_lambda(alpha, batch_size, device):
    """Sample λ from Beta(α, α) for each sample in batch.

    Returns: λ of shape [batch_size, 1, 1] for broadcasting
    """
    # Use Beta distribution
    gamma1 = torch._standard_gamma(torch.full((batch_size,), alpha, device=device))
    gamma2 = torch._standard_gamma(torch.full((batch_size,), alpha, device=device))
    lam = gamma1 / (gamma1 + gamma2)
    return lam.view(batch_size, 1, 1)


class LearnedBetaPSLNKhlFftMixupCfCStackedNetwork(nn.Module):
    """Round 187 winner + Mixup data augmentation.

    Args:
        input_size, hidden_size, output_size, num_layers
        mixup_alpha: Beta distribution α (0=disabled)
        Kh_ladder, Kx, etc.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=3,
        mixup_alpha=0.2,
        Kh_ladder=None,
        Kx=5,
        mode_x="diff",
        mode_h="diff",
        beta_x_init=0.75,
        beta_h_init=0.75,
        return_sequences=True,
        n_fft=None,
    ):
        super().__init__()
        self.mixup_alpha = mixup_alpha
        # Reuse round 187 stack
        self.cfc_net = LearnedBetaPSLNKhlFftCfCStackedNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
            Kh_ladder=Kh_ladder,
            Kx=Kx,
            mode_x=mode_x,
            mode_h=mode_h,
            beta_x_init=beta_x_init,
            beta_h_init=beta_h_init,
            return_sequences=return_sequences,
            n_fft=n_fft,
        )

    def forward(self, x):
        if not (self.training and self.mixup_alpha > 0):
            return self.cfc_net(x)
        # Mixup is applied at the loss level (we don't mix the input here,
        # we mix the targets). Actually for sample-level augmentation,
        # we need to mix the input. But since we want to keep the API
        # simple (forward takes only x), we can't access targets.
        # Workaround: apply mixup to input only, return mixed output.
        B = x.shape[0]
        device = x.device
        # Sample λ
        lam = sample_mixup_lambda(self.mixup_alpha, B, device)
        # Permute batch
        idx = torch.randperm(B, device=device)
        # Mix input only (target mixing done in loss)
        x_mixed = lam * x + (1.0 - lam) * x[idx]
        return self.cfc_net(x_mixed), idx, lam

    def forward_eval(self, x):
        """Eval forward (no mixup)."""
        return self.cfc_net(x)


def mixup_loss(y, target, idx, lam, base_loss_fn=None):
    """Compute mixup loss: lam * loss(y, t) + (1-lam) * loss(y, t[idx]).

    Args:
        y: model output [B, T, 1]
        target: target [B, T, 1]
        idx: permutation from forward
        lam: mixing coefficient [B, 1, 1]
        base_loss_fn: optional loss function (default F.mse_loss)
    """
    if base_loss_fn is None:
        base_loss_fn = F.mse_loss
    # Average lam across batch for scalar loss
    lam_scalar = lam.mean()
    loss_a = base_loss_fn(y, target)
    loss_b = base_loss_fn(y, target[idx])
    return lam_scalar * loss_a + (1.0 - lam_scalar) * loss_b


def make_lbps_lnkhlfft_mixup_5_3_2(input_size, hidden_size, output_size, num_layers=3, mixup_alpha=0.2):
    """Kh=[5,3,2] + LN + FFT + Mixup."""
    return LearnedBetaPSLNKhlFftMixupCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kh_ladder=[5, 3, 2], Kx=5,
        mixup_alpha=mixup_alpha,
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
    )


__all__ = [
    "LearnedBetaPSLNKhlFftMixupCfCStackedNetwork",
    "make_lbps_lnkhlfft_mixup_5_3_2",
    "mixup_loss",
    "sample_mixup_lambda",
]
