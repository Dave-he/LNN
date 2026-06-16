"""LearnedBetaPS+LN+Khl+FFT+4ScaleSpectralBiasDropout3-CfC (PRD #10-180, Round 218, 2026-06-16).

Same as r216 (4-scale + bias + dropout) but with dropout p=0.3
(instead of p=0.2). Tests if more aggressive regularization helps.

Mechanism: identical to r216 except dropout_p=0.3.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lnn.core.learned_beta_ps_ln_khlfft_4spectralbiasdrop_cfc import (
    FourScaleSpectralBiasDropCfCCell,
    FourScaleSpectralBiasDropCfCStackedNetwork,
)


def make_lbps_lnkhlfft_4spectralbiasdrop3_5_3_2(input_size, hidden_size, output_size, num_layers=3):
    """Kh=[5,3,2] + LN + FFT + 4-Scale Spectral Gating + per-frequency bias + dropout p=0.3."""
    return FourScaleSpectralBiasDropCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, Kh_ladder=[5, 3, 2], Kx=5,
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
        dropout_p=0.3,
    )


__all__ = [
    "make_lbps_lnkhlfft_4spectralbiasdrop3_5_3_2",
]
