"""LearnedBetaPS+LN+Khl+FFT+PerLayerScaleCfC (PRD #10-185, Round 223, 2026-06-16).

Different scale count per layer:
- Layer 0: 2 scales (full, half)
- Layer 1: 3 scales (full, half, quarter)
- Layer 2: 4 scales (full, half, quarter, eighth)

Tests if hierarchical scale allocation across layers helps.
Each cell uses its own scale count.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lnn.core.learned_beta_ps_ln_fft_cfc import FFTInputEncoder
from lnn.core.learned_beta_ps_ln_khlfft_2spectralbiasdrop_cfc import TwoScaleSpectralBiasDropCfCCell
from lnn.core.learned_beta_ps_ln_khlfft_3spectralbiasdrop_cfc import ThreeScaleSpectralBiasDropCfCCell
from lnn.core.learned_beta_ps_ln_khlfft_4spectralbiasdrop_cfc import FourScaleSpectralBiasDropCfCCell


class PerLayerScaleCfCStackedNetwork(nn.Module):
    """Stacked CfC with different scale count per layer (2, 3, 4)."""

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=3,
        scale_counts=None,  # [2, 3, 4] by default
        Kh_ladder=None,
        Kx=5,
        mode_x="diff",
        mode_h="diff",
        beta_x_init=0.75,
        beta_h_init=0.75,
        return_sequences=True,
        n_fft=None,
        dropout_p=0.2,
    ):
        super().__init__()
        self.input_size = input_size
        self.augmented_input_size = 2 * input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.Kx = Kx
        self.scale_counts = scale_counts if scale_counts is not None else [2, 3, 4]
        assert len(self.scale_counts) == num_layers
        self.Kh_ladder = list(Kh_ladder) if Kh_ladder is not None else [3] * num_layers
        assert len(self.Kh_ladder) == num_layers
        self.return_sequences = return_sequences

        self.fft_encoder = FFTInputEncoder(n_fft=n_fft)

        # Build cells with appropriate scale count
        cell_classes = {2: TwoScaleSpectralBiasDropCfCCell, 3: ThreeScaleSpectralBiasDropCfCCell, 4: FourScaleSpectralBiasDropCfCCell}
        self.cells = nn.ModuleList()
        for l in range(num_layers):
            in_size = self.augmented_input_size if l == 0 else hidden_size
            cell_cls = cell_classes[self.scale_counts[l]]
            self.cells.append(
                cell_cls(
                    in_size, hidden_size, Kx, self.Kh_ladder[l],
                    mode_x=mode_x, mode_h=mode_h,
                    beta_x_init=beta_x_init,
                    beta_h_init=beta_h_init,
                    dropout_p=dropout_p,
                ),
            )

        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        B, T, _ = x.shape
        device = x.device
        x_aug = self.fft_encoder(x)
        hs = [torch.zeros(B, self.hidden_size, device=device) for _ in range(self.num_layers)]
        emas_x = [
            [torch.zeros(B, self.cells[l].input_size, device=device) for _ in range(self.Kx)]
            for l in range(self.num_layers)
        ]
        emas_h = [
            [torch.zeros(B, self.hidden_size, device=device) for _ in range(self.Kh_ladder[l])]
            for l in range(self.num_layers)
        ]
        outputs = []
        for t in range(T):
            inp = x_aug[:, t, :]
            for l, cell in enumerate(self.cells):
                hs[l], emas_x[l], emas_h[l] = cell(
                    inp, hs[l], emas_x[l], emas_h[l],
                )
                inp = hs[l]
            outputs.append(self.head(hs[-1]))
        outputs = torch.stack(outputs, dim=1)
        if self.return_sequences:
            return outputs
        return outputs[:, -1, :]


def make_lbps_lnkhlfft_perlayer_234_5_3_2(input_size, hidden_size, output_size, num_layers=3, dropout_p=0.2):
    """Kh=[5,3,2] + LN + FFT + per-layer scale (2, 3, 4) + bias + dropout p=0.2."""
    return PerLayerScaleCfCStackedNetwork(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        num_layers=num_layers, scale_counts=[2, 3, 4], Kh_ladder=[5, 3, 2], Kx=5,
        beta_x_init=0.75, beta_h_init=0.75, return_sequences=True,
        dropout_p=dropout_p,
    )


__all__ = [
    "PerLayerScaleCfCStackedNetwork",
    "make_lbps_lnkhlfft_perlayer_234_5_3_2",
]
