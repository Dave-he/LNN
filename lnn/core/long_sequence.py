from __future__ import annotations

import torch
import torch.nn as nn


def parallel_liquid_relaxation(retain: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Parallel form of h[t] = retain[t] * h[t-1] + (1-retain[t]) * value[t]."""
    retain = retain.clamp(0.02, 0.98)
    log_prefix = torch.cumsum(torch.log(retain), dim=1).clamp_min(-30.0)
    prefix = torch.exp(log_prefix)
    drive = (1.0 - retain) * value
    return prefix * torch.cumsum(drive / prefix.clamp_min(1e-8), dim=1)


class LiquidS4Block(nn.Module):
    """
    Lightweight Liquid-S4-style block.

    This is not a full S4 implementation. It keeps the core engineering idea
    used for this repository's smoke runs: learn a liquid exponential relaxation
    and evaluate it over the sequence in parallel, then mix channels with a
    depthwise temporal convolution and residual feed-forward layer.
    """

    def __init__(self, input_size: int, hidden_size: int = 64, kernel_size: int = 7, dropout: float = 0.0) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for same-length padding")
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.retain_proj = nn.Linear(hidden_size, hidden_size)
        self.depthwise = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_size,
        )
        self.mix = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("x must have shape [batch, time, features]")
        hidden = self.input_proj(x)
        value = torch.tanh(self.value_proj(hidden))
        retain = torch.sigmoid(self.retain_proj(hidden))
        relaxed = parallel_liquid_relaxation(retain, value)
        conv = self.depthwise(relaxed.transpose(1, 2)).transpose(1, 2)
        output = self.output_norm(hidden + conv + self.mix(relaxed))
        if mask is not None:
            output = output * mask.to(device=output.device, dtype=output.dtype).unsqueeze(-1)
        return output


class LongSequenceLiquidClassifier(nn.Module):
    """Long-sequence classifier backed by Liquid-S4-style relaxation blocks."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 64,
        num_blocks: int = 2,
        kernel_size: int = 7,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        blocks = []
        for index in range(num_blocks):
            blocks.append(
                LiquidS4Block(
                    input_size if index == 0 else hidden_size,
                    hidden_size=hidden_size,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.readout = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = x
        for block in self.blocks:
            hidden = block(hidden, mask=mask)
        if mask is None:
            pooled = hidden.mean(dim=1)
        else:
            weights = mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.readout(pooled)


class LiquidTADHead(nn.Module):
    """Small LiquidTAD-style temporal action detection head over feature sequences."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 64,
        num_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.backbone = LongSequenceLiquidClassifier(
            input_size=input_size,
            num_classes=num_classes,
            hidden_size=hidden_size,
            num_blocks=num_blocks,
        )
        self.frame_classifier = nn.Linear(hidden_size, num_classes)
        self.boundary_head = nn.Linear(hidden_size, 2)

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = x
        for block in self.backbone.blocks:
            hidden = block(hidden, mask=mask)
        return hidden

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        hidden = self.encode(x, mask=mask)
        return {
            "frame_logits": self.frame_classifier(hidden),
            "boundaries": torch.sigmoid(self.boundary_head(hidden)),
        }
