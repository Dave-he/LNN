from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.cfc import CfCNetwork
from lnn.core.ltc import LTCNetwork


class MultimodalFusionLNN(nn.Module):
    """
    Encode sensor, image, and text modalities into a fused sequence for LNNs.

    Sensor data remains time-aligned. Image and text are encoded as static
    context vectors and broadcast across the sequence before recurrent fusion.
    """

    def __init__(
        self,
        sensor_dim: int,
        image_channels: int = 1,
        vocab_size: int = 48,
        num_classes: int = 3,
        fusion_size: int = 32,
        hidden_size: int = 48,
        text_embed_size: int = 16,
        recurrent_type: str = "cfc",
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        recurrent_type = recurrent_type.lower()
        if recurrent_type not in {"cfc", "ltc"}:
            raise ValueError("recurrent_type must be either 'cfc' or 'ltc'")

        self.recurrent_type = recurrent_type
        self.sensor_encoder = nn.Sequential(
            nn.Linear(sensor_dim, fusion_size),
            nn.LayerNorm(fusion_size),
            nn.SiLU(),
        )
        self.image_encoder = nn.Sequential(
            nn.Conv2d(image_channels, 8, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, fusion_size),
            nn.SiLU(),
        )
        self.token_embedding = nn.Embedding(vocab_size, text_embed_size, padding_idx=0)
        self.text_encoder = nn.Sequential(
            nn.Linear(text_embed_size, fusion_size),
            nn.LayerNorm(fusion_size),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(fusion_size * 3, fusion_size),
            nn.LayerNorm(fusion_size),
            nn.SiLU(),
        )

        if recurrent_type == "cfc":
            self.recurrent = CfCNetwork(
                input_size=fusion_size,
                hidden_size=hidden_size,
                output_size=num_classes,
                num_layers=num_layers,
                return_sequences=False,
            )
        else:
            self.recurrent = LTCNetwork(
                input_size=fusion_size,
                hidden_size=hidden_size,
                output_size=num_classes,
                num_layers=num_layers,
                ode_method="euler",
            )

    def encode_modalities(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        sensor = batch["sensor"]
        image = batch["image"]
        tokens = batch["tokens"]

        if sensor.dim() != 3:
            raise ValueError("sensor must have shape (batch, seq_len, sensor_dim)")
        if image.dim() != 4:
            raise ValueError("image must have shape (batch, channels, height, width)")
        if tokens.dim() != 2:
            raise ValueError("tokens must have shape (batch, text_len)")

        sensor_features = self.sensor_encoder(sensor)
        image_features = self.image_encoder(image).unsqueeze(1).expand(-1, sensor.shape[1], -1)

        embedded = self.token_embedding(tokens)
        token_mask = (tokens != 0).unsqueeze(-1)
        token_count = token_mask.sum(dim=1).clamp_min(1)
        text_features = (embedded * token_mask).sum(dim=1) / token_count
        text_features = self.text_encoder(text_features).unsqueeze(1).expand(-1, sensor.shape[1], -1)

        fused = torch.cat([sensor_features, image_features, text_features], dim=-1)
        return self.fusion(fused)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        fused_sequence = self.encode_modalities(batch)
        logits = self.recurrent(fused_sequence)
        if self.recurrent_type == "ltc":
            logits = logits[:, -1, :]
        return logits
