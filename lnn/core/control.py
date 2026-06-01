from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.cfc import CfCNetwork
from lnn.core.ltc import LTCNetwork
from lnn.core.mdn import MDNHead, mdn_mean
from lnn.ncps_integration.ncps_models import NCPSAutoNCP


class LNNImitationPolicy(nn.Module):
    """
    Low-dimensional behavior cloning policy with an LNN recurrent core.

    Inputs are state/proprioceptive sequences with shape [batch, time, state_dim].
    The policy can emit either a deterministic action or MDN parameters for
    multi-modal continuous actions.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 64,
        encoder_size: int | None = None,
        recurrent_type: str = "cfc",
        head_type: str = "mse",
        num_mixtures: int = 5,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        recurrent_type = recurrent_type.lower()
        head_type = head_type.lower()
        if recurrent_type not in {"cfc", "ltc", "autoncp"}:
            raise ValueError("recurrent_type must be cfc, ltc, or autoncp")
        if head_type not in {"mse", "mdn"}:
            raise ValueError("head_type must be mse or mdn")

        encoder_size = encoder_size or hidden_size
        self.recurrent_type = recurrent_type
        self.head_type = head_type
        self.action_dim = action_dim

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, encoder_size),
            nn.LayerNorm(encoder_size),
            nn.SiLU(),
        )

        if recurrent_type == "cfc":
            self.recurrent = CfCNetwork(
                input_size=encoder_size,
                hidden_size=hidden_size,
                output_size=hidden_size,
                num_layers=num_layers,
                return_sequences=False,
            )
        elif recurrent_type == "ltc":
            self.recurrent = LTCNetwork(
                input_size=encoder_size,
                hidden_size=hidden_size,
                output_size=hidden_size,
                num_layers=num_layers,
                ode_method="euler",
            )
        else:
            if hidden_size < 4:
                raise ValueError("AutoNCP recurrent_type requires hidden_size >= 4")
            ncp_output_size = max(1, min(hidden_size // 2, hidden_size - 3))
            self.recurrent = NCPSAutoNCP(
                input_size=encoder_size,
                hidden_size=hidden_size,
                output_size=ncp_output_size,
                model_type="cfc",
                return_sequences=False,
            )
        recurrent_output_size = hidden_size if recurrent_type in {"cfc", "ltc"} else ncp_output_size
        self.feature_adapter = (
            nn.Identity()
            if recurrent_output_size == hidden_size
            else nn.Linear(recurrent_output_size, hidden_size)
        )

        if head_type == "mdn":
            self.action_head = MDNHead(hidden_size, action_dim, num_mixtures=num_mixtures)
        else:
            self.action_head = nn.Linear(hidden_size, action_dim)

    def encode_sequence(self, states: torch.Tensor) -> torch.Tensor:
        if states.dim() != 3:
            raise ValueError("states must have shape [batch, time, state_dim]")
        return self.encoder(torch.nan_to_num(states))

    def apply_state_mask(
        self,
        states: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if mask is None:
            return states, None
        mask = mask.to(device=states.device)
        if mask.dim() == 2:
            step_mask = mask.unsqueeze(-1)
            return states * step_mask.to(dtype=states.dtype), step_mask
        if mask.dim() == 3:
            if mask.shape[-1] == states.shape[-1]:
                feature_mask = mask.to(dtype=states.dtype)
                step_mask = (feature_mask > 0).any(dim=-1, keepdim=True).to(dtype=states.dtype)
                return states * feature_mask, step_mask
            if mask.shape[-1] == 1:
                step_mask = mask.to(dtype=states.dtype)
                return states * step_mask, step_mask
        raise ValueError("mask must have shape [batch, time], [batch, time, 1], or [batch, time, state_dim]")

    def recurrent_features(
        self,
        states: torch.Tensor,
        dt: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        states, recurrent_mask = self.apply_state_mask(states, mask)
        encoded = self.encode_sequence(states)
        if self.recurrent_type in {"cfc", "ltc"}:
            features = self.recurrent(encoded, dt=dt, mask=recurrent_mask)
        else:
            features = self.recurrent(encoded)
        if features.dim() == 3:
            features = features[:, -1, :]
        return self.feature_adapter(features)

    def forward(
        self,
        states: torch.Tensor,
        dt: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        features = self.recurrent_features(states, dt=dt, mask=mask)
        return self.action_head(features)

    @torch.no_grad()
    def predict_action(
        self,
        states: torch.Tensor,
        dt: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = self.forward(states, dt=dt, mask=mask)
        if self.head_type == "mdn":
            return mdn_mean(output)
        return output
