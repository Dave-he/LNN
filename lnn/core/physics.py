from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lnn.core.cfc import CfCNetwork
from lnn.core.ltc import LTCNetwork


class PhysicsInformedLNN(nn.Module):
    """Recover oscillator parameters and future rollout from observed states."""

    def __init__(
        self,
        state_size: int = 2,
        hidden_size: int = 48,
        horizon: int = 12,
        recurrent_type: str = "ltc",
    ) -> None:
        super().__init__()
        recurrent_type = recurrent_type.lower()
        if recurrent_type not in {"cfc", "ltc", "gru"}:
            raise ValueError("recurrent_type must be cfc, ltc, or gru")
        self.horizon = horizon
        self.recurrent_type = recurrent_type
        self.encoder = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        if recurrent_type == "cfc":
            self.recurrent = CfCNetwork(hidden_size, hidden_size, hidden_size, return_sequences=False)
        elif recurrent_type == "ltc":
            self.recurrent = LTCNetwork(hidden_size, hidden_size, hidden_size, ode_method="euler")
        else:
            self.recurrent = nn.GRU(hidden_size, hidden_size, batch_first=True)

        self.param_head = nn.Sequential(nn.Linear(hidden_size, 2), nn.Softplus())
        self.rollout_head = nn.Linear(hidden_size, horizon * state_size)

    def forward(
        self,
        states: torch.Tensor,
        dt: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encoder(torch.nan_to_num(states))
        if self.recurrent_type == "gru":
            output, _ = self.recurrent(encoded)
            features = output[:, -1, :]
        else:
            features = self.recurrent(encoded, dt=dt, mask=mask)
            if features.dim() == 3:
                features = features[:, -1, :]
        params = self.param_head(features)
        rollout = self.rollout_head(features).view(states.shape[0], self.horizon, states.shape[-1])
        return {"params": params, "rollout": rollout}


def damped_oscillator_residual(
    rollout: torch.Tensor,
    params: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    """Finite-difference residual for the damped oscillator ODE."""
    if rollout.shape[1] < 2:
        return rollout.new_tensor(0.0)
    step_dt = dt[:, 1:, :].clamp_min(1e-6)
    position = rollout[:, :-1, 0]
    velocity = rollout[:, :-1, 1]
    next_position = rollout[:, 1:, 0]
    next_velocity = rollout[:, 1:, 1]
    omega = params[:, 0].view(-1, 1)
    damping = params[:, 1].view(-1, 1)

    dxdt = (next_position - position) / step_dt.squeeze(-1)
    dvdt = (next_velocity - velocity) / step_dt.squeeze(-1)
    acceleration = -2.0 * damping * omega * velocity - omega.pow(2) * position
    position_residual = dxdt - next_velocity
    velocity_residual = dvdt - acceleration
    return position_residual.pow(2).mean() + velocity_residual.pow(2).mean()


def physics_informed_loss(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    param_weight: float = 1.0,
    rollout_weight: float = 1.0,
    residual_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    param_loss = F.mse_loss(prediction["params"], target["params"])
    rollout_loss = F.mse_loss(prediction["rollout"], target["rollout"])
    residual_loss = damped_oscillator_residual(
        prediction["rollout"],
        prediction["params"],
        target["rollout_dt"],
    )
    loss = param_weight * param_loss + rollout_weight * rollout_loss + residual_weight * residual_loss
    metrics = {
        "param_loss": float(param_loss.detach().cpu()),
        "rollout_loss": float(rollout_loss.detach().cpu()),
        "residual_loss": float(residual_loss.detach().cpu()),
    }
    return loss, metrics
