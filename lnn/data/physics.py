from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset, random_split


class DampedOscillatorDataset(Dataset):
    """
    Parameterized damped oscillator sequences for physics-informed LNN tests.

    State is [position, velocity]. Parameters are [omega, damping], where
    acceleration follows:
        d2x/dt2 = -2*damping*omega*dx/dt - omega^2*x
    """

    def __init__(
        self,
        num_samples: int = 800,
        seq_len: int = 24,
        horizon: int = 12,
        dt: float = 0.05,
        noise_std: float = 0.01,
        seed: int = 42,
    ) -> None:
        if seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        if horizon < 2:
            raise ValueError("horizon must be >= 2")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.horizon = horizon
        self.dt = dt

        generator = torch.Generator().manual_seed(seed)
        self.observed, self.rollout, self.params, self.dt_input, self.dt_rollout = self._make_data(noise_std, generator)

    def _step(self, state: torch.Tensor, params: torch.Tensor, dt: float) -> torch.Tensor:
        position = state[:, 0]
        velocity = state[:, 1]
        omega = params[:, 0]
        damping = params[:, 1]
        acceleration = -2.0 * damping * omega * velocity - omega.pow(2) * position
        next_velocity = velocity + dt * acceleration
        next_position = position + dt * next_velocity
        return torch.stack([next_position, next_velocity], dim=-1)

    def _make_data(self, noise_std: float, generator: torch.Generator) -> tuple[torch.Tensor, ...]:
        omega = torch.rand(self.num_samples, generator=generator) * 1.2 + 0.8
        damping = torch.rand(self.num_samples, generator=generator) * 0.23 + 0.02
        params = torch.stack([omega, damping], dim=-1)
        state = torch.stack(
            [
                torch.rand(self.num_samples, generator=generator) * 2.0 - 1.0,
                torch.rand(self.num_samples, generator=generator) * 0.8 - 0.4,
            ],
            dim=-1,
        )

        states = []
        for _ in range(self.seq_len + self.horizon):
            states.append(state)
            state = self._step(state, params, self.dt)
        trajectory = torch.stack(states, dim=1)
        observed = trajectory[:, : self.seq_len]
        rollout = trajectory[:, self.seq_len : self.seq_len + self.horizon]
        observed = observed + noise_std * torch.randn(observed.shape, generator=generator)
        dt_input = torch.ones(self.num_samples, self.seq_len, 1) * self.dt
        dt_rollout = torch.ones(self.num_samples, self.horizon, 1) * self.dt
        return observed.float(), rollout.float(), params.float(), dt_input.float(), dt_rollout.float()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target = {
            "params": self.params[index],
            "rollout": self.rollout[index],
            "dt": self.dt_input[index],
            "rollout_dt": self.dt_rollout[index],
            "mask": torch.ones(self.seq_len, 1),
        }
        return self.observed[index], target


def create_physics_dataloaders(
    dataset: DampedOscillatorDataset,
    batch_size: int = 32,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    if train_fraction <= 0.0 or val_fraction < 0.0 or train_fraction + val_fraction >= 1.0:
        raise ValueError("fractions must leave a non-empty test split")
    train_size = int(len(dataset) * train_fraction)
    val_size = int(len(dataset) * val_fraction)
    test_size = len(dataset) - train_size - val_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size], generator=generator)
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set, batch_size=batch_size, shuffle=False),
        DataLoader(test_set, batch_size=batch_size, shuffle=False),
    )
