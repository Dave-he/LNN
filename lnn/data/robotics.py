from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader, Dataset, random_split


class SyntheticImitationDataset(Dataset):
    """
    Synthetic low-dimensional imitation learning task.

    Each sample is a short state history and a continuous expert action. The
    hidden expert mode is intentionally omitted from the state, creating a
    multi-modal action distribution that is useful for testing MDN heads.
    """

    def __init__(
        self,
        num_samples: int = 1200,
        context_len: int = 16,
        state_dim: int = 6,
        action_dim: int = 2,
        noise_std: float = 0.03,
        seed: int = 42,
        return_metadata: bool = False,
    ) -> None:
        if state_dim < 4:
            raise ValueError("state_dim must be >= 4")
        if action_dim != 2:
            raise ValueError("SyntheticImitationDataset currently supports action_dim=2")
        self.num_samples = num_samples
        self.context_len = context_len
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.return_metadata = return_metadata

        generator = torch.Generator().manual_seed(seed)
        self.states, self.actions, self.delta_t, self.mask = self._make_data(noise_std, generator)

    def _make_data(self, noise_std: float, generator: torch.Generator) -> tuple[torch.Tensor, ...]:
        start = torch.rand(self.num_samples, 2, generator=generator) * 2.0 - 1.0
        goal = torch.rand(self.num_samples, 2, generator=generator) * 2.0 - 1.0
        mode = torch.randint(0, 2, (self.num_samples, 1), generator=generator).float() * 2.0 - 1.0

        dt_values = torch.rand(self.num_samples, self.context_len, 1, generator=generator) * 0.06 + 0.04
        time_axis = torch.cumsum(dt_values.squeeze(-1), dim=1)
        time_axis = time_axis / time_axis[:, -1:].clamp_min(1e-6)

        displacement = goal - start
        pos = start.unsqueeze(1) + time_axis.unsqueeze(-1) * displacement.unsqueeze(1)
        bend = torch.stack([-displacement[:, 1], displacement[:, 0]], dim=-1)
        bend = bend / bend.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        pos = pos + 0.18 * mode.unsqueeze(1) * torch.sin(math.pi * time_axis).unsqueeze(-1) * bend.unsqueeze(1)
        vel = torch.diff(pos, dim=1, prepend=pos[:, :1, :]) / dt_values.clamp_min(1e-6)

        base_state = torch.cat([pos, vel, goal.unsqueeze(1).expand(-1, self.context_len, -1)], dim=-1)
        if self.state_dim > base_state.shape[-1]:
            extra_dim = self.state_dim - base_state.shape[-1]
            extra = torch.zeros(self.num_samples, self.context_len, extra_dim)
            extra[..., 0::2] = torch.sin(2.0 * math.pi * time_axis).unsqueeze(-1)
            if extra_dim > 1:
                extra[..., 1::2] = torch.cos(2.0 * math.pi * time_axis).unsqueeze(-1)
            base_state = torch.cat([base_state, extra], dim=-1)

        states = base_state + noise_std * torch.randn(base_state.shape, generator=generator)
        to_goal = goal - pos[:, -1, :]
        to_goal = to_goal / to_goal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        lateral = torch.stack([-to_goal[:, 1], to_goal[:, 0]], dim=-1)
        actions = to_goal + 0.55 * mode * lateral
        actions = actions + noise_std * torch.randn(actions.shape, generator=generator)

        mask = torch.ones(self.num_samples, self.context_len, self.state_dim)
        dropout = torch.rand(mask.shape, generator=generator) < 0.03
        mask = mask.masked_fill(dropout, 0.0)
        states = states * mask
        return states.float(), actions.float(), dt_values.float(), mask.float()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if self.return_metadata:
            metadata = {"dt": self.delta_t[index], "mask": self.mask[index]}
            return self.states[index], self.actions[index], metadata
        return self.states[index], self.actions[index]


def create_imitation_dataloaders(
    dataset: SyntheticImitationDataset,
    batch_size: int = 64,
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
    splits = random_split(dataset, [train_size, val_size, test_size], generator=generator)
    train_set, val_set, test_set = splits
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set, batch_size=batch_size, shuffle=False),
        DataLoader(test_set, batch_size=batch_size, shuffle=False),
    )
