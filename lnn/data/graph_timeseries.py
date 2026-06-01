from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader, Dataset, random_split


class SyntheticGraphTimeSeriesDataset(Dataset):
    """
    Dynamic graph sequence dataset for GNN + LNN smoke experiments.

    Node states evolve through local diffusion on a ring graph with occasional
    long-range edges. The target is the next-step graph-level load.
    """

    def __init__(
        self,
        num_samples: int = 600,
        seq_len: int = 12,
        num_nodes: int = 8,
        node_feature_size: int = 3,
        noise_std: float = 0.03,
        seed: int = 42,
    ) -> None:
        if num_nodes < 3:
            raise ValueError("num_nodes must be >= 3")
        if node_feature_size < 1:
            raise ValueError("node_feature_size must be >= 1")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        self.node_feature_size = node_feature_size

        generator = torch.Generator().manual_seed(seed)
        data = self._make_data(noise_std, generator)
        self.node_features, self.adjacency, self.delta_t, self.mask, self.targets = data

    def _base_adjacency(self) -> torch.Tensor:
        adjacency = torch.zeros(self.num_nodes, self.num_nodes)
        for node in range(self.num_nodes):
            adjacency[node, (node - 1) % self.num_nodes] = 1.0
            adjacency[node, (node + 1) % self.num_nodes] = 1.0
        return adjacency

    def _make_data(self, noise_std: float, generator: torch.Generator) -> tuple[torch.Tensor, ...]:
        base_adjacency = self._base_adjacency()
        node_axis = torch.linspace(0.0, 1.0, self.num_nodes)
        features = torch.zeros(self.num_samples, self.seq_len, self.num_nodes, self.node_feature_size)
        adjacency = torch.zeros(self.num_samples, self.seq_len, self.num_nodes, self.num_nodes)
        dt_values = torch.rand(self.num_samples, self.seq_len, 1, generator=generator) * 0.08 + 0.04
        targets = torch.zeros(self.num_samples, 1)

        for sample in range(self.num_samples):
            phase = torch.rand(1, generator=generator).item() * 2.0 * math.pi
            amplitude = torch.rand(1, generator=generator).item() * 0.7 + 0.6
            trend = torch.rand(1, generator=generator).item() * 0.25 - 0.1
            state = amplitude * torch.sin(2.0 * math.pi * node_axis + phase)
            sample_adjacency = base_adjacency.clone()

            for step in range(self.seq_len + 1):
                if step < self.seq_len:
                    if step % 4 == 0:
                        skip = (step // 4 + sample) % self.num_nodes
                        sample_adjacency[skip, (skip + 3) % self.num_nodes] = 1.0
                        sample_adjacency[(skip + 3) % self.num_nodes, skip] = 1.0
                    adjacency[sample, step] = sample_adjacency
                    features[sample, step, :, 0] = state
                    if self.node_feature_size > 1:
                        features[sample, step, :, 1] = node_axis
                    if self.node_feature_size > 2:
                        features[sample, step, :, 2] = trend
                    if self.node_feature_size > 3:
                        features[sample, step, :, 3:] = state.unsqueeze(-1)

                degree = sample_adjacency.sum(dim=-1).clamp_min(1.0)
                neighbor_mean = sample_adjacency.matmul(state) / degree
                dt_step = dt_values[sample, min(step, self.seq_len - 1), 0]
                state = state + dt_step * (0.9 * (neighbor_mean - state) + trend)
                state = state + noise_std * torch.randn(state.shape, generator=generator)

            targets[sample, 0] = state.mean()

        mask = torch.ones(self.num_samples, self.seq_len, 1)
        features += noise_std * torch.randn(features.shape, generator=generator)
        return features.float(), adjacency.float(), dt_values.float(), mask.float(), targets.float()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        sample = {
            "node_features": self.node_features[index],
            "adjacency": self.adjacency[index],
            "dt": self.delta_t[index],
            "mask": self.mask[index],
        }
        return sample, self.targets[index]


def create_graph_dataloaders(
    dataset: SyntheticGraphTimeSeriesDataset,
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
