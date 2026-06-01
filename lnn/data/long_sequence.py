from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset, random_split


class SyntheticLongSequenceDataset(Dataset):
    """
    Synthetic long-context sequence classification and TAD-style labels.

    A class-specific action segment is inserted into a noisy feature sequence.
    Sequence classification predicts the action class, while frame labels and
    boundary targets support lightweight LiquidTAD smoke tests.
    """

    def __init__(
        self,
        num_samples: int = 600,
        seq_len: int = 256,
        feature_size: int = 8,
        num_classes: int = 3,
        min_segment_fraction: float = 0.12,
        max_segment_fraction: float = 0.28,
        noise_std: float = 0.08,
        seed: int = 42,
    ) -> None:
        if seq_len < 32:
            raise ValueError("seq_len must be >= 32")
        if feature_size < num_classes:
            raise ValueError("feature_size must be >= num_classes")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.feature_size = feature_size
        self.num_classes = num_classes

        generator = torch.Generator().manual_seed(seed)
        self.features, self.labels, self.frame_labels, self.boundaries, self.mask = self._make_data(
            min_segment_fraction,
            max_segment_fraction,
            noise_std,
            generator,
        )

    def _make_data(
        self,
        min_segment_fraction: float,
        max_segment_fraction: float,
        noise_std: float,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, ...]:
        features = noise_std * torch.randn(self.num_samples, self.seq_len, self.feature_size, generator=generator)
        labels = torch.randint(0, self.num_classes, (self.num_samples,), generator=generator)
        frame_labels = torch.zeros(self.num_samples, self.seq_len, dtype=torch.long)
        boundaries = torch.zeros(self.num_samples, self.seq_len, 2)
        mask = torch.ones(self.num_samples, self.seq_len)
        time_axis = torch.linspace(0.0, 1.0, self.seq_len)

        for index, label in enumerate(labels.tolist()):
            segment_len = int(
                self.seq_len
                * (
                    min_segment_fraction
                    + (max_segment_fraction - min_segment_fraction) * torch.rand(1, generator=generator).item()
                )
            )
            start_max = max(self.seq_len - segment_len - 1, 1)
            start = int(torch.randint(0, start_max, (1,), generator=generator).item())
            end = start + segment_len
            frame_labels[index, start:end] = label + 1
            boundaries[index, start, 0] = 1.0
            boundaries[index, end - 1, 1] = 1.0
            features[index, start:end, label] += 1.0
            features[index, :, -1] = time_axis
            if self.feature_size > self.num_classes + 1:
                features[index, :, self.num_classes] = torch.sin(6.28 * time_axis * (label + 1))

        return features.float(), labels.long(), frame_labels, boundaries.float(), mask.float()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target = {
            "label": self.labels[index],
            "frame_labels": self.frame_labels[index],
            "boundaries": self.boundaries[index],
            "mask": self.mask[index],
        }
        return self.features[index], target


def create_long_sequence_dataloaders(
    dataset: SyntheticLongSequenceDataset,
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
