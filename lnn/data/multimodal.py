from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader, Dataset, random_split


class SyntheticMultimodalDataset(Dataset):
    """
    Local multimodal benchmark for LNN experiments.

    Each sample contains:
        - sensor: time-aligned numeric sequence, shape (seq_len, sensor_dim)
        - image: small grayscale pattern, shape (1, image_size, image_size)
        - tokens: short token sequence, shape (text_len,)
        - label: class id

    The class signal is intentionally present in all three modalities so this
    dataset can run quickly without external downloads.
    """

    def __init__(
        self,
        num_samples: int = 900,
        seq_len: int = 24,
        sensor_dim: int = 4,
        image_size: int = 16,
        text_len: int = 12,
        vocab_size: int = 48,
        num_classes: int = 3,
        noise_std: float = 0.08,
        seed: int = 42,
    ) -> None:
        if sensor_dim < 1:
            raise ValueError("sensor_dim must be >= 1")
        if image_size < 8:
            raise ValueError("image_size must be >= 8")
        if vocab_size <= num_classes * 4:
            raise ValueError("vocab_size must be larger than num_classes * 4")

        self.num_samples = num_samples
        self.seq_len = seq_len
        self.sensor_dim = sensor_dim
        self.image_size = image_size
        self.text_len = text_len
        self.vocab_size = vocab_size
        self.num_classes = num_classes

        generator = torch.Generator().manual_seed(seed)
        labels = torch.randint(0, num_classes, (num_samples,), generator=generator)
        self.labels = labels.long()
        self.sensor = self._make_sensor(labels, noise_std, generator)
        self.images = self._make_images(labels, noise_std, generator)
        self.tokens = self._make_tokens(labels, generator)

    def _make_sensor(self, labels: torch.Tensor, noise_std: float, generator: torch.Generator) -> torch.Tensor:
        t = torch.linspace(0.0, 1.0, self.seq_len)
        phase = torch.rand(self.num_samples, 1, generator=generator) * (2.0 * math.pi)
        freq = 1.0 + labels.float().unsqueeze(1)
        trend = (labels.float().unsqueeze(1) - (self.num_classes - 1) / 2.0) * t.unsqueeze(0)

        features = []
        for index in range(self.sensor_dim):
            if index % 4 == 0:
                value = torch.sin(2.0 * math.pi * freq * t.unsqueeze(0) + phase)
            elif index % 4 == 1:
                value = torch.cos(2.0 * math.pi * (freq + 0.5) * t.unsqueeze(0) + phase / 2.0)
            elif index % 4 == 2:
                value = trend
            else:
                amplitude = 0.5 + labels.float().unsqueeze(1) * 0.25
                value = torch.sin(2.0 * math.pi * (freq * 2.0) * t.unsqueeze(0)) * amplitude
            features.append(value)

        sensor = torch.stack(features, dim=-1)
        sensor += noise_std * torch.randn(sensor.shape, generator=generator)
        return sensor.float()

    def _make_images(self, labels: torch.Tensor, noise_std: float, generator: torch.Generator) -> torch.Tensor:
        images = torch.zeros(self.num_samples, 1, self.image_size, self.image_size)
        center = self.image_size // 2
        width = max(2, self.image_size // 8)

        for index, label in enumerate(labels.tolist()):
            pattern = label % 3
            if pattern == 0:
                images[index, :, :, center - width : center + width] = 1.0
            elif pattern == 1:
                images[index, :, center - width : center + width, :] = 1.0
            else:
                diag = torch.arange(self.image_size)
                for offset in range(-width, width + 1):
                    cols = (diag + offset).clamp(0, self.image_size - 1)
                    images[index, 0, diag, cols] = 1.0

            images[index] += 0.15 * float(label) / max(self.num_classes - 1, 1)

        images += noise_std * torch.randn(images.shape, generator=generator)
        return images.clamp(0.0, 1.0).float()

    def _make_tokens(self, labels: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        tokens = torch.zeros(self.num_samples, self.text_len, dtype=torch.long)
        class_block = max(4, (self.vocab_size - 1) // self.num_classes)

        for index, label in enumerate(labels.tolist()):
            low = 1 + label * class_block
            high = min(self.vocab_size, low + class_block)
            tokens[index] = torch.randint(low, high, (self.text_len,), generator=generator)

            noise_count = max(1, self.text_len // 4)
            noise_positions = torch.randperm(self.text_len, generator=generator)[:noise_count]
            tokens[index, noise_positions] = torch.randint(1, self.vocab_size, (noise_count,), generator=generator)

        return tokens

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        sample = {
            "sensor": self.sensor[index],
            "image": self.images[index],
            "tokens": self.tokens[index],
        }
        return sample, self.labels[index]


def create_multimodal_dataloaders(
    dataset: SyntheticMultimodalDataset,
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

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
