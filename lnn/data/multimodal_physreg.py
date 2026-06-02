"""Synthetic multimodal physics dataset for damped-oscillator parameter regression.

This generator is inspired by EMMA (CVPR 2026, arXiv 2605.24047) which feeds
video+audio into an LTC to recover physical parameters. Here we synthesise a
two-modality "video + audio" dataset that is intentionally complementary:

- **Video** stream: noisy position trajectory ``x(t)`` of a damped harmonic
  oscillator. When damping is high the envelope decays quickly, leaving only a
  few useful oscillations to read the spring constant ``k`` from.

- **Audio** stream: noisy instantaneous frequency of the oscillator's
  hypothetical "motor tone" (analogous to the rover wheel-tone in EMMA),
  ``f_inst(t) = omega_d / (2*pi)`` where ``omega_d = omega * sqrt(1 - zeta**2)``.
  This directly encodes ``k`` even when the position envelope has decayed.

Targets are the (k, c) parameters of the oscillator (continuous, not classes).

The dataset returns a dict ``{"video": ..., "audio": ...}`` and a target
``{"params": [k, c], "omega": omega, "damping": zeta, "omega_d": omega_d}``.
"""

from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader, Dataset, random_split


def _integrate_damped(
    x0: torch.Tensor,
    v0: torch.Tensor,
    omega: torch.Tensor,
    zeta: torch.Tensor,
    seq_len: int,
    dt: float,
) -> torch.Tensor:
    """Closed-form trajectory of an underdamped harmonic oscillator.

    Closed form: ``x(t) = A * exp(-zeta * omega * t) * cos(omega_d * t + phi)``.
    For each sample we pick ``A, phi`` so that ``x(0) = x0, x'(0) = v0``.
    Returns positions of shape ``[num_samples, seq_len]``.
    """
    omega_d = omega * torch.sqrt(torch.clamp(1.0 - zeta.pow(2), min=1e-8))
    alpha = zeta * omega
    A = torch.sqrt(x0.pow(2) + ((v0 + alpha * x0) / omega_d).pow(2))
    phi = torch.atan2(-(v0 + alpha * x0) / omega_d, x0)
    t = torch.arange(seq_len, dtype=x0.dtype) * dt
    decay = torch.exp(-alpha.unsqueeze(-1) * t.unsqueeze(0))
    arg = omega_d.unsqueeze(-1) * t.unsqueeze(0) + phi.unsqueeze(-1)
    return A.unsqueeze(-1) * decay * torch.cos(arg)


class MultimodalPhysicsDataset(Dataset):
    """Multimodal (video+audio) dataset for damped-oscillator parameter regression.

    Parameters
    ----------
    num_samples:
        Number of (random) oscillator realisations.
    seq_len:
        Length of the per-modality sequence.
    dt:
        Time step between consecutive samples.
    mass:
        Mass of the oscillator (held constant within a dataset so that
        ``k = mass * omega**2`` and ``c = 2 * mass * zeta * omega`` are
        trivially recovered from ``(omega, zeta)``).  Default ``1.0`` keeps
        the parameter range readable.
    omega_range:
        ``(low, high)`` uniform range for the natural frequency.
    zeta_range:
        ``(low, high)`` uniform range for the damping ratio (kept strictly
        under 1 to remain in the underdamped regime).
    video_noise_std:
        Gaussian noise added to the position trajectory.
    audio_noise_std:
        Gaussian noise added to the frequency stream.
    seed:
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        num_samples: int = 600,
        seq_len: int = 32,
        dt: float = 0.05,
        mass: float = 1.0,
        omega_range: tuple[float, float] = (0.8, 2.0),
        zeta_range: tuple[float, float] = (0.05, 0.45),
        video_noise_std: float = 0.05,
        audio_noise_std: float = 0.05,
        seed: int = 42,
    ) -> None:
        if seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        if not (0.0 < zeta_range[0] < zeta_range[1] < 1.0):
            raise ValueError("zeta_range must lie in (0, 1)")
        if not (omega_range[0] > 0.0 and omega_range[1] > omega_range[0]):
            raise ValueError("omega_range must be positive and ordered")

        self.num_samples = num_samples
        self.seq_len = seq_len
        self.dt = dt
        self.mass = mass
        self.video_noise_std = video_noise_std
        self.audio_noise_std = audio_noise_std

        generator = torch.Generator().manual_seed(seed)
        self.video, self.audio, self.targets = self._make_data(
            mass=mass,
            omega_range=omega_range,
            zeta_range=zeta_range,
            video_noise_std=video_noise_std,
            audio_noise_std=audio_noise_std,
            generator=generator,
        )

    def _make_data(
        self,
        mass: float,
        omega_range: tuple[float, float],
        zeta_range: tuple[float, float],
        video_noise_std: float,
        audio_noise_std: float,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        omega = torch.rand(self.num_samples, generator=generator) * (omega_range[1] - omega_range[0]) + omega_range[0]
        zeta = torch.rand(self.num_samples, generator=generator) * (zeta_range[1] - zeta_range[0]) + zeta_range[0]
        omega_d = omega * torch.sqrt(torch.clamp(1.0 - zeta.pow(2), min=1e-8))
        x0 = torch.rand(self.num_samples, generator=generator) * 2.0 - 1.0
        v0 = torch.rand(self.num_samples, generator=generator) * 1.6 - 0.8
        position = _integrate_damped(x0, v0, omega, zeta, self.seq_len, self.dt)
        # audio = instantaneous damped frequency in Hz, with mild per-step drift
        # to mimic a synthetic motor tone that the EMMA rover uses to recover
        # hidden motor speed. The base value is omega_d / (2*pi).
        base_freq = (omega_d / (2.0 * math.pi)).unsqueeze(-1).expand(-1, self.seq_len)
        drift = 0.02 * torch.randn(self.num_samples, self.seq_len, generator=generator)
        audio = base_freq + drift
        if video_noise_std > 0:
            position = position + video_noise_std * torch.randn(position.shape, generator=generator)
        if audio_noise_std > 0:
            audio = audio + audio_noise_std * torch.randn(audio.shape, generator=generator)
        k = mass * omega.pow(2)
        c = 2.0 * mass * zeta * omega
        params = torch.stack([k, c], dim=-1)
        extras = torch.stack([omega, zeta, omega_d], dim=-1)
        targets = torch.cat([params, extras], dim=-1)
        return (
            position.unsqueeze(-1).float(),
            audio.unsqueeze(-1).float(),
            targets.float(),
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        sample = {
            "video": self.video[index],
            "audio": self.audio[index],
        }
        target = {
            "params": self.targets[index, :2],  # [k, c]
            "omega": self.targets[index, 2],
            "zeta": self.targets[index, 3],
            "omega_d": self.targets[index, 4],
        }
        return sample, target


def create_multimodal_physics_dataloaders(
    dataset: MultimodalPhysicsDataset,
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
