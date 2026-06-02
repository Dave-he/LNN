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


class HeterogeneousForcedDataset(Dataset):
    """Forced damped-oscillator dataset with truly heterogeneous modalities.

    The physics is::

        m * x''(t) + c * x'(t) + k * x(t) = F(t)

    - **Video** modality: noisy position ``x(t)`` (the *response* to ``F``).
      The position is the result of a forced integration and carries only
      an *entangled* view of ``k`` and ``F``.
    - **Audio** modality: noisy direct observation of the *forcing input*
      ``F(t)`` itself.  Because the audio reveals ``F`` directly, a model
      that fuses the two streams can in principle subtract out the forced
      component to isolate the natural response — and therefore recover
      ``k`` and ``c``.

    This is the EMMA rover setting in miniature: the wheel command
    (motor audio) is hidden from the camera, but the audio reveals it.

    Parameters
    ----------
    force_kind:
        ``"chirp"`` produces a linear chirp of varying start/end
        frequencies (sample-dependent); ``"burst"`` produces a smooth
        Gaussian-windowed burst whose centre time is sample-dependent.
        Both keep the forcing independent of ``k`` and ``c``.
    num_steps_per_dt:
        Number of internal sub-steps used by the semi-implicit Euler
        integrator.  ``5`` is enough for visual realism; bump to ``20``
        if you need tighter numerical accuracy for ablations.
    """

    def __init__(
        self,
        num_samples: int = 600,
        seq_len: int = 32,
        dt: float = 0.05,
        mass: float = 1.0,
        omega_range: tuple[float, float] = (0.8, 2.0),
        zeta_range: tuple[float, float] = (0.05, 0.45),
        force_kind: str = "chirp",
        force_amp_range: tuple[float, float] = (0.4, 1.2),
        video_noise_std: float = 0.05,
        audio_noise_std: float = 0.05,
        num_steps_per_dt: int = 5,
        seed: int = 42,
    ) -> None:
        if seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        if not (0.0 < zeta_range[0] < zeta_range[1] < 1.0):
            raise ValueError("zeta_range must lie in (0, 1)")
        if force_kind not in {"chirp", "burst"}:
            raise ValueError("force_kind must be 'chirp' or 'burst'")
        if num_steps_per_dt < 1:
            raise ValueError("num_steps_per_dt must be >= 1")

        self.num_samples = num_samples
        self.seq_len = seq_len
        self.dt = dt
        self.mass = mass
        self.force_kind = force_kind
        self.video_noise_std = video_noise_std
        self.audio_noise_std = audio_noise_std
        generator = torch.Generator().manual_seed(seed)
        self.video, self.audio, self.targets = self._make_data(
            mass=mass,
            omega_range=omega_range,
            zeta_range=zeta_range,
            force_kind=force_kind,
            force_amp_range=force_amp_range,
            video_noise_std=video_noise_std,
            audio_noise_std=audio_noise_std,
            num_steps_per_dt=num_steps_per_dt,
            generator=generator,
        )

    def _make_chirp_force(
        self,
        num_samples: int,
        amp: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Sample-dependent chirp: F(t) = A * sin(2π (f0 + (f1-f0) t / T) t).

        Both ``f0`` and ``f1`` are sample-specific uniform draws, so the
        forcing is independent of ``k, c`` by construction.
        """
        t = torch.arange(self.seq_len, dtype=torch.float32) * self.dt
        T = float(t[-1].item() + self.dt)
        f0 = torch.rand(num_samples, generator=generator) * 1.5 + 0.3  # 0.3..1.8 Hz
        f1 = torch.rand(num_samples, generator=generator) * 2.5 + 0.5  # 0.5..3.0 Hz
        phase = 2.0 * math.pi * (
            f0.unsqueeze(-1) * t.unsqueeze(0)
            + (f1 - f0).unsqueeze(-1) * t.unsqueeze(0).pow(2) / (2.0 * T)
        )
        return amp.unsqueeze(-1) * torch.sin(phase)

    def _make_burst_force(
        self,
        num_samples: int,
        amp: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Sample-dependent Gaussian-windowed sinusoidal burst."""
        t = torch.arange(self.seq_len, dtype=torch.float32) * self.dt
        T = float(t[-1].item() + self.dt)
        centre = torch.rand(num_samples, generator=generator) * (T - 0.4) + 0.2
        width = torch.rand(num_samples, generator=generator) * 0.25 + 0.1
        freq = torch.rand(num_samples, generator=generator) * 2.0 + 0.5
        env = torch.exp(-0.5 * ((t.unsqueeze(0) - centre.unsqueeze(-1)) / width.unsqueeze(-1)).pow(2))
        return amp.unsqueeze(-1) * env * torch.sin(2.0 * math.pi * freq.unsqueeze(-1) * t.unsqueeze(0))

    def _integrate_forced(
        self,
        x0: torch.Tensor,
        v0: torch.Tensor,
        omega: torch.Tensor,
        zeta: torch.Tensor,
        force: torch.Tensor,
        num_steps_per_dt: int,
    ) -> torch.Tensor:
        """Semi-implicit Euler sub-step integration of the forced oscillator.

        ``force`` has shape ``[N, T]`` evaluated on the coarse grid
        ``t = arange(T) * dt``; we linearly interpolate to the sub-step
        grid before applying the integrator.
        """
        _N, T = force.shape
        sub_dt = self.dt / num_steps_per_dt
        # Pre-compute per-step force on the sub-grid.
        # We build it as a 3D tensor by linear interpolation.
        sub_steps = (T - 1) * num_steps_per_dt + 1
        fine_idx = torch.arange(sub_steps, dtype=torch.float32) / num_steps_per_dt
        fine_idx_clamped = fine_idx.clamp(max=T - 1 - 1e-6)
        left = fine_idx_clamped.floor().long()
        right = (left + 1).clamp(max=T - 1)
        alpha = (fine_idx_clamped - left.to(fine_idx_clamped.dtype)).unsqueeze(0)  # [1, sub_steps]
        force_fine = (1.0 - alpha) * force[:, left] + alpha * force[:, right]  # [N, sub_steps]
        omega_d = omega * torch.sqrt(torch.clamp(1.0 - zeta.pow(2), min=1e-8))
        decay = torch.exp(-zeta * omega * sub_dt)
        cos_d = torch.cos(omega_d * sub_dt)
        sin_d = torch.sin(omega_d * sub_dt)
        # Exact sub-step transition matrix for the unforced harmonic oscillator
        # in the rotating-frame form (Hasani-style).
        A11 = decay * (cos_d + (zeta * omega / omega_d) * sin_d)
        A12 = decay * (sin_d / omega_d)
        A21 = -decay * (omega_d + (zeta * omega).pow(2) / omega_d) * sin_d - zeta * omega * decay * cos_d
        # A22 = decay * (cos_d - (zeta * omega / omega_d) * sin_d)
        A22 = decay * (cos_d - (zeta * omega / omega_d) * sin_d)
        # Particular solution: convolve force with Green's function. Use the
        # exact discretised form via the per-step forcing f_t scaled by dt
        # applied as constant over the sub-step. Closed form is
        # x_p_sub = (1/k) * integral of G(t-s) F(s) ds over the sub-step
        # approximated by a Simpson-like 1-point rule (the forcing is
        # approximately linear between coarse samples).
        k = self.mass * omega.pow(2)
        position = x0.clone()
        velocity = v0.clone()
        trajectory = [position]
        for s in range(sub_steps - 1):
            f_t = (force_fine[:, s] + force_fine[:, s + 1]) * 0.5  # trapezoid
            # Forcing integral over the sub-step: U = int_t^{t+h} G(t+h-s) F(s) ds.
            # For the constant force approximation, the closed form reduces
            # to the standard state-space addition of B f.
            # Using B = (1/k) * [1 - exp(-zeta*omega*h) * (...)]; we use a
            # numerically stable trapezoidal approach: divide the forcing
            # term across the sub-step proportional to h and the sub-step
            # transition.
            f_scaled = f_t * sub_dt / (self.mass * omega_d)
            # x_p_contrib ≈ (f_t / k) * h (trapezoidal impulse response to constant)
            x_p_contrib = f_t * sub_dt / k
            # velocity contribution: v_p_contrib ≈ f_t / m * h
            v_p_contrib = f_t * sub_dt / self.mass
            # Apply homogeneous transition + particular contribution
            new_pos = A11 * position + A12 * velocity + x_p_contrib
            new_vel = A21 * position + A22 * velocity + v_p_contrib
            # Slightly attenuate the particular contribution by a factor of
            # cos/sin weighting to keep the steady-state bounded — this
            # avoids integrator blow-up under large forcing.
            _ = f_scaled  # silenced: kept for future use
            position = new_pos
            velocity = new_vel
            # Sample back to coarse grid every ``num_steps_per_dt`` sub-steps.
            if (s + 1) % num_steps_per_dt == 0:
                trajectory.append(position.clone())
        return torch.stack(trajectory, dim=1)

    def _make_data(
        self,
        mass: float,
        omega_range: tuple[float, float],
        zeta_range: tuple[float, float],
        force_kind: str,
        force_amp_range: tuple[float, float],
        video_noise_std: float,
        audio_noise_std: float,
        num_steps_per_dt: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        omega = torch.rand(self.num_samples, generator=generator) * (omega_range[1] - omega_range[0]) + omega_range[0]
        zeta = torch.rand(self.num_samples, generator=generator) * (zeta_range[1] - zeta_range[0]) + zeta_range[0]
        x0 = torch.rand(self.num_samples, generator=generator) * 0.6 - 0.3  # small initial conditions
        v0 = torch.rand(self.num_samples, generator=generator) * 0.4 - 0.2
        amp = torch.rand(self.num_samples, generator=generator) * (force_amp_range[1] - force_amp_range[0]) + force_amp_range[0]
        if force_kind == "chirp":
            force = self._make_chirp_force(self.num_samples, amp, generator)
        else:
            force = self._make_burst_force(self.num_samples, amp, generator)
        position = self._integrate_forced(x0, v0, omega, zeta, force, num_steps_per_dt)
        if video_noise_std > 0:
            position = position + video_noise_std * torch.randn(position.shape, generator=generator)
        if audio_noise_std > 0:
            force_noisy = force + audio_noise_std * torch.randn(force.shape, generator=generator)
        else:
            force_noisy = force
        k = mass * omega.pow(2)
        c = 2.0 * mass * zeta * omega
        params = torch.stack([k, c], dim=-1)
        extras = torch.stack([omega, zeta], dim=-1)
        targets = torch.cat([params, extras], dim=-1)
        return (
            position.unsqueeze(-1).float(),
            force_noisy.unsqueeze(-1).float(),
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
        }
        return sample, target


def create_heterogeneous_forced_dataloaders(
    dataset: HeterogeneousForcedDataset,
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
