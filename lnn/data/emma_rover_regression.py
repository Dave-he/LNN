"""EMMA rover real-data regression dataset.

The rover video is a single 4-second clip; we treat the 60-frame feature
stream as the *ground-truth trajectory* and synthesise additional samples
by adding Gaussian noise to each frame's features.  The regression
target is the rover's 5 known ground-truth parameters from EMMA paper
Table 4(c):

    a   (X-arm length)    = 0.178 m
    b   (Y-arm length)    = 0.144 m
    r   (wheel radius)     = 0.201 m
    m   (mass)             = 26.88 kg
    CM  (CoM height)       = 0.112 m

Because every sample is the same physical system, the *interesting*
benchmark is whether the model fits the trajectory consistently
(low val param MSE = good generalisation to noisy re-observations),
not whether it learns to discriminate systems.  The 5-dim parameter
vector is the target; we report MSE on it as ``param_mse`` (sum-of-
squared error across all 5 dims).
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from lnn.data.emma_rover_features import (
    DEFAULT_FPS,
    DEFAULT_SR,
    extract_rover_features,
)


# EMMA paper Table 4(c) - 5 ground-truth parameters for the differential-drive
# rover.  Used as the regression target for every sample.
EMMA_ROVER_GROUND_TRUTH = {
    "a": 0.178,   # X-arm length (m)
    "b": 0.144,   # Y-arm length (m)
    "r": 0.201,   # wheel radius (m)
    "m": 26.88,   # mass (kg)
    "CM": 0.112,  # CoM height (m)
}


def _build_augmented_features(
    video_path: str,
    cache_dir: str,
    cache_file: str,
    num_samples: int,
    window: int,
    feature_noise_std: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (video, audio, params) tensors of shapes ``[N, W, 3]``, ``[N, W]`` and ``[N, 5]``.

    The underlying 60-frame trajectory is extracted once and cached; we
    then draw ``num_samples`` random windows of size ``window`` and add
    per-frame Gaussian noise (``feature_noise_std``) to obtain an
    effectively-unlimited number of noisy re-observations of the same
    physical system.
    """
    if os.path.exists(cache_file):
        d = np.load(cache_file)
        video = d["video"]
        audio = d["audio"]
    else:
        feats = extract_rover_features(video_path, cache_dir)
        video = feats["video"]
        audio = feats["audio"]
        np.savez(cache_file, video=video, audio=audio)
    T = video.shape[0]
    if window > T:
        raise ValueError(f"window {window} exceeds trajectory length {T}")
    rng = np.random.default_rng(seed)
    gt = np.array([EMMA_ROVER_GROUND_TRUTH[k] for k in ("a", "b", "r", "m", "CM")], dtype=np.float32)
    starts = rng.integers(low=0, high=T - window + 1, size=num_samples)
    vid_out = np.zeros((num_samples, window, 3), dtype=np.float32)
    aud_out = np.zeros((num_samples, window), dtype=np.float32)
    for i, s in enumerate(starts):
        vid_out[i] = video[s : s + window] + feature_noise_std * rng.standard_normal((window, 3)).astype(np.float32)
        aud_out[i] = audio[s : s + window] + feature_noise_std * rng.standard_normal(window).astype(np.float32)
    params_out = np.broadcast_to(gt, (num_samples, 5)).copy()
    return vid_out, aud_out, params_out


class EmmaRoverRegressionDataset(Dataset):
    """EMMA rover video+audio regression dataset with sliding-window augmentation.

    Every sample targets the same 5-dim ground-truth parameter vector
    (the rover's known physical constants).  Samples differ only in
    which 16-frame window of the 60-frame trajectory they observe and
    what per-frame noise was added.
    """

    def __init__(
        self,
        num_samples: int = 200,
        window: int = 16,
        feature_noise_std: float = 0.02,
        seed: int = 42,
        video_path: str = "/tmp/RoverVideo.mp4",
        cache_dir: str = "/tmp/emma_features",
        cache_file: str = "/tmp/emma_features/features.npz",
        video_channels: tuple[int, ...] | None = None,
        audio_mode: str = "normal",
    ) -> None:
        if window < 4:
            raise ValueError("window must be >= 4")
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1")
        if video_channels is not None:
            video_channels = tuple(int(c) for c in video_channels)
            if not all(0 <= c < 3 for c in video_channels):
                raise ValueError(
                    f"video_channels must be a subset of {{0,1,2}}, got {video_channels}"
                )
            if len(video_channels) == 0:
                raise ValueError("video_channels must be non-empty")
        if audio_mode not in {"normal", "zero", "random", "lowpass"}:
            raise ValueError(
                f"audio_mode must be one of {{normal, zero, random, lowpass}}, "
                f"got {audio_mode!r}"
            )
        self.num_samples = num_samples
        self.window = window
        self.video_path = video_path
        self.video_channels = video_channels
        self.audio_mode = audio_mode
        self.video, self.audio, self.params = _build_augmented_features(
            video_path=video_path,
            cache_dir=cache_dir,
            cache_file=cache_file,
            num_samples=num_samples,
            window=window,
            feature_noise_std=feature_noise_std,
            seed=seed,
        )
        if video_channels is not None:
            # Slice along the channel axis ([N, W, 3] -> [N, W, len(video_channels)]).
            self.video = self.video[:, :, list(video_channels)]
        if audio_mode != "normal":
            self.audio = self._transform_audio(self.audio, audio_mode, seed)

    @staticmethod
    def _transform_audio(audio: np.ndarray, mode: str, seed: int) -> np.ndarray:
        """Round-16 symmetric audio ablation. Preserves shape; replaces content.

        - ``zero``    : replace with all zeros → no signal at all.
        - ``random``  : replace with i.i.d. Gaussian noise of the original
                        per-sample variance → preserves bandwidth and power
                        but destroys content correlation with motor RPM.
        - ``lowpass`` : keep only the per-sample mean (rolling DC) → preserves
                        the global scalar (mean motor RPM) the round-14
                        attention-viz analysis identified as the dominant
                        usable signal; destroys time-step variation.
        """
        rng = np.random.default_rng(seed + 1)
        if mode == "zero":
            return np.zeros_like(audio)
        if mode == "random":
            # Match per-sample std so the model can't trivially detect that
            # the random stream has different power than the normal one.
            std = audio.std(axis=-1, keepdims=True) + 1e-8
            return (rng.standard_normal(audio.shape).astype(np.float32) * std)
        if mode == "lowpass":
            # Per-sample mean broadcast — keeps the "global motor RPM scalar"
            # the round-14 attention visualization identified as the key.
            mean = audio.mean(axis=-1, keepdims=True)
            return np.broadcast_to(mean, audio.shape).astype(np.float32).copy()
        raise ValueError(f"unhandled audio_mode {mode!r}")

    @property
    def video_dim(self) -> int:
        return self.video.shape[-1]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        return (
            {
                "video": torch.from_numpy(self.video[index]),  # [W, 3]
                "audio": torch.from_numpy(self.audio[index]).unsqueeze(-1),  # [W, 1]
            },
            {
                "params": torch.from_numpy(self.params[index]),  # [5]
            },
        )


def create_emma_rover_dataloaders(
    dataset: EmmaRoverRegressionDataset,
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
