"""EMMA rover temporal-fold LOO dataset.

EMMA's Dropbox release only contains a single 4-second rover video,
so we cannot do *real* leave-one-video-out.  Instead we simulate
"different trajectory segments" by partitioning the single 60-frame
trajectory into 4 *temporal folds* of 15 frames each:

    fold 0: frames  0-14
    fold 1: frames 15-29
    fold 2: frames 30-44
    fold 3: frames 45-59

For LOO, the test split draws all windows whose midpoint falls in the
held-out fold; training uses windows with midpoints in the other
folds.  This is a meaningful cross-segment generalization test
(e.g. "does a model trained on early video segments work on later
ones?").

The SOTA recipe (h=64, ep=80, K=40, freeze=audio_only) is the
reference; the LOO test asks whether the recipe's MSE 0.31 is
robust across folds or whether some folds regress badly.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from lnn.data.emma_rover_features import extract_rover_features
from lnn.data.emma_rover_regression import (
    EMMA_ROVER_GROUND_TRUTH,
    EmmaRoverRegressionDataset,
    create_emma_rover_dataloaders,
)


N_FOLDS = 4
WINDOW = 16  # keep the same window as the round-26 SOTA recipe


def _build_window_fold_assignment(num_windows: int, num_folds: int = N_FOLDS) -> np.ndarray:
    """Assign each window index to a fold by its *midpoint* frame.

    Returns an array of length ``num_windows`` with values in
    ``[0, num_folds)`` indicating the fold of each window.
    """
    fold_size = 60 / num_folds
    mids = np.arange(num_windows) + (WINDOW - 1) / 2.0
    return (mids / fold_size).astype(int).clip(0, num_folds - 1)


def temporal_fold_indices(dataset: EmmaRoverRegressionDataset) -> Tuple[np.ndarray, int]:
    """Return (fold_assignments, n_folds) for a pre-built dataset.

    The dataset was built by sliding windows across the 60-frame
    trajectory with stride 1, so window ``i`` has midpoint at
    frame ``i + (WINDOW-1)/2`` of the underlying video.
    """
    return _build_window_fold_assignment(num_windows=dataset.num_samples), N_FOLDS


def create_loo_dataloaders(
    dataset: EmmaRoverRegressionDataset,
    held_out_fold: int,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader]:
    """Build train/val dataloaders for a single LOO fold.

    Train uses windows whose fold != held_out_fold.  Test is the
    held-out fold.  No val split inside the train partition (we
    trust the SOTA recipe's hyperparameters).
    """
    fold_assignments, _ = temporal_fold_indices(dataset)
    train_indices = [i for i, f in enumerate(fold_assignments) if f != held_out_fold]
    test_indices = [i for i, f in enumerate(fold_assignments) if f == held_out_fold]
    train_set = Subset(dataset, train_indices)
    test_set = Subset(dataset, test_indices)
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(test_set, batch_size=batch_size, shuffle=False),
    )


class TemporalSegmentRegressionDataset(Dataset):
    """One window drawn from a single 15-frame temporal segment.

    The EMMA rover trajectory is partitioned into 4 disjoint 15-frame
    segments.  Each window is the full 16-frame run starting at the
    segment's first frame.  Different segments are non-overlapping so
    LOO is a clean cross-segment test (model trained on 3 segments
    is tested on the 4th).

    Compared with the random-window :class:`EmmaRoverRegressionDataset`
    used for the SOTA recipe (which mixes all frames), this dataset
    is *segment-pure* and a stricter LOO test.
    """

    def __init__(
        self,
        cache_file: str = "/tmp/emma_features/features.npz",
        feature_noise_std: float = 0.02,
        seed: int = 42,
        audio_mode: str = "normal",
    ) -> None:
        import os
        if audio_mode not in {"normal", "zero", "random", "lowpass"}:
            raise ValueError(f"audio_mode must be one of normal/zero/random/lowpass, got {audio_mode!r}")
        if os.path.exists(cache_file):
            d = np.load(cache_file)
            video = d["video"]  # [60, 3]
            audio = d["audio"]  # [60]
        else:
            feats = extract_rover_features("/tmp/RoverVideo.mp4", "/tmp/emma_features")
            video = feats["video"]
            audio = feats["audio"]
        # Round 35 audio_mode transformation, applied to the FULL 60-frame
        # audio trace before slicing into segments. Mirrors round 16's
        # EmmaRoverRegressionDataset.audio_mode semantics so LOO can also
        # explore audio ablations.
        if audio_mode == "zero":
            audio = np.zeros_like(audio)
        elif audio_mode == "random":
            rng_audio = np.random.default_rng(seed + 12345)
            std = audio.std() + 1e-8
            audio = (rng_audio.standard_normal(audio.shape).astype(np.float32) * std)
        elif audio_mode == "lowpass":
            # broadcast per-array mean (single scalar for whole 60-frame trace)
            audio = np.full_like(audio, audio.mean())
        # 4 segments, each 15 frames.  We pad to 16 (window size) by
        # copying the last frame so window = whole segment.
        segs_video = []
        segs_audio = []
        for k in range(4):
            seg_v = video[k * 15 : (k + 1) * 15]  # [15, 3]
            seg_a = audio[k * 15 : (k + 1) * 15]  # [15]
            # pad to 16 by repeating the last frame
            seg_v = np.concatenate([seg_v, seg_v[-1:]], axis=0)  # [16, 3]
            seg_a = np.concatenate([seg_a, seg_a[-1:]], axis=0)  # [16]
            segs_video.append(seg_v)
            segs_audio.append(seg_a)
        self.segments_video = np.stack(segs_video)  # [4, 16, 3]
        self.segments_audio = np.stack(segs_audio)  # [4, 16]
        self.feature_noise_std = feature_noise_std
        self.seed = seed
        self.audio_mode = audio_mode
        self.fold_assignments = np.array([0, 1, 2, 3])

    def __len__(self) -> int:
        return 4

    def get_fold(self, fold: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (video, audio, params) for one segment, with optional noise."""
        rng = np.random.default_rng(self.seed + fold)
        v = self.segments_video[fold] + self.feature_noise_std * rng.standard_normal(self.segments_video[fold].shape).astype(np.float32)
        a = self.segments_audio[fold] + self.feature_noise_std * rng.standard_normal(self.segments_audio[fold].shape).astype(np.float32)
        gt = np.array([EMMA_ROVER_GROUND_TRUTH[k] for k in ("a", "b", "r", "m", "CM")], dtype=np.float32)
        return v, a, gt

    def __getitem__(self, index: int) -> Tuple[dict, dict]:
        v, a, gt = self.get_fold(index)
        return (
            {
                "video": torch.from_numpy(v),       # [16, 3]
                "audio": torch.from_numpy(a).unsqueeze(-1),  # [16, 1]
            },
            {
                "params": torch.from_numpy(gt),  # [5]
            },
        )


def create_segment_loo_dataloaders(
    dataset: TemporalSegmentRegressionDataset,
    held_out_fold: int,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader]:
    """Train = all but held_out segment; test = held_out segment.

    Note: 4 segments, batch_size=32 -> the train dataloader has 3
    samples (a batch of 3).  We pad the train side with a small
    random-noise repeat so the optimizer can run >1 step per epoch.
    """
    train_indices = [i for i in range(4) if i != held_out_fold]
    test_indices = [held_out_fold]
    # Repeat train indices 8x so the dataloader has 24 samples -> 3 batches
    # of 8 (avoids the trivial batch_size=3 issue).
    train_indices = train_indices * 8
    train_set = Subset(dataset, train_indices)
    test_set = Subset(dataset, test_indices)
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(test_set, batch_size=batch_size, shuffle=False),
    )
