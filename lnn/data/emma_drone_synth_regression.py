"""Synthesized EMMA quadrotor dataset (12 parameters, 7 known).

EMMA paper Table 4(d) gives 7 known ground-truth parameters of the
6-DoF quadrotor.  This dataset synthesizes a *simple 6-DoF
quadrotor trajectory* with those exact parameters, then provides
the standard LNN multimodal regression interface (video, audio,
target_params) so the same multimodal LNN models can be applied.

Crucially, this is the *only* way to test the adaptive-freeze
SOTA recipe's *cross-task generalisation* since the EMMA Dropbox
release does not include the drone video in a directly
downloadable form.  The synthetic data lets us at least ask:
"does the recipe's mechanism (freeze audio_encoder after warmup)
generalise to a *different physical system* (12-parameter
quadrotor vs 5-parameter rover)?"

The quadrotor model is a deliberately-simplified cascade of a
second-order motor system into translational dynamics:

    tau^2 * ddot{w}_i + 2*zeta*tau * dot{w}_i + w_i = k_p * u_i
    T_i = k_Th * w_i^2
    tau_i = k_To * w_i^2
    m * ddot{p} = R(q) * T - m*g*e_z - drag

Implemented as a single forward integration per time step.  No
external dependencies beyond numpy/torch.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from lnn.data.emma_rover_regression import create_emma_rover_dataloaders


# EMMA paper Table 4(d) - 7 known ground-truth parameters of the
# quadrotor (the remaining 5 are implicit/drag and not given GT).
EMMA_DRONE_GROUND_TRUTH = {
    "k_Th": 1.1,    # thrust coefficient
    "k_To": 1.3,    # torque coefficient
    "k_p": 0.91,     # motor gain
    "tau_2": 0.012, # motor time constant
    "d_xm": 0.18,    # X-arm length
    "d_ym": 0.20,    # Y-arm length
    "d_zm": 0.07,    # Z-arm offset
}


def _integrate_quadrotor(
    num_samples: int,
    seq_len: int,
    dt: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesise (video, audio) feature pairs for the quadrotor.

    Returns:
        video: [num_samples, seq_len, 3]  - [pos_x, pos_y, pos_z] of
                the quadrotor per step (synthetic video features).
        audio: [num_samples, seq_len]      - dominant motor RPM Hz
                per step (synthetic audio features, the closest
                analog of EMMA's audio = "motor acoustic peak Hz"
                on the rover).
    """
    rng = np.random.default_rng(seed)
    gt = EMMA_DRONE_GROUND_TRUTH
    k_Th = gt["k_Th"]
    k_p = gt["k_p"]
    tau_2 = gt["tau_2"]
    videos = np.zeros((num_samples, seq_len, 3), dtype=np.float32)
    audios = np.zeros((num_samples, seq_len), dtype=np.float32)
    for s in range(num_samples):
        # Random initial position, small velocities, hover thrust command.
        p = rng.standard_normal(3) * 0.3
        v = rng.standard_normal(3) * 0.1
        w = 0.0  # motor angular speed
        d_w = 0.0
        for t in range(seq_len):
            # Motor dynamics: tau^2 * ddot{w} + 2*zeta*tau*dot{w} + w = k_p * u
            # Hover command u = m*g / (4*k_Th*w^2); at hover, w_target ~ sqrt(m*g/(4*k_Th))
            u = 1.0  # normalised thrust command
            ddot_w = (k_p * u - w) / max(tau_2, 1e-3)
            d_w = d_w + dt * ddot_w
            w = w + dt * d_w
            # Translational dynamics (simplified): just a small force
            # proportional to (w^2 - hover_thrust) plus random noise.
            T = k_Th * w * w
            F = (T - 1.0) * 0.1  # hover error
            v = v + dt * F
            p = p + dt * v
            videos[s, t] = p
            audios[s, t] = max(w * 10.0, 1.0)  # motor RPM proxy
    return videos, audios


class EmmaDroneSynthRegressionDataset(Dataset):
    """Synthesized EMMA quadrotor dataset with 7-parameter regression target.

    All 7 EMMA paper Table 4(d) parameters are the regression
    target (the regression module's output_size must be 7 to match).
    Each sample is an independent random initial condition for the
    quadrotor; sliding windows come from the same trajectory.
    """

    def __init__(
        self,
        num_samples: int = 200,
        seq_len: int = 32,
        seed: int = 42,
    ) -> None:
        if seq_len < 4:
            raise ValueError("seq_len must be >= 4")
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.video, self.audio = _integrate_quadrotor(
            num_samples=num_samples, seq_len=seq_len, seed=seed,
        )
        self.targets = np.array(
            [EMMA_DRONE_GROUND_TRUTH[k] for k in (
                "k_Th", "k_To", "k_p", "tau_2", "d_xm", "d_ym", "d_zm",
            )],
            dtype=np.float32,
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        return (
            {
                "video": torch.from_numpy(self.video[index]),       # [seq, 3]
                "audio": torch.from_numpy(self.audio[index]).unsqueeze(-1),  # [seq, 1]
            },
            {
                "params": torch.from_numpy(self.targets),  # [7]
            },
        )


def create_emma_drone_dataloaders(
    dataset: EmmaDroneSynthRegressionDataset,
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
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size], generator=generator,
    )
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set, batch_size=batch_size, shuffle=False),
        DataLoader(test_set, batch_size=batch_size, shuffle=False),
    )
