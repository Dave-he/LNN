import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TimeSeriesDataset(Dataset):
    """
    Sliding-window dataset for time series prediction.

    Creates (input, target) pairs using a sliding window approach:
        input  = x[t : t + seq_len]
        target = x[t + seq_len : t + seq_len + horizon]
    """

    def __init__(
        self,
        data: np.ndarray | torch.Tensor,
        seq_len: int = 32,
        horizon: int = 1,
        stride: int = 1,
    ):
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)
        if data.dim() == 1:
            data = data.unsqueeze(-1)

        self.data = data
        self.seq_len = seq_len
        self.horizon = horizon
        self.stride = stride

        self.indices = list(range(0, len(data) - seq_len - horizon + 1, stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.indices[idx]
        x = self.data[start : start + self.seq_len]
        y = self.data[start + self.seq_len : start + self.seq_len + self.horizon]
        if self.horizon == 1:
            y = y.squeeze(0)
        return x, y


def create_dataloader(
    data: np.ndarray | torch.Tensor,
    seq_len: int = 32,
    horizon: int = 1,
    stride: int = 1,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    dataset = TimeSeriesDataset(data, seq_len=seq_len, horizon=horizon, stride=stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def generate_sine_data(
    num_samples: int = 1000,
    freq: float = 0.1,
    noise_std: float = 0.05,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(num_samples)
    data = np.sin(2 * np.pi * freq * t) + rng.normal(0, noise_std, num_samples)
    return data.astype(np.float32)


def generate_mackey_glass(
    num_samples: int = 1000,
    tau: int = 17,
    beta: float = 0.2,
    gamma: float = 0.1,
    n: int = 10,
    dt: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    history_len = tau + 1
    total_len = num_samples + history_len
    x = np.zeros(total_len, dtype=np.float32)
    x[:history_len] = 1.0 + 0.2 * rng.standard_normal(history_len)

    for i in range(history_len, total_len):
        x_delayed = x[i - tau]
        dxdt = beta * x_delayed / (1.0 + x_delayed**n) - gamma * x[i - 1]
        x[i] = x[i - 1] + dxdt * dt

    data = x[history_len:]
    data = (data - data.mean()) / (data.std() + 1e-8)
    return data


def generate_ood_sine(
    num_train: int = 1000,
    num_ood: int = 500,
    train_freq: float = 0.05,
    train_amp: float = 1.0,
    train_noise: float = 0.05,
    ood_freq_shift: float = 0.03,
    ood_amp_shift: float = 0.5,
    ood_noise_shift: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate in-distribution (train) and out-of-distribution (test) sine data.

    OOD shift is applied by changing frequency, amplitude, and noise level,
    simulating real-world distribution shift (e.g., different weather conditions,
    different market regimes).

    Returns:
        (train_data, ood_data) tuple of 1D numpy arrays
    """
    rng = np.random.default_rng(seed)
    t_train = np.arange(num_train)
    train_data = train_amp * np.sin(2 * np.pi * train_freq * t_train) + rng.normal(
        0, train_noise, num_train
    )

    t_ood = np.arange(num_ood)
    ood_data = (train_amp + ood_amp_shift) * np.sin(
        2 * np.pi * (train_freq + ood_freq_shift) * t_ood
    ) + rng.normal(0, train_noise + ood_noise_shift, num_ood)

    return train_data.astype(np.float32), ood_data.astype(np.float32)


def generate_concept_drift(
    num_samples: int = 2000,
    drift_point: int = 1000,
    freq_before: float = 0.05,
    freq_after: float = 0.12,
    amp_before: float = 1.0,
    amp_after: float = 0.6,
    noise_std: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate time series with concept drift (regime change).

    Before drift_point: low frequency, high amplitude (regime A)
    After drift_point:  high frequency, low amplitude (regime B)

    This simulates real-world scenarios like:
    - Market regime changes (bull -> bear)
    - Sensor degradation
    - Seasonal transitions

    Returns:
        (full_data, drift_labels) where drift_labels is 0 or 1 indicating regime
    """
    rng = np.random.default_rng(seed)
    t = np.arange(num_samples)

    data = np.zeros(num_samples, dtype=np.float32)
    labels = np.zeros(num_samples, dtype=np.float32)

    before = t < drift_point
    after = ~before

    data[before] = amp_before * np.sin(2 * np.pi * freq_before * t[before]) + rng.normal(
        0, noise_std, before.sum()
    )
    data[after] = amp_after * np.sin(2 * np.pi * freq_after * t[after]) + rng.normal(
        0, noise_std, after.sum()
    )
    labels[after] = 1.0

    return data, labels


def generate_lorenz(
    num_samples: int = 2000,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    dt: float = 0.01,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate Lorenz attractor time series (x-component).

    The Lorenz system is a classic chaotic system:
        dx/dt = sigma * (y - x)
        dy/dt = x * (rho - z) - y
        dz/dt = x * y - beta * z

    This provides a more challenging chaotic benchmark than Mackey-Glass.

    Returns:
        1D numpy array of shape (num_samples,) - normalized x-component
    """
    rng = np.random.default_rng(seed)
    x, y, z = 1.0 + rng.standard_normal(3) * 0.1

    data = np.zeros(num_samples, dtype=np.float32)
    for i in range(num_samples):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dx * dt
        y += dy * dt
        z += dz * dt
        data[i] = x

    data = (data - data.mean()) / (data.std() + 1e-8)
    return data
