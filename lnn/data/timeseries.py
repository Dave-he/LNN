import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TimeSeriesDataset(Dataset):
    """
    Sliding-window dataset for time series prediction.

    Creates (input, target) pairs using a sliding window approach:
        input  = x[t : t + seq_len]
        target = x[t + seq_len : t + seq_len + horizon]

    Args:
        data: Time series data of shape (T,) or (T, F)
        seq_len: Input sequence length
        horizon: Prediction horizon
        stride: Step size between windows
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

        self.indices = list(
            range(0, len(data) - seq_len - horizon + 1, stride)
        )

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
    """
    Generate a noisy sine wave for testing.

    Args:
        num_samples: Number of time steps
        freq: Frequency of the sine wave
        noise_std: Standard deviation of Gaussian noise
        seed: Random seed

    Returns:
        1D numpy array of shape (num_samples,)
    """
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
    """
    Generate Mackey-Glass chaotic time series.

    The Mackey-Glass equation is a classic benchmark for time series
    prediction due to its chaotic dynamics:

        dx/dt = beta * x(t-tau) / (1 + x(t-tau)^n) - gamma * x(t)

    Args:
        num_samples: Number of time steps to generate
        tau: Delay parameter (larger = more chaotic)
        beta: Nonlinearity strength
        gamma: Decay rate
        n: Nonlinearity exponent
        dt: Integration time step
        seed: Random seed for initial conditions

    Returns:
        1D numpy array of shape (num_samples,)
    """
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
