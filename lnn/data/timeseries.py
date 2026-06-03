import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _as_float_tensor(values: np.ndarray | torch.Tensor, name: str) -> torch.Tensor:
    if isinstance(values, np.ndarray):
        values = torch.tensor(values, dtype=torch.float32)
    elif not torch.is_tensor(values):
        values = torch.tensor(values, dtype=torch.float32)
    else:
        values = values.to(dtype=torch.float32)
    if values.dim() == 0:
        raise ValueError(f"{name} must contain at least one time step")
    return values


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
        delta_t: np.ndarray | torch.Tensor | None = None,
        mask: np.ndarray | torch.Tensor | None = None,
        return_metadata: bool = False,
    ):
        data = _as_float_tensor(data, "data")
        if data.dim() == 1:
            data = data.unsqueeze(-1)
        if data.dim() != 2:
            raise ValueError(f"data must have shape [time] or [time, features], got {tuple(data.shape)}")

        inferred_mask = torch.isfinite(data).to(dtype=torch.float32)
        data = torch.nan_to_num(data)

        if mask is None:
            mask_tensor = inferred_mask
        else:
            mask_tensor = _as_float_tensor(mask, "mask")
            if mask_tensor.dim() == 1:
                mask_tensor = mask_tensor.unsqueeze(-1)
            if mask_tensor.shape[0] != data.shape[0]:
                raise ValueError(
                    f"mask time dimension must match data; got {tuple(mask_tensor.shape)} and {tuple(data.shape)}"
                )
            if mask_tensor.shape[1] == 1 and data.shape[1] != 1:
                mask_tensor = mask_tensor.expand(-1, data.shape[1])
            if mask_tensor.shape != data.shape:
                raise ValueError(
                    f"mask must have shape {tuple(data.shape)} or [time, 1], got {tuple(mask_tensor.shape)}"
                )
            mask_tensor = (mask_tensor > 0).to(dtype=torch.float32)

        if delta_t is None:
            delta_t_tensor = torch.ones(data.shape[0], 1, dtype=torch.float32)
        else:
            delta_t_tensor = _as_float_tensor(delta_t, "delta_t")
            if delta_t_tensor.dim() == 1:
                delta_t_tensor = delta_t_tensor.unsqueeze(-1)
            if delta_t_tensor.shape[0] != data.shape[0]:
                raise ValueError(
                    f"delta_t time dimension must match data; got {tuple(delta_t_tensor.shape)} and {tuple(data.shape)}"
                )
            if delta_t_tensor.shape[1] != 1:
                raise ValueError(f"delta_t must have shape [time] or [time, 1], got {tuple(delta_t_tensor.shape)}")
            delta_t_tensor = delta_t_tensor.clamp_min(0.0)

        self.data = data
        self.delta_t = delta_t_tensor
        self.mask = mask_tensor
        self.seq_len = seq_len
        self.horizon = horizon
        self.stride = stride
        self.return_metadata = return_metadata

        self.indices = list(range(0, len(data) - seq_len - horizon + 1, stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        start = self.indices[idx]
        x = self.data[start : start + self.seq_len]
        y = self.data[start + self.seq_len : start + self.seq_len + self.horizon]
        if self.horizon == 1:
            y = y.squeeze(0)
        if self.return_metadata:
            metadata = {
                "dt": self.delta_t[start : start + self.seq_len],
                "mask": self.mask[start : start + self.seq_len],
            }
            return x, y, metadata
        return x, y


def create_dataloader(
    data: np.ndarray | torch.Tensor,
    seq_len: int = 32,
    horizon: int = 1,
    stride: int = 1,
    batch_size: int = 32,
    shuffle: bool = True,
    delta_t: np.ndarray | torch.Tensor | None = None,
    mask: np.ndarray | torch.Tensor | None = None,
    return_metadata: bool | None = None,
) -> DataLoader:
    if return_metadata is None:
        return_metadata = delta_t is not None or mask is not None
    dataset = TimeSeriesDataset(
        data,
        seq_len=seq_len,
        horizon=horizon,
        stride=stride,
        delta_t=delta_t,
        mask=mask,
        return_metadata=return_metadata,
    )
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


def generate_gradual_multi_regime(
    num_samples: int = 2000,
    num_regimes: int = 4,
    transition_frac: float = 0.15,
    freq_range: tuple[float, float] = (0.04, 0.18),
    amp_range: tuple[float, float] = (0.5, 1.4),
    noise_std: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate time series with ``num_regimes`` regimes that **gradually**
    transition into each other.

    The single sharp drift used by :func:`generate_concept_drift` is a
    pessimistic case for any continuous-time model — when the dynamics jump
    in one step the model has zero data to interpolate from. This generator
    models the more realistic clinical-style non-stationarity the LiquidNN
    paper claims its advantage on:

    * sequence is split into ``num_regimes`` equal segments;
    * each segment is a sine with its own ``(freq, amp)`` drawn uniformly
      from the supplied ranges;
    * neighbouring segments are blended via a cosine ramp over a
      ``transition_frac`` window so signals overlap rather than jump.

    Returns:
        ``(data, regime_id)`` where ``regime_id[i]`` ∈ ``[0, num_regimes)``.
    """
    if num_regimes < 1:
        raise ValueError("num_regimes must be >= 1")
    if not 0.0 < transition_frac < 0.5:
        raise ValueError("transition_frac must be in (0, 0.5)")

    rng = np.random.default_rng(seed)
    segment_len = num_samples // num_regimes
    transition_len = max(1, int(segment_len * transition_frac))

    freqs = rng.uniform(freq_range[0], freq_range[1], size=num_regimes)
    amps = rng.uniform(amp_range[0], amp_range[1], size=num_regimes)
    phases = rng.uniform(0, 2 * np.pi, size=num_regimes)

    t = np.arange(num_samples)
    data = np.zeros(num_samples, dtype=np.float32)
    regime_id = np.zeros(num_samples, dtype=np.float32)

    for r in range(num_regimes):
        start = r * segment_len
        end = num_samples if r == num_regimes - 1 else (r + 1) * segment_len
        idx = np.arange(start, end)
        primary = amps[r] * np.sin(2 * np.pi * freqs[r] * t[idx] + phases[r])
        data[idx] = primary
        regime_id[idx] = float(r)

    # cosine-blend neighbouring regimes inside the transition window.
    for r in range(1, num_regimes):
        boundary = r * segment_len
        lo = max(0, boundary - transition_len)
        hi = min(num_samples, boundary + transition_len)
        win = np.arange(lo, hi)
        weight = 0.5 * (1.0 - np.cos(np.pi * (win - lo) / max(hi - lo - 1, 1)))
        # blend from regime r-1 (computed for segment_len start) into r.
        prev_signal = amps[r - 1] * np.sin(2 * np.pi * freqs[r - 1] * t[win] + phases[r - 1])
        next_signal = amps[r] * np.sin(2 * np.pi * freqs[r] * t[win] + phases[r])
        data[win] = (1.0 - weight) * prev_signal + weight * next_signal
        # regime label fractionally tracks the blend so labels are continuous.
        regime_id[win] = (r - 1) + weight

    data = data + rng.normal(0, noise_std, num_samples).astype(np.float32)
    return data.astype(np.float32), regime_id


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
