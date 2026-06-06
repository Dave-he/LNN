"""Head-to-head benchmarking harness for LNN/SSM backbones (round 73).

Provides a unified interface to train and evaluate four backbones on the
same datasets, with the same seeds, so the numbers are directly
comparable.

Supported backbones (canonical, each lives in its own module):
    - "cfc"   — CfCNetwork           (lnn/core/cfc.py)
    - "gru"   — vanilla PyTorch nn.GRU + linear head
    - "dss"   — DiagonalSSMNetwork   (lnn/core/dss_cell.py)
    - "mamba" — SelectiveScanMambaNetwork (lnn/core/mamba_simple.py)

Supported datasets (round 73 subset, no external download required):
    - "mackey_glass" — chaotic 1-D time series, regression target
    - "sine"         — synthetic sine + noise, regression target
    - "toy_class"    — 2-D Gaussian mixture classification toy

sMNIST / permuted-MNIST / seq-CIFAR are deferred to round 74 (they
need either torchvision download or a pre-cached .pt file and a wider
training budget; we get the harness right first, then plug in real
benchmarks in the next round).

Usage
-----
>>> from lnn.core.bench_suite import run_suite, list_backbones, list_datasets
>>> list_backbones()  # ['cfc', 'dss', 'gru', 'mamba']
>>> list_datasets()   # ['mackey_glass', 'sine', 'toy_class']
>>> result = run_suite("cfc", "mackey_glass", seed=0, hidden=32, epochs=5)
>>> result.test_loss
0.0123
>>> result.to_dict()
{'backbone': 'cfc', 'dataset': 'mackey_glass', 'seed': 0, 'hidden': 32,
 'epochs': 5, 'train_loss': ..., 'test_loss': ..., 'test_acc_or_mse': ...,
 'n_params': ..., 'wall_clock_s': ...}
"""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from lnn.core.cfc import CfCNetwork
from lnn.core.dss_cell import DiagonalSSMNetwork
from lnn.core.mamba_simple import SelectiveScanMambaNetwork


# --- Backbone registry --------------------------------------------------------

SUPPORTED_BACKBONES = ("cfc", "gru", "dss", "mamba")
SUPPORTED_DATASETS = ("mackey_glass", "sine", "toy_class")


@dataclass
class BenchResult:
    """One training run result. JSON-serialisable via ``to_dict()``."""

    backbone: str
    dataset: str
    seed: int
    hidden: int
    epochs: int
    train_loss: float
    test_loss: float
    test_metric_name: str  # 'mse' for regression, 'acc' for classification
    test_metric_value: float
    n_params: int
    wall_clock_s: float
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def list_backbones() -> tuple[str, ...]:
    return SUPPORTED_BACKBONES


def list_datasets() -> tuple[str, ...]:
    return SUPPORTED_DATASETS


# --- Model factory ------------------------------------------------------------


def _build_gru(input_size: int, hidden_size: int, output_size: int) -> nn.Module:
    """A vanilla nn.GRU + linear head, mirroring the public surface of
    CfCNetwork / Mamba / DSS networks. We use bidirectional=False for
    parity with the other unidirectional cells."""

    class GRUNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=False,
            )
            self.head = nn.Linear(hidden_size, output_size)
            self.return_sequences = True  # default; run_suite may override

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.gru(x)
            if not self.return_sequences:
                out = out[:, -1, :]
            return self.head(out)

    return GRUNetwork()


def _build_model(backbone: str, input_size: int, hidden_size: int, output_size: int) -> nn.Module:
    if backbone == "cfc":
        return CfCNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=1,
            return_sequences=True,
        )
    if backbone == "gru":
        return _build_gru(input_size, hidden_size, output_size)
    if backbone == "dss":
        return DiagonalSSMNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=1,
            return_sequences=True,
        )
    if backbone == "mamba":
        return SelectiveScanMambaNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=1,
            return_sequences=True,
        )
    raise ValueError(
        f"Unknown backbone {backbone!r}. Supported: {list(SUPPORTED_BACKBONES)}"
    )


# --- Dataset loaders ----------------------------------------------------------


@dataclass
class DatasetBundle:
    """Train / test tensors in (X, y) format. y is 1-D for regression
    and 1-D class index for classification."""

    X_train: torch.Tensor  # (N_train, T, F)
    y_train: torch.Tensor  # (N_train,) or (N_train, 1)
    X_test: torch.Tensor
    y_test: torch.Tensor
    num_classes: int  # 1 for regression
    task: str  # 'regression' or 'classification'
    feature_size: int
    seq_len: int
    name: str


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_mackey_glass(seed: int) -> DatasetBundle:
    """Mackey-Glass next-step prediction: predict x[t+1] from x[t-T+1:t]."""
    from lnn.data.timeseries import generate_mackey_glass

    _seed_everything(seed)
    series = generate_mackey_glass(num_samples=2000, tau=17, seed=seed)
    series = (series - series.mean()) / (series.std() + 1e-8)

    T = 100  # input window length
    X_list, y_list = [], []
    for i in range(len(series) - T):
        X_list.append(series[i : i + T])
        y_list.append(series[i + T])
    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.float32)

    # Train / test split 80/20.
    n_train = int(0.8 * len(X))
    X_train = torch.from_numpy(X[:n_train]).unsqueeze(-1).float()  # (N, T, 1)
    y_train = torch.from_numpy(y[:n_train]).float()
    X_test = torch.from_numpy(X[n_train:]).unsqueeze(-1).float()
    y_test = torch.from_numpy(y[n_train:]).float()
    return DatasetBundle(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        num_classes=1,
        task="regression",
        feature_size=1,
        seq_len=T,
        name="mackey_glass",
    )


def _load_sine(seed: int) -> DatasetBundle:
    """Sine-wave next-step prediction."""
    from lnn.data.timeseries import generate_sine_data

    _seed_everything(seed)
    series = generate_sine_data(num_samples=2000, freq=0.05, noise_std=0.1, seed=seed)
    series = (series - series.mean()) / (series.std() + 1e-8)

    T = 50
    X_list, y_list = [], []
    for i in range(len(series) - T):
        X_list.append(series[i : i + T])
        y_list.append(series[i + T])
    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.float32)

    n_train = int(0.8 * len(X))
    return DatasetBundle(
        X_train=torch.from_numpy(X[:n_train]).unsqueeze(-1).float(),
        y_train=torch.from_numpy(y[:n_train]).float(),
        X_test=torch.from_numpy(X[n_train:]).unsqueeze(-1).float(),
        y_test=torch.from_numpy(y[n_train:]).float(),
        num_classes=1,
        task="regression",
        feature_size=1,
        seq_len=T,
        name="sine",
    )


def _load_toy_class(seed: int) -> DatasetBundle:
    """2-class 2-D Gaussian-mixture sequence classification.

    Each sample is a (T, 2) trajectory of a noisy 2-D random walk, with
    the class label determined by which mixture the drift is sampled
    from. This is small enough to run in seconds but non-trivial
    because the class is only determined by the *integrated* drift, not
    any single timestep — recurrent cells are needed.
    """
    _seed_everything(seed)
    rng = np.random.default_rng(seed)
    n_per_class = 500
    T = 32
    D = 2
    n_classes = 2
    X = np.zeros((n_per_class * n_classes, T, D), dtype=np.float32)
    y = np.zeros(n_per_class * n_classes, dtype=np.int64)
    for cls in range(n_classes):
        drift = rng.normal(loc=cls * 1.5, scale=0.5, size=(1, D)).astype(np.float32)
        for i in range(n_per_class):
            noise = rng.normal(scale=0.5, size=(T, D)).astype(np.float32)
            X[cls * n_per_class + i] = np.cumsum(noise + drift, axis=0)
            y[cls * n_per_class + i] = cls
    n_train = int(0.8 * len(X))
    return DatasetBundle(
        X_train=torch.from_numpy(X[:n_train]),
        y_train=torch.from_numpy(y[:n_train]),
        X_test=torch.from_numpy(X[n_train:]),
        y_test=torch.from_numpy(y[n_train:]),
        num_classes=n_classes,
        task="classification",
        feature_size=D,
        seq_len=T,
        name="toy_class",
    )


_LOADERS: dict[str, Callable[[int], DatasetBundle]] = {
    "mackey_glass": _load_mackey_glass,
    "sine": _load_sine,
    "toy_class": _load_toy_class,
}


def load_dataset(name: str, seed: int) -> DatasetBundle:
    if name not in _LOADERS:
        raise ValueError(
            f"Unknown dataset {name!r}. Supported: {list(SUPPORTED_DATASETS)}"
        )
    return _LOADERS[name](seed)


# --- Training loop ------------------------------------------------------------


def _align_y_for_loss(y: torch.Tensor, task: str) -> torch.Tensor:
    """CrossEntropyLoss wants 1-D class indices; MSELoss wants (N, 1)."""
    if task == "classification":
        return y.long()  # 1-D
    # regression
    if y.dim() == 1:
        return y.unsqueeze(-1)
    return y.float()


def _train_one_epoch(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    task: str,
    batch_size: int = 32,
) -> float:
    model.train()
    perm = torch.randperm(len(X))
    total_loss = 0.0
    n_batches = 0
    for i in range(0, len(X), batch_size):
        idx = perm[i : i + batch_size]
        xb, yb = X[idx], y[idx]
        optimizer.zero_grad()
        # Models return (B, T, output_size). We use the last step.
        out = model(xb)[:, -1, :]
        yb_loss = _align_y_for_loss(yb, task)
        loss = loss_fn(out, yb_loss)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    loss_fn: Callable,
    task: str,
) -> tuple[float, str, float]:
    model.eval()
    out = model(X)[:, -1, :]
    y_loss = _align_y_for_loss(y, task)
    loss = float(loss_fn(out, y_loss).item())
    if task == "classification":
        preds = out.argmax(dim=-1)
        acc = float((preds == y.long()).float().mean().item())
        return loss, "acc", acc
    # regression: report MSE as the metric
    mse = float(((out.squeeze(-1) - y.float()) ** 2).mean().item())
    return loss, "mse", mse


def run_suite(
    backbone: str,
    dataset: str,
    *,
    seed: int = 0,
    hidden: int = 32,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-2,
    verbose: bool = False,
) -> BenchResult:
    """Train ``backbone`` on ``dataset`` with the given seed + hidden size,
    return a ``BenchResult``.

    The training loop is intentionally short (5 epochs by default) so
    a full 4-backbone × 3-dataset × 3-seed sweep finishes in minutes
    on a CPU. This is enough to expose the relative trends; for
    publication-quality numbers, bump ``epochs``.
    """
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unknown backbone {backbone!r}")
    _seed_everything(seed)

    data = load_dataset(dataset, seed)
    model = _build_model(backbone, data.feature_size, hidden, data.num_classes)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if data.task == "classification":
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    t0 = time.time()
    for epoch in range(epochs):
        train_loss = _train_one_epoch(
            model,
            data.X_train,
            data.y_train,
            optimizer,
            loss_fn,
            data.task,
            batch_size=batch_size,
        )
        if verbose:
            print(f"  epoch {epoch + 1}/{epochs}  train_loss={train_loss:.4f}")
    elapsed = time.time() - t0

    test_loss, metric_name, metric_val = _evaluate(
        model, data.X_test, data.y_test, loss_fn, data.task
    )
    return BenchResult(
        backbone=backbone,
        dataset=dataset,
        seed=seed,
        hidden=hidden,
        epochs=epochs,
        train_loss=train_loss,
        test_loss=test_loss,
        test_metric_name=metric_name,
        test_metric_value=metric_val,
        n_params=n_params,
        wall_clock_s=elapsed,
        notes=(
            f"task={data.task}  T={data.seq_len}  F={data.feature_size}  "
            f"lr={lr}  batch={batch_size}"
        ),
    )
