"""Falsifiable bench: Bi-CfC-NAD vs Uni-CfC-NAD on windowed-median regression.

The target at time t is the median of ``x[t-k : t+k+1]`` (centred window). Such
a target requires lookahead — a unidirectional model cannot know the right side
of the window. The falsifiable claim is that bi-CfC-NAD should beat uni-CfC-NAD
on validation MSE by at least 25%.

Usage:
    python scripts/benchmark_bi_cfc_nad.py --epochs 6 --hidden 16
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from lnn.core.noise_adaptive_cfc import (
    BidirectionalNoiseAdaptiveCfC,
    NoiseAdaptiveCfCNetwork,
)
from lnn.data.timeseries import generate_mackey_glass


def build_windowed_median(num_samples: int, seq_len: int, window_half: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate (x, y) pairs where y[t] is the median of x[t-k:t+k+1].

    Both x and y are returned with shape ``[num_samples, seq_len, 1]``. Blocks
    use a stride of ``seq_len // 4`` (overlapping) so a moderate raw series
    produces enough training samples.
    """

    # Generate a raw series long enough to fit ``num_samples`` overlapping blocks
    # of length ``seq_len`` with a stride of seq_len // 4.
    stride = max(1, seq_len // 4)
    raw_len = num_samples * stride + seq_len + 2 * window_half
    series = generate_mackey_glass(num_samples=raw_len).astype(np.float32)
    series = (series - series.mean()) / (series.std() + 1e-8)
    targets = np.zeros_like(series)
    n = len(series)
    for t in range(n):
        lo = max(0, t - window_half)
        hi = min(n, t + window_half + 1)
        targets[t] = float(np.median(series[lo:hi]))

    blocks_x: list[np.ndarray] = []
    blocks_y: list[np.ndarray] = []
    start = 0
    while start + seq_len <= n and len(blocks_x) < num_samples:
        blocks_x.append(series[start : start + seq_len])
        blocks_y.append(targets[start : start + seq_len])
        start += stride
    x = torch.tensor(np.stack(blocks_x), dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(np.stack(blocks_y), dtype=torch.float32).unsqueeze(-1)
    return x, y


def make_loaders(x: torch.Tensor, y: torch.Tensor, batch_size: int, val_split: float = 0.2) -> tuple[DataLoader, DataLoader]:
    cut = int(len(x) * (1 - val_split))
    train_ds = TensorDataset(x[:cut], y[:cut])
    val_ds = TensorDataset(x[cut:], y[cut:])
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


def train_and_eval(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> tuple[float, float, float]:
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
    train_seconds = time.time() - t0

    model.eval()
    total_sq, count = 0.0, 0
    t1 = time.time()
    infer_steps = 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            total_sq += float(((pred - y) ** 2).sum())
            count += y.numel()
            infer_steps += x.shape[0] * x.shape[1]
    infer_time = time.time() - t1
    return total_sq / max(count, 1), train_seconds, (infer_time / max(infer_steps, 1)) * 1e6


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=400)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--window-half", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x, y = build_windowed_median(args.num_samples, args.seq_len, args.window_half)
    print(f"# data: x={tuple(x.shape)} y={tuple(y.shape)} window_half={args.window_half}")
    train_loader, val_loader = make_loaders(x, y, args.batch_size)

    torch.manual_seed(args.seed)
    uni = NoiseAdaptiveCfCNetwork(
        input_size=1, hidden_size=args.hidden, output_size=1, num_layers=1, return_sequences=True
    )
    uni_mse, uni_train_s, uni_infer = train_and_eval(uni, train_loader, val_loader, args.epochs, args.lr, device)

    torch.manual_seed(args.seed)
    bi = BidirectionalNoiseAdaptiveCfC(
        input_size=1, hidden_size=args.hidden, output_size=1, num_layers=1, return_sequences=True
    )
    bi_mse, bi_train_s, bi_infer = train_and_eval(bi, train_loader, val_loader, args.epochs, args.lr, device)

    print()
    print(f"  Uni-CfC-NAD : params={count_params(uni):>5}  val_mse={uni_mse:.5f}  "
          f"train_s={uni_train_s:.2f}  infer={uni_infer:.2f} µs/step")
    print(f"  Bi-CfC-NAD  : params={count_params(bi):>5}  val_mse={bi_mse:.5f}  "
          f"train_s={bi_train_s:.2f}  infer={bi_infer:.2f} µs/step")
    drop = (1.0 - bi_mse / uni_mse) * 100
    claim = drop >= 25.0
    print(f"\n  Bi vs Uni val MSE drop: {drop:+.1f}% "
          f"(falsifiable claim ≥25%): {'PASS' if claim else 'FAIL'}")

    out_dir = Path("analysis/cfc_nad")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date.today().isoformat()}_bi_cfc_nad_benchmark.json"
    payload = {
        "config": vars(args),
        "uni": {
            "params": count_params(uni),
            "val_mse": uni_mse,
            "train_s": uni_train_s,
            "infer_us_per_step": uni_infer,
        },
        "bi": {
            "params": count_params(bi),
            "val_mse": bi_mse,
            "train_s": bi_train_s,
            "infer_us_per_step": bi_infer,
        },
        "summary": {
            "mse_drop_pct": drop,
            "claim_threshold_pct": 25.0,
            "claim_passed": bool(claim),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nJSON written to {out_path}")


if __name__ == "__main__":
    main()
