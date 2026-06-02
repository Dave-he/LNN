"""Falsifiable benchmark: centered vs independent noise EMA in Bi-CfC-NAD.

Compares three configurations on a noisy windowed-median regression task:
    Uni-CfC-NAD                            (single direction, causal noise EMA)
    Bi-CfC-NAD noise_aggregation=indep     (each branch keeps its own EMA)
    Bi-CfC-NAD noise_aggregation=centered  (shared non-causal EMA)

Falsifiable claim:
    Under AWGN at SNR 20 dB, bi_centered val MSE <= bi_indep val MSE * 0.90.
    (centered noise should yield >=10% MSE reduction by exploiting both past
    and future noise context.)

Usage:
    python scripts/benchmark_bi_cfc_nad_centered.py --epochs 8 --hidden 16
"""

from __future__ import annotations

import argparse
import json
import math
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


def add_awgn(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    power = float(np.mean(signal**2))
    noise_power = power / (10.0 ** (snr_db / 10.0))
    noise = rng.standard_normal(signal.shape) * math.sqrt(noise_power)
    return signal + noise.astype(signal.dtype)


def build_noisy_windowed_median(
    num_samples: int, seq_len: int, window_half: int, snr_db: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    stride = max(1, seq_len // 4)
    raw_len = num_samples * stride + seq_len + 2 * window_half
    clean = generate_mackey_glass(num_samples=raw_len).astype(np.float32)
    clean = (clean - clean.mean()) / (clean.std() + 1e-8)
    # Targets computed from the CLEAN signal so noise robustness is what we
    # actually probe (model gets noisy input, must recover the clean median).
    targets = np.zeros_like(clean)
    n = len(clean)
    for t in range(n):
        lo = max(0, t - window_half)
        hi = min(n, t + window_half + 1)
        targets[t] = float(np.median(clean[lo:hi]))
    noisy = add_awgn(clean, snr_db, rng)

    blocks_x: list[np.ndarray] = []
    blocks_y: list[np.ndarray] = []
    start = 0
    while start + seq_len <= n and len(blocks_x) < num_samples:
        blocks_x.append(noisy[start : start + seq_len])
        blocks_y.append(targets[start : start + seq_len])
        start += stride
    x = torch.tensor(np.stack(blocks_x), dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(np.stack(blocks_y), dtype=torch.float32).unsqueeze(-1)
    return x, y


def make_loaders(x: torch.Tensor, y: torch.Tensor, batch_size: int, val_split: float = 0.2):
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
    parser.add_argument("--snr-db", type=float, default=20.0)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x, y = build_noisy_windowed_median(
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        window_half=args.window_half,
        snr_db=args.snr_db,
        seed=args.seed,
    )
    train_loader, val_loader = make_loaders(x, y, args.batch_size)
    print(
        f"# data: x={tuple(x.shape)} y={tuple(y.shape)} "
        f"window_half={args.window_half} snr_db={args.snr_db}"
    )

    results: dict[str, dict[str, float]] = {}

    torch.manual_seed(args.seed)
    uni = NoiseAdaptiveCfCNetwork(
        input_size=1,
        hidden_size=args.hidden,
        output_size=1,
        num_layers=1,
        return_sequences=True,
    )
    uni_mse, uni_train_s, uni_infer = train_and_eval(
        uni, train_loader, val_loader, args.epochs, args.lr, device
    )
    results["uni"] = {
        "params": count_params(uni),
        "val_mse": uni_mse,
        "train_s": uni_train_s,
        "infer_us_per_step": uni_infer,
    }
    print(f"  Uni-CfC-NAD       : params={results['uni']['params']:>5}  "
          f"val_mse={uni_mse:.5f}  train_s={uni_train_s:.2f}  "
          f"infer={uni_infer:.2f} µs/step")

    torch.manual_seed(args.seed)
    bi_indep = BidirectionalNoiseAdaptiveCfC(
        input_size=1,
        hidden_size=args.hidden,
        output_size=1,
        num_layers=1,
        return_sequences=True,
        noise_aggregation="independent",
    )
    bi_indep_mse, bi_indep_train_s, bi_indep_infer = train_and_eval(
        bi_indep, train_loader, val_loader, args.epochs, args.lr, device
    )
    results["bi_indep"] = {
        "params": count_params(bi_indep),
        "val_mse": bi_indep_mse,
        "train_s": bi_indep_train_s,
        "infer_us_per_step": bi_indep_infer,
    }
    print(f"  Bi-CfC-NAD indep  : params={results['bi_indep']['params']:>5}  "
          f"val_mse={bi_indep_mse:.5f}  train_s={bi_indep_train_s:.2f}  "
          f"infer={bi_indep_infer:.2f} µs/step")

    torch.manual_seed(args.seed)
    bi_centered = BidirectionalNoiseAdaptiveCfC(
        input_size=1,
        hidden_size=args.hidden,
        output_size=1,
        num_layers=1,
        return_sequences=True,
        noise_aggregation="centered",
    )
    bi_cen_mse, bi_cen_train_s, bi_cen_infer = train_and_eval(
        bi_centered, train_loader, val_loader, args.epochs, args.lr, device
    )
    results["bi_centered"] = {
        "params": count_params(bi_centered),
        "val_mse": bi_cen_mse,
        "train_s": bi_cen_train_s,
        "infer_us_per_step": bi_cen_infer,
    }
    print(f"  Bi-CfC-NAD center : params={results['bi_centered']['params']:>5}  "
          f"val_mse={bi_cen_mse:.5f}  train_s={bi_cen_train_s:.2f}  "
          f"infer={bi_cen_infer:.2f} µs/step")

    drop_centered_vs_indep = (1.0 - bi_cen_mse / bi_indep_mse) * 100
    drop_bi_vs_uni = (1.0 - bi_indep_mse / uni_mse) * 100
    drop_centered_vs_uni = (1.0 - bi_cen_mse / uni_mse) * 100
    claim_passed = drop_centered_vs_indep >= 10.0
    print()
    print(f"  Bi-indep vs Uni                  : {drop_bi_vs_uni:+.1f}% MSE drop")
    print(f"  Bi-centered vs Bi-indep          : {drop_centered_vs_indep:+.1f}% MSE drop "
          f"(falsifiable claim ≥10%): {'PASS' if claim_passed else 'FAIL'}")
    print(f"  Bi-centered vs Uni (cumulative)  : {drop_centered_vs_uni:+.1f}% MSE drop")

    out_dir = Path("analysis/cfc_nad")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date.today().isoformat()}_bi_centered_noise_benchmark.json"
    payload = {
        "config": vars(args),
        "results": results,
        "summary": {
            "drop_bi_indep_vs_uni_pct": drop_bi_vs_uni,
            "drop_centered_vs_indep_pct": drop_centered_vs_indep,
            "drop_centered_vs_uni_pct": drop_centered_vs_uni,
            "claim_threshold_pct": 10.0,
            "claim_passed": bool(claim_passed),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nJSON written to {out_path}")


if __name__ == "__main__":
    main()
