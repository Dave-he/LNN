"""Falsifiable benchmark: CfC-NAD + MDN head produces calibrated heteroscedastic
uncertainty.

Each training sample is generated with its own SNR drawn uniformly from
[snr_low, snr_high] dB. Targets are the *clean* windowed median, inputs are
noisy. The model is trained with the MDN negative log-likelihood loss.

Falsifiable claim: on the held-out split, the Pearson correlation between the
per-sample mean predicted std and the per-sample ground-truth noise std must
satisfy ``r >= 0.5``.

Usage:
    python scripts/benchmark_cfcnad_mdn_uncertainty.py --epochs 16 --hidden 16
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
from torch.utils.data import DataLoader, TensorDataset

from lnn.core.mdn import mdn_negative_log_likelihood
from lnn.core.noise_adaptive_cfc import CfCNADWithMDN, mdn_predicted_std
from lnn.data.timeseries import generate_mackey_glass


def add_awgn(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Add AWGN at the requested SNR and return (noisy_signal, noise_std)."""

    power = float(np.mean(signal**2))
    noise_power = power / (10.0 ** (snr_db / 10.0))
    noise_std = math.sqrt(noise_power)
    noise = rng.standard_normal(signal.shape) * noise_std
    return signal + noise.astype(signal.dtype), noise_std


def build_heteroscedastic_dataset(
    num_samples: int,
    seq_len: int,
    window_half: int,
    snr_low: float,
    snr_high: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (x_noisy, y_clean_median, noise_std_per_sample)."""

    rng = np.random.default_rng(seed)
    stride = max(1, seq_len // 4)
    raw_len = num_samples * stride + seq_len + 2 * window_half
    clean = generate_mackey_glass(num_samples=raw_len).astype(np.float32)
    clean = (clean - clean.mean()) / (clean.std() + 1e-8)
    targets = np.zeros_like(clean)
    n = len(clean)
    for t in range(n):
        lo = max(0, t - window_half)
        hi = min(n, t + window_half + 1)
        targets[t] = float(np.median(clean[lo:hi]))

    blocks_x: list[np.ndarray] = []
    blocks_y: list[np.ndarray] = []
    blocks_noise_std: list[float] = []
    start = 0
    while start + seq_len <= n and len(blocks_x) < num_samples:
        snr_db = float(rng.uniform(snr_low, snr_high))
        clean_block = clean[start : start + seq_len]
        noisy_block, noise_std = add_awgn(clean_block, snr_db, rng)
        blocks_x.append(noisy_block)
        blocks_y.append(targets[start : start + seq_len])
        blocks_noise_std.append(noise_std)
        start += stride

    x = torch.tensor(np.stack(blocks_x), dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(np.stack(blocks_y), dtype=torch.float32).unsqueeze(-1)
    noise_std = torch.tensor(blocks_noise_std, dtype=torch.float32)
    return x, y, noise_std


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(1e-12)
    return float((a * b).sum() / denom)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--window-half", type=int, default=3)
    parser.add_argument("--snr-low", type=float, default=5.0)
    parser.add_argument("--snr-high", type=float, default=25.0)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--num-mixtures", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    x, y, noise_std = build_heteroscedastic_dataset(
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        window_half=args.window_half,
        snr_low=args.snr_low,
        snr_high=args.snr_high,
        seed=args.seed,
    )
    cut = int(len(x) * 0.8)
    train_x, val_x = x[:cut], x[cut:]
    train_y, val_y = y[:cut], y[cut:]
    val_noise_std = noise_std[cut:]
    print(
        f"# data: train={len(train_x)} val={len(val_x)} "
        f"snr=[{args.snr_low},{args.snr_high}] dB "
        f"noise_std range={float(noise_std.min()):.3f}..{float(noise_std.max()):.3f}"
    )

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=args.batch_size, shuffle=False)

    torch.manual_seed(args.seed)
    model = CfCNADWithMDN(
        input_size=1,
        hidden_size=args.hidden,
        output_size=1,
        num_mixtures=args.num_mixtures,
        num_layers=1,
        return_sequences=True,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            params = model(xb)
            loss = mdn_negative_log_likelihood(params, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += float(loss)
            n_batches += 1
        if (epoch + 1) % max(1, args.epochs // 4) == 0:
            print(f"  epoch {epoch + 1:>3}/{args.epochs}  train NLL={epoch_loss / max(n_batches,1):.4f}")
    train_seconds = time.time() - t0

    # Evaluation: per-sample mean predicted std.
    model.eval()
    val_predicted_std: list[float] = []
    val_mean_mse_terms: list[float] = []
    with torch.no_grad():
        idx = 0
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            params = model(xb)
            std_per_step = mdn_predicted_std(params)  # [B, T]
            # Use mid-sequence average to avoid edge-of-sequence artefacts.
            mid = std_per_step.shape[1] // 4
            sample_std = std_per_step[:, mid:-mid].mean(dim=-1).cpu()
            val_predicted_std.append(sample_std)
            # Track point-prediction quality via mixture mean.
            from lnn.core.mdn import mdn_mean
            point_mean = mdn_mean(params)
            val_mean_mse_terms.append(((point_mean - yb) ** 2).mean(dim=(-1, -2)).cpu())
            idx += xb.shape[0]
    val_pred_std_tensor = torch.cat(val_predicted_std)
    val_point_mse = float(torch.cat(val_mean_mse_terms).mean())

    r = pearson(val_pred_std_tensor, val_noise_std)
    print()
    print(f"  Pearson r(predicted σ, true noise σ) = {r:.3f}  "
          f"(falsifiable claim ≥0.5): {'PASS' if r >= 0.5 else 'FAIL'}")
    print(f"  point-prediction val MSE (mdn_mean vs target) = {val_point_mse:.5f}")
    print(f"  train time: {train_seconds:.2f} s")

    out_dir = Path("analysis/cfc_nad")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date.today().isoformat()}_cfcnad_mdn_uncertainty.json"
    payload = {
        "config": vars(args),
        "results": {
            "pearson_r_pred_std_vs_true_noise_std": r,
            "val_point_mse": val_point_mse,
            "train_seconds": train_seconds,
            "predicted_std_min": float(val_pred_std_tensor.min()),
            "predicted_std_max": float(val_pred_std_tensor.max()),
            "true_noise_std_min": float(val_noise_std.min()),
            "true_noise_std_max": float(val_noise_std.max()),
        },
        "summary": {
            "claim_threshold_r": 0.5,
            "claim_passed": bool(r >= 0.5),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nJSON written to {out_path}")


if __name__ == "__main__":
    main()
