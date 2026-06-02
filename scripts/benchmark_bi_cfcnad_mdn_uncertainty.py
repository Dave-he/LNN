"""Bi-CfC-NAD + MDN vs Uni-CfC-NAD + MDN on the heteroscedastic uncertainty
task. Follow-up to round 4 (2026-06-02 appendix C) where Uni+MDN reached
Pearson r = 0.426, below the 0.5 claim threshold.

Hypothesis: a bidirectional backbone gives both halves of the noise EMA per
step, so the MDN head can calibrate its predicted std more accurately and
push Pearson r past 0.5.

Usage:
    python scripts/benchmark_bi_cfcnad_mdn_uncertainty.py --epochs 32 --hidden 16
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
from lnn.core.noise_adaptive_cfc import (
    BiCfCNADWithMDN,
    CfCNADWithMDN,
    mdn_predicted_std,
)
from lnn.data.timeseries import generate_mackey_glass


def add_awgn(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> tuple[np.ndarray, float]:
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
):
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


def train_and_score(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    val_noise_std: torch.Tensor,
    epochs: int,
    lr: float,
    device: torch.device,
) -> dict[str, float]:
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            params = model(xb)
            loss = mdn_negative_log_likelihood(params, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
    train_seconds = time.time() - t0

    model.eval()
    with torch.no_grad():
        xb = val_x.to(device)
        yb = val_y.to(device)
        params = model(xb)
        std_per_step = mdn_predicted_std(params)  # [B, T]
        mid = std_per_step.shape[1] // 4
        sample_std = std_per_step[:, mid:-mid].mean(dim=-1).cpu()
        from lnn.core.mdn import mdn_mean
        point_mean = mdn_mean(params)
        point_mse = float(((point_mean - yb) ** 2).mean())
    r = pearson(sample_std, val_noise_std)
    return {
        "train_seconds": train_seconds,
        "pearson_r": r,
        "val_point_mse": point_mse,
        "params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--window-half", type=int, default=3)
    parser.add_argument("--snr-low", type=float, default=5.0)
    parser.add_argument("--snr-high", type=float, default=25.0)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--num-mixtures", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=32)
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
    train_loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True
    )
    print(
        f"# data: train={len(train_x)} val={len(val_x)} K={args.num_mixtures} "
        f"epochs={args.epochs} hidden={args.hidden}"
    )

    results = {}

    print("\n-- Uni-CfC-NAD + MDN --")
    torch.manual_seed(args.seed)
    uni = CfCNADWithMDN(
        input_size=1,
        hidden_size=args.hidden,
        output_size=1,
        num_mixtures=args.num_mixtures,
        num_layers=1,
        return_sequences=True,
    ).to(device)
    uni_res = train_and_score(
        uni, train_loader, val_x, val_y, val_noise_std, args.epochs, args.lr, device
    )
    results["uni_mdn"] = uni_res
    print(f"  params={uni_res['params']:>6}  r={uni_res['pearson_r']:.3f}  "
          f"point_mse={uni_res['val_point_mse']:.5f}  train_s={uni_res['train_seconds']:.1f}")

    print("\n-- Bi-CfC-NAD (independent noise) + MDN --")
    torch.manual_seed(args.seed)
    bi = BiCfCNADWithMDN(
        input_size=1,
        hidden_size=args.hidden,
        output_size=1,
        num_mixtures=args.num_mixtures,
        num_layers=1,
        return_sequences=True,
        noise_aggregation="independent",
    ).to(device)
    bi_res = train_and_score(
        bi, train_loader, val_x, val_y, val_noise_std, args.epochs, args.lr, device
    )
    results["bi_mdn_indep"] = bi_res
    print(f"  params={bi_res['params']:>6}  r={bi_res['pearson_r']:.3f}  "
          f"point_mse={bi_res['val_point_mse']:.5f}  train_s={bi_res['train_seconds']:.1f}")

    print("\n-- Bi-CfC-NAD (centered noise) + MDN --")
    torch.manual_seed(args.seed)
    bi_c = BiCfCNADWithMDN(
        input_size=1,
        hidden_size=args.hidden,
        output_size=1,
        num_mixtures=args.num_mixtures,
        num_layers=1,
        return_sequences=True,
        noise_aggregation="centered",
    ).to(device)
    bi_c_res = train_and_score(
        bi_c, train_loader, val_x, val_y, val_noise_std, args.epochs, args.lr, device
    )
    results["bi_mdn_centered"] = bi_c_res
    print(f"  params={bi_c_res['params']:>6}  r={bi_c_res['pearson_r']:.3f}  "
          f"point_mse={bi_c_res['val_point_mse']:.5f}  train_s={bi_c_res['train_seconds']:.1f}")

    print("\n===== Summary =====")
    best_name = max(results, key=lambda k: results[k]["pearson_r"])
    best_r = results[best_name]["pearson_r"]
    print(f"  best Pearson r: {best_r:.3f} ({best_name})")
    print(f"  falsifiable claim (r ≥ 0.5): {'PASS' if best_r >= 0.5 else 'FAIL'}")

    out_dir = Path("analysis/cfc_nad")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date.today().isoformat()}_bi_cfcnad_mdn_uncertainty.json"
    payload = {
        "config": vars(args),
        "results": results,
        "summary": {
            "best_model": best_name,
            "best_pearson_r": best_r,
            "claim_threshold_r": 0.5,
            "claim_passed": bool(best_r >= 0.5),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nJSON written to {out_path}")


if __name__ == "__main__":
    main()
