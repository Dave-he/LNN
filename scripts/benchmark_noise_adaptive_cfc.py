"""Benchmark Noise-Adaptive CfC vs vanilla CfC on noisy Mackey-Glass.

Usage:
    python scripts/benchmark_noise_adaptive_cfc.py --epochs 8 --hidden 16

Outputs:
    analysis/cfc_nad/<date>_cfc_nad_benchmark.json
    Console table summarising MSE per SNR for CfC, CfC-NAD, and LSTM baseline.

The benchmark protocol matches the falsifiable claim in the 2026-06-01 research
report: across 5 SNR bands (clean / 30 / 20 / 10 / 5 dB), CfC-NAD should beat
CfC on >= 3 bands at parity hidden size, with parameter overhead < 50%.
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
from torch.utils.data import DataLoader

from lnn.core.cfc import CfCNetwork
from lnn.core.noise_adaptive_cfc import NoiseAdaptiveCfCNetwork
from lnn.data.timeseries import TimeSeriesDataset, generate_mackey_glass


SNR_BANDS_DB: list[float | None] = [None, 30.0, 20.0, 10.0, 5.0]


def add_awgn(signal: np.ndarray, snr_db: float | None, rng: np.random.Generator) -> np.ndarray:
    """Add additive white Gaussian noise at the requested SNR (None = clean)."""

    if snr_db is None:
        return signal.copy()
    sig_power = float(np.mean(signal**2))
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = rng.standard_normal(signal.shape) * math.sqrt(noise_power)
    return signal + noise.astype(signal.dtype)


def build_loaders(
    series: np.ndarray,
    seq_len: int,
    horizon: int,
    batch_size: int,
    val_split: float = 0.2,
) -> tuple[DataLoader, DataLoader]:
    cut = int(len(series) * (1.0 - val_split))
    train = TimeSeriesDataset(series[:cut], seq_len=seq_len, horizon=horizon)
    val = TimeSeriesDataset(series[cut:], seq_len=seq_len, horizon=horizon)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(val, batch_size=batch_size, shuffle=False),
    )


class LSTMBaseline(nn.Module):
    """A small LSTM head with the same I/O shape as CfC/CfC-NAD."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.proj = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, dt=None, mask=None) -> torch.Tensor:  # noqa: ARG002
        out, _ = self.lstm(x)
        return self.proj(out[:, -1, :])


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    seq_only: bool = False,
) -> tuple[float, float, float]:
    """Train ``model`` for ``epochs`` and return (val MSE, train_seconds, infer_us_per_step)."""

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
            if pred.dim() == 3:  # return_sequences=True -> take last step
                pred = pred[:, -1, :]
            if y.dim() == 1:
                y = y.unsqueeze(-1)
            loss = loss_fn(pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
    train_seconds = time.time() - t0

    # Validation
    model.eval()
    total_sq, count = 0.0, 0
    infer_steps = 0
    t1 = time.time()
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            if pred.dim() == 3:
                pred = pred[:, -1, :]
            if y.dim() == 1:
                y = y.unsqueeze(-1)
            total_sq += float(((pred - y) ** 2).sum())
            count += y.numel()
            infer_steps += x.shape[0] * x.shape[1]
    infer_time = time.time() - t1
    val_mse = total_sq / max(count, 1)
    infer_us = (infer_time / max(infer_steps, 1)) * 1e6
    return val_mse, train_seconds, infer_us


def run_band(
    snr_db: float | None,
    clean: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, dict[str, float]]:
    noisy = add_awgn(clean, snr_db, rng)
    train_loader, val_loader = build_loaders(noisy, args.seq_len, 1, args.batch_size)
    val_clean = TimeSeriesDataset(clean[int(len(clean) * 0.8) :], seq_len=args.seq_len, horizon=1)
    clean_loader = DataLoader(val_clean, batch_size=args.batch_size, shuffle=False)

    band_label = "clean" if snr_db is None else f"{int(snr_db)}dB"
    print(f"\n=== SNR band: {band_label} ===")
    results: dict[str, dict[str, float]] = {}

    torch.manual_seed(args.seed)
    cfc = CfCNetwork(input_size=1, hidden_size=args.hidden, output_size=1, return_sequences=False)
    mse_n, ts_n, infer_n = train_one(cfc, train_loader, val_loader, args.epochs, args.lr, device)
    mse_c, _, _ = train_one(cfc, train_loader, clean_loader, 0, args.lr, device)  # noisy-trained, clean eval
    results["cfc"] = {
        "params": count_parameters(cfc),
        "val_mse_noisy": mse_n,
        "val_mse_clean_target": mse_c,
        "train_s": ts_n,
        "infer_us_per_step": infer_n,
    }
    print(f"  CfC      params={results['cfc']['params']:>5}  mse_noisy={mse_n:.5f}  "
          f"mse_clean={mse_c:.5f}  train_s={ts_n:.2f}  infer={infer_n:.2f} µs/step")

    torch.manual_seed(args.seed)
    cfc_nad = NoiseAdaptiveCfCNetwork(
        input_size=1, hidden_size=args.hidden, output_size=1, return_sequences=False
    )
    mse_n2, ts_n2, infer_n2 = train_one(
        cfc_nad, train_loader, val_loader, args.epochs, args.lr, device
    )
    mse_c2, _, _ = train_one(cfc_nad, train_loader, clean_loader, 0, args.lr, device)
    results["cfc_nad"] = {
        "params": count_parameters(cfc_nad),
        "val_mse_noisy": mse_n2,
        "val_mse_clean_target": mse_c2,
        "train_s": ts_n2,
        "infer_us_per_step": infer_n2,
    }
    print(f"  CfC-NAD  params={results['cfc_nad']['params']:>5}  mse_noisy={mse_n2:.5f}  "
          f"mse_clean={mse_c2:.5f}  train_s={ts_n2:.2f}  infer={infer_n2:.2f} µs/step")

    torch.manual_seed(args.seed)
    lstm = LSTMBaseline(input_size=1, hidden_size=args.hidden, output_size=1)
    mse_n3, ts_n3, infer_n3 = train_one(lstm, train_loader, val_loader, args.epochs, args.lr, device)
    mse_c3, _, _ = train_one(lstm, train_loader, clean_loader, 0, args.lr, device)
    results["lstm"] = {
        "params": count_parameters(lstm),
        "val_mse_noisy": mse_n3,
        "val_mse_clean_target": mse_c3,
        "train_s": ts_n3,
        "infer_us_per_step": infer_n3,
    }
    print(f"  LSTM     params={results['lstm']['params']:>5}  mse_noisy={mse_n3:.5f}  "
          f"mse_clean={mse_c3:.5f}  train_s={ts_n3:.2f}  infer={infer_n3:.2f} µs/step")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1500)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None, help="optional override for JSON output path")
    args = parser.parse_args()

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    series = generate_mackey_glass(num_samples=args.samples).astype(np.float32)
    # Normalise to zero mean / unit std so SNR has consistent meaning.
    series = (series - series.mean()) / (series.std() + 1e-8)

    all_results: dict[str, dict[str, dict[str, float]]] = {}
    for snr_db in SNR_BANDS_DB:
        label = "clean" if snr_db is None else f"{int(snr_db)}dB"
        all_results[label] = run_band(snr_db, series, args, device, rng)

    # Summary table
    print("\n===== Summary: val MSE on noisy validation =====")
    print(f"{'SNR':<8}{'CfC':>14}{'CfC-NAD':>14}{'LSTM':>14}{'NAD wins':>12}")
    nad_wins = 0
    for label, group in all_results.items():
        cfc_mse = group["cfc"]["val_mse_noisy"]
        nad_mse = group["cfc_nad"]["val_mse_noisy"]
        lstm_mse = group["lstm"]["val_mse_noisy"]
        win = "yes" if nad_mse < cfc_mse else "no"
        if nad_mse < cfc_mse:
            nad_wins += 1
        print(f"{label:<8}{cfc_mse:>14.5f}{nad_mse:>14.5f}{lstm_mse:>14.5f}{win:>12}")
    print(f"\nCfC-NAD beats CfC on {nad_wins}/{len(SNR_BANDS_DB)} SNR bands "
          f"(falsifiable claim: >= 3/5).")

    out_dir = Path("analysis/cfc_nad")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        Path(args.output)
        if args.output
        else out_dir / f"{date.today().isoformat()}_cfc_nad_benchmark.json"
    )
    payload = {
        "config": vars(args),
        "snr_bands": [None if b is None else float(b) for b in SNR_BANDS_DB],
        "results": all_results,
        "summary": {
            "nad_wins_vs_cfc": nad_wins,
            "nad_total_bands": len(SNR_BANDS_DB),
            "claim_passed": nad_wins >= 3,
        },
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nJSON written to {out_path}")


if __name__ == "__main__":
    main()
