"""Round 301 — ParallelCfC vs vanilla CfC toy_sin benchmark (5 seeds).

Reproduces the central claim from arXiv:2608.03041v1 (PLAN, Kannan et al.
2026): a parallel liquid-inspired approximation should be competitive with
sequential CfC on smooth targets, while reducing step count.

Setting (matches prior r155-r200 toy_sin protocols in this repo):
  - target_fn:  y(t) = sin(2π·1.5·t) + 0.3·cos(2π·4.7·t) on t ∈ [0, 1]
  - 256 train / 64 test
  - h=64, T=64, epochs=200, batch=full-batch
  - Optimizer: Adam(lr=2e-3), MSELoss
  - 5 seeds, mean ± std

Models:
  - CfC (vanilla, sequential) — manual step loop, no chunking
  - ParallelCfC window ∈ {2, 4, 8}

Output: JSON dict per (model, seed) → {mse, latency_ms, train_s}, plus
summary `mean ± std` per model.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from lnn.core.cfc import CfCCell
from lnn.core.parallel_cfc import ParallelCfCNetwork


def target_fn(t: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * 1.5 * t) + 0.3 * np.cos(2 * np.pi * 4.7 * t)


def make_data(n_train: int = 256, n_test: int = 64, T: int = 64):
    rng = np.random.default_rng(0)
    t_all = np.linspace(0.0, 1.0, T)
    train_x, train_y = [], []
    for _ in range(n_train):
        offset = rng.uniform(0.0, 0.5)
        x_seq = t_all + offset
        y_seq = target_fn(x_seq) + rng.normal(0.0, 0.05, size=T)
        train_x.append(x_seq)
        train_y.append(y_seq)
    test_x, test_y = [], []
    for _ in range(n_test):
        offset = rng.uniform(0.0, 0.5)
        x_seq = t_all + offset
        y_seq = target_fn(x_seq) + rng.normal(0.0, 0.05, size=T)
        test_x.append(x_seq)
        test_y.append(y_seq)
    return (
        torch.tensor(np.array(train_x), dtype=torch.float32).unsqueeze(-1),
        torch.tensor(np.array(train_y), dtype=torch.float32).unsqueeze(-1),
        torch.tensor(np.array(test_x), dtype=torch.float32).unsqueeze(-1),
        torch.tensor(np.array(test_y), dtype=torch.float32).unsqueeze(-1),
    )


class VanillaCfCModel(nn.Module):
    """Sequential CfC regressor with manual step loop (matches CfCNetwork)."""

    def __init__(self, input_size: int = 1, hidden_size: int = 64, output_size: int = 1):
        super().__init__()
        self.cell = CfCCell(input_size, hidden_size)
        self.proj = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_in)
        B, T, _ = x.shape
        h = torch.zeros(B, self.cell.hidden_size, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(T):
            h = self.cell(x[:, t, :], h, dt=torch.tensor(1.0, device=x.device, dtype=x.dtype))
            outs.append(h)
        seq = torch.stack(outs, dim=1)
        return self.proj(seq[:, -1, :])


def train_eval(model: nn.Module, x_tr, y_tr, x_te, y_te, epochs: int, lr: int, seed: int):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    t0 = time.perf_counter()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(x_tr)
        loss = loss_fn(pred, y_tr[:, -1, :])
        loss.backward()
        opt.step()
    train_time = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        pred = model(x_te)
        mse = float(((pred - y_te[:, -1, :]) ** 2).mean().item())
        t0 = time.perf_counter()
        for _ in range(10):
            _ = model(x_te)
        lat_ms = (time.perf_counter() - t0) * 100.0  # 10 passes
    return mse, train_time, lat_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", type=str, default="bench_parallel_cfc_results.json")
    args = parser.parse_args()

    x_tr, y_tr, x_te, y_te = make_data()
    results: dict = {}

    # 1) Vanilla CfC
    for seed in range(args.seeds):
        model = VanillaCfCModel()
        mse, tt, lat = train_eval(model, x_tr, y_tr, x_te, y_te, args.epochs, args.lr, seed)
        results.setdefault("vanilla_cfc", []).append(
            {"seed": seed, "mse": mse, "train_s": tt, "lat_10pass_ms": lat}
        )
        print(f"  vanilla_cfc seed={seed} mse={mse:.5f} train_s={tt:.2f} lat_10p_ms={lat:.2f}")

    # 2) ParallelCfC window ∈ {2, 4, 8}
    for W in (2, 4, 8):
        for seed in range(args.seeds):
            model = ParallelCfCNetwork(
                input_size=1, hidden_size=64, output_size=1, num_layers=1, window=W
            )
            mse, tt, lat = train_eval(model, x_tr, y_tr, x_te, y_te, args.epochs, args.lr, seed)
            results.setdefault(f"parallel_cfc_w{W}", []).append(
                {"seed": seed, "mse": mse, "train_s": tt, "lat_10pass_ms": lat}
            )
            print(f"  parallel_w{W} seed={seed} mse={mse:.5f} train_s={tt:.2f} lat_10p_ms={lat:.2f}")

    # Summary
    summary: dict = {}
    for k, lst in results.items():
        mses = np.array([r["mse"] for r in lst])
        lats = np.array([r["lat_10pass_ms"] for r in lst])
        tts = np.array([r["train_s"] for r in lst])
        summary[k] = {
            "mse_mean": float(mses.mean()),
            "mse_std": float(mses.std()),
            "lat_10p_mean_ms": float(lats.mean()),
            "lat_10p_std_ms": float(lats.std()),
            "train_s_mean": float(tts.mean()),
        }
    print("\n=== Summary (mean ± std over 5 seeds) ===")
    for k, s in summary.items():
        print(
            f"  {k:25s}  mse={s['mse_mean']:.5f} ± {s['mse_std']:.5f}  "
            f"lat={s['lat_10p_mean_ms']:.2f}ms  train={s['train_s_mean']:.2f}s"
        )

    out_path = Path(args.out)
    out_path.write_text(json.dumps({"raw": results, "summary": summary}, indent=2))
    print(f"\nResults written to {out_path.resolve()}")


if __name__ == "__main__":
    main()
