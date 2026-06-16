"""Round 134 — Bench: LiquidTAD-style PLR vs CfC baseline.

Bench plan (12-16 cells):

- Models:
    M1 CfC          (baseline; round 3 CfCCell)
    M2 PLR          (PLREncoder, no CfC head)
    M3 PLR+CfC      (PLRCfCCell two-axis)
    M4 PLR (HDRS)   (PLREncoder with share_alpha_across_layers=True)

- Tasks (4):
    T1 multi_sin     : sum of two sines at different frequencies
    T2 structured_irr: regime-switch between slow drift and fast transient
    T3 mackey_glass  : classic chaotic time series
    T4 noise_decor   : noisy step function (PLR should low-pass cleanly)

- Metrics per cell:
    mse  : mean squared error on a held-out validation tail
    mae  : mean absolute error
    n_params
    n_flops_estimate (analytical; per-step multiply-add count)

- Verdict:
    write bench_liquid_tad_results.md with per-cell tables and a
    one-line TL;DR (POSITIVE / NEUTRAL / NEGATIVE-WITH-NUANCE /
    STRICTLY NEGATIVE).
"""
from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from lnn.core.cfc import CfCNetwork
from lnn.core.liquid_tad import PLRCell, PLRConfig, PLREncoder, PLRCfCCell


# ---------------------------------------------------------------------------
# Synthetic tasks
# ---------------------------------------------------------------------------


def task_multi_sin(T: int = 400, in_channels: int = 4, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 12 * math.pi, T)
    out = np.zeros((T, in_channels), dtype=np.float32)
    for c in range(in_channels):
        f1 = 0.3 + 0.05 * c
        f2 = 2.0 + 0.2 * c
        out[:, c] = 0.6 * np.sin(f1 * t) + 0.3 * np.sin(f2 * t + 0.7 * c)
    out += 0.05 * rng.standard_normal(out.shape)
    return torch.from_numpy(out).unsqueeze(0)  # (1, T, C)


def task_structured_irr(T: int = 400, in_channels: int = 4, seed: int = 1) -> torch.Tensor:
    """Regime-switch between slow drift and fast transient."""
    rng = np.random.default_rng(seed)
    out = np.zeros((T, in_channels), dtype=np.float32)
    regime = np.zeros(T, dtype=np.int32)
    regime[100:180] = 1
    regime[260:340] = 1
    for c in range(in_channels):
        slow = np.cumsum(0.01 * rng.standard_normal(T)) + 0.5
        fast = 0.3 * np.sin(8.0 * np.linspace(0, 30, T) + c) * rng.standard_normal(1).item()
        out[:, c] = np.where(regime == 0, slow, slow + fast)
    return torch.from_numpy(out).unsqueeze(0)


def task_mackey_glass(T: int = 400, seed: int = 2) -> torch.Tensor:
    """Mackey-Glass classical time series."""
    rng = np.random.default_rng(seed)
    tau = 17
    n = 200 + T
    x = np.zeros(n, dtype=np.float32)
    x[:tau] = 1.2 + 0.1 * rng.standard_normal(tau)
    for t in range(tau, n):
        xtau = x[t - tau]
        x[t] = x[t - 1] + 0.2 * (0.2 * xtau / (1.0 + xtau**10) - 0.1 * x[t - 1])
    out = x[200:].reshape(-1, 1)
    out = np.repeat(out, 4, axis=1)
    return torch.from_numpy(out).unsqueeze(0)


def task_noise_decor(T: int = 400, in_channels: int = 4, seed: int = 3) -> torch.Tensor:
    """Step function + heavy noise. PLR should low-pass to the step."""
    rng = np.random.default_rng(seed)
    step = np.zeros(T, dtype=np.float32)
    step[100:200] = 1.0
    step[300:] = -0.5
    out = np.tile(step[:, None], (1, in_channels))
    out += 0.3 * rng.standard_normal(out.shape).astype(np.float32)
    return torch.from_numpy(out).unsqueeze(0)


TASKS: Dict[str, Callable[[], torch.Tensor]] = {
    "multi_sin": task_multi_sin,
    "structured_irr": task_structured_irr,
    "mackey_glass": task_mackey_glass,
    "noise_decor": task_noise_decor,
}


# ---------------------------------------------------------------------------
# Models (each maps (B, T, C) -> (B, T, C))
# ---------------------------------------------------------------------------


class CfCBaseline(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int = 32) -> None:
        super().__init__()
        self.net = CfCNetwork(
            input_size=in_channels,
            hidden_size=hidden_size,
            output_size=in_channels,
            num_layers=1,
            return_sequences=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PLROnly(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 32) -> None:
        super().__init__()
        self.cfg = PLRConfig(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            n_layers=2,
            use_cfc_head=False,
            share_alpha_across_layers=False,
        )
        self.enc = PLREncoder(self.cfg)
        self.out = nn.Linear(hidden_channels, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.enc(x))


class PLRHDRS(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 32) -> None:
        super().__init__()
        self.cfg = PLRConfig(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            n_layers=2,
            use_cfc_head=False,
            share_alpha_across_layers=True,
        )
        self.enc = PLREncoder(self.cfg)
        self.out = nn.Linear(hidden_channels, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.enc(x))


class PLRCfCCombo(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 32) -> None:
        super().__init__()
        self.cell = PLRCfCCell(in_channels=in_channels, out_channels=hidden_channels, cfc_hidden=hidden_channels)
        self.out = nn.Linear(hidden_channels, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.cell(x))


MODELS: Dict[str, Callable[[int], nn.Module]] = {
    "cfc": lambda c: CfCBaseline(c, 32),
    "plr": lambda c: PLROnly(c, 32),
    "plr_hdrs": lambda c: PLRHDRS(c, 32),
    "plr_cfc": lambda c: PLRCfCCombo(c, 32),
}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    model: str
    task: str
    mse: float
    mae: float
    n_params: int
    seconds: float


def _train_and_eval(
    model: nn.Module,
    x: torch.Tensor,
    epochs: int = 60,
    lr: float = 1e-2,
    train_frac: float = 0.75,
) -> Tuple[float, float, float]:
    T = x.size(1)
    cut = int(train_frac * T)
    x_train = x[:, :cut, :]
    x_val = x[:, cut:, :]

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    for _ in range(epochs):
        opt.zero_grad()
        y = model(x_train)
        # Next-step prediction: predict x_{t+1} from x_t.
        pred = y[:, :-1, :]
        tgt = x_train[:, 1:, :]
        loss = ((pred - tgt) ** 2).mean()
        loss.backward()
        opt.step()
    elapsed = time.time() - t0

    model.eval()
    with torch.no_grad():
        y_val = model(x_val)
        pred_v = y_val[:, :-1, :]
        tgt_v = x_val[:, 1:, :]
        mse = ((pred_v - tgt_v) ** 2).mean().item()
        mae = (pred_v - tgt_v).abs().mean().item()
    return mse, mae, elapsed


def run_one(model_name: str, task_name: str, seed: int = 42) -> BenchResult:
    torch.manual_seed(seed)
    x = TASKS[task_name]()
    in_channels = x.size(-1)
    model = MODELS[model_name](in_channels)
    mse, mae, secs = _train_and_eval(model, x)
    n_params = sum(p.numel() for p in model.parameters())
    return BenchResult(model_name, task_name, mse, mae, n_params, secs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Reduce epochs for a smoke run.")
    args = parser.parse_args()

    results: List[BenchResult] = []
    for model_name in MODELS:
        for task_name in TASKS:
            r = run_one(model_name, task_name)
            results.append(r)
            print(
                f"[{model_name:8s} | {task_name:14s}] "
                f"mse={r.mse:.5f} mae={r.mae:.5f} "
                f"params={r.n_params:6d} t={r.seconds:5.2f}s"
            )

    # Write summary.
    out_path = os.path.join(os.path.dirname(__file__), "..", "analysis", "bench_liquid_tad_results.md")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# Bench results — LiquidTAD PLR (round 134)\n\n")
        f.write("| Model | Task | MSE | MAE | Params | Train s |\n")
        f.write("|---|---|---:|---:|---:|---:|\n")
        for r in results:
            f.write(f"| {r.model} | {r.task} | {r.mse:.5f} | {r.mae:.5f} | {r.n_params} | {r.seconds:.2f} |\n")
        f.write("\nAuto-generated by `scripts/bench_liquid_tad.py`.\n")
    print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()
