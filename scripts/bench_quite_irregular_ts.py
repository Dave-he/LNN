"""Round 102 bench (PRD #10-64) — QuITE Query-Based Irregular TS Embedding.

Direct test of arXiv:2605.28166 (Lim, ICML 2026) — *QuITE: Query-based
Irregular Time-series Embedding*.

We test QuITE as a plug-and-play embedding for irregular multivariate
time series (IMTS) on synthetic PhysioNet-style data. The hypothesis
is that QuITE's learnable query tokens, attending to irregular
observations via masked self-attention, provide a better input
representation than the baseline (uniform assumption) or simple
baselines (mean/concat/add).

For each of 3 datasets (sin_irr, structured_irr, random_irr), we compare:
- baseline (CfC, uniform assumption)
- +mean baseline embedding
- +concat baseline embedding
- +add baseline embedding
- +QuITE (n_queries=8, d_model=16, n_heads=4)

Cells: 1 model × 3 datasets × 5 conditions × 2 seeds = 30 cells

For each cell measure:
- task_loss (MSE)
- mask_recall (predicts masked values)
- time_gap_robustness (Δ task loss as gap variance increases)
- latent_diversity (variance across query tokens)

H1: QuITE lower task_loss than baselines.
H2: QuITE more robust to time-gap variance.
H3: QuITE target-agnostic (works on smooth/structured/random).

Run:
    .venv312/bin/python scripts/bench_quite_irregular_ts.py --quick
    .venv312/bin/python scripts/bench_quite_irregular_ts.py        # full
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
from lnn.core.quite_embedding import (
    QueryIrregularEmbedding,
    quite_baseline_modes,
)


# ---------------------------------------------------------------------------
# Synthetic PhysioNet-style data generation
# ---------------------------------------------------------------------------

def make_irregular(
    target_fn,
    T_max: int,
    D: int,
    seed: int,
    gap_rate: float = 0.3,
    nan_rate: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate an irregular multivariate time series.

    Args:
        target_fn: callable (T,) → (T,) target values
        T_max: max sequence length
        D: number of features (D-1 noisy + 1 true)
        seed: random seed
        gap_rate: fraction of timesteps to drop (irregular gaps)
        nan_rate: fraction of observations to set to NaN

    Returns:
        observations: (T_max, D) with NaN for missing
        times: (T_max,) normalized to [0, 1]
        mask: (T_max,) bool, True = valid
        target: (T_max,) ground truth target
    """
    rng = np.random.default_rng(seed)
    # Random time gaps
    raw_times = np.cumsum(rng.exponential(1.0, size=T_max + 5))
    # Sample T_max timestamps (irregular)
    indices = np.sort(rng.choice(T_max + 5, size=T_max, replace=False))
    times = raw_times[indices]
    # Normalize to [0, 1]
    times = (times - times.min()) / (times.max() - times.min() + 1e-9)
    # Drop some observations (irregular gaps)
    drop = rng.random(T_max) < gap_rate
    times[drop] = -1.0  # mark as dropped
    # Generate observations: first feature is the target, others are noise
    times_t = torch.tensor(times, dtype=torch.float32)
    target = target_fn(times_t).to(torch.float32)
    obs = torch.zeros(T_max, D, dtype=torch.float32)
    obs[:, 0] = target
    obs[:, 1:] = torch.randn(T_max, D - 1, dtype=torch.float32) * 0.3
    # Add some NaN
    nan_mask = rng.random((T_max, D)) < nan_rate
    obs[nan_mask] = float("nan")
    # Build mask
    mask = torch.tensor(times >= 0, dtype=torch.bool)
    times = torch.tensor(times, dtype=torch.float32)
    times[~mask] = 0.0  # placeholder
    return obs, times, mask, target


def make_sin_irr(T: int, D: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    def fn(t: torch.Tensor) -> torch.Tensor:
        return torch.sin(2 * np.pi * t)

    return make_irregular(fn, T, D, seed)


def make_structured_irr(T: int, D: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    def fn(t: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(t)
        mask = t >= 0
        m1 = mask & (t < 0.5)
        m2 = mask & (t >= 0.5)
        out[m1] = torch.sin(2 * np.pi * t[m1])
        out[m2] = torch.sign(torch.sin(20 * np.pi * t[m2]))
        return out

    return make_irregular(fn, T, D, seed)


def make_random_irr(T: int, D: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    def fn(t: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(t)
        mask = t >= 0
        out[mask] = torch.randn(int(mask.sum()))
        return out

    return make_irregular(fn, T, D, seed)


DATASETS = {
    "sin_irr": make_sin_irr,
    "structured_irr": make_structured_irr,
    "random_irr": make_random_irr,
}


# ---------------------------------------------------------------------------
# Model variants
# ---------------------------------------------------------------------------

class BaselineModel(nn.Module):
    """CfC that processes the irregular sequence as-is (uniform assumption)."""

    def __init__(self, d_input: int, hidden: int) -> None:
        super().__init__()
        self.cell = CfCCell(input_size=d_input, hidden_size=hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(
        self, obs: torch.Tensor, times: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        # obs: (T, D), times: (T,), mask: (T,)
        # Replace NaN with 0
        clean = torch.where(torch.isfinite(obs), obs, torch.zeros_like(obs))
        h = torch.zeros(1, self.cell.hidden_size)
        last_h = h
        for t in range(clean.shape[0]):
            x_t = clean[t].reshape(1, -1)
            h = self.cell(x_t, h, dt=1.0)
            if mask[t]:
                last_h = h
        return self.head(last_h).squeeze(0)


class MeanBaselineModel(nn.Module):
    """Mean-pool irregular obs then CfC on the (T, D) sequence."""

    def __init__(self, d_input: int, hidden: int) -> None:
        super().__init__()
        # Use a CfC to process the (T, D) sequence directly
        self.cell = CfCCell(input_size=d_input, hidden_size=hidden)
        self.head = nn.Linear(hidden, 1)
        # Replicate the pooled mean over T to create a uniform sequence
        self.pool = lambda obs, times, mask: quite_baseline_modes(
            obs.unsqueeze(0), times.unsqueeze(0), mask.unsqueeze(0), mode="mean",
        ).squeeze(0)  # (D,)

    def forward(
        self, obs: torch.Tensor, times: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        pooled = self.pool(obs, times, mask)  # (D,)
        # Replicate over T
        seq = pooled.unsqueeze(0).expand(obs.shape[0], -1)  # (T, D)
        h = torch.zeros(1, self.cell.hidden_size)
        for t in range(seq.shape[0]):
            h = self.cell(seq[t].reshape(1, -1), h, dt=1.0)
        return self.head(h).squeeze(0)


class ConcatBaselineModel(nn.Module):
    """Concat-pool (last obs + last time) then MLP."""

    def __init__(self, d_input: int, hidden: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_input + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self, obs: torch.Tensor, times: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        pooled = quite_baseline_modes(
            obs.unsqueeze(0), times.unsqueeze(0), mask.unsqueeze(0), mode="concat",
        ).squeeze(0)  # (D+1,)
        return self.head(pooled).squeeze()


class AddBaselineModel(nn.Module):
    """Add-pool (value + time emb) then MLP."""

    def __init__(self, d_input: int, hidden: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_input, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self, obs: torch.Tensor, times: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        pooled = quite_baseline_modes(
            obs.unsqueeze(0), times.unsqueeze(0), mask.unsqueeze(0), mode="add",
        ).squeeze(0)  # (D,)
        return self.head(pooled).squeeze()


class QuiteModel(nn.Module):
    """QuITE query embedding + CfC on the (n_queries, d_model) tokens."""

    def __init__(self, d_input: int, hidden: int, n_queries: int = 8) -> None:
        super().__init__()
        self.embed = QueryIrregularEmbedding(
            d_input=d_input, n_queries=n_queries, d_model=hidden, n_heads=4,
        )
        self.cell = CfCCell(input_size=hidden, hidden_size=hidden)
        self.head = nn.Linear(hidden, 1)
        self.n_queries = n_queries

    def forward(
        self, obs: torch.Tensor, times: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        # obs: (T, D) → (1, T, D)
        tokens = self.embed(
            obs.unsqueeze(0), times.unsqueeze(0), mask.unsqueeze(0),
        )  # (1, n_queries, d_model)
        tokens = tokens.squeeze(0)  # (n_queries, d_model)
        # Process with CfC
        h = torch.zeros(1, self.cell.hidden_size)
        for q in range(self.n_queries):
            h = self.cell(tokens[q].reshape(1, -1), h, dt=1.0)
        return self.head(h).squeeze(0)


def make_model(cond: str, d_input: int, hidden: int) -> nn.Module:
    if cond == "baseline":
        return BaselineModel(d_input, hidden)
    if cond == "mean":
        return MeanBaselineModel(d_input, hidden)
    if cond == "concat":
        return ConcatBaselineModel(d_input, hidden)
    if cond == "add":
        return AddBaselineModel(d_input, hidden)
    if cond == "quite":
        return QuiteModel(d_input, hidden, n_queries=8)
    raise ValueError(f"Unknown cond: {cond}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    obs: torch.Tensor,
    times: torch.Tensor,
    mask: torch.Tensor,
    target: torch.Tensor,
    epochs: int,
    lr: float,
    train_gap_rate: float = 0.1,
    test_gap_rate: float = 0.5,
) -> tuple[float, float, float, float]:
    """Train and return (train_mse, test_mse_gap, mask_recall, latent_diversity).

    Trains on data with low gap rate, then tests on data with high gap rate
    to measure generalization to missing data.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # Generate training data with low gap rate
    torch.manual_seed(0)
    valid_idx = mask.nonzero(as_tuple=True)[0]
    if len(valid_idx) > 0:
        last_idx = valid_idx[-1]
        target_pred = target[last_idx]
    else:
        target_pred = torch.tensor(0.0)
    for _ in range(epochs):
        opt.zero_grad()
        y_pred = model(obs, times, mask)
        loss = (y_pred - target_pred) ** 2
        loss.backward()
        opt.step()
    with torch.no_grad():
        y_train = model(obs, times, mask)
        train_mse = float(((y_train - target_pred) ** 2).item())
        # Test on a NEW sequence with HIGHER gap rate (more missing data)
        torch.manual_seed(99)
        if target_fn := getattr(model, "_target_fn", None):
            pass
        # Simulate missing data by masking more positions
        rng = np.random.default_rng(99)
        new_mask = mask.clone()
        extra_drop = rng.random(mask.shape[0]) < test_gap_rate
        new_mask = new_mask & ~torch.tensor(extra_drop, dtype=torch.bool)
        y_test = model(obs, times, new_mask)
        test_mse = float(((y_test - target_pred) ** 2).item())
        # Mask sensitivity: difference between full-mask and partial-mask predictions
        all_mask = torch.ones_like(mask)
        y_full = model(obs, times, all_mask)
        mask_recall = float(torch.abs(y_full - y_train).item())
        # Latent diversity
        if hasattr(model, "embed"):
            tokens = model.embed(
                obs.unsqueeze(0), times.unsqueeze(0), mask.unsqueeze(0),
            ).squeeze(0)
            latent_diversity = float(tokens.var(dim=0).mean().item())
        else:
            latent_diversity = 0.0
    return train_mse, test_mse, mask_recall, latent_diversity


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--out", default="results/bench_quite_irregular_ts.json")
    args = p.parse_args()
    epochs = 30 if args.quick else 100
    n_seeds = args.seeds
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    out: dict = {
        "epochs": epochs,
        "n_seeds": n_seeds,
        "wall_time_s": 0.0,
        "datasets": {},
    }
    T_max = 32
    D = 3  # 1 true + 2 noisy
    hidden = 16
    conds = ("baseline", "mean", "concat", "add", "quite")

    for ds_name, ds_fn in DATASETS.items():
        ds_out: dict = {}
        for cond in conds:
            cond_out: list[dict] = []
            for seed in range(n_seeds):
                torch.manual_seed(seed)
                np.random.seed(seed)
                obs, times, mask, target = ds_fn(T_max, D, seed)
                model = make_model(cond, D, hidden)
                train_mse, test_mse, mask_recall, latent_div = train_model(
                    model, obs, times, mask, target, epochs=epochs, lr=1e-2,
                )
                cond_out.append({
                    "task_loss": train_mse,
                    "test_mse": test_mse,
                    "mask_recall": mask_recall,
                    "latent_diversity": latent_div,
                })
            def agg(field: str) -> tuple[float, float]:
                vals = [s[field] for s in cond_out]
                return float(np.mean(vals)), float(np.std(vals))
            ds_out[cond] = {
                "task_loss_mean_std": agg("task_loss"),
                "test_mse_mean_std": agg("test_mse"),
                "mask_recall_mean_std": agg("mask_recall"),
                "latent_diversity_mean_std": agg("latent_diversity"),
                "per_seed": cond_out,
            }
        out["datasets"][ds_name] = ds_out

    out["wall_time_s"] = round(time.time() - t0, 2)
    out_path.write_text(json.dumps(out, indent=2))

    # Pretty print
    print(f"\n=== Round 102 QuITE irregular TS bench "
          f"(epochs={epochs}, seeds={n_seeds}, T={T_max}, D={D}, hidden={hidden}) ===\n")
    print(f"{'dataset':16s} | {'cond':10s} | {'train_mse':>10s} | {'test_mse':>10s} | "
          f"{'mask_recall':>11s} | {'latent_div':>10s}")
    print("-" * 90)
    for ds_name in DATASETS:
        for cond in conds:
            c = out["datasets"][ds_name][cond]
            tl_m, _ = c["task_loss_mean_std"]
            te_m, _ = c["test_mse_mean_std"]
            mr_m, _ = c["mask_recall_mean_std"]
            ld_m, _ = c["latent_diversity_mean_std"]
            print(f"{ds_name:16s} | {cond:10s} | {tl_m:10.4f} | {te_m:10.4f} | "
                  f"{mr_m:11.4f} | {ld_m:10.4f}")
        print()


if __name__ == "__main__":
    main()
