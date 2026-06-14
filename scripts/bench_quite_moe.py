"""Round 103 — Bench: QuITE+MoE vs FAME baseline on irregular TS (PRD #10-65).

Measures test_mse, expert utilization, dead experts, and training stability
for 2 conditions (FAMECfC baseline, QuiteMoECfC) × 3 datasets
(sin_irr / structured_irr / random_irr) × 2 K settings (K=2, K=3)
× 2 seeds × 100 epochs = 24 cells.

Tests with HIGHER missing rate (50%) than training (30%) to measure
generalization, mirroring round 102's bench design.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


# Make repo root importable when running as a script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_sin_irr(T: int, D: int, gap_rate: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Smooth sin with irregular gaps."""
    t = np.linspace(0, 2 * np.pi, T)
    base = np.stack([np.sin(t + i * 0.5) for i in range(D)], axis=-1)  # (T, D)
    mask = rng.random((T, D)) > gap_rate  # (T, D)
    obs = base * mask
    return obs.astype(np.float32), mask


def make_structured_irr(T: int, D: int, gap_rate: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Regime-switched signal with irregular gaps."""
    t = np.linspace(0, 4 * np.pi, T)
    base = np.zeros((T, D), dtype=np.float32)
    for i in range(D):
        regime = (t // (2 * np.pi)) % 2
        base[:, i] = np.sin(t + i) * (1.0 + 0.5 * regime)
    mask = rng.random((T, D)) > gap_rate
    obs = base * mask
    return obs.astype(np.float32), mask


def make_random_irr(T: int, D: int, gap_rate: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Noisy random walk with irregular gaps."""
    t = np.linspace(0, 1, T)
    base = np.cumsum(rng.normal(0, 0.1, (T, D)), axis=0)
    mask = rng.random((T, D)) > gap_rate
    obs = base * mask
    return obs.astype(np.float32), mask


def make_dataset(name: str, T: int, D: int, gap_rate: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if name == "sin_irr":
        return make_sin_irr(T, D, gap_rate, rng)
    if name == "structured_irr":
        return make_structured_irr(T, D, gap_rate, rng)
    if name == "random_irr":
        return make_random_irr(T, D, gap_rate, rng)
    raise ValueError(f"Unknown dataset: {name}")


def get_target(obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Target = value at last valid timestep per feature.

    Forces the model to learn a meaningful mapping rather than always
    predicting zero.  Shape (D,).
    """
    T, D = obs.shape
    # For each feature, find the last valid timestep
    target = np.zeros(D, dtype=np.float32)
    for d in range(D):
        valid_idx = np.where(mask[:, d])[0]
        if len(valid_idx) > 0:
            target[d] = obs[valid_idx[-1], d]
    return target


def make_batch(B: int, T: int, D: int, gap_rate: float, dataset: str, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Make a batch of irregular-TS data.

    Returns (observations, times, mask, target) where target is per-batch
    the last valid value per feature, broadcast to all timesteps for loss
    computation.
    """
    obs_list, mask_list, tgt_list = [], [], []
    for _ in range(B):
        obs, mask = make_dataset(dataset, T, D, gap_rate, rng)
        tgt = get_target(obs, mask)
        obs_list.append(obs)
        mask_list.append(mask)
        tgt_list.append(tgt)
    obs = torch.from_numpy(np.stack(obs_list))  # (B, T, D)
    mask = torch.from_numpy(np.stack(mask_list).any(axis=-1))  # (B, T) — any feature valid
    times = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1)
    target = torch.from_numpy(np.stack(tgt_list)).unsqueeze(1).expand(-1, T, -1)  # (B, T, D)
    return obs, times, mask, target


def build_model(cond: str, input_size: int, hidden_size: int, K: int, top_k: int, d_context: int = 16):
    from lnn.core.fame_cfc import FAMECfCNetwork
    from lnn.core.quite_moe import QuiteMoECfCNetwork
    if cond == "fame":
        # Use FAMECfCNetwork: n_experts=K, top_k=top_k
        return FAMECfCNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=input_size,  # predict per-feature value
            n_experts=K,
            top_k=top_k,
        )
    if cond == "quite_moe":
        return QuiteMoECfCNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            n_experts=K,
            top_k=top_k,
            n_queries=4,
            d_context=d_context,
            n_heads=4,
            output_size=input_size,
        )
    raise ValueError(f"Unknown condition: {cond}")


def compute_expert_utilization(model, all_top_idx: list) -> dict:
    """Compute utilization statistics from collected top-K indices."""
    if not all_top_idx:
        return {"usage_per_expert": [], "dead_experts": 0, "entropy": 0.0}
    all_idx = torch.cat([t.flatten() for t in all_top_idx])  # (N*top_k,)
    K = model.n_experts if hasattr(model, "n_experts") else model.cell.n_experts
    counts = torch.bincount(all_idx, minlength=K).float()
    usage = (counts / counts.sum()).tolist()
    dead = int((counts == 0).sum())
    # Entropy
    p = counts / counts.sum()
    p_safe = p[p > 0]
    H = -(p_safe * p_safe.log()).sum().item()
    return {"usage_per_expert": usage, "dead_experts": dead, "entropy": H}


def run_cell(
    cond: str,
    dataset: str,
    K: int,
    top_k: int,
    seed: int,
    n_epochs: int,
    T: int,
    D: int,
    hidden: int,
    train_gap: float,
    test_gap: float,
    device: torch.device,
) -> dict:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = build_model(cond, D, hidden, K, top_k).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Train
    t0 = time.time()
    train_losses = []
    all_top_idx = []
    grad_norm = 0.0
    for epoch in range(n_epochs):
        obs, times, mask, target = make_batch(32, T, D, train_gap, dataset, rng)
        obs, times, mask, target = obs.to(device), times.to(device), mask.to(device), target.to(device)
        opt.zero_grad()
        if cond == "fame":
            # FAMECfCNetwork: (x, h0=None, dt=None, mask=None)
            pred = model(obs, mask=mask)  # (B, T, D)
        else:  # quite_moe
            pred = model(obs, times, mask=mask)
        # Loss: MSE on valid positions only
        valid_mask = mask.unsqueeze(-1).expand_as(pred)  # (B, T, D)
        loss = ((pred - target) ** 2 * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1)
        if torch.isnan(loss):
            return {
                "cond": cond, "dataset": dataset, "K": K, "top_k": top_k, "seed": seed,
                "train_mse": float("nan"), "test_mse": float("nan"),
                "test_robust_mse": float("nan"),
                "dead_experts": K, "entropy": 0.0, "usage_per_expert": [0.0] * K,
                "training_stable": False, "grad_norm": float("nan"),
                "elapsed_s": time.time() - t0,
            }
        loss.backward()
        grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
        opt.step()
        train_losses.append(loss.item())
        # Collect expert usage
        if hasattr(model, "cell"):
            all_top_idx.append(model.cell.router.last_top_idx.detach().cpu())
    train_mse = float(np.mean(train_losses[-10:]))
    # Test on harder missing rate
    rng_test = np.random.default_rng(seed + 10000)
    with torch.no_grad():
        obs_te, times_te, mask_te, target_te = make_batch(32, T, D, test_gap, dataset, rng_test)
        obs_te, times_te, mask_te, target_te = obs_te.to(device), times_te.to(device), mask_te.to(device), target_te.to(device)
        if cond == "fame":
            pred_te = model(obs_te, mask=mask_te)
        else:
            pred_te = model(obs_te, times_te, mask=mask_te)
        valid_te = mask_te.unsqueeze(-1).expand_as(pred_te)
        test_mse = ((pred_te - target_te) ** 2 * valid_te.float()).sum() / valid_te.float().sum().clamp(min=1)
        test_mse = test_mse.item()
    # Test on even harder (extreme missing)
    rng_test2 = np.random.default_rng(seed + 20000)
    with torch.no_grad():
        obs_te2, times_te2, mask_te2, target_te2 = make_batch(32, T, D, 0.7, dataset, rng_test2)
        obs_te2, times_te2, mask_te2, target_te2 = obs_te2.to(device), times_te2.to(device), mask_te2.to(device), target_te2.to(device)
        if cond == "fame":
            pred_te2 = model(obs_te2, mask=mask_te2)
        else:
            pred_te2 = model(obs_te2, times_te2, mask=mask_te2)
        valid_te2 = mask_te2.unsqueeze(-1).expand_as(pred_te2)
        robust_mse = ((pred_te2 - target_te2) ** 2 * valid_te2.float()).sum() / valid_te2.float().sum().clamp(min=1)
        robust_mse = robust_mse.item()
    util = compute_expert_utilization(model, all_top_idx)
    return {
        "cond": cond, "dataset": dataset, "K": K, "top_k": top_k, "seed": seed,
        "train_mse": train_mse, "test_mse": test_mse, "test_robust_mse": robust_mse,
        "dead_experts": util["dead_experts"], "entropy": util["entropy"],
        "usage_per_expert": util["usage_per_expert"],
        "training_stable": grad_norm < 10.0 and not math.isnan(grad_norm),
        "grad_norm": grad_norm, "elapsed_s": time.time() - t0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/bench_quite_moe.json")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--T", type=int, default=32)
    p.add_argument("--D", type=int, default=2)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--train-gap", type=float, default=0.3)
    p.add_argument("--test-gap", type=float, default=0.5)
    p.add_argument("--datasets", nargs="+", default=["sin_irr", "structured_irr", "random_irr"])
    p.add_argument("--conds", nargs="+", default=["fame", "quite_moe"])
    p.add_argument("--K-topk", nargs="+", default=["2-1", "3-2"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, epochs: {args.epochs}, T={args.T}, D={args.D}")

    all_results = []
    for dataset in args.datasets:
        for cond in args.conds:
            for k_setting in args.K_topk:
                K, top_k = map(int, k_setting.split("-"))
                for seed in args.seeds:
                    print(f"  [{cond}|{dataset}|K={K},top_k={top_k}|s={seed}] running...")
                    r = run_cell(
                        cond=cond, dataset=dataset, K=K, top_k=top_k,
                        seed=seed, n_epochs=args.epochs, T=args.T, D=args.D,
                        hidden=args.hidden, train_gap=args.train_gap,
                        test_gap=args.test_gap, device=device,
                    )
                    all_results.append(r)
                    print(
                        f"    train_mse={r['train_mse']:.4f} test_mse={r['test_mse']:.4f} "
                        f"robust_mse={r['test_robust_mse']:.4f} "
                        f"dead={r['dead_experts']} H={r['entropy']:.3f} "
                        f"stable={r['training_stable']} grad={r['grad_norm']:.3f}"
                    )
    # Aggregate
    print("\n=== Aggregate (mean over seeds) ===")
    for cond in args.conds:
        for dataset in args.datasets:
            for k_setting in args.K_topk:
                K, top_k = map(int, k_setting.split("-"))
                rows = [r for r in all_results if r["cond"] == cond and r["dataset"] == dataset
                        and r["K"] == K and r["top_k"] == top_k]
                if not rows:
                    continue
                train_mse = np.mean([r["train_mse"] for r in rows])
                test_mse = np.mean([r["test_mse"] for r in rows])
                robust_mse = np.mean([r["test_robust_mse"] for r in rows])
                dead = np.mean([r["dead_experts"] for r in rows])
                H = np.mean([r["entropy"] for r in rows])
                stable = all(r["training_stable"] for r in rows)
                print(
                    f"  {cond:12s} {dataset:14s} K={K},top_k={top_k} | "
                    f"train={train_mse:.4f} test={test_mse:.4f} "
                    f"robust={robust_mse:.4f} dead={dead:.1f} H={H:.3f} stable={stable}"
                )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
