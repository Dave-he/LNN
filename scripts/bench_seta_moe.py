"""Round 105 — Bench: SETA sparse shared + unique experts vs QuITE+MoE (PRD #10-67).

Measures test_mse, full-system entropy (shared + unique), shared
utilization, unique utilization, and training stability for:
  - 3 conditions: quite_moe (baseline round 103), seta_only_shared, seta_full
  - 3 datasets: sin_irr, structured_irr, random_irr
  - 2 K settings: S=2+U=3 (K=5), S=1+U=4 (K=5)
  - 2 seeds × 100 epochs = 36 cells

Tests on data with HIGHER missing rate (50% vs train 30%).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# Data generators (reused from rounds 102-104)
def make_sin_irr(T, D, gap_rate, rng):
    t = np.linspace(0, 2 * np.pi, T)
    base = np.stack([np.sin(t + i * 0.5) for i in range(D)], axis=-1)
    mask = rng.random((T, D)) > gap_rate
    obs = base * mask
    return obs.astype(np.float32), mask


def make_structured_irr(T, D, gap_rate, rng):
    t = np.linspace(0, 4 * np.pi, T)
    base = np.zeros((T, D), dtype=np.float32)
    for i in range(D):
        regime = (t // (2 * np.pi)) % 2
        base[:, i] = np.sin(t + i) * (1.0 + 0.5 * regime)
    mask = rng.random((T, D)) > gap_rate
    obs = base * mask
    return obs.astype(np.float32), mask


def make_random_irr(T, D, gap_rate, rng):
    t = np.linspace(0, 1, T)
    base = np.cumsum(rng.normal(0, 0.1, (T, D)), axis=0)
    mask = rng.random((T, D)) > gap_rate
    obs = base * mask
    return obs.astype(np.float32), mask


def make_dataset(name, T, D, gap_rate, rng):
    if name == "sin_irr":
        return make_sin_irr(T, D, gap_rate, rng)
    if name == "structured_irr":
        return make_structured_irr(T, D, gap_rate, rng)
    if name == "random_irr":
        return make_random_irr(T, D, gap_rate, rng)
    raise ValueError(f"Unknown dataset: {name}")


def get_target(obs, mask):
    T, D = obs.shape
    target = np.zeros(D, dtype=np.float32)
    for d in range(D):
        valid_idx = np.where(mask[:, d])[0]
        if len(valid_idx) > 0:
            target[d] = obs[valid_idx[-1], d]
    return target


def make_batch(B, T, D, gap_rate, dataset, rng):
    obs_list, mask_list, tgt_list = [], [], []
    for _ in range(B):
        obs, mask = make_dataset(dataset, T, D, gap_rate, rng)
        tgt = get_target(obs, mask)
        obs_list.append(obs)
        mask_list.append(mask)
        tgt_list.append(tgt)
    obs = torch.from_numpy(np.stack(obs_list))
    mask = torch.from_numpy(np.stack(mask_list).any(axis=-1))
    times = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1)
    target = torch.from_numpy(np.stack(tgt_list)).unsqueeze(1).expand(-1, T, -1)
    return obs, times, mask, target


def build_model(cond, input_size, hidden_size, n_shared, n_unique, top_k):
    from lnn.core.quite_moe import QuiteMoECfCNetwork
    from lnn.core.seta_moe import SETAConfig, SETAMoECfCNetwork
    K_total = n_shared + n_unique
    if cond == "quite_moe":
        return QuiteMoECfCNetwork(
            input_size=input_size, hidden_size=hidden_size,
            n_experts=K_total, top_k=top_k, n_queries=4, d_context=16,
            n_heads=4, output_size=input_size,
        )
    if cond == "seta_only_shared":
        # No SETA regularization — just shared + unique architecture
        cfg = SETAConfig(
            n_shared=n_shared, n_unique=n_unique, top_k=top_k,
            elastic_lambda=0.0, routing_lambda=0.0, use_ema_anchor=False,
        )
        return SETAMoECfCNetwork(
            input_size=input_size, hidden_size=hidden_size,
            sdta_config=cfg, n_queries=4, d_context=16,
            n_heads=4, output_size=input_size,
        )
    if cond == "seta_full":
        cfg = SETAConfig(
            n_shared=n_shared, n_unique=n_unique, top_k=top_k,
            elastic_lambda=1e-3, routing_lambda=1e-2, use_ema_anchor=True,
            ema_decay=0.99,
        )
        return SETAMoECfCNetwork(
            input_size=input_size, hidden_size=hidden_size,
            sdta_config=cfg, n_queries=4, d_context=16,
            n_heads=4, output_size=input_size,
        )
    raise ValueError(f"Unknown condition: {cond}")


def run_cell(cond, dataset, n_shared, n_unique, top_k, seed, n_epochs,
             T, D, hidden, train_gap, test_gap, device):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = build_model(cond, D, hidden, n_shared, n_unique, top_k).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0 = time.time()
    train_losses = []
    grad_norm = 0.0
    nan_loss = False
    shared_H_list, unique_H_list = [], []
    for epoch in range(n_epochs):
        obs, times, mask, target = make_batch(32, T, D, train_gap, dataset, rng)
        obs, times, mask, target = obs.to(device), times.to(device), mask.to(device), target.to(device)
        opt.zero_grad()
        pred = model(obs, times, mask=mask)
        valid_mask = mask.unsqueeze(-1).expand_as(pred)
        loss = ((pred - target) ** 2 * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1)
        if torch.isnan(loss):
            nan_loss = True
            break
        # Add SETA regularization for seta_full
        if cond == "seta_full":
            reg_loss = model.regularization_loss()
            total = loss + reg_loss
        else:
            total = loss
        total.backward()
        grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
        opt.step()
        train_losses.append(loss.item())
        # Collect utilization snapshot every 10 epochs
        if epoch % 10 == 0 and hasattr(model, "get_utilization"):
            util = model.get_utilization()
            shared_H_list.append(util["shared_entropy"])
            unique_H_list.append(util["unique_entropy"])
    train_mse = float(np.mean(train_losses[-10:])) if train_losses else float("nan")
    # Test
    rng_test = np.random.default_rng(seed + 10000)
    with torch.no_grad():
        obs_te, times_te, mask_te, target_te = make_batch(32, T, D, test_gap, dataset, rng_test)
        obs_te, times_te, mask_te, target_te = obs_te.to(device), times_te.to(device), mask_te.to(device), target_te.to(device)
        pred_te = model(obs_te, times_te, mask=mask_te)
        valid_te = mask_te.unsqueeze(-1).expand_as(pred_te)
        test_mse = ((pred_te - target_te) ** 2 * valid_te.float()).sum() / valid_te.float().sum().clamp(min=1)
        test_mse = test_mse.item()
    # Test robust
    rng_test2 = np.random.default_rng(seed + 20000)
    with torch.no_grad():
        obs_te2, times_te2, mask_te2, target_te2 = make_batch(32, T, D, 0.7, dataset, rng_test2)
        obs_te2, times_te2, mask_te2, target_te2 = obs_te2.to(device), times_te2.to(device), mask_te2.to(device), target_te2.to(device)
        pred_te2 = model(obs_te2, times_te2, mask=mask_te2)
        valid_te2 = mask_te2.unsqueeze(-1).expand_as(pred_te2)
        robust_mse = ((pred_te2 - target_te2) ** 2 * valid_te2.float()).sum() / valid_te2.float().sum().clamp(min=1)
        robust_mse = robust_mse.item()
    return {
        "cond": cond, "dataset": dataset, "n_shared": n_shared, "n_unique": n_unique,
        "top_k": top_k, "seed": seed,
        "train_mse": train_mse, "test_mse": test_mse, "test_robust_mse": robust_mse,
        "shared_entropy_mean": float(np.mean(shared_H_list)) if shared_H_list else 0.0,
        "unique_entropy_mean": float(np.mean(unique_H_list)) if unique_H_list else 0.0,
        "training_stable": (not nan_loss) and grad_norm < 10.0 and not math.isnan(grad_norm),
        "grad_norm": grad_norm, "elapsed_s": time.time() - t0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/bench_seta_moe.json")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--T", type=int, default=32)
    p.add_argument("--D", type=int, default=2)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--train-gap", type=float, default=0.3)
    p.add_argument("--test-gap", type=float, default=0.5)
    p.add_argument("--datasets", nargs="+", default=["sin_irr", "structured_irr", "random_irr"])
    p.add_argument("--conds", nargs="+", default=["quite_moe", "seta_only_shared", "seta_full"])
    p.add_argument("--K-topk", nargs="+", default=["2-3-2", "1-4-2"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, epochs: {args.epochs}, T={args.T}, D={args.D}")

    all_results = []
    for dataset in args.datasets:
        for cond in args.conds:
            for k_setting in args.K_topk:
                n_shared, n_unique, top_k = map(int, k_setting.split("-"))
                for seed in args.seeds:
                    print(f"  [{cond}|{dataset}|S={n_shared},U={n_unique},k={top_k}|s={seed}] running...")
                    r = run_cell(
                        cond=cond, dataset=dataset, n_shared=n_shared,
                        n_unique=n_unique, top_k=top_k, seed=seed,
                        n_epochs=args.epochs, T=args.T, D=args.D, hidden=args.hidden,
                        train_gap=args.train_gap, test_gap=args.test_gap, device=device,
                    )
                    all_results.append(r)
                    print(
                        f"    train_mse={r['train_mse']:.4f} test_mse={r['test_mse']:.4f} "
                        f"robust_mse={r['test_robust_mse']:.4f} "
                        f"shared_H={r['shared_entropy_mean']:.3f} "
                        f"unique_H={r['unique_entropy_mean']:.3f} "
                        f"stable={r['training_stable']} grad={r['grad_norm']:.3f}"
                    )
    # Aggregate
    print("\n=== Aggregate (mean over seeds) ===")
    for cond in args.conds:
        for dataset in args.datasets:
            for k_setting in args.K_topk:
                n_shared, n_unique, top_k = map(int, k_setting.split("-"))
                rows = [r for r in all_results if r["cond"] == cond and r["dataset"] == dataset
                        and r["n_shared"] == n_shared and r["n_unique"] == n_unique
                        and r["top_k"] == top_k]
                if not rows:
                    continue
                train_mse = np.mean([r["train_mse"] for r in rows])
                test_mse = np.mean([r["test_mse"] for r in rows])
                robust_mse = np.mean([r["test_robust_mse"] for r in rows])
                shared_H = np.mean([r["shared_entropy_mean"] for r in rows])
                unique_H = np.mean([r["unique_entropy_mean"] for r in rows])
                stable = all(r["training_stable"] for r in rows)
                print(
                    f"  {cond:18s} {dataset:14s} S={n_shared},U={n_unique},k={top_k} | "
                    f"train={train_mse:.4f} test={test_mse:.4f} "
                    f"robust={robust_mse:.4f} shared_H={shared_H:.3f} unique_H={unique_H:.3f} "
                    f"stable={stable}"
                )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
