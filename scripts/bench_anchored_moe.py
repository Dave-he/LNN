"""Round 108 — Bench: Anchored MoE with structural prior (PRD #10-70).

12 cells: 4 conditions × 3 datasets × 1 K × 1 seed × 100 epochs.
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


# Data generators (reused from rounds 102-107)
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
    target = np.zeros(obs.shape[1], dtype=np.float32)
    for d in range(obs.shape[1]):
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
    target = torch.from_numpy(np.stack(tgt_list)).unsqueeze(1).expand(-1, T, -1)
    return obs, mask, target


def build_model(cond, input_size, hidden_size, n_experts, top_k):
    from lnn.core.anchored_moe import AnchoredMoECfCNetwork
    if cond == "baseline":
        return AnchoredMoECfCNetwork(
            input_size=input_size, hidden_size=hidden_size,
            n_experts=n_experts, top_k=top_k, output_size=input_size,
            anchor_mode="logit",  # logit mode is the simplest; we'll set anchor_alpha=0
        ) if False else AnchoredMoECfCNetwork(
            input_size=input_size, hidden_size=hidden_size,
            n_experts=n_experts, top_k=top_k, output_size=input_size,
            anchor_mode="logit",
            anchor_alpha=0.0,  # disable anchoring
        )
    if cond == "anchor_logit":
        return AnchoredMoECfCNetwork(
            input_size=input_size, hidden_size=hidden_size,
            n_experts=n_experts, top_k=top_k, output_size=input_size,
            anchor_mode="logit", anchor_alpha=0.5,
        )
    if cond == "anchor_mix":
        return AnchoredMoECfCNetwork(
            input_size=input_size, hidden_size=hidden_size,
            n_experts=n_experts, top_k=top_k, output_size=input_size,
            anchor_mode="mix", anchor_alpha=0.5,
        )
    if cond == "anchor_kl":
        return AnchoredMoECfCNetwork(
            input_size=input_size, hidden_size=hidden_size,
            n_experts=n_experts, top_k=top_k, output_size=input_size,
            anchor_mode="kl", anchor_lambda=0.1,
        )
    raise ValueError(f"Unknown condition: {cond}")


def run_cell(cond, dataset, n_experts, top_k, seed, n_epochs,
             T, D, hidden, train_gap, test_gap, device):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = build_model(cond, D, hidden, n_experts, top_k).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0 = time.time()
    train_losses = []
    grad_norm = 0.0
    nan_loss = False
    routing_H_list, prior_H_list = [], []
    max_min_list = []
    for epoch in range(n_epochs):
        obs, mask, target = make_batch(32, T, D, train_gap, dataset, rng)
        obs, mask, target = obs.to(device), mask.to(device), target.to(device)
        opt.zero_grad()
        pred = model(obs)
        valid_mask = mask.unsqueeze(-1).expand_as(pred)
        loss = ((pred - target) ** 2 * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1)
        reg = model.get_regularization_loss()
        total = loss + reg
        if torch.isnan(total):
            nan_loss = True
            break
        total.backward()
        grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
        opt.step()
        train_losses.append(loss.item())
        if epoch % 10 == 0:
            util = model.get_utilization()
            routing_H_list.append(util["routing_entropy"])
            prior_H_list.append(util["prior_entropy"])
            max_min_list.append(util["routing_max_min_ratio"])
    train_mse = float(np.mean(train_losses[-10:])) if train_losses else float("nan")
    rng_test = np.random.default_rng(seed + 10000)
    with torch.no_grad():
        obs_te, mask_te, target_te = make_batch(32, T, D, test_gap, dataset, rng_test)
        obs_te, mask_te, target_te = obs_te.to(device), mask_te.to(device), target_te.to(device)
        pred_te = model(obs_te)
        valid_te = mask_te.unsqueeze(-1).expand_as(pred_te)
        test_mse = ((pred_te - target_te) ** 2 * valid_te.float()).sum() / valid_te.float().sum().clamp(min=1)
        test_mse = test_mse.item()
    rng_test2 = np.random.default_rng(seed + 20000)
    with torch.no_grad():
        obs_te2, mask_te2, target_te2 = make_batch(32, T, D, 0.7, dataset, rng_test2)
        obs_te2, mask_te2, target_te2 = obs_te2.to(device), mask_te2.to(device), target_te2.to(device)
        pred_te2 = model(obs_te2)
        valid_te2 = mask_te2.unsqueeze(-1).expand_as(pred_te2)
        robust_mse = ((pred_te2 - target_te2) ** 2 * valid_te2.float()).sum() / valid_te2.float().sum().clamp(min=1)
        robust_mse = robust_mse.item()
    return {
        "cond": cond, "dataset": dataset, "n_experts": n_experts, "top_k": top_k, "seed": seed,
        "train_mse": train_mse, "test_mse": test_mse, "test_robust_mse": robust_mse,
        "routing_H_mean": float(np.mean(routing_H_list)) if routing_H_list else 0.0,
        "prior_H_mean": float(np.mean(prior_H_list)) if prior_H_list else 0.0,
        "max_min_mean": float(np.mean(max_min_list)) if max_min_list else 0.0,
        "training_stable": (not nan_loss) and grad_norm < 10.0 and not math.isnan(grad_norm),
        "grad_norm": grad_norm, "elapsed_s": time.time() - t0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/bench_anchored_moe.json")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--T", type=int, default=32)
    p.add_argument("--D", type=int, default=2)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--train-gap", type=float, default=0.3)
    p.add_argument("--test-gap", type=float, default=0.5)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, epochs: {args.epochs}, T={args.T}, D={args.D}, K={args.K}, top_k={args.topk}")

    conds = ["baseline", "anchor_logit", "anchor_mix", "anchor_kl"]
    datasets = ["sin_irr", "structured_irr", "random_irr"]
    all_results = []
    for dataset in datasets:
        for cond in conds:
            for seed in args.seeds:
                print(f"  [{cond}|{dataset}|K={args.K},k={args.topk}|s={seed}] running...")
                r = run_cell(
                    cond=cond, dataset=dataset, n_experts=args.K, top_k=args.topk,
                    seed=seed, n_epochs=args.epochs, T=args.T, D=args.D,
                    hidden=args.hidden, train_gap=args.train_gap, test_gap=args.test_gap, device=device,
                )
                all_results.append(r)
                print(
                    f"    train_mse={r['train_mse']:.4f} test_mse={r['test_mse']:.4f} "
                    f"robust_mse={r['test_robust_mse']:.4f} "
                    f"routing_H={r['routing_H_mean']:.3f} "
                    f"prior_H={r['prior_H_mean']:.3f} "
                    f"max_min={r['max_min_mean']:.2f} "
                    f"stable={r['training_stable']} grad={r['grad_norm']:.3f}"
                )
    print("\n=== Aggregate (mean over seeds) ===")
    for cond in conds:
        for dataset in datasets:
            rows = [r for r in all_results if r["cond"] == cond and r["dataset"] == dataset]
            if not rows:
                continue
            train_mse = np.mean([r["train_mse"] for r in rows])
            test_mse = np.mean([r["test_mse"] for r in rows])
            robust_mse = np.mean([r["test_robust_mse"] for r in rows])
            routing_H = np.mean([r["routing_H_mean"] for r in rows])
            prior_H = np.mean([r["prior_H_mean"] for r in rows])
            max_min = np.mean([r["max_min_mean"] for r in rows])
            stable = all(r["training_stable"] for r in rows)
            print(
                f"  {cond:15s} {dataset:14s} | train={train_mse:.4f} test={test_mse:.4f} "
                f"robust={robust_mse:.4f} routing_H={routing_H:.3f} prior_H={prior_H:.3f} "
                f"max_min={max_min:.2f} stable={stable}"
            )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
