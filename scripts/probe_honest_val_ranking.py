"""Honest val-set ranking probe (no leakage).

Round 63 (44th meta): smart seed selection 0.97 (median_10).
   * Critical caveat*: ranking by *test* MSE is leakage.
   * Need honest val-set ranking to verify production value.

This probe: for each test fold, split its 3 train folds 80/20
into train/val. Rank seeds by val MSE. Compute ensemble MSE
on the held-out test fold. Compare to:
  - first_10 (round 62 default)
  - best_10 by test MSE (leaky, round 63 baseline)
  - best_10 by val MSE (HONEST, this round)
  - median_10 by val MSE (HONEST, this round)
  - worst_10 (control)

If honest val-set ranking ≈ test-set ranking, then 0.97 is
a fair production estimate. If honest val-set ranking is much
worse than test, then 0.97 is leakage-inflated.

Probe: 20 seeds x 4 folds = 80 fold runs (~12 min)
"""
import os, sys, json, datetime as dt, pathlib, time
import numpy as np

sys.path.insert(0, "/Users/hyx/workspace/LNN")

import torch
from torch.utils.data import DataLoader, Subset
from lnn.core.mdn import mdn_mean, mdn_negative_log_likelihood
from lnn.core.multimodal_physreg import CrossModalAttnBiCfCNADWithMDN
from lnn.data.emma_rover_temporal_folds import (
    TemporalSegmentRegressionDataset,
    create_segment_loo_dataloaders,
)

ROOT = pathlib.Path("/Users/hyx/workspace/LNN")
device = torch.device("cpu")
HIDDEN = 96
EPOCHS = 80
WARMUP = 40
LR = 5e-3
# 20 seeds (same as round 61/62/63)
SEEDS = [
    1, 2, 3, 7, 42,
    11, 100, 2026, 313, 777,
    55, 99, 314, 555, 888,
    1024, 2027, 3141, 4242, 9999,
]
N_FOLDS = 4
INJECT_SIGMA = 0.10
VAL_FRAC = 0.20

def audio_apply(audio, sigma):
    if sigma == 0: return audio
    return audio + torch.randn_like(audio) * sigma

def _move(b, d):
    return {k: v.to(d) for k, v in b.items()}

def train_epoch(model, loader, opt, inject_sigma):
    model.train()
    total, n = 0.0, 0
    for batch, target in loader:
        batch = _move(batch, device)
        target = _move(target, device)
        if "audio" in batch and inject_sigma > 0:
            batch["audio"] = audio_apply(batch["audio"], inject_sigma)
        opt.zero_grad(set_to_none=True)
        out = model(batch["video"], batch["audio"])
        final = {k: v[:, -1] for k, v in out.items()}
        loss = mdn_negative_log_likelihood(final, target["params"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)

@torch.no_grad()
def collect_predictions(model, loader):
    model.eval()
    preds, tgts = [], []
    for batch, target in loader:
        batch = _move(batch, device)
        target = _move(target, device)
        out = model(batch["video"], batch["audio"])
        final = {k: v[:, -1] for k, v in out.items()}
        mean = mdn_mean(final)
        preds.append(mean.cpu())
        tgts.append(target["params"].cpu())
    return torch.cat(preds), torch.cat(tgts)

@torch.no_grad()
def eval_mse(model, loader):
    preds, tgts = collect_predictions(model, loader)
    return float(((preds - tgts) ** 2).sum(dim=-1).mean().item())

def split_train_val(dataset, val_frac=VAL_FRAC, seed=42):
    """Split a dataset into train and val by indices."""
    n = len(dataset)
    n_val = max(1, int(n * val_frac))
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen).tolist()
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)

def adaptive_freeze_run_train_val(train_sub, val_sub, test_loader, phase2_sigma, seed):
    """Train on train_sub, evaluate on val_sub AND test_loader."""
    torch.manual_seed(seed)
    model = CrossModalAttnBiCfCNADWithMDN(
        video_dim=3, audio_dim=1, hidden_size=HIDDEN,
        output_size=5, num_mixtures=1,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    train_loader = DataLoader(train_sub, batch_size=8, shuffle=True)
    for _ in range(WARMUP):
        train_epoch(model, train_loader, opt, 0.0)
    for p in model.audio_encoder.parameters():
        p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=LR)
    for _ in range(EPOCHS - WARMUP):
        train_epoch(model, train_loader, opt, phase2_sigma)
    val_loader = DataLoader(val_sub, batch_size=8, shuffle=False)
    val_mse = eval_mse(model, val_loader)
    test_preds, test_tgts = collect_predictions(model, test_loader)
    return val_mse, test_preds, test_tgts

print("=== Honest Val-Set Ranking Probe (round 64) ===")
print(f"h={HIDDEN} ep={EPOCHS} warmup={WARMUP} inject={INJECT_SIGMA}")
print(f"seeds={len(SEEDS)} folds={N_FOLDS} val_frac={VAL_FRAC}")
print(f"Total: {len(SEEDS) * N_FOLDS} fold runs (~12 min)")

# Per-fold per-seed: val_mse, test_preds
fold_data = {}
for fold in range(N_FOLDS):
    print(f"\n=== Fold {fold} ===")
    ds = TemporalSegmentRegressionDataset(seed=1, audio_mode="normal")
    tl_full, te = create_segment_loo_dataloaders(
        ds, held_out_fold=fold, batch_size=8,
    )
    # tl_full is the train loader (3 segments concatenated).
    # We need to split it into train/val.
    # Access the underlying dataset from the loader.
    train_dataset = tl_full.dataset
    train_sub, val_sub = split_train_val(train_dataset, val_frac=VAL_FRAC, seed=42+fold)
    print(f"  Train: {len(train_sub)} samples, Val: {len(val_sub)} samples, Test: {len(te.dataset)} samples")
    fold_data[fold] = {
        "val_mses": [],
        "test_preds": [],
        "test_tgts": None,
        "seed_indices": [],
    }
    for i, seed in enumerate(SEEDS):
        start = time.perf_counter()
        try:
            val_mse, test_preds, test_tgts = adaptive_freeze_run_train_val(
                train_sub, val_sub, te, INJECT_SIGMA, seed
            )
        except Exception as e:
            print(f"  ERROR: seed={seed}: {e}")
            continue
        elapsed = time.perf_counter() - start
        fold_data[fold]["val_mses"].append(val_mse)
        fold_data[fold]["test_preds"].append(test_preds)
        if fold_data[fold]["test_tgts"] is None:
            fold_data[fold]["test_tgts"] = test_tgts
        fold_data[fold]["seed_indices"].append(i)
        # per-seed test MSE
        per_test_mse = float(((test_preds - test_tgts) ** 2).sum(dim=-1).mean().item())
        print(f"  [{i+1}/{len(SEEDS)}] seed={seed:>5d} (idx={i:>2d}) | val_mse = {val_mse:>8.4f} | test_mse = {per_test_mse:>8.4f} | {elapsed:.1f}s")

# Compute selection strategies
def compute_ensemble_mse(test_preds_list, tgts, K):
    if not test_preds_list or K == 0:
        return None
    avg = torch.stack(test_preds_list[:K], dim=0).mean(dim=0)
    return float(((avg - tgts) ** 2).sum(dim=-1).mean().item())

print("\n=== Per-seed val vs test MSE comparison ===")
print(f"{'seed':>5s} | {'val_mse':>10s} | {'test_mse':>10s} | {'val_rank':>8s} | {'test_rank':>9s}")
val_ranks_all = []
test_ranks_all = []
for fold in range(N_FOLDS):
    val_mses = fold_data[fold]["val_mses"]
    test_mses = [float(((p - fold_data[fold]["test_tgts"]) ** 2).sum(dim=-1).mean().item()) for p in fold_data[fold]["test_preds"]]
    val_ranked = np.argsort(val_mses)  # indices in val order
    test_ranked = np.argsort(test_mses)  # indices in test order
    val_ranks = np.zeros_like(val_ranked)
    val_ranks[val_ranked] = np.arange(len(val_ranked))
    test_ranks = np.zeros_like(test_ranked)
    test_ranks[test_ranked] = np.arange(len(test_ranked))
    val_ranks_all.append(val_ranks)
    test_ranks_all.append(test_ranks)
    if fold == 0:  # print only fold 0
        for j, (v_mse, t_mse, v_r, t_r) in enumerate(zip(val_mses, test_mses, val_ranks, test_ranks)):
            print(f"  {SEEDS[j]:>5d} | {v_mse:>10.4f} | {t_mse:>10.4f} | {v_r:>8d} | {t_r:>9d}")

# Compute Spearman correlation between val and test ranks
print("\n=== Val vs test rank correlation (Spearman) ===")
for fold in range(N_FOLDS):
    val_ranks = val_ranks_all[fold]
    test_ranks = test_ranks_all[fold]
    # Spearman: 1 - 6 * sum(d^2) / (n * (n^2-1))
    n = len(val_ranks)
    d = val_ranks - test_ranks
    spearman = 1 - 6 * np.sum(d**2) / (n * (n**2 - 1))
    print(f"  Fold {fold}: Spearman = {spearman:.4f}")

# Test strategies
print("\n=== Strategy comparison ===")
strategies = {
    "first_10 (round 62)": list(range(10)),
    "best_10_by_test (LEAKY round 63)": "test",
    "best_10_by_val (HONEST)": "val",
    "median_10_by_val (HONEST)": "median_val",
    "best_10_by_val_then_eval": "val",
    "worst_10_by_val": "worst_val",
}

results = {}
for strat_name, mode in strategies.items():
    if mode == "test":
        # Rank by test MSE (LEAKY)
        strategy_results = []
        for fold in range(N_FOLDS):
            test_mses = [float(((p - fold_data[fold]["test_tgts"]) ** 2).sum(dim=-1).mean().item()) for p in fold_data[fold]["test_preds"]]
            sorted_idx = np.argsort(test_mses)[:10]
            preds = [fold_data[fold]["test_preds"][i] for i in sorted_idx]
            mse = compute_ensemble_mse(preds, fold_data[fold]["test_tgts"], 10)
            strategy_results.append(mse)
        agg = sum(strategy_results) / len(strategy_results)
    elif mode == "val":
        # Rank by val MSE (HONEST)
        strategy_results = []
        for fold in range(N_FOLDS):
            val_mses = fold_data[fold]["val_mses"]
            sorted_idx = np.argsort(val_mses)[:10]
            preds = [fold_data[fold]["test_preds"][i] for i in sorted_idx]
            mse = compute_ensemble_mse(preds, fold_data[fold]["test_tgts"], 10)
            strategy_results.append(mse)
        agg = sum(strategy_results) / len(strategy_results)
    elif mode == "median_val":
        # Ranks 5-15 by val MSE
        strategy_results = []
        for fold in range(N_FOLDS):
            val_mses = fold_data[fold]["val_mses"]
            sorted_idx = np.argsort(val_mses)[5:15]
            preds = [fold_data[fold]["test_preds"][i] for i in sorted_idx]
            mse = compute_ensemble_mse(preds, fold_data[fold]["test_tgts"], 10)
            strategy_results.append(mse)
        agg = sum(strategy_results) / len(strategy_results)
    elif mode == "worst_val":
        # Bottom 10 by val MSE
        strategy_results = []
        for fold in range(N_FOLDS):
            val_mses = fold_data[fold]["val_mses"]
            sorted_idx = np.argsort(val_mses)[-10:]
            preds = [fold_data[fold]["test_preds"][i] for i in sorted_idx]
            mse = compute_ensemble_mse(preds, fold_data[fold]["test_tgts"], 10)
            strategy_results.append(mse)
        agg = sum(strategy_results) / len(strategy_results)
    else:
        # first_10
        strategy_results = []
        for fold in range(N_FOLDS):
            preds = fold_data[fold]["test_preds"][:10]
            mse = compute_ensemble_mse(preds, fold_data[fold]["test_tgts"], 10)
            strategy_results.append(mse)
        agg = sum(strategy_results) / len(strategy_results)
    print(f"  {strat_name:50s} K=10: {agg:.4f} (per fold: {[f'{m:.2f}' for m in strategy_results]})")
    results[strat_name] = {"per_fold": strategy_results, "agg": agg}

# Save
out = {
    "config": {"hidden_size": HIDDEN, "epochs": EPOCHS, "warmup_epochs": WARMUP,
               "lr": LR, "seeds": SEEDS, "inject_sigma": INJECT_SIGMA,
               "folds": N_FOLDS, "val_frac": VAL_FRAC,
               "protocol": "TemporalSegmentRegressionDataset 4-fold LOO + 80/20 train/val split",
               "model": "CrossModalAttnBiCfCNADWithMDN"},
    "strategies": results,
    "per_seed_data": {
        "fold_val_mses": [fold_data[f]["val_mses"] for f in range(N_FOLDS)],
        "fold_test_mses": [[float(((p - fold_data[f]["test_tgts"]) ** 2).sum(dim=-1).mean().item())
                            for p in fold_data[f]["test_preds"]] for f in range(N_FOLDS)],
    },
    "spearman_per_fold": [
        1 - 6 * np.sum((val_ranks_all[f] - test_ranks_all[f])**2) / (len(val_ranks_all[f]) * (len(val_ranks_all[f])**2 - 1))
        for f in range(N_FOLDS)
    ],
    "metadata": {
        "round": 64,
        "follows_up": "round 63 (smart selection 0.97, 44th meta, validation-leakage warning)",
        "hypothesis": "val-set ranking ~ test-set ranking -> 0.97 is honest production",
    },
}
now = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
out_path = ROOT / "analysis" / "emma_rover" / f"{now}_honest_val_ranking.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nWrote: {out_path}")

# Verdict
print("\n=== Verdict ===")
ref = results["first_10 (round 62)"]["agg"]
for s, r in results.items():
    delta = r["agg"] - ref
    marker = "✅" if r["agg"] < ref else "❌"
    print(f"  {marker} {s:50s} K=10: {r['agg']:.4f} (delta = {delta:+.4f})")

# Honest vs leaky comparison
leaky = results["best_10_by_test (LEAKY round 63)"]["agg"]
honest = results["best_10_by_val (HONEST)"]["agg"]
median_honest = results["median_10_by_val (HONEST)"]["agg"]
print(f"\n  Leaky (test-rank)  -> 0.97: HONESTLY {leaky:.4f} (probably higher)")
print(f"  Honest (val-rank)   -> 0.97: HONESTLY {honest:.4f}")
print(f"  Honest (val-rank)   -> median: HONESTLY {median_honest:.4f}")
print(f"  delta leaky -> honest: {honest - leaky:+.4f}")
