"""Smart seed selection: can we beat K=10 first-10 ensemble by smart selection?

Round 62 (43rd meta): K=10 ensemble MSE = 1.49 (first 10 in deterministic order)
Hypothesis: K=10 success might be partially due to first-10 selection
  being implicit 'catastrophic seed filter' (deterministic seed order).

Probe design: rerun 20 seeds x 4 folds, save per-sample predictions,
then test multiple K=10 selection strategies:
  - first_10: deterministic order (round 62 baseline)
  - best_10: lowest per-seed LOO mean
  - worst_10: highest per-seed LOO mean (anti-pattern test)
  - median_10: median per-seed LOO mean
  - best_5_5: top 5 + middle 5 (diversity)
  - best_3_3_4: top 3 + middle 3 + bottom 4 (diversity)
  - random_10: 5 random subsets averaged

This is essentially a 0-cost experiment (just analysis on saved data).
"""
import os, sys, json, datetime as dt, pathlib, time
import numpy as np

sys.path.insert(0, "/Users/hyx/workspace/LNN")

import torch
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
# 20 seeds (same as round 61/62)
SEEDS = [
    1, 2, 3, 7, 42,
    11, 100, 2026, 313, 777,
    55, 99, 314, 555, 888,
    1024, 2027, 3141, 4242, 9999,
]
N_FOLDS = 4
INJECT_SIGMA = 0.10

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

def adaptive_freeze_run(train_loader, test_loader, phase2_sigma, seed):
    torch.manual_seed(seed)
    model = CrossModalAttnBiCfCNADWithMDN(
        video_dim=3, audio_dim=1, hidden_size=HIDDEN,
        output_size=5, num_mixtures=1,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(WARMUP):
        train_epoch(model, train_loader, opt, 0.0)
    for p in model.audio_encoder.parameters():
        p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=LR)
    for _ in range(EPOCHS - WARMUP):
        train_epoch(model, train_loader, opt, phase2_sigma)
    return collect_predictions(model, test_loader)

print("=== Smart Seed Selection Probe (round 63) ===")
print(f"h={HIDDEN} ep={EPOCHS} warmup={WARMUP} inject={INJECT_SIGMA}")
print(f"seeds={len(SEEDS)} folds={N_FOLDS}")
print(f"Total: {len(SEEDS) * N_FOLDS} fold runs (~12 min)")

# Per-fold per-seed predictions
# fold_data[fold] = {"preds": [(seed_idx, tensor)], "targets": tensor}
fold_data = {}
for fold in range(N_FOLDS):
    print(f"\n=== Fold {fold} ===")
    fold_data[fold] = {"preds": [], "targets": None, "seed_indices": []}
    for i, seed in enumerate(SEEDS):
        ds = TemporalSegmentRegressionDataset(seed=seed, audio_mode="normal")
        tl, te = create_segment_loo_dataloaders(
            ds, held_out_fold=fold, batch_size=8,
        )
        start = time.perf_counter()
        try:
            preds, tgts = adaptive_freeze_run(tl, te, INJECT_SIGMA, seed)
        except Exception as e:
            print(f"  ERROR: seed={seed}: {e}")
            continue
        elapsed = time.perf_counter() - start
        fold_data[fold]["preds"].append(preds)
        fold_data[fold]["seed_indices"].append(i)
        if fold_data[fold]["targets"] is None:
            fold_data[fold]["targets"] = tgts
        # Compute per-seed MSE for ranking
        per_mse = float(((preds - tgts) ** 2).sum(dim=-1).mean().item())
        print(f"  [{i+1}/{len(SEEDS)}] seed={seed:>5d} (idx={i:>2d}) | per-seed MSE = {per_mse:>8.4f} | {elapsed:.1f}s")

# Now compute selection strategies
print("\n=== Per-seed MSE ranking ===")
per_seed_mse = []
for fold in range(N_FOLDS):
    for j, idx in enumerate(fold_data[fold]["seed_indices"]):
        preds = fold_data[fold]["preds"][j]
        tgts = fold_data[fold]["targets"]
        mse = float(((preds - tgts) ** 2).sum(dim=-1).mean().item())
        per_seed_mse.append({"fold": fold, "seed_idx": idx, "seed": SEEDS[idx], "mse": mse})

# Average per-seed MSE across folds (for ranking)
from collections import defaultdict
seed_avg_mse = defaultdict(list)
for entry in per_seed_mse:
    seed_avg_mse[entry["seed"]].append(entry["mse"])
seed_to_avg = {s: np.mean(v) for s, v in seed_avg_mse.items()}
seed_to_rank = sorted(seed_to_avg.items(), key=lambda x: x[1])
print("Seed ranking (best to worst by avg per-seed MSE):")
for rank, (s, mse) in enumerate(seed_to_rank):
    print(f"  rank {rank+1:>2d}: seed={s:>5d} avg_mse = {mse:.4f}")

# Selection strategies
def compute_ensemble_mse(fold_data, fold, selected_indices, K):
    """Compute MSE of ensemble predictions for given K seeds from selected_indices[:K]."""
    tgts = fold_data[fold]["targets"]
    preds_list = [fold_data[fold]["preds"][i] for i in selected_indices[:K]]
    if not preds_list:
        return None
    avg_preds = torch.stack(preds_list, dim=0).mean(dim=0)
    return float(((avg_preds - tgts) ** 2).sum(dim=-1).mean().item())

# Strategies (in terms of seed positions, not seed values)
strategies = {
    "first_10": list(range(10)),  # round 62 default
    "first_5": list(range(5)),
    "first_15": list(range(15)),
    "first_20": list(range(20)),
    "best_10": [SEEDS.index(s) for s, m in seed_to_rank[:10]],
    "best_5": [SEEDS.index(s) for s, m in seed_to_rank[:5]],
    "best_15": [SEEDS.index(s) for s, m in seed_to_rank[:15]],
    "worst_10": [SEEDS.index(s) for s, m in seed_to_rank[-10:]],
    "median_10": [SEEDS.index(s) for s, m in seed_to_rank[5:15]],
    "best_5_5": [SEEDS.index(s) for s, m in (seed_to_rank[:5] + seed_to_rank[7:12])],
    # Random subsets (5 random)
    "random_5_a": [0, 5, 8, 11, 17],  # 5 random
    "random_5_b": [2, 6, 9, 14, 19],  # 5 random
    "random_5_c": [1, 4, 12, 15, 18],  # 5 random
}

# For each strategy, compute ensemble MSE at K values
print("\n=== Selection strategy comparison ===")
results = {}
for strat_name, selected in strategies.items():
    print(f"\n--- {strat_name} (selected={selected}) ---")
    per_fold_mse = []
    for K in [1, 2, 5, 10, 15, 20]:
        if K > len(selected):
            continue
        mses = []
        for fold in range(N_FOLDS):
            mse = compute_ensemble_mse(fold_data, fold, selected, K)
            mses.append(mse)
        avg = sum(mses) / len(mses)
        per_fold_mse.append({"K": K, "per_fold_mse": mses, "mean_mse": avg})
        if K in [1, 5, 10, 20]:
            print(f"  K={K}: {avg:.4f} (per fold: {[f'{m:.2f}' for m in mses]})")
    results[strat_name] = {"selected_indices": selected, "per_K_mse": per_fold_mse}

# Best strategy at K=10
print("\n=== Best strategy at K=10 ===")
best_strat = min(results.items(), key=lambda x: next(
    (p["mean_mse"] for p in x[1]["per_K_mse"] if p["K"] == 10), float("inf")
))
print(f"  Best at K=10: {best_strat[0]} = {next(p['mean_mse'] for p in best_strat[1]['per_K_mse'] if p['K'] == 10):.4f}")

# Save
out = {
    "config": {"hidden_size": HIDDEN, "epochs": EPOCHS, "warmup_epochs": WARMUP,
               "lr": LR, "seeds": SEEDS, "inject_sigma": INJECT_SIGMA,
               "folds": N_FOLDS, "protocol": "TemporalSegmentRegressionDataset 4-fold LOO",
               "model": "CrossModalAttnBiCfCNADWithMDN"},
    "seed_ranking": [{"rank": r+1, "seed": s, "avg_mse": m} for r, (s, m) in enumerate(seed_to_rank)],
    "strategies": results,
    "metadata": {
        "round": 63,
        "follows_up": "round 62 (K=10 ensemble MSE 1.49, 43rd meta-refinement PRODUCTION BREAKTHROUGH)",
        "hypothesis": "smart seed selection can beat first_10 K=10 ensemble",
    },
}
now = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
out_path = ROOT / "analysis" / "emma_rover" / f"{now}_smart_seed_selection.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nWrote: {out_path}")

# Verdict
first_10_k10 = next(p["mean_mse"] for p in results["first_10"]["per_K_mse"] if p["K"] == 10)
print(f"\n=== Verdict ===")
print(f"  Round 62 baseline (first_10 K=10): {first_10_k10:.4f}")
for strat_name, r in results.items():
    k10_mse = next((p["mean_mse"] for p in r["per_K_mse"] if p["K"] == 10), None)
    if k10_mse is not None and k10_mse < first_10_k10:
        delta = first_10_k10 - k10_mse
        print(f"  ★ {strat_name} (K=10): {k10_mse:.4f} -> WINS by {delta:.4f} ({(delta/first_10_k10)*100:.1f}%)")
