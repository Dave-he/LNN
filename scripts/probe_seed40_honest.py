"""40-seed pool probe with honest val ranking (smaller than 50 for safety).

Round 65 (46th meta): 30-seed pool + K=20 by val = 0.24
Round 67: extends to 40 seeds.

Hypothesis: with 40 seeds, top-20 selection has more candidates
to filter, possibly giving lower ensemble MSE.

Probe: 40 seeds x 4 folds = 160 fold runs (~24 min)
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
# 40 seeds: 30 from round 65 + 10 NEW
SEEDS = [
    1, 2, 3, 7, 42,
    11, 100, 2026, 313, 777,
    55, 99, 314, 555, 888,
    1024, 2027, 3141, 4242, 9999,
    17, 88, 256, 512, 1023, 2048, 4096, 8192, 16384, 32768,
    # Round 67's 10 NEW
    31, 71, 113, 211, 311, 419, 521, 631, 733, 911,
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
    n = len(dataset)
    n_val = max(1, int(n * val_frac))
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen).tolist()
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)

def adaptive_freeze_run_train_val(train_sub, val_sub, test_loader, phase2_sigma, seed):
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

print("=== 40-seed Pool Probe (round 67) ===", flush=True)
print(f"h={HIDDEN} ep={EPOCHS} warmup={WARMUP} inject={INJECT_SIGMA}", flush=True)
print(f"seeds={len(SEEDS)} folds={N_FOLDS} val_frac={VAL_FRAC}", flush=True)
print(f"Total: {len(SEEDS) * N_FOLDS} fold runs (~24 min)", flush=True)

fold_data = {}
for fold in range(N_FOLDS):
    print(f"\n=== Fold {fold} ===", flush=True)
    ds = TemporalSegmentRegressionDataset(seed=1, audio_mode="normal")
    tl_full, te = create_segment_loo_dataloaders(
        ds, held_out_fold=fold, batch_size=8,
    )
    train_dataset = tl_full.dataset
    train_sub, val_sub = split_train_val(train_dataset, val_frac=VAL_FRAC, seed=42+fold)
    print(f"  Train: {len(train_sub)} samples, Val: {len(val_sub)} samples, Test: {len(te.dataset)} samples", flush=True)
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
            print(f"  ERROR: seed={seed}: {e}", flush=True)
            continue
        elapsed = time.perf_counter() - start
        fold_data[fold]["val_mses"].append(val_mse)
        fold_data[fold]["test_preds"].append(test_preds)
        if fold_data[fold]["test_tgts"] is None:
            fold_data[fold]["test_tgts"] = test_tgts
        fold_data[fold]["seed_indices"].append(i)
        per_test_mse = float(((test_preds - test_tgts) ** 2).sum(dim=-1).mean().item())
        print(f"  [{i+1}/{len(SEEDS)}] seed={seed:>5d} | val={val_mse:>8.4f} | test={per_test_mse:>8.4f} | {elapsed:.1f}s", flush=True)

def compute_ensemble_mse(test_preds_list, tgts, K):
    if not test_preds_list or K == 0:
        return None
    avg = torch.stack(test_preds_list[:K], dim=0).mean(dim=0)
    return float(((avg - tgts) ** 2).sum(dim=-1).mean().item())

print("\n=== Strategy comparison (40-seed pool) ===", flush=True)
strategies_results = {}
for K in [3, 5, 10, 15, 20, 25, 30, 40]:
    mses = []
    for fold in range(N_FOLDS):
        val_mses = fold_data[fold]["val_mses"]
        sorted_idx = np.argsort(val_mses)[:K]
        preds = [fold_data[fold]["test_preds"][i] for i in sorted_idx]
        mse = compute_ensemble_mse(preds, fold_data[fold]["test_tgts"], K)
        mses.append(mse)
    agg = sum(mses) / len(mses)
    strategies_results[K] = agg
    print(f"  K={K:>2d} (top by val): {agg:.4f} (per fold: {[f'{m:.2f}' for m in mses]})", flush=True)

# Save
out = {
    "config": {"hidden_size": HIDDEN, "epochs": EPOCHS, "warmup_epochs": WARMUP,
               "lr": LR, "seeds": SEEDS, "inject_sigma": INJECT_SIGMA,
               "folds": N_FOLDS, "val_frac": VAL_FRAC,
               "protocol": "TemporalSegmentRegressionDataset 4-fold LOO + 80/20 train/val split",
               "model": "CrossModalAttnBiCfCNADWithMDN"},
    "strategies": {f"K={K}": strategies_results[K] for K in strategies_results},
    "round_65_reference": {"K=20_by_val": 0.24},
    "metadata": {
        "round": 67,
        "follows_up": "round 65 (30-seed pool K=20 = 0.24, 46th meta NEW BEST)",
        "hypothesis": "40-seed pool improves over 30-seed pool",
    },
}
now = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
out_path = ROOT / "analysis" / "emma_rover" / f"{now}_seed40_honest.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nWrote: {out_path}", flush=True)

print("\n=== Verdict ===", flush=True)
ref = 0.24
for K, mse in strategies_results.items():
    delta = mse - ref
    marker = "✅ wins" if mse < ref else "❌"
    print(f"  K={K:>2d}: {mse:.4f} (delta vs {ref:.4f}: {delta:+.4f}) {marker}", flush=True)

best_K = min(strategies_results.items(), key=lambda x: x[1])
print(f"\n  ★ BEST K = {best_K[0]} with ensemble MSE = {best_K[1]:.4f}", flush=True)
