"""Vanilla CfC 30-seed K=20 ensemble probe (round 71).

Tests if v15 recipe (30 seeds + K=20 by val + phase2 inject=0.10 +
freeze audio_encoder) generalizes from Bi-CfC-NAD to vanilla CfC.

Round 70 (BiCfCEnsemble 30-seed K=20): 0.24 honest LOO MSE
Round 71 (this, vanilla_cfc): ???

If vanilla_cfc 30-seed K=20 << vanilla_cfc single-seed mean (e.g. ~10-15),
then v15 recipe is generalizable to vanilla_cfc.

Probe: 30 seeds x 4 folds = 120 fold runs (~25 min)
"""
import os, sys, json, time, datetime as dt, pathlib
sys.path.insert(0, "/Users/hyx/workspace/LNN")

import torch
from torch.utils.data import DataLoader, Subset
from lnn.core.mdn import mdn_mean, mdn_negative_log_likelihood
from lnn.core.multimodal_physreg import VanillaCfCXAttnWithMDN
from lnn.data.emma_rover_temporal_folds import (
    TemporalSegmentRegressionDataset,
    create_segment_loo_dataloaders,
)

# 30 seeds from round 65
SEEDS = [
    1, 2, 3, 7, 42,
    11, 100, 2026, 313, 777,
    55, 99, 314, 555, 888,
    1024, 2027, 3141, 4242, 9999,
    17, 88, 256, 512, 1023, 2048, 4096, 8192, 16384, 32768,
]
N_FOLDS = 4
EPOCHS = 80
WARMUP = 40
LR = 5e-3
HIDDEN = 96
K = 20
INJECT_SIGMA = 0.10
VAL_FRAC = 0.20

def audio_apply(audio, sigma):
    if sigma == 0: return audio
    return audio + torch.randn_like(audio) * sigma

def _move(b, d):
    return {k: v.to(d) for k, v in b.items()}

def make_model(seed):
    torch.manual_seed(seed)
    return VanillaCfCXAttnWithMDN(
        video_dim=3, audio_dim=1,  # audio_dim ignored per round 22 protocol
        hidden_size=HIDDEN, output_size=5, num_mixtures=1, num_layers=1,
    )

def train_one_seed(model, train_loader, val_loader):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(WARMUP):
        model.train()
        for batch, target in train_loader:
            batch = _move(batch, "cpu"); target = _move(target, "cpu")
            if "audio" in batch and INJECT_SIGMA > 0:
                batch["audio"] = audio_apply(batch["audio"], INJECT_SIGMA)
            opt.zero_grad()
            out = model(batch["video"], batch["audio"])
            final = {k: v[:, -1] for k, v in out.items()}
            loss = mdn_negative_log_likelihood(final, target["params"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
    # Freeze audio_encoder
    for p in model._inner.audio_encoder.parameters():
        p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=LR)
    for _ in range(EPOCHS - WARMUP):
        model.train()
        for batch, target in train_loader:
            batch = _move(batch, "cpu"); target = _move(target, "cpu")
            if "audio" in batch and INJECT_SIGMA > 0:
                batch["audio"] = audio_apply(batch["audio"], INJECT_SIGMA)
            opt.zero_grad()
            out = model(batch["video"], batch["audio"])
            final = {k: v[:, -1] for k, v in out.items()}
            loss = mdn_negative_log_likelihood(final, target["params"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
    # Compute val MSE
    model.eval()
    sq = []
    with torch.no_grad():
        for batch, target in val_loader:
            batch = _move(batch, "cpu"); target = _move(target, "cpu")
            out = model(batch["video"], batch["audio"])
            final = {k: v[:, -1] for k, v in out.items()}
            mean = mdn_mean(final)
            sq.append((mean - target["params"]).pow(2).sum(dim=-1))
    return float(torch.cat(sq).mean().item())

@torch.no_grad()
def get_test_preds(model, test_loader):
    model.eval()
    preds, tgts = [], []
    for batch, target in test_loader:
        batch = _move(batch, "cpu"); target = _move(target, "cpu")
        out = model(batch["video"], batch["audio"])
        final = {k: v[:, -1] for k, v in out.items()}
        mean = mdn_mean(final)
        preds.append(mean.cpu())
        tgts.append(target["params"].cpu())
    return torch.cat(preds), torch.cat(tgts)

def split_train_val(dataset, val_frac, seed):
    n = len(dataset)
    n_val = max(1, int(n * val_frac))
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen).tolist()
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)

print("=== Vanilla CfC 30-seed K=20 ensemble probe (round 71) ===")
print(f"Seeds: {len(SEEDS)}")
print(f"Folds: {N_FOLDS}")
print(f"Total: {len(SEEDS) * N_FOLDS} = 120 fold runs (~25 min)")

ensemble_mses = []
per_seed_mean_mses = []

for fold in range(N_FOLDS):
    print(f"\n=== Fold {fold} ===")
    ds = TemporalSegmentRegressionDataset(seed=1, audio_mode="normal")
    tl_full, te = create_segment_loo_dataloaders(
        ds, held_out_fold=fold, batch_size=8,
    )
    train_dataset = tl_full.dataset
    train_sub, val_sub = split_train_val(train_dataset, val_frac=VAL_FRAC, seed=42+fold)
    train_loader = DataLoader(train_sub, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=8, shuffle=False)

    val_mses = []
    test_preds_list = []
    for i, seed in enumerate(SEEDS):
        start = time.perf_counter()
        model = make_model(seed).to("cpu")
        val_mse = train_one_seed(model, train_loader, val_loader)
        test_preds, test_tgts = get_test_preds(model, te)
        elapsed = time.perf_counter() - start
        val_mses.append(val_mse)
        test_preds_list.append(test_preds)
        if i % 5 == 0 or i == len(SEEDS) - 1:
            print(f"  [{i+1}/{len(SEEDS)}] seed={seed:>5d} | val={val_mse:>8.4f} | {elapsed:>5.1f}s")

    # Per-seed test MSE
    per_seed_test = []
    for preds in test_preds_list:
        mse = float(((preds - test_tgts) ** 2).sum(dim=-1).mean().item())
        per_seed_test.append(mse)
    per_seed_mean = sum(per_seed_test) / len(per_seed_test)

    # Top K by val
    sorted_idx = sorted(range(len(SEEDS)), key=lambda i: val_mses[i])[:K]
    top_preds = [test_preds_list[i] for i in sorted_idx]
    avg_preds = torch.stack(top_preds, dim=0).mean(dim=0)
    ensemble_mse = float(((avg_preds - test_tgts) ** 2).sum(dim=-1).mean().item())

    ensemble_mses.append(ensemble_mse)
    per_seed_mean_mses.append(per_seed_mean)
    print(f"  ensemble MSE (K={K}): {ensemble_mse:.4f}")
    print(f"  per-seed mean MSE:    {per_seed_mean:.4f}")

# Aggregate
import statistics
agg_ensemble = statistics.mean(ensemble_mses)
agg_per_seed = statistics.mean(per_seed_mean_mses)
delta = agg_per_seed - agg_ensemble
pct = (delta / agg_per_seed) * 100 if agg_per_seed else 0

print("\n=== Aggregate across 4 folds ===")
print(f"  Ensemble MSE (K={K}): {agg_ensemble:.4f}")
print(f"  Per-seed mean MSE:    {agg_per_seed:.4f}")
print(f"  Delta (per-seed - ensemble): {delta:+.4f} ({pct:+.1f}%)")
print(f"  Per-fold ensemble MSEs: {[f'{m:.4f}' for m in ensemble_mses]}")

print("\n=== Comparison to Bi-CfC reference (round 70) ===")
print(f"  Round 70 (Bi-CfC 30-seed K=20):  0.2359 honest LOO MSE")
print(f"  Round 71 (vanilla_cfc 30-seed K=20): {agg_ensemble:.4f}")
if agg_ensemble < agg_per_seed:
    print(f"  ★ Vanilla_cfc ensemble works (delta {delta:+.4f} vs per-seed mean)")
else:
    print(f"  ✗ Vanilla_cfc ensemble does NOT improve per-seed mean")

# Save
out = {
    "config": {
        "model": "VanillaCfCXAttnWithMDN", "n_seeds": 30, "K": 20,
        "hidden_size": HIDDEN, "epochs": EPOCHS, "warmup_epochs": WARMUP,
        "phase2_inject_sigma": INJECT_SIGMA, "freeze": "audio_only",
        "val_frac": VAL_FRAC, "lr": LR, "seeds": SEEDS,
    },
    "per_fold": {
        "ensemble_mses": ensemble_mses,
        "per_seed_mean_mses": per_seed_mean_mses,
    },
    "aggregate": {
        "ensemble_mse_mean": agg_ensemble,
        "per_seed_mean_mse_mean": agg_per_seed,
        "delta": delta,
        "delta_pct": pct,
    },
    "references": {
        "round_70_BiCfC_30seed_K20": 0.2359,
        "round_65_BiCfC_30seed_K20": 0.24,
    },
    "metadata": {
        "round": 71,
        "purpose": "Test if v15 recipe (30 seeds + K=20 + phase2 inject) generalizes to vanilla_cfc",
    },
}
now = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
out_path = f"analysis/emma_rover/{now}_vanilla_cfc_30seed.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nWrote: {out_path}")

# Verdict
print("\n=== Verdict ===")
if agg_ensemble < 1.0 and agg_ensemble < agg_per_seed:
    print(f"  ★ vanilla_cfc ensemble WORKS (MSE {agg_ensemble:.4f} < per-seed mean {agg_per_seed:.4f})")
    print(f"  ★ v15 recipe GENERALIZES to vanilla_cfc")
elif agg_ensemble < agg_per_seed:
    print(f"  ✓ vanilla_cfc ensemble helps somewhat (MSE {agg_ensemble:.4f} < per-seed mean {agg_per_seed:.4f})")
else:
    print(f"  ✗ vanilla_cfc ensemble does NOT help (MSE {agg_ensemble:.4f} >= per-seed mean {agg_per_seed:.4f})")
