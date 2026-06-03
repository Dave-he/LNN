"""Apply inject=0.1 to the round 38 SOTA recipe (h=96/ep=80/K=10/freeze)
and measure 5-seed LOO mean.

Round 38 SOTA: LOO MSE 0.42 (single-seed=42, h=96, ep=80, K=10, freeze=audio_only)
Round 43 refutation: 5-seed mean 8.16 ± 6.78 (3/4 new seeds 27-37x worse)
Round 55: inject=0.1 prevents catastrophic seed failure at large-budget

Probe: round 38 SOTA recipe + inject=0.1 vs inject=0.0
- 2 inject conditions x 5 seeds x 4 folds = 40 fold runs
- Each fold: 96 hidden, 80 epochs, 40 warmup, freeze=audio_only
- Hypothesis: inject=0.1 reduces 5-seed mean from 8.16 to <4

This is the most valuable test in the round 55 backlog because:
- It combines the inject recipe with the actual SOTA recipe
- A successful result would be a publishable improvement
"""
import os, sys, json, datetime as dt, pathlib, time
sys.path.insert(0, "/Users/hyx/workspace/LNN")

import torch
import torch.nn as nn
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
SEEDS = [1, 2, 3, 7, 42]
INJECT_CONDITIONS = [0.0, 0.1]
N_FOLDS = 4

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
        # Inject noise on audio (data augmentation)
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
def eval_mse(model, loader):
    model.eval()
    sq = []
    for batch, target in loader:
        batch = _move(batch, device)
        target = _move(target, device)
        out = model(batch["video"], batch["audio"])
        final = {k: v[:, -1] for k, v in out.items()}
        mean = mdn_mean(final)
        sq.append((mean - target["params"]).pow(2).sum(dim=-1))
    return float(torch.cat(sq).mean().item())

def adaptive_freeze_run(train_loader, test_loader, inject_sigma, seed):
    torch.manual_seed(seed)
    model = CrossModalAttnBiCfCNADWithMDN(
        video_dim=3, audio_dim=1, hidden_size=HIDDEN,
        output_size=5, num_mixtures=1,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    # Phase 1: warmup
    for _ in range(WARMUP):
        train_epoch(model, train_loader, opt, inject_sigma)
    # Freeze audio_encoder
    for p in model.audio_encoder.parameters():
        p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=LR)
    # Phase 2: continue with frozen audio_encoder
    for _ in range(EPOCHS - WARMUP):
        train_epoch(model, train_loader, opt, inject_sigma)
    return eval_mse(model, test_loader)

print("=== SOTA recipe + inject=0.1 5-seed LOO probe (round 56) ===")
print(f"h={HIDDEN} ep={EPOCHS} warmup={WARMUP} freeze=audio_only")
print(f"seeds={SEEDS} inject_conditions={INJECT_CONDITIONS} folds={N_FOLDS}")
print(f"Total: {len(SEEDS) * len(INJECT_CONDITIONS) * N_FOLDS} fold runs")

results = {}
for inject in INJECT_CONDITIONS:
    for seed in SEEDS:
        per_fold = []
        for fold in range(N_FOLDS):
            ds = TemporalSegmentRegressionDataset(seed=seed, audio_mode="normal")
            tl, te = create_segment_loo_dataloaders(
                ds, held_out_fold=fold, batch_size=8,
            )
            start = time.perf_counter()
            try:
                mse = adaptive_freeze_run(tl, te, inject, seed)
            except Exception as e:
                print(f"  ERROR: inject={inject} seed={seed} fold={fold}: {e}")
                mse = float("nan")
            elapsed = time.perf_counter() - start
            per_fold.append(mse)
            print(f"  inject={inject:>4.1f} | seed={seed:>3d} | fold={fold} | MSE={mse:>10.4f} | {elapsed:>5.1f}s")
        valid = [m for m in per_fold if m == m]  # filter NaN
        mean = sum(valid) / len(valid) if valid else float("nan")
        results[f"inject{inject}__seed{seed}"] = {
            "inject": inject, "seed": seed, "per_fold_mse": per_fold,
            "loo_mean": mean,
        }
        print(f"  inject={inject:>4.1f} | seed={seed:>3d} | LOO mean = {mean:.4f}")

# Per-inject summary
print("\n=== Per-inject summary (5-seed mean ± std) ===")
for inject in INJECT_CONDITIONS:
    loo_means = [results[f"inject{inject}__seed{s}"]["loo_mean"] for s in SEEDS]
    valid = [m for m in loo_means if m == m]
    mean = sum(valid) / len(valid) if valid else float("nan")
    std = (sum((m - mean) ** 2 for m in valid) / max(1, len(valid) - 1)) ** 0.5 if valid else float("nan")
    min_m = min(valid) if valid else float("nan")
    max_m = max(valid) if valid else float("nan")
    print(f"  inject={inject:>4.1f} | 5-seed mean = {mean:.4f} ± {std:.4f} | min={min_m:.4f} max={max_m:.4f}")
    # Per-seed for context
    for s in SEEDS:
        m = results[f"inject{inject}__seed{s}"]["loo_mean"]
        print(f"    seed={s:>3d}: LOO mean = {m:.4f}")

out = {
    "config": {
        "hidden_size": HIDDEN, "epochs": EPOCHS, "warmup_epochs": WARMUP,
        "freeze": "audio_only", "lr": LR,
        "seeds": SEEDS, "inject_conditions": INJECT_CONDITIONS,
        "folds": N_FOLDS, "protocol": "TemporalSegmentRegressionDataset 4-fold LOO",
        "model": "CrossModalAttnBiCfCNADWithMDN",
    },
    "results": results,
    "metadata": {
        "round": 56,
        "follows_up": "round 55 inject recipe large-budget verification",
        "combines": "round 38 SOTA recipe + round 54 inject=0.1",
        "round_38_sota_single_seed": 0.42,
        "round_43_5seed_mean_no_inject": 8.16,
        "hypotheses": {
            "H_a": "inject=0.1 reduces 5-seed mean from 8.16 to <4",
            "H_b": "inject=0.1 reduces std from 6.78 to <3",
            "H_c": "inject=0.1 prevents catastrophic seed failure",
        },
    },
}
now = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
out_path = ROOT / "analysis" / "emma_rover" / f"{now}_sota_inject_5seed_loo.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nWrote: {out_path}")

# Final verdict
print("\n=== Final Verdict ===")
for inject in INJECT_CONDITIONS:
    loo_means = [results[f"inject{inject}__seed{s}"]["loo_mean"] for s in SEEDS]
    valid = [m for m in loo_means if m == m]
    if valid:
        mean = sum(valid) / len(valid)
        std = (sum((m - mean) ** 2 for m in valid) / max(1, len(valid) - 1)) ** 0.5
        print(f"  inject={inject:>4.1f}: 5-seed LOO mean = {mean:.4f} ± {std:.4f}")
