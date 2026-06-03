"""Phase 2 inject sigma scan + cross-h validation.

Two probes to refine the round 57 production recipe:

Probe 1: phase2_only inject sigma scan at h=96/ep=80
  - 5 sigma values: 0.0 (baseline) / 0.05 / 0.1 / 0.15 / 0.2
  - 3 seeds x 4 folds = 60 fold runs
  - Find optimal phase2 inject value

Probe 2: phase2_only inject=0.1 across h values (32/64/96)
  - 3 h values x 3 seeds x 4 folds = 36 fold runs
  - Verify cross-h generalization

Hypotheses (falsifiable):
  H_a: 0.1 is optimal phase2 inject (V-shape around it)
  H_b: phase2_only inject=0.1 generalizes across h in [32, 64, 96]
"""
import os, sys, json, datetime as dt, pathlib, time
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
EPOCHS = 80
WARMUP = 40
LR = 5e-3
SEEDS = [1, 2, 3]  # 3 seeds for speed
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

def adaptive_freeze_run(train_loader, test_loader, hidden_size, phase2_sigma, seed):
    torch.manual_seed(seed)
    model = CrossModalAttnBiCfCNADWithMDN(
        video_dim=3, audio_dim=1, hidden_size=hidden_size,
        output_size=5, num_mixtures=1,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    # Phase 1: warmup clean
    for _ in range(WARMUP):
        train_epoch(model, train_loader, opt, 0.0)
    # Freeze audio_encoder
    for p in model.audio_encoder.parameters():
        p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=LR)
    # Phase 2: inject
    for _ in range(EPOCHS - WARMUP):
        train_epoch(model, train_loader, opt, phase2_sigma)
    return eval_mse(model, test_loader)

# Probe 1: phase2 sigma scan at h=96
print("=== Probe 1: phase2_only inject sigma scan at h=96/ep=80 ===")
print(f"seeds={SEEDS} folds={N_FOLDS}")
probe1_sigmas = [0.0, 0.05, 0.1, 0.15, 0.2]
probe1_results = {}
for sigma in probe1_sigmas:
    per_seed = []
    for seed in SEEDS:
        per_fold = []
        for fold in range(N_FOLDS):
            ds = TemporalSegmentRegressionDataset(seed=seed, audio_mode="normal")
            tl, te = create_segment_loo_dataloaders(
                ds, held_out_fold=fold, batch_size=8,
            )
            try:
                mse = adaptive_freeze_run(tl, te, hidden_size=96,
                                           phase2_sigma=sigma, seed=seed)
            except Exception as e:
                print(f"  ERROR: sigma={sigma} seed={seed} fold={fold}: {e}")
                mse = float("nan")
            per_fold.append(mse)
            print(f"  sigma={sigma:>4.2f} | seed={seed:>3d} | fold={fold} | MSE={mse:>10.4f}")
        valid = [m for m in per_fold if m == m]
        mean = sum(valid) / len(valid) if valid else float("nan")
        per_seed.append(mean)
        print(f"  sigma={sigma:>4.2f} | seed={seed:>3d} | LOO mean = {mean:.4f}")
    valid_p = [m for m in per_seed if m == m]
    mean = sum(valid_p) / len(valid_p) if valid_p else float("nan")
    std = (sum((m - mean) ** 2 for m in valid_p) / max(1, len(valid_p) - 1)) ** 0.5 if valid_p else float("nan")
    probe1_results[f"sigma{sigma}"] = {
        "sigma": sigma, "per_seed_loo_mean": per_seed,
        "mean": mean, "std": std,
    }
    print(f"  sigma={sigma:>4.2f} | 3-seed mean = {mean:.4f} ± {std:.4f}")

# Probe 2: cross-h with phase2_only inject=0.1
print("\n=== Probe 2: phase2_only inject=0.1 across h values ===")
probe2_h = [32, 64, 96]
probe2_results = {}
for h in probe2_h:
    per_seed = []
    for seed in SEEDS:
        per_fold = []
        for fold in range(N_FOLDS):
            ds = TemporalSegmentRegressionDataset(seed=seed, audio_mode="normal")
            tl, te = create_segment_loo_dataloaders(
                ds, held_out_fold=fold, batch_size=8,
            )
            try:
                mse = adaptive_freeze_run(tl, te, hidden_size=h,
                                           phase2_sigma=0.1, seed=seed)
            except Exception as e:
                print(f"  ERROR: h={h} seed={seed} fold={fold}: {e}")
                mse = float("nan")
            per_fold.append(mse)
            print(f"  h={h:>3d} | seed={seed:>3d} | fold={fold} | MSE={mse:>10.4f}")
        valid = [m for m in per_fold if m == m]
        mean = sum(valid) / len(valid) if valid else float("nan")
        per_seed.append(mean)
        print(f"  h={h:>3d} | seed={seed:>3d} | LOO mean = {mean:.4f}")
    valid_p = [m for m in per_seed if m == m]
    mean = sum(valid_p) / len(valid_p) if valid_p else float("nan")
    std = (sum((m - mean) ** 2 for m in valid_p) / max(1, len(valid_p) - 1)) ** 0.5 if valid_p else float("nan")
    probe2_results[f"h{h}"] = {
        "hidden_size": h, "per_seed_loo_mean": per_seed,
        "mean": mean, "std": std,
    }
    print(f"  h={h:>3d} | 3-seed mean = {mean:.4f} ± {std:.4f}")

# Save
out = {
    "config": {"epochs": EPOCHS, "warmup_epochs": WARMUP, "lr": LR,
               "seeds": SEEDS, "folds": N_FOLDS, "protocol": "TemporalSegmentRegressionDataset 4-fold LOO",
               "model": "CrossModalAttnBiCfCNADWithMDN"},
    "probe1_sigma_scan": {
        "description": "h=96, ep=80, phase2_only inject in {0.0, 0.05, 0.1, 0.15, 0.2}",
        "results": probe1_results,
    },
    "probe2_cross_h": {
        "description": "phase2_only inject=0.1, h in {32, 64, 96}",
        "results": probe2_results,
    },
    "metadata": {
        "round": 58,
        "follows_up": "round 57 (phase2-only inject wins -20.4%, 38th meta-refinement)",
        "hypotheses": {
            "H_a": "0.1 is optimal phase2 inject (V-shape around it)",
            "H_b": "phase2_only inject=0.1 generalizes across h in [32, 64, 96]",
        },
    },
}
now = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
out_path = ROOT / "analysis" / "emma_rover" / f"{now}_phase2_sweep.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nWrote: {out_path}")

# Verdict
print("\n=== Probe 1 verdict ===")
baseline = probe1_results["sigma0.0"]["mean"]
for sigma in probe1_sigmas:
    s = probe1_results[f"sigma{sigma}"]
    delta = s["mean"] - baseline
    pct = (delta / baseline) * 100 if baseline else 0
    marker = "✅ wins" if delta < 0 else "❌ hurts"
    print(f"  sigma={sigma:>4.2f} | mean = {s['mean']:.4f} | delta = {delta:+.4f} ({pct:+.1f}%) | {marker}")
best_sigma = min(probe1_results.values(), key=lambda v: v["mean"])
print(f"  BEST sigma = {best_sigma['sigma']}, mean = {best_sigma['mean']:.4f}")

print("\n=== Probe 2 verdict ===")
for h in probe2_h:
    s = probe2_results[f"h{h}"]
    print(f"  h={h:>3d} | 3-seed mean = {s['mean']:.4f} ± {s['std']:.4f}")
