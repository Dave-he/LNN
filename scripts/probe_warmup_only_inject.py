"""Warmup-only inject: only inject noise during phase 1 (warmup)
of the adaptive-freeze SOTA recipe.  Phase 2 (frozen audio_encoder)
uses CLEAN audio.

Round 56 (NEGATIVE result) found inject=0.1 + freeze hurts (+106% MSE).
Hypothesis: the hurt is because phase 2 injects noise but the audio_encoder
is frozen and learned 'denoising' during warmup -> clean phase 2 audio
gets over-denoised -> mismatch.

If we ONLY inject during warmup, the frozen audio_encoder learns
'denoise noisy audio', and at test time sees CLEAN audio (matching
phase 2 training) -> no mismatch.

Probe design:
  - 2 conditions x 5 seeds x 4 folds = 40 fold runs
  - condition A: baseline (no inject anywhere) = 8.88 from round 56
  - condition B: inject=0.1 in warmup ONLY (40 epochs), clean in phase 2 (40 epochs)
  - condition C (bonus): inject=0.1 throughout but very weak

Hypotheses (falsifiable):
  H_a: warmup-only inject <= baseline (8.88) -> freeze+inject is fixable
  H_b: warmup-only inject >= 8.88 -> freeze+inject mismatch is fundamental
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
HIDDEN = 96
EPOCHS = 80
WARMUP = 40
LR = 5e-3
SEEDS = [1, 2, 3, 7, 42]
INJECT_CONDITIONS = [
    ("none", 0.0, 0.0),       # baseline: no inject anywhere
    ("warmup_only", 0.1, 0.0),# inject 0.1 in warmup, 0 in phase 2
    ("warmup_only_05", 0.05, 0.0), # inject 0.05 in warmup
    ("phase2_only", 0.0, 0.1),# no inject in warmup, 0.1 in phase 2
]
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

def adaptive_freeze_run(train_loader, test_loader, warmup_sigma, phase2_sigma, seed):
    torch.manual_seed(seed)
    model = CrossModalAttnBiCfCNADWithMDN(
        video_dim=3, audio_dim=1, hidden_size=HIDDEN,
        output_size=5, num_mixtures=1,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    # Phase 1: warmup with warmup_sigma
    for _ in range(WARMUP):
        train_epoch(model, train_loader, opt, warmup_sigma)
    # Freeze audio_encoder
    for p in model.audio_encoder.parameters():
        p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=LR)
    # Phase 2: continue with phase2_sigma
    for _ in range(EPOCHS - WARMUP):
        train_epoch(model, train_loader, opt, phase2_sigma)
    return eval_mse(model, test_loader)

print("=== Warmup-only inject probe (round 57) ===")
print(f"h={HIDDEN} ep={EPOCHS} warmup={WARMUP} freeze=audio_only")
print(f"seeds={SEEDS} conditions={len(INJECT_CONDITIONS)} folds={N_FOLDS}")
print(f"Total: {len(SEEDS) * len(INJECT_CONDITIONS) * N_FOLDS} fold runs")

results = {}
for label, warmup_sigma, phase2_sigma in INJECT_CONDITIONS:
    for seed in SEEDS:
        per_fold = []
        for fold in range(N_FOLDS):
            ds = TemporalSegmentRegressionDataset(seed=seed, audio_mode="normal")
            tl, te = create_segment_loo_dataloaders(
                ds, held_out_fold=fold, batch_size=8,
            )
            start = time.perf_counter()
            try:
                mse = adaptive_freeze_run(tl, te, warmup_sigma, phase2_sigma, seed)
            except Exception as e:
                print(f"  ERROR: {label} seed={seed} fold={fold}: {e}")
                mse = float("nan")
            elapsed = time.perf_counter() - start
            per_fold.append(mse)
            print(f"  {label:18s} | seed={seed:>3d} | fold={fold} | MSE={mse:>10.4f} | {elapsed:>5.1f}s")
        valid = [m for m in per_fold if m == m]
        mean = sum(valid) / len(valid) if valid else float("nan")
        results[f"{label}__seed{seed}"] = {
            "label": label, "warmup_sigma": warmup_sigma, "phase2_sigma": phase2_sigma,
            "seed": seed, "per_fold_mse": per_fold, "loo_mean": mean,
        }
        print(f"  {label:18s} | seed={seed:>3d} | LOO mean = {mean:.4f}")

# Per-condition summary
print("\n=== Per-condition summary (5-seed mean ± std) ===")
summary = {}
for label, _, _ in INJECT_CONDITIONS:
    loo_means = [results[f"{label}__seed{s}"]["loo_mean"] for s in SEEDS]
    valid = [m for m in loo_means if m == m]
    mean = sum(valid) / len(valid) if valid else float("nan")
    std = (sum((m - mean) ** 2 for m in valid) / max(1, len(valid) - 1)) ** 0.5 if valid else float("nan")
    min_m = min(valid) if valid else float("nan")
    max_m = max(valid) if valid else float("nan")
    summary[label] = {"mean": mean, "std": std, "min": min_m, "max": max_m}
    print(f"  {label:18s} | 5-seed mean = {mean:.4f} ± {std:.4f} | min={min_m:.4f} max={max_m:.4f}")

out = {
    "config": {
        "hidden_size": HIDDEN, "epochs": EPOCHS, "warmup_epochs": WARMUP,
        "freeze": "audio_only", "lr": LR,
        "seeds": SEEDS, "conditions": INJECT_CONDITIONS,
        "folds": N_FOLDS, "protocol": "TemporalSegmentRegressionDataset 4-fold LOO",
        "model": "CrossModalAttnBiCfCNADWithMDN",
    },
    "results": results,
    "summary": summary,
    "metadata": {
        "round": 57,
        "follows_up": "round 56 (inject+freeze incompatible, 37th meta-refinement NEGATIVE)",
        "hypotheses": {
            "H_a": "warmup-only inject <= baseline 8.88 -> freeze+inject is fixable",
            "H_b": "warmup-only inject >= 8.88 -> freeze+inject mismatch is fundamental",
        },
    },
}
now = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
out_path = ROOT / "analysis" / "emma_rover" / f"{now}_warmup_only_inject.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nWrote: {out_path}")

# Verdict
print("\n=== Verdict ===")
baseline = summary["none"]["mean"]
for label in [l for l, _, _ in INJECT_CONDITIONS]:
    s = summary[label]
    delta = s["mean"] - baseline
    pct = (delta / baseline) * 100 if baseline else 0
    marker = "✅ wins" if delta < 0 else ("❌ hurts" if delta > 0 else "= baseline")
    print(f"  {label:18s} | mean = {s['mean']:.4f} | delta = {delta:+.4f} ({pct:+.1f}%) | {marker}")
