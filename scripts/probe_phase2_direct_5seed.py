"""5-seed direct comparison: phase2 inject=0.10 vs 0.15.

Round 57 (38th meta): 5-seed mean with 0.10 = 7.07 (NEW BEST)
Round 58 (39th meta): 3-seed mean with 0.15 = 8.19 (slight better than 0.10 = 9.30 in 3-seed)

This probe does a HEAD-TO-HEAD 5-seed comparison on BOTH 0.10 and 0.15
to definitively resolve which is the optimal phase2 inject value.

Probe: 2 sigma x 5 seeds x 4 folds = 40 fold runs
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
SEEDS = [1, 2, 3, 7, 42]  # 5 seeds (matches round 57)
N_FOLDS = 4
SIGMAS = [0.10, 0.15]

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
    return eval_mse(model, test_loader)

print("=== 5-seed direct comparison: phase2 inject=0.10 vs 0.15 ===")
print(f"h={HIDDEN} ep={EPOCHS} warmup={WARMUP}")
print(f"seeds={SEEDS} sigmas={SIGMAS} folds={N_FOLDS}")
print(f"Total: {len(SEEDS) * len(SIGMAS) * N_FOLDS} fold runs")

results = {}
for sigma in SIGMAS:
    per_seed = []
    for seed in SEEDS:
        per_fold = []
        for fold in range(N_FOLDS):
            ds = TemporalSegmentRegressionDataset(seed=seed, audio_mode="normal")
            tl, te = create_segment_loo_dataloaders(
                ds, held_out_fold=fold, batch_size=8,
            )
            start = time.perf_counter()
            try:
                mse = adaptive_freeze_run(tl, te, sigma, seed)
            except Exception as e:
                print(f"  ERROR: sigma={sigma} seed={seed} fold={fold}: {e}")
                mse = float("nan")
            elapsed = time.perf_counter() - start
            per_fold.append(mse)
            print(f"  sigma={sigma:>4.2f} | seed={seed:>3d} | fold={fold} | MSE={mse:>10.4f} | {elapsed:>5.1f}s")
        valid = [m for m in per_fold if m == m]
        mean = sum(valid) / len(valid) if valid else float("nan")
        per_seed.append(mean)
        print(f"  sigma={sigma:>4.2f} | seed={seed:>3d} | LOO mean = {mean:.4f}")
    valid_p = [m for m in per_seed if m == m]
    mean = sum(valid_p) / len(valid_p) if valid_p else float("nan")
    std = (sum((m - mean) ** 2 for m in valid_p) / max(1, len(valid_p) - 1)) ** 0.5 if valid_p else float("nan")
    results[f"sigma{sigma}"] = {
        "sigma": sigma, "per_seed_loo_mean": per_seed,
        "mean": mean, "std": std,
        "min": min(valid_p) if valid_p else float("nan"),
        "max": max(valid_p) if valid_p else float("nan"),
    }
    print(f"  sigma={sigma:>4.2f} | 5-seed mean = {mean:.4f} ± {std:.4f} | min={min(valid_p):.4f} max={max(valid_p):.4f}")

# Head-to-head verdict
print("\n=== Head-to-head verdict (per-seed) ===")
print(f"  {'seed':>5s} | {'0.10':>10s} | {'0.15':>10s} | {'winner':>10s} | {'delta':>10s}")
for i, seed in enumerate(SEEDS):
    m10 = results["sigma0.1"]["per_seed_loo_mean"][i]
    m15 = results["sigma0.15"]["per_seed_loo_mean"][i]
    winner = "0.10" if m10 < m15 else "0.15"
    print(f"  {seed:>5d} | {m10:>10.4f} | {m15:>10.4f} | {winner:>10s} | {m15-m10:>+10.4f}")

mean_10 = results["sigma0.1"]["mean"]
mean_15 = results["sigma0.15"]["mean"]
print(f"\n  OVERALL | {mean_10:>10.4f} | {mean_15:>10.4f} | {'0.10' if mean_10 < mean_15 else '0.15':>10s} | {mean_15-mean_10:>+10.4f}")

# Save
out = {
    "config": {"hidden_size": HIDDEN, "epochs": EPOCHS, "warmup_epochs": WARMUP,
               "lr": LR, "seeds": SEEDS, "sigmas": SIGMAS,
               "folds": N_FOLDS, "protocol": "TemporalSegmentRegressionDataset 4-fold LOO",
               "model": "CrossModalAttnBiCfCNADWithMDN"},
    "results": results,
    "metadata": {
        "round": 59,
        "follows_up": "round 58 (sweet spot [0.0, 0.15], 0.15 slightly better in 3-seed)",
        "hypothesis": "5-seed head-to-head 0.10 vs 0.15",
    },
}
now = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
out_path = ROOT / "analysis" / "emma_rover" / f"{now}_phase2_direct_5seed.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nWrote: {out_path}")

# Final verdict
print("\n=== Final Verdict ===")
delta = mean_15 - mean_10
pct = (delta / mean_10) * 100 if mean_10 else 0
if abs(delta) < 1.0:
    print(f"  ★ INDIFFERENT (delta {delta:+.4f} = {pct:+.1f}%): both 0.10 and 0.15 are acceptable")
else:
    winner = "0.10" if delta > 0 else "0.15"
    print(f"  ★ {winner} wins by {abs(delta):.4f} ({abs(pct):.1f}%)")
