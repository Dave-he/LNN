"""10-seed validation of phase2 inject=0.10.

Round 59 (40th meta): 5-seed mean 7.07 with inject=0.10
(reproducible from round 57). This probe extends to 10 seeds for
stronger statistical evidence.

5 new seeds: [11, 100, 2026, 313, 777] (excluding existing 5 seeds)
Combined with round 57/59's [1, 2, 3, 7, 42], total 10 seeds.

If 10-seed mean is similar to 5-seed mean (7.07), the production
recipe is robust. If 10-seed mean is significantly higher, the
5-seed mean was lucky.

Probe: 10 seeds x 4 folds = 40 fold runs (~9s each, ~7 min)
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
# 10 seeds: 5 from round 57/59 + 5 new
SEEDS = [1, 2, 3, 7, 42, 11, 100, 2026, 313, 777]
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

print("=== 10-seed validation: phase2 inject=0.10 ===")
print(f"h={HIDDEN} ep={EPOCHS} warmup={WARMUP} inject={INJECT_SIGMA}")
print(f"seeds={SEEDS} folds={N_FOLDS}")
print(f"Total: {len(SEEDS) * N_FOLDS} fold runs")

per_seed = []
per_seed_times = []
for seed in SEEDS:
    per_fold = []
    start_seed = time.perf_counter()
    for fold in range(N_FOLDS):
        ds = TemporalSegmentRegressionDataset(seed=seed, audio_mode="normal")
        tl, te = create_segment_loo_dataloaders(
            ds, held_out_fold=fold, batch_size=8,
        )
        try:
            mse = adaptive_freeze_run(tl, te, INJECT_SIGMA, seed)
        except Exception as e:
            print(f"  ERROR: seed={seed} fold={fold}: {e}")
            mse = float("nan")
        per_fold.append(mse)
        print(f"  seed={seed:>5d} | fold={fold} | MSE={mse:>10.4f}")
    valid = [m for m in per_fold if m == m]
    mean = sum(valid) / len(valid) if valid else float("nan")
    per_seed.append(mean)
    elapsed = time.perf_counter() - start_seed
    per_seed_times.append(elapsed)
    print(f"  seed={seed:>5d} | LOO mean = {mean:.4f} | {elapsed:.1f}s")

# Aggregate
valid_p = [m for m in per_seed if m == m]
mean_all = sum(valid_p) / len(valid_p) if valid_p else float("nan")
std_all = (sum((m - mean_all) ** 2 for m in valid_p) / max(1, len(valid_p) - 1)) ** 0.5 if valid_p else float("nan")

# Original 5 seeds from round 57/59
original_5_seeds = [1, 2, 3, 7, 42]
new_5_seeds = [11, 100, 2026, 313, 777]
orig_means = [per_seed[SEEDS.index(s)] for s in original_5_seeds if SEEDS.index(s) < len(per_seed)]
new_means = [per_seed[SEEDS.index(s)] for s in new_5_seeds if SEEDS.index(s) < len(per_seed)]
orig_mean = sum(orig_means) / len(orig_means) if orig_means else float("nan")
new_mean = sum(new_means) / len(new_means) if new_means else float("nan")

print("\n=== Per-seed 10-seed LOO means ===")
for seed, mse in zip(SEEDS, per_seed):
    print(f"  seed={seed:>5d} | LOO mean = {mse:.4f}")

print(f"\n=== Aggregate ===")
print(f"  10-seed mean = {mean_all:.4f} ± {std_all:.4f}")
print(f"  Original 5 seeds (1,2,3,7,42) mean = {orig_mean:.4f}")
print(f"  New 5 seeds (11,100,2026,313,777) mean = {new_mean:.4f}")
print(f"  Round 57/59 5-seed mean = 7.07 (REFERENCE)")

# Save
out = {
    "config": {"hidden_size": HIDDEN, "epochs": EPOCHS, "warmup_epochs": WARMUP,
               "lr": LR, "seeds": SEEDS, "inject_sigma": INJECT_SIGMA,
               "folds": N_FOLDS, "protocol": "TemporalSegmentRegressionDataset 4-fold LOO",
               "model": "CrossModalAttnBiCfCNADWithMDN"},
    "per_seed_loo_mean": per_seed,
    "per_seed_elapsed_s": per_seed_times,
    "metadata": {
        "round": 60,
        "follows_up": "round 59 (5-seed mean 7.07 with 0.10, 40th meta-refinement FINAL)",
        "reference_5seed_mean": 7.07,
    },
}
now = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
out_path = ROOT / "analysis" / "emma_rover" / f"{now}_phase2_10seed.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nWrote: {out_path}")

# Verdict
print("\n=== Verdict ===")
delta = mean_all - 7.07
pct = (delta / 7.07) * 100
if abs(delta) < 1.0:
    print(f"  ★ 10-seed mean ({mean_all:.4f}) ≈ 5-seed reference (7.07) -> ROBUST")
elif delta > 0:
    print(f"  10-seed mean ({mean_all:.4f}) HIGHER than 5-seed ref (7.07) by {delta:+.4f} ({pct:+.1f}%)")
    print(f"  5-seed ref may have been slightly lucky")
else:
    print(f"  10-seed mean ({mean_all:.4f}) LOWER than 5-seed ref (7.07) by {delta:+.4f} ({pct:+.1f}%) -> even better")
print(f"  Original 5 mean ({orig_mean:.4f}) vs Round 57 ref (7.07) -> delta = {orig_mean - 7.07:+.4f}")
print(f"  New 5 mean ({new_mean:.4f}) vs Round 57 ref (7.07) -> delta = {new_mean - 7.07:+.4f}")
