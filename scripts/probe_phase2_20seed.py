"""20-seed validation of phase2 inject=0.10.

Round 60 (41st meta): 10-seed mean 9.98 (up from 5-seed 7.07).
Round 61: extends to 20 seeds for stronger statistical evidence.

20 seeds = 5 from round 57/59 + 5 from round 60 + 10 NEW.
If 20-seed mean is similar to 10-seed (9.98), the mean is stable.
If 20-seed mean continues to rise, the 5-seed ref was substantially
lucky and the true production mean is higher.

Probe: 20 seeds x 4 folds = 80 fold runs (~9s each, ~12 min)
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
# 20 seeds: 10 from round 60 + 10 NEW
SEEDS = [
    1, 2, 3, 7, 42,        # original 5 (round 57/59)
    11, 100, 2026, 313, 777,  # round 60's new 5
    55, 99, 314, 555, 888,    # round 61 batch 1
    1024, 2027, 3141, 4242, 9999,  # round 61 batch 2
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

print("=== 20-seed validation: phase2 inject=0.10 ===")
print(f"h={HIDDEN} ep={EPOCHS} warmup={WARMUP} inject={INJECT_SIGMA}")
print(f"seeds={len(SEEDS)} folds={N_FOLDS}")
print(f"Total: {len(SEEDS) * N_FOLDS} fold runs")

per_seed = []
per_seed_times = []
for i, seed in enumerate(SEEDS):
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
    valid = [m for m in per_fold if m == m]
    mean = sum(valid) / len(valid) if valid else float("nan")
    per_seed.append(mean)
    elapsed = time.perf_counter() - start_seed
    per_seed_times.append(elapsed)
    print(f"  [{i+1}/{len(SEEDS)}] seed={seed:>5d} | LOO mean = {mean:.4f} | {elapsed:.1f}s")
    # Save partial result
    out_partial = {
        "config": {"hidden_size": HIDDEN, "epochs": EPOCHS, "warmup_epochs": WARMUP,
                   "lr": LR, "seeds": SEEDS, "inject_sigma": INJECT_SIGMA,
                   "folds": N_FOLDS, "protocol": "TemporalSegmentRegressionDataset 4-fold LOO",
                   "model": "CrossModalAttnBiCfCNADWithMDN"},
        "per_seed_loo_mean": per_seed,
        "partial": True,
        "metadata": {
            "round": 61,
            "follows_up": "round 60 (10-seed mean 9.98, 41st meta-refinement)",
        },
    }
    now = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_path_partial = ROOT / "analysis" / "emma_rover" / f"{now}_phase2_20seed_partial.json"
    with open(out_path_partial, "w") as f:
        json.dump(out_partial, f, indent=2)

# Aggregate
valid_p = [m for m in per_seed if m == m]
mean_all = sum(valid_p) / len(valid_p) if valid_p else float("nan")
std_all = (sum((m - mean_all) ** 2 for m in valid_p) / max(1, len(valid_p) - 1)) ** 0.5 if valid_p else float("nan")

# Subsets
original_5 = [per_seed[SEEDS.index(s)] for s in [1, 2, 3, 7, 42]]
round60_5 = [per_seed[SEEDS.index(s)] for s in [11, 100, 2026, 313, 777]]
new_10 = [per_seed[SEEDS.index(s)] for s in [55, 99, 314, 555, 888, 1024, 2027, 3141, 4242, 9999]]
orig_mean = sum(original_5) / len(original_5)
round60_mean = sum(round60_5) / len(round60_5)
new_mean = sum(new_10) / len(new_10)

print("\n=== Per-seed 20-seed LOO means ===")
for seed, mse in zip(SEEDS, per_seed):
    print(f"  seed={seed:>5d} | LOO mean = {mse:.4f}")

print(f"\n=== Aggregate ===")
print(f"  Original 5 (1,2,3,7,42) mean = {orig_mean:.4f} (round 57/59 ref)")
print(f"  Round 60's 5 (11,100,2026,313,777) mean = {round60_mean:.4f}")
print(f"  Round 61's NEW 10 (55,99,...) mean = {new_mean:.4f}")
print(f"  20-seed mean = {mean_all:.4f} ± {std_all:.4f}")

# Save final
out = {
    "config": {"hidden_size": HIDDEN, "epochs": EPOCHS, "warmup_epochs": WARMUP,
               "lr": LR, "seeds": SEEDS, "inject_sigma": INJECT_SIGMA,
               "folds": N_FOLDS, "protocol": "TemporalSegmentRegressionDataset 4-fold LOO",
               "model": "CrossModalAttnBiCfCNADWithMDN"},
    "per_seed_loo_mean": per_seed,
    "per_seed_elapsed_s": per_seed_times,
    "subset_means": {
        "original_5": {"seeds": [1, 2, 3, 7, 42], "mean": orig_mean, "individual": original_5},
        "round60_5": {"seeds": [11, 100, 2026, 313, 777], "mean": round60_mean, "individual": round60_5},
        "round61_new10": {"seeds": [55, 99, 314, 555, 888, 1024, 2027, 3141, 4242, 9999], "mean": new_mean, "individual": new_10},
    },
    "metadata": {
        "round": 61,
        "follows_up": "round 60 (10-seed mean 9.98, 41st meta-refinement)",
        "round_57_5seed_ref": 7.07,
        "round_60_10seed_ref": 9.98,
    },
}
now = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
out_path = ROOT / "analysis" / "emma_rover" / f"{now}_phase2_20seed.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nWrote: {out_path}")

# Verdict
print("\n=== Verdict ===")
print(f"  5-seed ref: 7.07 (round 57/59)")
print(f"  10-seed ref: 9.98 (round 60)")
print(f"  20-seed new: {mean_all:.4f} (round 61)")
print(f"  Delta 20-seed vs 5-seed: {mean_all - 7.07:+.4f} ({(mean_all - 7.07)/7.07*100:+.1f}%)")
print(f"  Delta 20-seed vs 10-seed: {mean_all - 9.98:+.4f} ({(mean_all - 9.98)/9.98*100:+.1f}%)")
if abs(mean_all - 9.98) < 1.0:
    print(f"  ★ 20-seed mean ≈ 10-seed mean (9.98) -> STABLE")
elif mean_all > 9.98:
    print(f"  20-seed mean still rising -> need even more seeds")
else:
    print(f"  20-seed mean LOWER than 10-seed -> 10-seed was slightly unlucky")
