# PDNA stage C — synthetic Pathfinder (LRA-style long-range) ablation

_Generated: 2026-06-08T07:28:40+00:00_

_Variants × seeds: 3 × 1_

_Hidden size: 32, Epochs: 1, Train samples: 200, Test samples: 80_

_Seq len: 1024 (= 32x32 pixel sequence), Grid: 32x32_


## Per-variant test accuracy

| Variant | n_params | test_acc (mean ± std) | train_s |
|---|---:|---:|---:|
| baseline_cfc ⚠️N<3 (n=1) | 3362 | 48.75±0.00 | 115.9 |
| cfc_pulse ⚠️N<3 (n=1) | 5540 | 48.75±0.00 | 140.4 |
| full_pdna ⚠️N<3 (n=1) | 5540 | 48.75±0.00 | 120.8 |

## Key deltas (vs baseline_cfc)

| Comparison | Δtest_acc (pp) | Verdict |
|---|---:|---|
| cfc_pulse | +0.00 | 🟰 mixed |
| full_pdna | +0.00 | 🟰 mixed |

## Per-seed raw accuracies


### baseline_cfc

| seed | test_acc |
|---:|---:|
| 42 | 48.75 |

### cfc_pulse

| seed | test_acc |
|---:|---:|
| 42 | 48.75 |

### full_pdna

| seed | test_acc |
|---:|---:|
| 42 | 48.75 |
