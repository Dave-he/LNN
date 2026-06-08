# PDNA stage C — synthetic Pathfinder (LRA-style long-range) ablation

_Generated: 2026-06-08T11:48:03+00:00_

_Variants × seeds: 3 × 2_

_Hidden size: 32, Epochs: 2, Train samples: 300, Test samples: 100_

_Seq len: 1024 (= 32x32 pixel sequence), Grid: 32x32_


## Per-variant test accuracy

| Variant | n_params | test_acc (mean ± std) | train_s |
|---|---:|---:|---:|
| baseline_cfc ⚠️N<3 (n=2) | 3362 | 51.00±1.41 | 317.8 |
| cfc_pulse ⚠️N<3 (n=2) | 5540 | 49.00±1.41 | 459.0 |
| full_pdna ⚠️N<3 (n=2) | 5540 | 49.00±1.41 | 881.3 |

## Key deltas (vs baseline_cfc)

| Comparison | Δtest_acc (pp) | Verdict |
|---|---:|---|
| cfc_pulse | -2.00 | ❌ worse |
| full_pdna | -2.00 | ❌ worse |

## Per-seed raw accuracies


### baseline_cfc

| seed | test_acc |
|---:|---:|
| 42 | 50.00 |
| 1153 | 52.00 |

### cfc_pulse

| seed | test_acc |
|---:|---:|
| 42 | 50.00 |
| 1153 | 48.00 |

### full_pdna

| seed | test_acc |
|---:|---:|
| 42 | 50.00 |
| 1153 | 48.00 |
