# Natural Gas LNN Forecaster ablation

_Generated: 2026-06-08T08:17:48+00:00_

_Backbones × seeds: 2 × 2_

_Window: 20 days, Hidden: 16, Epochs: 5, Train/Val split: 80/10 chronological_

_Data: synthetic Henry Hub from ``lnn.data.natural_gas_generator`` (2645 business days)_


## Per-backbone metrics (lower MAPE better, higher directional accuracy better)

| Backbone | n_params | median_mape (%) | dir_acc_7d (%) | train_s |
|---|---:|---:|---:|---:|
| gru ⚠️N<3 (n=2) | 929 | 101.13±2.15 | 54.49±0.98 | 11.1 |
| lstm ⚠️N<3 (n=2) | 1233 | 100.12±0.65 | 52.93±0.59 | 5.3 |

## Key deltas (vs LSTM)

| Comparison | Δmape (pp) | Δdir_acc_7d (pp) | Verdict |
|---|---:|---:|---|
| gru | +1.01 | +1.56 | 🟰 mixed |

## Per-seed raw metrics


### gru

| seed | median_mape | dir_acc_7d |
|---:|---:|---:|
| 42 | 98.97 | 55.47 |
| 1153 | 103.28 | 53.52 |

### lstm

| seed | median_mape | dir_acc_7d |
|---:|---:|---:|
| 42 | 99.47 | 52.34 |
| 1153 | 100.77 | 53.52 |
