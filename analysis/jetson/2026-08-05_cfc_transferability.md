---
title: CfC transferability on harder tasks (N16) — 2026-08-05
date: 2026-08-05
tags: [LNN, CfC, TFP, hybrid_gate, transferability, multi-regime, structural-generic, N16]
---

# CfC transferability on harder tasks (N16) — 2026-08-05

## Setup
- 6 task variants of increasing difficulty (all AR(2) family)
- Same dt distribution train (sigma=0.5) and test (sigma=0.5) — in-dist for dt
- 3 models x 6 tasks x 2 repeats x 4 epochs

## Tasks
- **3-regime (N12 baseline)**: {'n_regimes': 3, 'regime_overlap': False, 'intra_drift': False, 'seq_len': 32}
- **5-regime**: {'n_regimes': 5, 'regime_overlap': False, 'intra_drift': False, 'seq_len': 32}
- **8-regime**: {'n_regimes': 8, 'regime_overlap': False, 'intra_drift': False, 'seq_len': 32}
- **3-regime + intra-drift**: {'n_regimes': 3, 'regime_overlap': False, 'intra_drift': True, 'seq_len': 32}
- **3-regime + overlap**: {'n_regimes': 4, 'regime_overlap': True, 'intra_drift': False, 'seq_len': 32}
- **3-regime long (sl=96)**: {'n_regimes': 3, 'regime_overlap': False, 'intra_drift': False, 'seq_len': 96}

## Results (degradation ratio)

| task | cfc-baseline | mfc-tfp | mfc-hybrid_gate |
|---|---:|---:|---:|
| 3-regime (N12 baseline) | **1.00x** | 1.05x | 1.04x |
| 5-regime | **1.00x** | 1.05x | 1.03x |
| 8-regime | **1.00x** | 1.11x | 1.05x |
| 3-regime + intra-drift | **1.00x** | 1.05x | 1.00x |
| 3-regime + overlap | **1.00x** | 1.18x | 1.04x |
| 3-regime long (sl=96) | **1.00x** | 1.07x | 1.01x |

## Verdict (TBD)

A "structural-generic" mechanism (N12 finding) should maintain
≈ 1.00x degradation across ALL task variants. If cfc-baseline
breaks on harder tasks, the finding is limited to 3-regime AR(2).
