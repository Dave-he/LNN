---
title: MemoryFusionCfC Pareto Sweep — hidden × seq_len grid (2026-08-05)
date: 2026-08-05
tags: [LNN, CfC, TFP, NSFD, cross-paper, retention, memory-fusion, pareto, benchmark]
---

# MemoryFusionCfC Pareto Sweep — 2026-08-05

## Grid
- hidden sizes: [16, 32]
- seq lengths: [32, 64]
- repeats: 2, epochs: 2, batch: 8, lr: 0.01
- n_samples: 384
- task: synthetic non-stationary AR(2) + 3-regime

## Results table

| model | hidden | seq_len | params | test MSE (mean ± std) | inf steps/s | train s |
|---|---|---|---:|---:|---:|---:|
| cfc | 16 | 32 | 1041 | 0.0562 ± 0.0001 | 4316.6 | 6.80 |
| mfc-cfc | 16 | 32 | 1041 | 0.0561 ± 0.0001 | 3427.6 | 7.01 |
| mfc-tfp | 16 | 32 | 1025 | 0.0575 ± 0.0021 | 4023.0 | 6.82 |
| mfc-nsfd | 16 | 32 | 1361 | 0.0793 ± 0.0181 | 4101.6 | 7.77 |
| gru | 16 | 32 | 1073 | 0.0598 ± 0.0050 | 17160.5 | 3.01 |
| cfc | 16 | 64 | 1041 | 0.0566 ± 0.0001 | 4699.8 | 13.07 |
| mfc-cfc | 16 | 64 | 1041 | 0.0567 ± 0.0001 | 4455.5 | 13.71 |
| mfc-tfp | 16 | 64 | 1025 | 0.0573 ± 0.0016 | 4693.4 | 13.17 |
| mfc-nsfd | 16 | 64 | 1361 | 160.9607 ± 227.5163 | 3907.9 | 14.96 |
| gru | 16 | 64 | 1073 | 0.0612 ± 0.0060 | 18482.5 | 4.75 |
| cfc | 32 | 32 | 3617 | 0.0566 ± 0.0001 | 3660.7 | 9.25 |
| mfc-cfc | 32 | 32 | 3617 | 0.0570 ± 0.0002 | 3389.1 | 9.76 |
| mfc-tfp | 32 | 32 | 3585 | 0.0566 ± 0.0000 | 3980.2 | 9.38 |
| mfc-nsfd | 32 | 32 | 4769 | 0.0680 ± 0.0022 | 2365.0 | 12.46 |
| gru | 32 | 32 | 3681 | 0.0592 ± 0.0037 | 4557.8 | 8.92 |
| cfc | 32 | 64 | 3617 | 0.0572 ± 0.0000 | 1933.9 | 31.50 |
| mfc-cfc | 32 | 64 | 3617 | 0.0567 ± 0.0002 | 2202.1 | 28.13 |
| mfc-tfp | 32 | 64 | 3585 | 0.0564 ± 0.0006 | 1992.2 | 25.52 |
| mfc-nsfd | 32 | 64 | 4769 | 0.0744 ± 0.0020 | 1915.0 | 35.65 |
| gru | 32 | 64 | 3681 | 0.0573 ± 0.0016 | 10160.2 | 11.93 |

## Pareto-frontier analysis
For each (hidden, seq_len) cell, the model with the lowest **test_mse_mean** is the Pareto winner (within-seed noise band). Tied models are listed together.
- h=16, seq_len=32: min MSE = 0.0561  winners = ['mfc-cfc']
- h=16, seq_len=64: min MSE = 0.0566  winners = ['cfc']
- h=32, seq_len=32: min MSE = 0.0566  winners = ['cfc', 'mfc-tfp']
- h=32, seq_len=64: min MSE = 0.0564  winners = ['mfc-tfp']

## Verdict

TBD — see report.
