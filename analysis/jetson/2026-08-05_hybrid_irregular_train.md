---
title: MFC-Hybrid retention — trained on irregular Δt to verify α conditional gating (N9) — 2026-08-05
date: 2026-08-05
tags: [LNN, CfC, TFP, hybrid, retention, irregular-dt, conditional-gating, alpha-learning, N9]
---

# MFC-Hybrid retention — trained on irregular Δt (N9 verification)

## Setup
- **Training dt**: LogNormal(0, 0.5) — IRREGULAR, range [0.101, 6.860]
- **Test A (regular)**: dt=1.0 constant
- **Test B (irregular)**: LogNormal(0, 0.5) — same distribution as training

## Hypothesis
If hybrid's α learns **conditional gating** (input-dependent α switching), then:
- α should approach 1 (CfC path) under dt-jitter input that the model trained on
- α might approach 0 (TFP path) under regular dt input that the model didn't train on
Alternatively, α may converge to a single value (no per-input conditioning).

## Results (3 repeats × 4 epochs, mean±std)

| model | test_mse_regular | test_mse_irregular | degradation ratio |
|---|---:|---:|---:|

| cfc | 0.0573 ± 0.0000 | 0.0574 ± 0.0000 | **1.00×** |

| mfc-cfc | 0.0572 ± 0.0001 | 0.0573 ± 0.0002 | **1.00×** |

| mfc-tfp | 0.0575 ± 0.0001 | 0.0605 ± 0.0002 | **1.05×** |

| mfc-hybrid | 0.0576 ± 0.0001 | 0.0582 ± 0.0002 | **1.01×** |


**mfc-hybrid α trajectory (epoch-end mean over 3 runs)**: [0.501, 0.525, 0.557, 0.576]


## Verdict
TBD — see report.
