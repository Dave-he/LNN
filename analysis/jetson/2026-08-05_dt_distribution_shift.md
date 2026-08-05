---
title: dt distribution shift transferability (N12): hybrid_gate vs CfC vs TFP — 2026-08-05
date: 2026-08-05
tags: [LNN, CfC, TFP, hybrid_gate, dt-distribution-shift, transferability, robustness, N12]
---

# dt distribution shift transferability (N12) — 2026-08-05

## Setup
- Train dt: LogNormal(0, 0.5) — IRREGULAR (in-dist for σ_test=0.5)
- Test regular: dt = 1.0
- Test σ_test ∈ {0.3, 0.5, 1.0}: in-dist (0.5), similar (0.3), OOD (1.0)
- 2 repeats × 4 epochs

## Results

| model | σ_test=0.3 reg | irr | ratio | σ_test=0.5 reg | irr | ratio | σ_test=1.0 reg | irr | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cfc-baseline | 0.0564 | 0.0564 | **1.00x** | 0.0564 | 0.0565 | **1.00x** | 0.0564 | 0.0565 | **1.00x** |
| mfc-tfp | 0.0586 | 0.0599 | **1.02x** | 0.0586 | 0.0618 | **1.05x** | 0.0586 | 0.0654 | **1.12x** |
| mfc-hybrid | 0.0556 | 0.0563 | **1.01x** | 0.0556 | 0.0574 | **1.03x** | 0.0556 | 0.0604 | **1.09x** |
| mfc-hybrid_gate | 0.0558 | 0.0566 | **1.01x** | 0.0558 | 0.0579 | **1.04x** | 0.0558 | 0.0615 | **1.10x** |

## Interpretation

A model that *learns general dt-robustness* should have degradation ratio
roughly constant across σ_test values. A model that *overfits to training
dt-distribution* will see degradation increase with σ_test distance from
σ_train=0.5.

Key questions:
1. Does mfc-hybrid_gate transfer (ratio ≈ 1.00x across σ_test)?
2. Or does it overfit (ratio grows with |σ_test - 0.5|)?
3. Compare against mfc-tfp (which should overfit most strongly).
