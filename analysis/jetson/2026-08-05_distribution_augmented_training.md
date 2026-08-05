---
title: Distribution-augmented training (N15): does it fix hybrid_gate OOD transferability? — 2026-08-05
date: 2026-08-05
tags: [LNN, CfC, hybrid_gate, distribution-augmented-training, OOD, transferability, N15]
---

# Distribution-augmented training (N15) — 2026-08-05

## Setup
- **Train dt**: per-batch random sample from LogNormal(0, sigma) for sigma in [0.3, 0.5, 1.0]
  (mixed distributions, force the model to see all 3 distributions during training)
- **Test dt**: per-sigma LogNormal(0, sigma_test) for sigma_test in [0.3, 0.5, 1.0]
- 2 models x 3 sigma_test x 2 repeats x 4 epochs

## Results

| model | σ=0.3 | σ=0.5 | σ=1.0 |
|---|---:|---:|---:|
| cfc-baseline (regular train only) | reg=0.0565 irr=0.0565 **1.00x** | reg=0.0565 irr=0.0565 **1.00x** | reg=0.0565 irr=0.0566 **1.00x** |
| mfc-hybrid_gate (mixed dt train) | reg=0.0563 irr=0.0568 **1.01x** | reg=0.0563 irr=0.0576 **1.02x** | reg=0.0563 irr=0.0601 **1.07x** |

## N12 baseline (single-distribution training, for comparison)

| model | σ=0.3 | σ=0.5 | σ=1.0 |
|---|---:|---:|---:|
| cfc-baseline | **1.00x** | **1.00x** | **1.00x** |
| mfc-hybrid_gate (N11) | **1.01x** | **1.04x** | **1.10x** |

## Verdict (TBD — see results above)

If hybrid_gate mixed-train row stays around **1.00x across all sigma_test**,
distribution-augmented training fixes N12's finding (POSITIVE).
If it stays at 1.10x for OOD like N12, the finding stands (NEGATIVE).

The CfC row should be 1.00x across all (no learning, structural generic).
