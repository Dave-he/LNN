---
title: α MLP capacity hypothesis (N22) — does deeper/wider α break interpolation ceiling? — 2026-08-05
date: 2026-08-05
tags: [LNN, hybrid_gate, alpha-capacity, MLP-depth, OOD, N22]
---

# α MLP capacity hypothesis (N22) — 2026-08-05

## Setup
- Task: AR(2) + 3-regime, mixed-dt training (sigma in {0.3, 0.5, 1.0})
- Test on 3 sigma values (N15 setup for direct comparison)
- 5 α capacity variants + CfC baseline

## Results (degradation ratio)

| model | params | σ=0.3 | σ=0.5 | σ=1.0 (OOD) |
|---|---:|---:|---:|---:|
| cfc-baseline | 2137 | **1.00x** | **1.00x** | **1.00x** |
| mfc-hybrid_gate (depth=1, w=branch_dim, N11/N15 baseline) | 2977 | **1.01x** | **1.03x** | **1.07x** |
| mfc-hybrid_gate (depth=2, w=2*branch_dim) | 3577 | **1.01x** | **1.02x** | **1.07x** |
| mfc-hybrid_gate (depth=3, w=2*branch_dim) | 4177 | **1.01x** | **1.03x** | **1.08x** |
| mfc-hybrid_gate (depth=3, w=4*branch_dim) | 4177 | **1.01x** | **1.03x** | **1.08x** |
