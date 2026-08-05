---
title: MR routing on long-sequence / multi-scale tasks (N24) — 2026-08-05
date: 2026-08-05
tags: [LNN, MR-MoE, hybrid_gate, multi-rate, long-sequence, multi-scale, N24]
---

# MR routing on long-sequence / multi-scale tasks (N24) — 2026-08-05

## Setup
- Task: multi-scale non-stationary (8 regimes, sinusoidal + AR mixed)
- sl=96, h=64, n_tau=4
- 8 regimes with distinct *frequency content* (0.05–0.70 Hz carriers)

## Results

| model | per_expert | params | test MSE | train s |
|---|---:|---:|---:|---:|
| cfc | 64 | 7241 | 0.2496 ± 0.0062 | 32.4 |
| mfc-hybrid_gate | 64 | 12105 | 0.2692 ± 0.0071 | 46.2 |
| mr-hybrid-gate-cfc | 16 | 6993 | 0.1618 ± 0.0310 | 144.1 |
