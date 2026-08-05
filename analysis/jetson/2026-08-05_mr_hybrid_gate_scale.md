---
title: MR-hybrid_gate-CfC at h≥64 (N14) — verify N13 honest finding — 2026-08-05
date: 2026-08-05
tags: [LNN, MR-MoE, hybrid_gate, multi-rate, scale-up, N14, n_tau=4]
---

# MR-hybrid_gate-CfC at h≥64 (N14) — verify N13 honest finding

## Setup
- Task: AR(2) + 3-regime + irregular dt (sigma=0.5)
- Models: CfC, mfc-hybrid_gate (single expert), MR-hybrid_gate-CfC (n_tau=4)
- Sweep h ∈ [64]

## Results

| model | h | per_expert | params | reg MSE | irr MSE | train s |
|---|---:|---:|---:|---:|---:|---:|
| cfc | 64 | 64 | 7241 | 0.0618 ± 0.0004 | 0.0618 ± 0.0004 | 7.6 |
| mfc-hybrid_gate | 64 | 64 | 12105 | 0.0606 ± 0.0014 | 0.0606 ± 0.0016 | 11.4 |
| mr-hybrid-gate-cfc | 64 | 16 | 6993 | 0.0643 ± 0.0021 | 0.0645 ± 0.0021 | 33.5 |
