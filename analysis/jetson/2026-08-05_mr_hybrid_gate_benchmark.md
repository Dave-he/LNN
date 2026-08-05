---
title: MR-hybrid_gate-CfC (N13: MR × hybrid_gate × TFP × CfC) — three-layer synthesis — 2026-08-05
date: 2026-08-05
tags: [LNN, CfC, TFP, hybrid_gate, multi-rate, MR-MoE, three-layer-synthesis, N13]
---

# MR-hybrid_gate-CfC (N13) — three-layer synthesis — 2026-08-05

## Setup
- 7 models (5 single-expert + 2 multi-rate), irregular dt training
- 192 samples × 32 seq_len × 24 hidden (each MR expert gets 6 dim)
- 2 repeats × 4 epochs

## Results

| model | params | regular MSE | irregular MSE | degradation |
|---|---:|---:|---:|---:|
| cfc-baseline (h=24) | 2137 | 0.0564 ± 0.0004 | 0.0565 ± 0.0005 | **1.00×** |
| mfc-cfc (h=24) | 2137 | 0.0560 ± 0.0001 | 0.0560 ± 0.0001 | **1.00×** |
| mfc-tfp (h=24) | 2113 | 0.0586 ± 0.0039 | 0.0618 ± 0.0010 | **1.05×** |
| mfc-hybrid (static α, h=24) | 2857 | 0.0556 ± 0.0002 | 0.0574 ± 0.0000 | **1.03×** |
| mfc-hybrid_gate (input-dep α, h=24) | 3577 | 0.0558 ± 0.0003 | 0.0579 ± 0.0002 | **1.04×** |
| MR-TFP-CfC (n_tau=4, expert=tfp, h=24) | 833 | 0.0650 ± 0.0002 | 0.0649 ± 0.0002 | **1.00×** |
| MR-hybrid_gate-CfC (n_tau=4, expert=hybrid_gate, h=24) | 1433 | 0.0643 ± 0.0001 | 0.0643 ± 0.0002 | **1.00×** |

## Interpretation

Compares:
- Single-expert (h=24): cfc / mfc-cfc / mfc-tfp / mfc-hybrid / mfc-hybrid_gate
- Multi-rate (n_tau=4, h=24 split as 6 per expert): MR-TFP-CfC / **MR-hybrid_gate-CfC** (N13)

If MR-hybrid_gate-CfC has degradation ratio ≤ 1.00× AND irregular MSE ≤ CfC's,
the three-layer synthesis wins. If it has degradation > 1.05×, the multi-rate
structure at h=24/n_tau=4 (each expert gets only 6 hidden) is too small to
exploit (echoing the b8d8879 finding from round 282).
