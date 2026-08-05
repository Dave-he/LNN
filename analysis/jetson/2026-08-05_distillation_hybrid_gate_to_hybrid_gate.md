---
title: DLNet-style LNN Distillation Pareto Sweep — 2026-08-05 (teacher=hybrid_gate → student=hybrid_gate)
date: 2026-08-05
tags: [LNN, distillation, pareto, edge-ai, DLNet, knowledge-distillation, dual-stage, N1]
arxiv_refs: [2601.06227, 2106.13898]
---

# DLNet-style LNN Distillation Pareto Sweep — 2026-08-05

## Setup
- Teacher hidden: 32
- Student hiddens: [4, 8, 12, 16]
- Task: non-stationary AR(2) + 3-regime + irregular dt (sigma=0.5)
- 2 repeats × 4 epochs

## Results

| student hidden | params | test MSE | train seconds |
|---:|---:|---:|---:|
| 4 | 309 | 0.0699 ± 0.0017 | 12.13 |
| 8 | 745 | 0.0599 ± 0.0028 | 11.73 |
| 12 | 1309 | 0.0573 ± 0.0003 | 31.55 |
| 16 | 2001 | 0.0569 ± 0.0001 | 32.81 |
| 32 | 4993 | 0.0570 ± 0.0008 | 7.93 |

## Pareto frontier

- hidden=4: 309 params, MSE=0.0699
- hidden=8: 745 params, MSE=0.0599
- hidden=12: 1309 params, MSE=0.0573
- hidden=16: 2001 params, MSE=0.0569

## Compression ratio vs teacher

- h=4: 16.16× smaller, MSE delta=+0.0129
- h=8: 6.70× smaller, MSE delta=+0.0030
- h=12: 3.81× smaller, MSE delta=+0.0003
- h=16: 2.50× smaller, MSE delta=-0.0000
