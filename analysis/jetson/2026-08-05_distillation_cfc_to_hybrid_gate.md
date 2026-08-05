---
title: DLNet-style LNN Distillation Pareto Sweep — 2026-08-05 (teacher=cfc → student=hybrid_gate)
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
| 4 | 309 | 0.0685 ± 0.0003 | 10.04 |
| 8 | 745 | 0.0624 ± 0.0075 | 12.96 |
| 12 | 1309 | 0.0593 ± 0.0034 | 10.78 |
| 16 | 2001 | 0.0572 ± 0.0004 | 10.22 |
| 32 | 3617 | 0.0571 ± 0.0006 | 6.02 |

## Pareto frontier

- hidden=4: 309 params, MSE=0.0685
- hidden=8: 745 params, MSE=0.0624
- hidden=12: 1309 params, MSE=0.0593
- hidden=16: 2001 params, MSE=0.0572
- hidden=32: 3617 params, MSE=0.0571

## Compression ratio vs teacher

- h=4: 11.71× smaller, MSE delta=+0.0114
- h=8: 4.86× smaller, MSE delta=+0.0053
- h=12: 2.76× smaller, MSE delta=+0.0023
- h=16: 1.81× smaller, MSE delta=+0.0001
