---
title: DLNet-style LNN Distillation Pareto Sweep — 2026-08-05
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
| 4 | 249 | 0.0632 ± 0.0059 | 7.06 |
| 8 | 593 | 0.0570 ± 0.0004 | 6.20 |
| 12 | 1033 | 0.0570 ± 0.0002 | 6.87 |
| 16 | 1569 | 0.0563 ± 0.0005 | 6.53 |
| 32 | 3617 | 0.0571 ± 0.0006 | 4.78 |

## Pareto frontier

- hidden=4: 249 params, MSE=0.0632
- hidden=8: 593 params, MSE=0.0570
- hidden=16: 1569 params, MSE=0.0563

## Compression ratio vs teacher

- h=4: 14.53× smaller, MSE delta=+0.0061
- h=8: 6.10× smaller, MSE delta=-0.0001
- h=12: 3.50× smaller, MSE delta=-0.0001
- h=16: 2.31× smaller, MSE delta=-0.0008
