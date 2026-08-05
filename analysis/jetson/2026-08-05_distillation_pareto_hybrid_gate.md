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
| 4 | 249 | 0.0571 ± 0.0003 | 9.03 |
| 8 | 593 | 0.0569 ± 0.0009 | 9.63 |
| 12 | 1033 | 0.0571 ± 0.0006 | 9.21 |
| 16 | 1569 | 0.0563 ± 0.0002 | 9.03 |
| 32 | 6049 | 0.0572 ± 0.0008 | 8.59 |

## Pareto frontier

- hidden=4: 249 params, MSE=0.0571
- hidden=8: 593 params, MSE=0.0569
- hidden=16: 1569 params, MSE=0.0563

## Compression ratio vs teacher

- h=4: 24.29× smaller, MSE delta=-0.0001
- h=8: 10.20× smaller, MSE delta=-0.0003
- h=12: 5.86× smaller, MSE delta=-0.0002
- h=16: 3.86× smaller, MSE delta=-0.0010
