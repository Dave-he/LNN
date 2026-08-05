---
title: Lorenz attractor retention validation (N18) — 2026-08-05
date: 2026-08-05
tags: [LNN, CfC, TFP, hybrid_gate, lorenz, nonlinear-ODE, N18, real-world-validation]
---

# Lorenz attractor retention validation (N18) — 2026-08-05

## Setup
- Task: Lorenz attractor next-step prediction
  - dx/dt = sigma * (y - x)
  - dy/dt = x * (rho - z) - y
  - dz/dt = x * y - beta * z
  - sigma=10, rho=28, beta=8/3 → chaotic
- sl=96, h=32
- Train dt: regular (sigma=0)
- Test dt: regular, in-dist irregular (sigma=0.5), OOD irregular (sigma=1.0)

## Results

| model | regular MSE | in-dist irregular | OOD irregular |
|---|---:|---:|---:|
| cfc-baseline | 2.8871 | 3.1993 (1.11x) | 1.5202 (0.53x) |
| mfc-hybrid_gate | 3.8679 | 3.8962 (1.01x) | 0.2336 (0.06x) |
| mr-hybrid-gate-cfc (n_tau=4) | 19.9591 | 19.9605 (1.00x) | 5.4475 (0.27x) |
