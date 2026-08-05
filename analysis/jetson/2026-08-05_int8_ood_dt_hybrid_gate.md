---
title: Int8 quantization on OOD dt (N23) — hybrid_gate teacher
date: 2026-08-05
tags: [LNN, int8, quantization, distillation, OOD-dt, N23]
---

# Int8 quantization on OOD dt (N23) — hybrid_gate teacher

## Setup
- Teacher (hybrid_gate, h=32) → student (CfC, h=8)
- Train dt: regular (sigma=0)
- Test dt: regular (0), in-dist (0.5), OOD (1.0)

## Results

| test dt σ | fp32 MSE | int8 MSE | delta |
|---:|---:|---:|---:|
| 0.0 | 0.0520 | 0.0520 | **+0.0000** |
| 0.5 | 0.0526 | 0.0526 | **+0.0000** |
| 1.0 | 0.0535 | 0.0535 | **+0.0000** |
