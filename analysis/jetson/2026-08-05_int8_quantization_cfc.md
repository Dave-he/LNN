---
title: Int8 Quantization on Distillation Students (N20) — cfc teacher
date: 2026-08-05
tags: [LNN, int8, quantization, distillation, edge-ai, DLNet, N20, Stage-3]
---

# Int8 Quantization on Distillation Students (N20) — cfc teacher

## Setup
- Teacher (cfc, h=32) → student distillation
- Apply per-channel int8 quantization to student
- Compare float32 vs int8 MSE

## Results

| student h | params | fp32 MSE | int8 MSE | delta | int8 bytes | fp32 bytes | compression |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 249 | 0.0632 ± 0.0059 | 0.0632 ± 0.0060 | **-0.0000** | 113 | 452 | **4.0×** |
| 8 | 593 | 0.0570 ± 0.0004 | 0.0570 ± 0.0004 | **+0.0000** | 321 | 1284 | **4.0×** |
| 12 | 1033 | 0.0570 ± 0.0002 | 0.0570 ± 0.0002 | **+0.0000** | 625 | 2500 | **4.0×** |
| 16 | 1569 | 0.0563 ± 0.0005 | 0.0563 ± 0.0005 | **-0.0000** | 1025 | 4100 | **4.0×** |
