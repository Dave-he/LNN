---
title: Jetson LFM2.5 Benchmark - 2026-08-04 - LFM2.5-350M
date: 2026-08-04
tags: [LNN, LFM2.5, Jetson, benchmark, edge-ai]
---

# Jetson LFM2.5 Benchmark - 2026-08-04

## Environment
- Platform: Linux-5.15.148-tegra-aarch64-with-glibc2.35
- Python: 3.10.12
- PyTorch: 2.10.0
- CUDA available: True
- Jetson BSP:

```text
# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
```
- CUDA device: Orin
- Total memory: 7619.79 MB

## Model
- Name: LiquidAI/LFM2.5-350M
- Parameters: 354,483,968
- Size: 676.1 MB (0.66 GB)
- dtype: float16
- Device: cpu
- Load time: 13.20 s

## Performance
- Repeats: 3
- Prompts per repeat: 1
- Tokens generated per run: 64
- KV cache enabled: True

| Metric | Mean | Min | Max |
|---|---:|---:|---:|
| Tokens/s | 9.01 | 8.00 | 9.53 |
| Prefill time (s) | 1.012 | 0.681 | 1.497 |

## Example Generation

```text
Write a short poem about artificial intelligence:

The code whispers secrets in the dark,
Guiding the way, yet leaving no trace.
It learns from experience, it adapts,
A
```
