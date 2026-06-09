---
title: Jetson LNN 基准验证 - 2026-06-09_test_quick
date: 2026-06-09_test_quick
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-06-09_test_quick

## 环境
- 平台：Linux-5.15.148-tegra-aarch64-with-glibc2.35
- 设备树型号：NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
- PyTorch：2.11.0+cu130
- CUDA：False (13.0)
- Jetson BSP：

```text
# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
```

## 任务配置
- 数据：合成非平稳时间序列，一步预测
- Samples / Epoch：384 / 3
- Hidden sweep：[8, 16]
- SeqLen sweep：[16, 32]
- Seeds：[42]
- 设备：cpu

## Pareto 结果
| Front | 模型 | Hidden | SeqLen | Seed | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| yes | PDNAPulse | 8 | 32 | 42 | 418 | 0.401337 | 38910.2 | 1.41 |
| yes | CfCStyle | 16 | 32 | 42 | 1169 | 0.469870 | 46125.6 | 2.28 |
| yes | GRU | 8 | 32 | 42 | 273 | 0.483972 | 230064.1 | 0.41 |
| yes | GRU | 16 | 16 | 42 | 929 | 0.575272 | 98334.9 | 0.34 |
| yes | GRU | 8 | 16 | 42 | 273 | 0.601254 | 132639.9 | 0.27 |
| yes | LTC | 8 | 32 | 42 | 185 | 0.607213 | 14482.4 | 4.28 |
| yes | LTC | 8 | 16 | 42 | 185 | 0.654613 | 17502.5 | 2.14 |
|  | PDNAPulse | 16 | 32 | 42 | 1474 | 0.449496 | 38485.1 | 1.83 |
|  | LTC | 16 | 32 | 42 | 625 | 0.506501 | 12951.8 | 5.56 |
|  | PDNAPulse | 16 | 16 | 42 | 1474 | 0.525346 | 50150.2 | 1.09 |
|  | CfCStyle | 8 | 32 | 42 | 329 | 0.562039 | 37468.2 | 1.38 |
|  | GRU | 16 | 32 | 42 | 929 | 0.567335 | 150206.8 | 0.56 |
|  | LTC | 16 | 16 | 42 | 625 | 0.571133 | 13652.8 | 2.58 |
|  | CfCStyle | 8 | 16 | 42 | 329 | 0.591101 | 40371.7 | 0.68 |
|  | CfCStyle | 16 | 16 | 42 | 1169 | 0.622324 | 40214.0 | 0.79 |
|  | PDNAPulse | 8 | 16 | 42 | 418 | 0.721614 | 52349.2 | 0.66 |

## Pareto 图
![Jetson LNN Pareto](2026-06-09_test_quick_lnn_pareto.png)

## 解读
- Pareto front 表示没有其他配置能同时做到更低误差、更少参数、更短训练时间和更高吞吐。
- 该 sweep 是边缘筛选入口，正式实验应在真实 Jetson CUDA 路径上增加多 seed、能耗和导出后延迟。
