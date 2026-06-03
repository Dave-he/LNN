---
title: Jetson LNN 基准验证 - 2026-06-03
date: 2026-06-03
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-06-03

## 环境
- 平台：Linux-5.15.148-tegra-aarch64-with-glibc2.35
- 设备树型号：NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
- PyTorch：2.10.0
- CUDA：True (12.6)
- Jetson BSP：

```text
# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
```
- CUDA 设备：Orin，显存 7619.78 MB

## 任务配置
- 数据：合成非平稳时间序列，一步预测
- Samples / Epoch：96 / 2
- Hidden sweep：[8, 16, 24]
- SeqLen sweep：[16, 32]
- Seeds：[42]
- 设备：cpu

## Pareto 结果
| Front | 模型 | Hidden | SeqLen | Seed | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| yes | CfCStyle | 24 | 32 | 42 | 2521 | 0.428481 | 48081.0 | 0.43 |
| yes | CfCStyle | 24 | 16 | 42 | 2521 | 0.554439 | 47724.4 | 0.22 |
| yes | CfCStyle | 8 | 32 | 42 | 329 | 0.563227 | 47201.9 | 0.41 |
| yes | GRU | 8 | 32 | 42 | 273 | 0.584524 | 274086.6 | 0.15 |
| yes | GRU | 16 | 16 | 42 | 929 | 0.620457 | 221062.2 | 0.09 |
| yes | GRU | 8 | 16 | 42 | 273 | 0.651102 | 241817.5 | 0.09 |
|  | CfCStyle | 16 | 32 | 42 | 1169 | 0.610435 | 49155.4 | 0.43 |
|  | GRU | 16 | 32 | 42 | 929 | 0.612124 | 245763.7 | 0.17 |
|  | GRU | 24 | 32 | 42 | 1969 | 0.617864 | 188596.8 | 0.18 |
|  | CfCStyle | 8 | 16 | 42 | 329 | 0.632338 | 50900.5 | 0.34 |
|  | GRU | 24 | 16 | 42 | 1969 | 0.634277 | 197556.9 | 0.09 |
|  | CfCStyle | 16 | 16 | 42 | 1169 | 0.656239 | 49471.7 | 0.22 |

## 解读
- Pareto front 表示没有其他配置能同时做到更低误差、更少参数、更短训练时间和更高吞吐。
- 该 sweep 是边缘筛选入口，正式实验应在真实 Jetson CUDA 路径上增加多 seed、能耗和导出后延迟。
