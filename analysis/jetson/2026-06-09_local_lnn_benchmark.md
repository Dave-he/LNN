---
title: Jetson LNN 基准验证 - 2026-06-09_local
date: 2026-06-09_local
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-06-09_local

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
| yes | CfCStyle | 16 | 32 | 42 | 1169 | 0.470338 | 21960.3 | 4.11 |
| yes | GRU | 16 | 32 | 42 | 929 | 0.536350 | 98844.5 | 1.45 |
| yes | GRU | 16 | 16 | 42 | 929 | 0.547923 | 110075.8 | 0.99 |
| yes | GRU | 8 | 16 | 42 | 273 | 0.558936 | 92016.6 | 0.31 |
| yes | GRU | 8 | 32 | 42 | 273 | 0.565124 | 118700.2 | 0.65 |
|  | CfCStyle | 8 | 32 | 42 | 329 | 0.561916 | 28667.0 | 2.41 |
|  | CfCStyle | 8 | 16 | 42 | 329 | 0.590879 | 28377.1 | 1.22 |
|  | CfCStyle | 16 | 16 | 42 | 1169 | 0.622440 | 32303.7 | 2.13 |

## Pareto 图
![Jetson LNN Pareto](2026-06-09_local_lnn_pareto.png)

## 解读
- Pareto front 表示没有其他配置能同时做到更低误差、更少参数、更短训练时间和更高吞吐。
- 该 sweep 是边缘筛选入口，正式实验应在真实 Jetson CUDA 路径上增加多 seed、能耗和导出后延迟。
