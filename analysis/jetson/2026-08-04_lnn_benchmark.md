---
title: Jetson LNN 基准验证 - 2026-08-04
date: 2026-08-04
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-08-04

## 环境
- 平台：Linux-5.15.148-tegra-aarch64-with-glibc2.35
- 设备树型号：unknown
- PyTorch：2.10.0
- CUDA：True (12.6)
- Jetson BSP：

```text
# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
```
- CUDA 设备：Orin，显存 7619.79 MB

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
| yes | NCPS-CfC | 16 | 32 | 42 | 10577 | 0.217156 | 18891.9 | 2.25 |
| yes | NCPS-CfC | 8 | 32 | 42 | 5417 | 0.227616 | 24440.1 | 2.58 |
| yes | NCPS-CfC | 16 | 16 | 42 | 10577 | 0.363849 | 22609.9 | 1.20 |
| yes | NCPS-CfC | 8 | 16 | 42 | 5417 | 0.371024 | 25519.7 | 1.23 |
| yes | PDNAPulse | 8 | 32 | 42 | 418 | 0.401099 | 134253.2 | 0.87 |
| yes | CfCStyle | 16 | 32 | 42 | 1169 | 0.470368 | 136784.4 | 0.92 |
| yes | GRU | 8 | 32 | 42 | 273 | 0.483984 | 568409.5 | 0.41 |
| yes | GRU | 16 | 16 | 42 | 929 | 0.575463 | 275887.1 | 0.30 |
| yes | GRU | 8 | 16 | 42 | 273 | 0.601100 | 660580.7 | 0.22 |
| yes | LTC | 8 | 32 | 42 | 185 | 0.607747 | 32407.8 | 3.22 |
| yes | LTC | 8 | 16 | 42 | 185 | 0.654283 | 29655.1 | 1.67 |
|  | PDNAPulse | 16 | 32 | 42 | 1474 | 0.449062 | 32429.2 | 1.99 |
|  | LTC | 16 | 32 | 42 | 625 | 0.508113 | 30444.3 | 3.44 |
|  | PDNAPulse | 16 | 16 | 42 | 1474 | 0.525087 | 123323.6 | 0.50 |
|  | NCPS-LTC | 16 | 32 | 42 | 1187 | 0.539707 | 27971.7 | 6.01 |
|  | CfCStyle | 8 | 32 | 42 | 329 | 0.562273 | 154659.9 | 1.13 |
|  | GRU | 16 | 32 | 42 | 929 | 0.567266 | 351706.9 | 0.46 |
|  | NCPS-LTC | 16 | 16 | 42 | 1187 | 0.568193 | 30432.9 | 2.59 |
|  | LTC | 16 | 16 | 42 | 625 | 0.572051 | 33821.2 | 1.46 |
|  | NCPS-LTC | 8 | 16 | 42 | 339 | 0.572990 | 35436.8 | 2.28 |
|  | CfCStyle | 8 | 16 | 42 | 329 | 0.590963 | 147066.1 | 0.70 |
|  | CfCStyle | 16 | 16 | 42 | 1169 | 0.621814 | 144267.7 | 0.44 |
|  | NCPS-LTC | 8 | 32 | 42 | 339 | 0.678020 | 45067.1 | 4.11 |
|  | PDNAPulse | 8 | 16 | 42 | 418 | 0.721411 | 127784.5 | 0.47 |

## Pareto 图
![Jetson LNN Pareto](2026-08-04_lnn_pareto.png)

## 解读
- Pareto front 表示没有其他配置能同时做到更低误差、更少参数、更短训练时间和更高吞吐。
- 该 sweep 是边缘筛选入口，正式实验应在真实 Jetson CUDA 路径上增加多 seed、能耗和导出后延迟。
- `NCPS-LTC` / `NCPS-CfC` 是官方实现 (mlech26l/ncps)，与仓库内的近似实现做对比。
