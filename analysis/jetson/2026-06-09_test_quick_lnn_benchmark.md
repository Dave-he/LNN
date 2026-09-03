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
| Front | 模型 | Hidden | SeqLen | Seed | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 | VDD_IN mJ/步 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| yes | NCPS-CfC | 16 | 32 | 42 | 10577 | 0.217156 | 32271.9 | 1.80 | 0.34 |
| yes | NCPS-CfC | 8 | 32 | 42 | 5417 | 0.227616 | 49044.1 | 1.83 | 0.20 |
| yes | NCPS-CfC | 16 | 16 | 42 | 10577 | 0.363849 | 51340.4 | 0.90 | n/a |
| yes | NCPS-CfC | 8 | 16 | 42 | 5417 | 0.371024 | 42667.7 | 0.88 | 0.23 |
| yes | PDNAPulse | 8 | 32 | 42 | 418 | 0.401099 | 29655.6 | 1.32 | 0.33 |
| yes | PDNAPulse | 16 | 32 | 42 | 1474 | 0.449062 | 51282.7 | 2.02 | 0.19 |
| yes | CfCStyle | 16 | 32 | 42 | 1169 | 0.470368 | 33890.4 | 1.41 | 0.24 |
| yes | GRU | 8 | 32 | 42 | 273 | 0.483984 | 177735.0 | 0.46 | n/a |
| yes | GRU | 16 | 32 | 42 | 929 | 0.567266 | 251959.0 | 0.53 | n/a |
| yes | GRU | 16 | 16 | 42 | 929 | 0.575463 | 237340.1 | 0.42 | n/a |
| yes | GRU | 8 | 16 | 42 | 273 | 0.601100 | 198252.2 | 0.21 | n/a |
| yes | LTC | 8 | 32 | 42 | 185 | 0.607747 | 20735.1 | 3.79 | 0.42 |
| yes | LTC | 8 | 16 | 42 | 185 | 0.654283 | 20793.1 | 1.84 | 0.41 |
|  | LTC | 16 | 32 | 42 | 625 | 0.508113 | 19495.7 | 3.86 | 0.48 |
|  | PDNAPulse | 16 | 16 | 42 | 1474 | 0.525087 | 36925.9 | 1.00 | 0.27 |
|  | NCPS-LTC | 16 | 32 | 42 | 1187 | 0.539707 | 31837.8 | 4.58 | 0.25 |
|  | CfCStyle | 8 | 32 | 42 | 329 | 0.562273 | 42779.2 | 1.13 | 0.20 |
|  | NCPS-LTC | 16 | 16 | 42 | 1187 | 0.568193 | 31761.2 | 2.22 | 0.25 |
|  | LTC | 16 | 16 | 42 | 625 | 0.572051 | 20097.1 | 1.86 | 0.43 |
|  | NCPS-LTC | 8 | 16 | 42 | 339 | 0.572990 | 44454.0 | 1.74 | 0.17 |
|  | CfCStyle | 8 | 16 | 42 | 329 | 0.590963 | 66718.8 | 0.67 | n/a |
|  | CfCStyle | 16 | 16 | 42 | 1169 | 0.621814 | 57902.0 | 0.71 | n/a |
|  | NCPS-LTC | 8 | 32 | 42 | 339 | 0.678020 | 45405.7 | 3.65 | 0.17 |
|  | PDNAPulse | 8 | 16 | 42 | 418 | 0.721411 | 38831.6 | 0.92 | 0.27 |

## Pareto 图
![Jetson LNN Pareto](2026-06-09_test_quick_lnn_pareto.png)

## 解读
- Pareto front 表示没有其他配置能同时做到更低误差、更少参数、更短训练时间和更高吞吐。
- 该 sweep 是边缘筛选入口，正式实验应在真实 Jetson CUDA 路径上增加多 seed、能耗和导出后延迟。
- `NCPS-LTC` / `NCPS-CfC` 是官方实现 (mlech26l/ncps)，与仓库内的近似实现做对比。
