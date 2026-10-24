---
title: Jetson LNN 基准验证 - 2026-10-25
date: 2026-10-25
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-10-25

> ⚠️ **运行状态：`ok_cpu`**
> CUDA was unavailable, so these are CPU timings measured on the Jetson host. They are not GPU/edge performance evidence.

## 环境
- 平台：Linux-5.15.148-tegra-aarch64-with-glibc2.35
- 设备树型号：NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
- PyTorch：2.11.0+cu130
- CUDA：False (13.0)
- 系统内存：total 7620 MB / free 88 MB / available 709 MB
- Jetson BSP：

```text
# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
```

## 功耗与温度
- 功耗采样：可用 (2334 samples @ 100ms)
- 采样窗口时长：262.33s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3308/3809 mW
  - VDD_IN: 8539/9116 mW
  - VDD_SOC: 1590/1671 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 867843 mJ (867.843 J)
  - VDD_IN: 2240068 mJ (2240.068 J)
  - VDD_SOC: 417244 mJ (417.244 J)
- 温度：
  - cpu: 54.8°C (peak 55.6°C)
  - gpu: 54.9°C (peak 55.6°C)
  - soc0: 53.9°C (peak 54.5°C)
  - soc1: 54.4°C (peak 55.0°C)
  - soc2: 53.5°C (peak 54.0°C)
  - tj: 54.9°C (peak 55.6°C)
- GPU 利用率：mean 0%, peak 0%

## 各模型独立功耗

### CfCStyle
- 功耗采样：可用 (14 samples @ 100ms)
- 采样窗口时长：1.68s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3102/3179 mW
  - VDD_IN: 8290/8399 mW
  - VDD_SOC: 1592/1592 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 5216 mJ (5.216 J)
  - VDD_IN: 13939 mJ (13.939 J)
  - VDD_SOC: 2677 mJ (2.677 J)
- 温度：
  - cpu: 54.3°C (peak 54.5°C)
  - gpu: 54.4°C (peak 54.7°C)
  - soc0: 53.4°C (peak 53.5°C)
  - soc1: 53.9°C (peak 54.1°C)
  - soc2: 53.0°C (peak 53.1°C)
  - tj: 54.4°C (peak 54.7°C)
- GPU 利用率：mean 0%, peak 0%

### GRU
- 功耗采样：可用 (9 samples @ 100ms)
- 采样窗口时长：1.11s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3487/3571 mW
  - VDD_IN: 9045/9116 mW
  - VDD_SOC: 1629/1629 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 3856 mJ (3.857 J)
  - VDD_IN: 10003 mJ (10.003 J)
  - VDD_SOC: 1802 mJ (1.802 J)
- 温度：
  - cpu: 54.9°C (peak 55.2°C)
  - gpu: 54.9°C (peak 55.1°C)
  - soc0: 53.9°C (peak 54.0°C)
  - soc1: 54.4°C (peak 54.4°C)
  - soc2: 53.5°C (peak 53.6°C)
  - tj: 55.0°C (peak 55.2°C)
- GPU 利用率：mean 0%, peak 0%

### LTC
- 功耗采样：可用 (26 samples @ 100ms)
- 采样窗口时长：2.97s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2900/3100 mW
  - VDD_IN: 8006/8213 mW
  - VDD_SOC: 1560/1592 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 8622 mJ (8.622 J)
  - VDD_IN: 23803 mJ (23.803 J)
  - VDD_SOC: 4637 mJ (4.637 J)
- 温度：
  - cpu: 54.2°C (peak 54.7°C)
  - gpu: 54.5°C (peak 54.8°C)
  - soc0: 53.6°C (peak 53.7°C)
  - soc1: 54.0°C (peak 54.2°C)
  - soc2: 53.1°C (peak 53.2°C)
  - tj: 54.5°C (peak 54.8°C)
- GPU 利用率：mean 0%, peak 0%

### NCPS-CfC
- 功耗采样：可用 (33 samples @ 100ms)
- 采样窗口时长：3.80s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3241/3378 mW
  - VDD_IN: 8522/8717 mW
  - VDD_SOC: 1566/1592 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 12314 mJ (12.315 J)
  - VDD_IN: 32385 mJ (32.385 J)
  - VDD_SOC: 5953 mJ (5.953 J)
- 温度：
  - cpu: 54.9°C (peak 55.2°C)
  - gpu: 55.1°C (peak 55.4°C)
  - soc0: 54.1°C (peak 54.3°C)
  - soc1: 54.6°C (peak 54.8°C)
  - soc2: 53.7°C (peak 53.8°C)
  - tj: 55.1°C (peak 55.4°C)
- GPU 利用率：mean 0%, peak 0%

### NCPS-LTC
- 功耗采样：可用 (97 samples @ 100ms)
- 采样窗口时长：10.73s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3143/3298 mW
  - VDD_IN: 8128/8439 mW
  - VDD_SOC: 1531/1592 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 33719 mJ (33.719 J)
  - VDD_IN: 87208 mJ (87.207 J)
  - VDD_SOC: 16426 mJ (16.426 J)
- 温度：
  - cpu: 54.9°C (peak 55.3°C)
  - gpu: 55.1°C (peak 55.4°C)
  - soc0: 54.1°C (peak 54.4°C)
  - soc1: 54.6°C (peak 54.8°C)
  - soc2: 53.7°C (peak 53.9°C)
  - tj: 55.1°C (peak 55.4°C)
- GPU 利用率：mean 0%, peak 0%

### PDNAPulse
- 功耗采样：可用 (14 samples @ 100ms)
- 采样窗口时长：1.69s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 3288/3417 mW
  - VDD_IN: 8602/8797 mW
  - VDD_SOC: 1609/1671 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 5572 mJ (5.572 J)
  - VDD_IN: 14576 mJ (14.576 J)
  - VDD_SOC: 2727 mJ (2.727 J)
- 温度：
  - cpu: 54.7°C (peak 55.2°C)
  - gpu: 54.8°C (peak 55.1°C)
  - soc0: 53.8°C (peak 53.9°C)
  - soc1: 54.3°C (peak 54.5°C)
  - soc2: 53.4°C (peak 53.5°C)
  - tj: 54.8°C (peak 55.1°C)
- GPU 利用率：mean 0%, peak 0%

## 任务配置
- 数据：合成非平稳时间序列，一步预测
- 样本 / 序列长度：384 / 48
- 隐藏维度 / Epoch：24 / 3
- 设备：cpu

## 结果
| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 | VDD_IN mJ/步 |
|---|---:|---:|---:|---:|---:|
| CfCStyle | 2521 | 0.312067 | 11778.3 | 20.68 | 0.94 |
| LTC | 1321 | 0.465366 | 6028.3 | 38.88 | 1.61 |
| PDNAPulse | 3170 | 0.286117 | 11801.3 | 24.79 | 0.99 |
| GRU | 1969 | 0.393358 | 17237.4 | 6.99 | 0.68 |
| NCPS-LTC | 2547 | 0.621254 | 1705.7 | 124.00 | 5.90 |
| NCPS-CfC | 15737 | 0.106344 | 4791.1 | 22.31 | 2.19 |

## 解读
- `CfCStyle` 是闭式连续时间思想的轻量实现，用于快速验证 LNN 类动态门控在边缘设备上的训练与推理成本。
- `NCPS-LTC` / `NCPS-CfC` 是 mlech26l/ncps 官方实现，便于比较。
- `GRU` 是同等隐藏维度的传统循环网络基线。
- 该脚本是 smoke benchmark；正式论文复现应替换为论文数据集、固定随机种子、多次重复和置信区间。

## ⚠️ CPU-only 运行
- CUDA was unavailable, so these are CPU timings measured on the Jetson host. They are not GPU/edge performance evidence.
- 上方功耗/温度采样来自真实 Jetson 传感器，有效；但下方**速度与精度不是边缘推理性能证据**。
