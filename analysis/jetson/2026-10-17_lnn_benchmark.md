---
title: Jetson LNN 基准验证 - 2026-10-17
date: 2026-10-17
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-10-17

> ⚠️ **运行状态：`ok_cpu`**
> CUDA was unavailable, so these are CPU timings measured on the Jetson host. They are not GPU/edge performance evidence.

## 环境
- 平台：Linux-5.15.148-tegra-aarch64-with-glibc2.35
- 设备树型号：NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
- PyTorch：2.11.0+cu130
- CUDA：False (13.0)
- 系统内存：total 7620 MB / free 230 MB / available 2790 MB
- Jetson BSP：

```text
# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
```

## 功耗与温度
- 功耗采样：可用 (918 samples @ 100ms)
- 采样窗口时长：98.02s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2921/3304 mW
  - VDD_IN: 7671/8133 mW
  - VDD_SOC: 1483/1594 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 286360 mJ (286.360 J)
  - VDD_IN: 751952 mJ (751.952 J)
  - VDD_SOC: 145365 mJ (145.365 J)
- 温度：
  - cpu: 52.0°C (peak 53.0°C)
  - gpu: 52.1°C (peak 53.2°C)
  - soc0: 51.2°C (peak 52.1°C)
  - soc1: 51.6°C (peak 52.5°C)
  - soc2: 50.7°C (peak 51.6°C)
  - tj: 52.2°C (peak 53.2°C)
- GPU 利用率：mean 0%, peak 0%

## 各模型独立功耗

### CfCStyle
- 功耗采样：可用 (5 samples @ 100ms)
- 采样窗口时长：0.62s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2921/2945 mW
  - VDD_IN: 7659/7707 mW
  - VDD_SOC: 1475/1475 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 1809 mJ (1.809 J)
  - VDD_IN: 4743 mJ (4.743 J)
  - VDD_SOC: 914 mJ (0.913 J)
- 温度：
  - cpu: 51.3°C (peak 51.4°C)
  - gpu: 51.3°C (peak 51.4°C)
  - soc0: 50.3°C (peak 50.4°C)
  - soc1: 50.7°C (peak 50.8°C)
  - soc2: 49.9°C (peak 50.0°C)
  - tj: 51.3°C (peak 51.4°C)
- GPU 利用率：mean 0%, peak 0%

### GRU
- 功耗采样：可用 (3 samples @ 100ms)
- 采样窗口时长：0.36s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2932/2945 mW
  - VDD_IN: 7667/7707 mW
  - VDD_SOC: 1475/1475 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 1054 mJ (1.054 J)
  - VDD_IN: 2756 mJ (2.756 J)
  - VDD_SOC: 530 mJ (0.530 J)
- 温度：
  - cpu: 51.7°C (peak 51.9°C)
  - gpu: 52.0°C (peak 52.2°C)
  - soc0: 51.0°C (peak 51.0°C)
  - soc1: 51.4°C (peak 51.6°C)
  - soc2: 50.5°C (peak 50.6°C)
  - tj: 52.0°C (peak 52.2°C)
- GPU 利用率：mean 0%, peak 0%

### LTC
- 功耗采样：可用 (18 samples @ 100ms)
- 采样窗口时长：2.01s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2780/2945 mW
  - VDD_IN: 7523/7667 mW
  - VDD_SOC: 1486/1515 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 5598 mJ (5.598 J)
  - VDD_IN: 15148 mJ (15.149 J)
  - VDD_SOC: 2992 mJ (2.993 J)
- 温度：
  - cpu: 51.5°C (peak 51.8°C)
  - gpu: 51.8°C (peak 52.1°C)
  - soc0: 50.9°C (peak 51.0°C)
  - soc1: 51.2°C (peak 51.3°C)
  - soc2: 50.3°C (peak 50.4°C)
  - tj: 51.8°C (peak 52.1°C)
- GPU 利用率：mean 0%, peak 0%

### NCPS-CfC
- 功耗采样：可用 (13 samples @ 100ms)
- 采样窗口时长：1.49s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2958/3065 mW
  - VDD_IN: 7705/7814 mW
  - VDD_SOC: 1490/1515 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 4416 mJ (4.416 J)
  - VDD_IN: 11505 mJ (11.505 J)
  - VDD_SOC: 2226 mJ (2.226 J)
- 温度：
  - cpu: 52.7°C (peak 52.9°C)
  - gpu: 52.8°C (peak 53.0°C)
  - soc0: 52.0°C (peak 52.1°C)
  - soc1: 52.3°C (peak 52.5°C)
  - soc2: 51.4°C (peak 51.5°C)
  - tj: 52.8°C (peak 53.0°C)
- GPU 利用率：mean 0%, peak 0%

### NCPS-LTC
- 功耗采样：可用 (58 samples @ 100ms)
- 采样窗口时长：6.26s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2906/3065 mW
  - VDD_IN: 7633/7775 mW
  - VDD_SOC: 1478/1515 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 18199 mJ (18.199 J)
  - VDD_IN: 47814 mJ (47.814 J)
  - VDD_SOC: 9256 mJ (9.257 J)
- 温度：
  - cpu: 52.5°C (peak 52.7°C)
  - gpu: 52.7°C (peak 53.2°C)
  - soc0: 51.8°C (peak 51.9°C)
  - soc1: 52.2°C (peak 52.4°C)
  - soc2: 51.3°C (peak 51.4°C)
  - tj: 52.7°C (peak 53.2°C)
- GPU 利用率：mean 0%, peak 0%

### PDNAPulse
- 功耗采样：可用 (5 samples @ 100ms)
- 采样窗口时长：0.58s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2953/2985 mW
  - VDD_IN: 7676/7695 mW
  - VDD_SOC: 1475/1475 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 1708 mJ (1.708 J)
  - VDD_IN: 4440 mJ (4.440 J)
  - VDD_SOC: 853 mJ (0.853 J)
- 温度：
  - cpu: 51.8°C (peak 52.0°C)
  - gpu: 52.0°C (peak 52.2°C)
  - soc0: 50.9°C (peak 51.1°C)
  - soc1: 51.3°C (peak 51.3°C)
  - soc2: 50.5°C (peak 50.5°C)
  - tj: 52.0°C (peak 52.2°C)
- GPU 利用率：mean 0%, peak 0%

## 任务配置
- 数据：合成非平稳时间序列，一步预测
- 样本 / 序列长度：384 / 48
- 隐藏维度 / Epoch：24 / 3
- 设备：cpu

## 结果
| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 | VDD_IN mJ/步 |
|---|---:|---:|---:|---:|---:|
| CfCStyle | 2521 | 0.312067 | 32420.8 | 6.19 | 0.32 |
| LTC | 1321 | 0.465366 | 9345.3 | 14.97 | 1.02 |
| PDNAPulse | 3170 | 0.286117 | 32745.2 | 5.44 | 0.30 |
| GRU | 1969 | 0.393358 | 53274.5 | 2.38 | 0.19 |
| NCPS-LTC | 2547 | 0.621254 | 3066.6 | 45.65 | 3.23 |
| NCPS-CfC | 15737 | 0.106344 | 12447.8 | 9.69 | 0.78 |

## 解读
- `CfCStyle` 是闭式连续时间思想的轻量实现，用于快速验证 LNN 类动态门控在边缘设备上的训练与推理成本。
- `NCPS-LTC` / `NCPS-CfC` 是 mlech26l/ncps 官方实现，便于比较。
- `GRU` 是同等隐藏维度的传统循环网络基线。
- 该脚本是 smoke benchmark；正式论文复现应替换为论文数据集、固定随机种子、多次重复和置信区间。

## ⚠️ CPU-only 运行
- CUDA was unavailable, so these are CPU timings measured on the Jetson host. They are not GPU/edge performance evidence.
- 上方功耗/温度采样来自真实 Jetson 传感器，有效；但下方**速度与精度不是边缘推理性能证据**。
