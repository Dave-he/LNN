---
title: Jetson LNN 基准验证 - 2026-10-16
date: 2026-10-16
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-10-16

> ⚠️ **运行状态：`ok_cpu`**
> CUDA was unavailable, so these are CPU timings measured on the Jetson host. They are not GPU/edge performance evidence.

## 环境
- 平台：Linux-5.15.148-tegra-aarch64-with-glibc2.35
- 设备树型号：NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
- PyTorch：2.11.0+cu130
- CUDA：False (13.0)
- 系统内存：total 7620 MB / free 228 MB / available 4768 MB
- Jetson BSP：

```text
# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
```

## 功耗与温度
- 功耗采样：可用 (907 samples @ 100ms)
- 采样窗口时长：96.89s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2922/3259 mW
  - VDD_IN: 8008/8452 mW
  - VDD_SOC: 1560/1634 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 283128 mJ (283.128 J)
  - VDD_IN: 775902 mJ (775.902 J)
  - VDD_SOC: 151106 mJ (151.106 J)
- 温度：
  - cpu: 52.4°C (peak 53.5°C)
  - gpu: 52.6°C (peak 53.7°C)
  - soc0: 51.8°C (peak 52.7°C)
  - soc1: 52.0°C (peak 53.1°C)
  - soc2: 51.2°C (peak 52.1°C)
  - tj: 52.6°C (peak 53.7°C)
- GPU 利用率：mean 0%, peak 0%

## 各模型独立功耗

### CfCStyle
- 功耗采样：可用 (6 samples @ 100ms)
- 采样窗口时长：0.75s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2839/2905 mW
  - VDD_IN: 7967/8014 mW
  - VDD_SOC: 1581/1594 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 2127 mJ (2.127 J)
  - VDD_IN: 5970 mJ (5.970 J)
  - VDD_SOC: 1185 mJ (1.185 J)
- 温度：
  - cpu: 51.8°C (peak 52.1°C)
  - gpu: 51.9°C (peak 52.1°C)
  - soc0: 51.1°C (peak 51.2°C)
  - soc1: 51.3°C (peak 51.4°C)
  - soc2: 50.4°C (peak 50.5°C)
  - tj: 51.9°C (peak 52.1°C)
- GPU 利用率：mean 0%, peak 0%

### GRU
- 功耗采样：可用 (2 samples @ 100ms)
- 采样窗口时长：0.26s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2985/2985 mW
  - VDD_IN: 8054/8054 mW
  - VDD_SOC: 1555/1555 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 783 mJ (0.783 J)
  - VDD_IN: 2112 mJ (2.112 J)
  - VDD_SOC: 408 mJ (0.408 J)
- 温度：
  - cpu: 52.0°C (peak 52.1°C)
  - gpu: 52.5°C (peak 52.5°C)
  - soc0: 51.5°C (peak 51.5°C)
  - soc1: 51.9°C (peak 52.0°C)
  - soc2: 51.0°C (peak 51.0°C)
  - tj: 52.5°C (peak 52.5°C)
- GPU 利用率：mean 0%, peak 0%

### LTC
- 功耗采样：可用 (14 samples @ 100ms)
- 采样窗口时长：1.61s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2815/2866 mW
  - VDD_IN: 7897/7934 mW
  - VDD_SOC: 1555/1555 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 4519 mJ (4.519 J)
  - VDD_IN: 12678 mJ (12.678 J)
  - VDD_SOC: 2496 mJ (2.497 J)
- 温度：
  - cpu: 51.9°C (peak 52.1°C)
  - gpu: 52.2°C (peak 52.5°C)
  - soc0: 51.3°C (peak 51.4°C)
  - soc1: 51.6°C (peak 51.8°C)
  - soc2: 50.7°C (peak 50.8°C)
  - tj: 52.2°C (peak 52.5°C)
- GPU 利用率：mean 0%, peak 0%

### NCPS-CfC
- 功耗采样：可用 (13 samples @ 100ms)
- 采样窗口时长：1.41s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2866/2945 mW
  - VDD_IN: 7943/8054 mW
  - VDD_SOC: 1555/1555 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 4035 mJ (4.035 J)
  - VDD_IN: 11186 mJ (11.185 J)
  - VDD_SOC: 2190 mJ (2.190 J)
- 温度：
  - cpu: 53.1°C (peak 53.3°C)
  - gpu: 53.4°C (peak 53.7°C)
  - soc0: 52.5°C (peak 52.6°C)
  - soc1: 52.9°C (peak 53.1°C)
  - soc2: 52.0°C (peak 52.0°C)
  - tj: 53.4°C (peak 53.7°C)
- GPU 利用率：mean 0%, peak 0%

### NCPS-LTC
- 功耗采样：可用 (60 samples @ 100ms)
- 采样窗口时长：6.54s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2933/3105 mW
  - VDD_IN: 8001/8173 mW
  - VDD_SOC: 1557/1594 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 19166 mJ (19.166 J)
  - VDD_IN: 52292 mJ (52.291 J)
  - VDD_SOC: 10175 mJ (10.175 J)
- 温度：
  - cpu: 52.9°C (peak 53.2°C)
  - gpu: 53.1°C (peak 53.5°C)
  - soc0: 52.3°C (peak 52.5°C)
  - soc1: 52.7°C (peak 52.9°C)
  - soc2: 51.8°C (peak 51.9°C)
  - tj: 53.2°C (peak 53.5°C)
- GPU 利用率：mean 0%, peak 0%

### PDNAPulse
- 功耗采样：可用 (7 samples @ 100ms)
- 采样窗口时长：0.80s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2928/2985 mW
  - VDD_IN: 7997/8054 mW
  - VDD_SOC: 1555/1555 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 2335 mJ (2.335 J)
  - VDD_IN: 6379 mJ (6.379 J)
  - VDD_SOC: 1240 mJ (1.240 J)
- 温度：
  - cpu: 52.1°C (peak 52.2°C)
  - gpu: 52.2°C (peak 52.6°C)
  - soc0: 51.5°C (peak 51.6°C)
  - soc1: 51.7°C (peak 51.8°C)
  - soc2: 50.9°C (peak 51.0°C)
  - tj: 52.3°C (peak 52.6°C)
- GPU 利用率：mean 0%, peak 0%

## 任务配置
- 数据：合成非平稳时间序列，一步预测
- 样本 / 序列长度：384 / 48
- 隐藏维度 / Epoch：24 / 3
- 设备：cpu

## 结果
| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 | VDD_IN mJ/步 |
|---|---:|---:|---:|---:|---:|
| CfCStyle | 2521 | 0.312067 | 26640.7 | 6.58 | 0.40 |
| LTC | 1321 | 0.465366 | 11489.6 | 14.67 | 0.86 |
| PDNAPulse | 3170 | 0.286117 | 22738.8 | 5.46 | 0.43 |
| GRU | 1969 | 0.393358 | 71682.5 | 1.92 | 0.14 |
| NCPS-LTC | 2547 | 0.621254 | 2856.3 | 45.20 | 3.54 |
| NCPS-CfC | 15737 | 0.106344 | 12911.3 | 9.39 | 0.76 |

## 解读
- `CfCStyle` 是闭式连续时间思想的轻量实现，用于快速验证 LNN 类动态门控在边缘设备上的训练与推理成本。
- `NCPS-LTC` / `NCPS-CfC` 是 mlech26l/ncps 官方实现，便于比较。
- `GRU` 是同等隐藏维度的传统循环网络基线。
- 该脚本是 smoke benchmark；正式论文复现应替换为论文数据集、固定随机种子、多次重复和置信区间。

## ⚠️ CPU-only 运行
- CUDA was unavailable, so these are CPU timings measured on the Jetson host. They are not GPU/edge performance evidence.
- 上方功耗/温度采样来自真实 Jetson 传感器，有效；但下方**速度与精度不是边缘推理性能证据**。
