---
title: Jetson LNN 基准验证 - 2026-10-18
date: 2026-10-18
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-10-18

> ⚠️ **运行状态：`ok_cpu`**
> CUDA was unavailable, so these are CPU timings measured on the Jetson host. They are not GPU/edge performance evidence.

## 环境
- 平台：Linux-5.15.148-tegra-aarch64-with-glibc2.35
- 设备树型号：NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
- PyTorch：2.11.0+cu130
- CUDA：False (13.0)
- 系统内存：total 7620 MB / free 203 MB / available 3366 MB
- Jetson BSP：

```text
# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
```

## 功耗与温度
- 功耗采样：可用 (280 samples @ 100ms)
- 采样窗口时长：29.73s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2493/3105 mW
  - VDD_IN: 7190/7787 mW
  - VDD_SOC: 1471/1517 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 74132 mJ (74.132 J)
  - VDD_IN: 213763 mJ (213.763 J)
  - VDD_SOC: 43749 mJ (43.749 J)
- 温度：
  - cpu: 48.8°C (peak 50.0°C)
  - gpu: 49.0°C (peak 49.9°C)
  - soc0: 48.1°C (peak 48.9°C)
  - soc1: 48.3°C (peak 49.0°C)
  - soc2: 47.5°C (peak 48.2°C)
  - tj: 49.0°C (peak 50.0°C)
- GPU 利用率：mean 0%, peak 11%

## 各模型独立功耗

### CfCStyle
- 功耗采样：可用 (2 samples @ 100ms)
- 采样窗口时长：0.32s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2671/2671 mW
  - VDD_IN: 7308/7308 mW
  - VDD_SOC: 1437/1437 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 868 mJ (0.868 J)
  - VDD_IN: 2374 mJ (2.374 J)
  - VDD_SOC: 467 mJ (0.467 J)
- 温度：
  - cpu: 48.5°C (peak 48.5°C)
  - gpu: 48.3°C (peak 48.4°C)
  - soc0: 47.6°C (peak 47.7°C)
  - soc1: 47.8°C (peak 47.8°C)
  - soc2: 47.1°C (peak 47.2°C)
  - tj: 48.5°C (peak 48.5°C)
- GPU 利用率：mean 0%, peak 0%

### GRU
- 功耗采样：不可用
  - tegrastats produced no parseable samples (window too short? try a longer run or smaller --interval)

### LTC
- 功耗采样：可用 (8 samples @ 100ms)
- 采样窗口时长：0.95s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2247/2312 mW
  - VDD_IN: 6878/6908 mW
  - VDD_SOC: 1462/1477 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 2137 mJ (2.137 J)
  - VDD_IN: 6542 mJ (6.542 J)
  - VDD_SOC: 1390 mJ (1.391 J)
- 温度：
  - cpu: 48.4°C (peak 48.6°C)
  - gpu: 48.7°C (peak 48.8°C)
  - soc0: 47.9°C (peak 48.0°C)
  - soc1: 48.1°C (peak 48.2°C)
  - soc2: 47.3°C (peak 47.3°C)
  - tj: 48.7°C (peak 48.8°C)
- GPU 利用率：mean 0%, peak 0%

### NCPS-CfC
- 功耗采样：可用 (3 samples @ 100ms)
- 采样窗口时长：0.40s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2972/2985 mW
  - VDD_IN: 7667/7667 mW
  - VDD_SOC: 1475/1475 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 1182 mJ (1.182 J)
  - VDD_IN: 3050 mJ (3.050 J)
  - VDD_SOC: 587 mJ (0.587 J)
- 温度：
  - cpu: 49.6°C (peak 49.7°C)
  - gpu: 49.6°C (peak 49.8°C)
  - soc0: 48.7°C (peak 48.8°C)
  - soc1: 48.8°C (peak 48.9°C)
  - soc2: 48.2°C (peak 48.2°C)
  - tj: 49.7°C (peak 49.8°C)
- GPU 利用率：mean 0%, peak 0%

### NCPS-LTC
- 功耗采样：可用 (8 samples @ 100ms)
- 采样窗口时长：0.90s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2769/3105 mW
  - VDD_IN: 7452/7787 mW
  - VDD_SOC: 1476/1477 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 2505 mJ (2.505 J)
  - VDD_IN: 6742 mJ (6.742 J)
  - VDD_SOC: 1335 mJ (1.335 J)
- 温度：
  - cpu: 49.7°C (peak 49.9°C)
  - gpu: 49.5°C (peak 49.7°C)
  - soc0: 48.5°C (peak 48.6°C)
  - soc1: 48.8°C (peak 48.9°C)
  - soc2: 48.1°C (peak 48.2°C)
  - tj: 49.7°C (peak 50.0°C)
- GPU 利用率：mean 0%, peak 0%

### PDNAPulse
- 功耗采样：可用 (3 samples @ 100ms)
- 采样窗口时长：0.35s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2764/2791 mW
  - VDD_IN: 7401/7428 mW
  - VDD_SOC: 1437/1437 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 963 mJ (0.963 J)
  - VDD_IN: 2579 mJ (2.579 J)
  - VDD_SOC: 501 mJ (0.501 J)
- 温度：
  - cpu: 48.8°C (peak 48.8°C)
  - gpu: 48.9°C (peak 49.0°C)
  - soc0: 48.0°C (peak 48.1°C)
  - soc1: 48.2°C (peak 48.3°C)
  - soc2: 47.6°C (peak 47.6°C)
  - tj: 48.9°C (peak 49.0°C)
- GPU 利用率：mean 0%, peak 0%

## 任务配置
- 数据：合成非平稳时间序列，一步预测
- 样本 / 序列长度：384 / 48
- 隐藏维度 / Epoch：24 / 3
- 设备：cpu

## 结果
| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 | VDD_IN mJ/步 |
|---|---:|---:|---:|---:|---:|
| CfCStyle | 2521 | 0.312067 | 60496.7 | 2.13 | 0.16 |
| LTC | 1321 | 0.465366 | 19323.9 | 6.53 | 0.44 |
| PDNAPulse | 3170 | 0.286117 | 51812.2 | 1.99 | 0.17 |
| GRU | 1969 | 0.393358 | 249427.3 | 0.63 | n/a |
| NCPS-LTC | 2547 | 0.621254 | 20690.5 | 10.78 | 0.46 |
| NCPS-CfC | 15737 | 0.106344 | 46957.1 | 2.39 | 0.21 |

## 解读
- `CfCStyle` 是闭式连续时间思想的轻量实现，用于快速验证 LNN 类动态门控在边缘设备上的训练与推理成本。
- `NCPS-LTC` / `NCPS-CfC` 是 mlech26l/ncps 官方实现，便于比较。
- `GRU` 是同等隐藏维度的传统循环网络基线。
- 该脚本是 smoke benchmark；正式论文复现应替换为论文数据集、固定随机种子、多次重复和置信区间。

## ⚠️ CPU-only 运行
- CUDA was unavailable, so these are CPU timings measured on the Jetson host. They are not GPU/edge performance evidence.
- 上方功耗/温度采样来自真实 Jetson 传感器，有效；但下方**速度与精度不是边缘推理性能证据**。
