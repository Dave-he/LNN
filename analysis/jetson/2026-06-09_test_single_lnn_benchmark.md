---
title: Jetson LNN 基准验证 - 2026-06-09_test_single
date: 2026-06-09_test_single
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-06-09_test_single

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

## 功耗与温度
- 功耗采样：可用 (295 samples @ 100ms)
- 采样窗口时长：31.36s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2541/3144 mW
  - VDD_IN: 7274/7934 mW
  - VDD_SOC: 1496/1594 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 79706 mJ (79.706 J)
  - VDD_IN: 228147 mJ (228.147 J)
  - VDD_SOC: 46920 mJ (46.920 J)
- 温度：
  - cpu: 53.0°C (peak 53.8°C)
  - gpu: 53.3°C (peak 53.8°C)
  - soc0: 52.4°C (peak 52.8°C)
  - soc1: 52.9°C (peak 53.3°C)
  - soc2: 51.8°C (peak 52.3°C)
  - tj: 53.3°C (peak 53.8°C)
- GPU 利用率：mean 0%, peak 0%

## 各模型独立功耗

### CfCStyle
- 功耗采样：可用 (3 samples @ 100ms)
- 采样窗口时长：0.40s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2680/2706 mW
  - VDD_IN: 7375/7428 mW
  - VDD_SOC: 1475/1475 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 1066 mJ (1.066 J)
  - VDD_IN: 2933 mJ (2.933 J)
  - VDD_SOC: 587 mJ (0.587 J)
- 温度：
  - cpu: 52.9°C (peak 53.0°C)
  - gpu: 53.0°C (peak 53.3°C)
  - soc0: 52.2°C (peak 52.2°C)
  - soc1: 52.6°C (peak 52.7°C)
  - soc2: 51.6°C (peak 51.7°C)
  - tj: 53.1°C (peak 53.3°C)
- GPU 利用率：mean 0%, peak 0%

### GRU
- 功耗采样：可用 (2 samples @ 100ms)
- 采样窗口时长：0.25s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2806/2866 mW
  - VDD_IN: 7714/7854 mW
  - VDD_SOC: 1555/1555 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 696 mJ (0.697 J)
  - VDD_IN: 1915 mJ (1.915 J)
  - VDD_SOC: 386 mJ (0.386 J)
- 温度：
  - cpu: 53.4°C (peak 53.5°C)
  - gpu: 53.3°C (peak 53.4°C)
  - soc0: 52.4°C (peak 52.4°C)
  - soc1: 52.9°C (peak 52.9°C)
  - soc2: 51.9°C (peak 51.9°C)
  - tj: 53.4°C (peak 53.5°C)
- GPU 利用率：mean 0%, peak 0%

### LTC
- 功耗采样：可用 (9 samples @ 100ms)
- 采样窗口时长：1.08s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2259/2272 mW
  - VDD_IN: 6939/6988 mW
  - VDD_SOC: 1477/1477 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 2431 mJ (2.431 J)
  - VDD_IN: 7468 mJ (7.468 J)
  - VDD_SOC: 1590 mJ (1.590 J)
- 温度：
  - cpu: 52.8°C (peak 53.0°C)
  - gpu: 53.1°C (peak 53.4°C)
  - soc0: 52.3°C (peak 52.5°C)
  - soc1: 52.8°C (peak 52.8°C)
  - soc2: 51.6°C (peak 51.7°C)
  - tj: 53.1°C (peak 53.4°C)
- GPU 利用率：mean 0%, peak 0%

### NCPS-CfC
- 功耗采样：可用 (3 samples @ 100ms)
- 采样窗口时长：0.39s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2958/2985 mW
  - VDD_IN: 7682/7695 mW
  - VDD_SOC: 1488/1515 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 1161 mJ (1.161 J)
  - VDD_IN: 3014 mJ (3.014 J)
  - VDD_SOC: 584 mJ (0.584 J)
- 温度：
  - cpu: 53.3°C (peak 53.3°C)
  - gpu: 53.5°C (peak 53.5°C)
  - soc0: 52.7°C (peak 52.8°C)
  - soc1: 53.2°C (peak 53.3°C)
  - soc2: 52.2°C (peak 52.2°C)
  - tj: 53.5°C (peak 53.5°C)
- GPU 利用率：mean 0%, peak 0%

### NCPS-LTC
- 功耗采样：可用 (19 samples @ 100ms)
- 采样窗口时长：2.12s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2878/3144 mW
  - VDD_IN: 7635/7814 mW
  - VDD_SOC: 1500/1555 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 6111 mJ (6.111 J)
  - VDD_IN: 16211 mJ (16.211 J)
  - VDD_SOC: 3185 mJ (3.185 J)
- 温度：
  - cpu: 53.3°C (peak 53.8°C)
  - gpu: 53.6°C (peak 53.9°C)
  - soc0: 52.6°C (peak 52.8°C)
  - soc1: 53.1°C (peak 53.3°C)
  - soc2: 52.1°C (peak 52.3°C)
  - tj: 53.6°C (peak 53.9°C)
- GPU 利用率：mean 0%, peak 0%

### PDNAPulse
- 功耗采样：可用 (2 samples @ 100ms)
- 采样窗口时长：0.31s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2766/2786 mW
  - VDD_IN: 7456/7456 mW
  - VDD_SOC: 1475/1475 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 869 mJ (0.869 J)
  - VDD_IN: 2343 mJ (2.343 J)
  - VDD_SOC: 464 mJ (0.464 J)
- 温度：
  - cpu: 53.0°C (peak 53.1°C)
  - gpu: 53.2°C (peak 53.2°C)
  - soc0: 52.4°C (peak 52.4°C)
  - soc1: 52.8°C (peak 52.8°C)
  - soc2: 51.9°C (peak 51.9°C)
  - tj: 53.2°C (peak 53.2°C)
- GPU 利用率：mean 0%, peak 0%

## 任务配置
- 数据：合成非平稳时间序列，一步预测
- 样本 / 序列长度：384 / 48
- 隐藏维度 / Epoch：24 / 3
- 设备：cpu

## 结果
| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 | VDD_IN mJ/步 |
|---|---:|---:|---:|---:|---:|
| CfCStyle | 2521 | 0.312067 | 45276.1 | 2.11 | 0.20 |
| LTC | 1321 | 0.465366 | 16917.4 | 6.68 | 0.51 |
| PDNAPulse | 3170 | 0.286117 | 58878.2 | 1.87 | 0.16 |
| GRU | 1969 | 0.393358 | 79381.0 | 0.93 | 0.13 |
| NCPS-LTC | 2547 | 0.621254 | 13750.1 | 10.92 | 1.10 |
| NCPS-CfC | 15737 | 0.106344 | 47320.1 | 2.42 | 0.20 |

## Benchmark 图
![Jetson LNN Benchmark](2026-06-09_test_single_lnn_benchmark.png)

## 解读
- `CfCStyle` 是闭式连续时间思想的轻量实现，用于快速验证 LNN 类动态门控在边缘设备上的训练与推理成本。
- `NCPS-LTC` / `NCPS-CfC` 是 mlech26l/ncps 官方实现，便于比较。
- `GRU` 是同等隐藏维度的传统循环网络基线。
- 该脚本是 smoke benchmark；正式论文复现应替换为论文数据集、固定随机种子、多次重复和置信区间。
