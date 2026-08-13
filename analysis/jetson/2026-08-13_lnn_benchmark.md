---
title: Jetson LNN 基准验证 - 2026-08-13
date: 2026-08-13
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-08-13

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
- 功耗采样：可用 (467 samples @ 100ms)
- 采样窗口时长：49.65s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2544/3264 mW
  - VDD_IN: 7380/8452 mW
  - VDD_SOC: 1508/1634 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 126305 mJ (126.305 J)
  - VDD_IN: 366403 mJ (366.403 J)
  - VDD_SOC: 74855 mJ (74.855 J)
- 温度：
  - cpu: 51.4°C (peak 52.5°C)
  - gpu: 51.7°C (peak 52.6°C)
  - soc0: 50.8°C (peak 51.5°C)
  - soc1: 51.1°C (peak 51.8°C)
  - soc2: 50.2°C (peak 50.9°C)
  - tj: 51.7°C (peak 52.6°C)
- GPU 利用率：mean 0%, peak 0%

## 各模型独立功耗

### CfCStyle
- 功耗采样：可用 (5 samples @ 100ms)
- 采样窗口时长：0.63s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2961/2985 mW
  - VDD_IN: 7830/7854 mW
  - VDD_SOC: 1515/1515 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 1879 mJ (1.879 J)
  - VDD_IN: 4970 mJ (4.970 J)
  - VDD_SOC: 962 mJ (0.962 J)
- 温度：
  - cpu: 51.2°C (peak 51.4°C)
  - gpu: 51.4°C (peak 51.6°C)
  - soc0: 50.4°C (peak 50.5°C)
  - soc1: 50.7°C (peak 50.8°C)
  - soc2: 49.9°C (peak 50.0°C)
  - tj: 51.4°C (peak 51.6°C)
- GPU 利用率：mean 0%, peak 0%

### GRU
- 功耗采样：可用 (1 samples @ 100ms)
- 采样窗口时长：0.20s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2551/2551 mW
  - VDD_IN: 7308/7308 mW
  - VDD_SOC: 1475/1475 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 510 mJ (0.510 J)
  - VDD_IN: 1462 mJ (1.462 J)
  - VDD_SOC: 295 mJ (0.295 J)
- 温度：
  - cpu: 51.5°C (peak 51.5°C)
  - gpu: 51.6°C (peak 51.6°C)
  - soc0: 50.8°C (peak 50.8°C)
  - soc1: 51.2°C (peak 51.2°C)
  - soc2: 50.2°C (peak 50.2°C)
  - tj: 51.6°C (peak 51.6°C)
- GPU 利用率：mean 0%, peak 0%

### LTC
- 功耗采样：可用 (13 samples @ 100ms)
- 采样窗口时长：1.44s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2561/2985 mW
  - VDD_IN: 7512/8054 mW
  - VDD_SOC: 1549/1594 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 3690 mJ (3.690 J)
  - VDD_IN: 10821 mJ (10.821 J)
  - VDD_SOC: 2232 mJ (2.232 J)
- 温度：
  - cpu: 51.5°C (peak 51.7°C)
  - gpu: 51.7°C (peak 52.0°C)
  - soc0: 50.7°C (peak 51.0°C)
  - soc1: 51.1°C (peak 51.2°C)
  - soc2: 50.2°C (peak 50.4°C)
  - tj: 51.7°C (peak 52.0°C)
- GPU 利用率：mean 0%, peak 0%

### NCPS-CfC
- 功耗采样：可用 (5 samples @ 100ms)
- 采样窗口时长：0.58s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2794/2826 mW
  - VDD_IN: 7507/7547 mW
  - VDD_SOC: 1475/1475 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 1625 mJ (1.625 J)
  - VDD_IN: 4365 mJ (4.365 J)
  - VDD_SOC: 858 mJ (0.858 J)
- 温度：
  - cpu: 52.2°C (peak 52.2°C)
  - gpu: 52.4°C (peak 52.6°C)
  - soc0: 51.4°C (peak 51.4°C)
  - soc1: 51.8°C (peak 51.9°C)
  - soc2: 50.9°C (peak 50.9°C)
  - tj: 52.4°C (peak 52.6°C)
- GPU 利用率：mean 0%, peak 0%

### NCPS-LTC
- 功耗采样：可用 (14 samples @ 100ms)
- 采样窗口时长：1.57s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2742/2985 mW
  - VDD_IN: 7487/7695 mW
  - VDD_SOC: 1490/1517 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 4305 mJ (4.305 J)
  - VDD_IN: 11757 mJ (11.757 J)
  - VDD_SOC: 2339 mJ (2.339 J)
- 温度：
  - cpu: 52.0°C (peak 52.3°C)
  - gpu: 52.1°C (peak 52.4°C)
  - soc0: 51.2°C (peak 51.3°C)
  - soc1: 51.6°C (peak 51.7°C)
  - soc2: 50.7°C (peak 50.9°C)
  - tj: 52.2°C (peak 52.4°C)
- GPU 利用率：mean 0%, peak 0%

### PDNAPulse
- 功耗采样：可用 (3 samples @ 100ms)
- 采样窗口时长：0.40s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2655/2667 mW
  - VDD_IN: 7388/7388 mW
  - VDD_SOC: 1475/1475 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 1064 mJ (1.064 J)
  - VDD_IN: 2960 mJ (2.960 J)
  - VDD_SOC: 591 mJ (0.591 J)
- 温度：
  - cpu: 51.4°C (peak 51.5°C)
  - gpu: 51.8°C (peak 52.0°C)
  - soc0: 50.8°C (peak 50.8°C)
  - soc1: 51.2°C (peak 51.2°C)
  - soc2: 50.2°C (peak 50.3°C)
  - tj: 51.8°C (peak 52.0°C)
- GPU 利用率：mean 0%, peak 0%

## 任务配置
- 数据：合成非平稳时间序列，一步预测
- 样本 / 序列长度：384 / 48
- 隐藏维度 / Epoch：24 / 3
- 设备：cpu

## 结果
| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 | VDD_IN mJ/步 |
|---|---:|---:|---:|---:|---:|
| CfCStyle | 2521 | 0.312067 | 29948.3 | 5.18 | 0.34 |
| LTC | 1321 | 0.465366 | 13280.2 | 11.10 | 0.73 |
| PDNAPulse | 3170 | 0.286117 | 47224.7 | 2.46 | 0.20 |
| GRU | 1969 | 0.393358 | 80923.7 | 0.95 | 0.10 |
| NCPS-LTC | 2547 | 0.621254 | 13691.7 | 16.35 | 0.80 |
| NCPS-CfC | 15737 | 0.106344 | 34005.7 | 3.59 | 0.30 |

## Benchmark 图
![Jetson LNN Benchmark](2026-08-13_lnn_benchmark.png)

## 解读
- `CfCStyle` 是闭式连续时间思想的轻量实现，用于快速验证 LNN 类动态门控在边缘设备上的训练与推理成本。
- `NCPS-LTC` / `NCPS-CfC` 是 mlech26l/ncps 官方实现，便于比较。
- `GRU` 是同等隐藏维度的传统循环网络基线。
- 该脚本是 smoke benchmark；正式论文复现应替换为论文数据集、固定随机种子、多次重复和置信区间。
