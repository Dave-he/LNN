---
title: Jetson LNN 基准验证 - 2026-09-01
date: 2026-09-01
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-09-01

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
- 功耗采样：可用 (537 samples @ 100ms)
- 采样窗口时长：57.18s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2405/3025 mW
  - VDD_IN: 7253/8014 mW
  - VDD_SOC: 1511/1594 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 137531 mJ (137.531 J)
  - VDD_IN: 414767 mJ (414.767 J)
  - VDD_SOC: 86418 mJ (86.418 J)
- 温度：
  - cpu: 51.1°C (peak 52.3°C)
  - gpu: 51.4°C (peak 52.2°C)
  - soc0: 50.5°C (peak 51.2°C)
  - soc1: 50.8°C (peak 51.6°C)
  - soc2: 49.9°C (peak 50.7°C)
  - tj: 51.4°C (peak 52.3°C)
- GPU 利用率：mean 0%, peak 0%

## 各模型独立功耗

### CfCStyle
- 功耗采样：可用 (4 samples @ 100ms)
- 采样窗口时长：0.50s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2577/2587 mW
  - VDD_IN: 7408/7468 mW
  - VDD_SOC: 1475/1475 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 1279 mJ (1.279 J)
  - VDD_IN: 3678 mJ (3.678 J)
  - VDD_SOC: 732 mJ (0.732 J)
- 温度：
  - cpu: 50.6°C (peak 51.0°C)
  - gpu: 51.0°C (peak 51.2°C)
  - soc0: 50.1°C (peak 50.2°C)
  - soc1: 50.4°C (peak 50.4°C)
  - soc2: 49.5°C (peak 49.6°C)
  - tj: 51.0°C (peak 51.2°C)
- GPU 利用率：mean 0%, peak 0%

### GRU
- 功耗采样：可用 (1 samples @ 100ms)
- 采样窗口时长：0.18s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2627/2627 mW
  - VDD_IN: 7507/7507 mW
  - VDD_SOC: 1515/1515 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 478 mJ (0.478 J)
  - VDD_IN: 1366 mJ (1.366 J)
  - VDD_SOC: 276 mJ (0.276 J)
- 温度：
  - cpu: 51.1°C (peak 51.1°C)
  - gpu: 51.5°C (peak 51.5°C)
  - soc0: 50.4°C (peak 50.4°C)
  - soc1: 51.0°C (peak 51.0°C)
  - soc2: 50.0°C (peak 50.0°C)
  - tj: 51.5°C (peak 51.5°C)
- GPU 利用率：mean 0%, peak 0%

### LTC
- 功耗采样：可用 (11 samples @ 100ms)
- 采样窗口时长：1.23s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2319/2392 mW
  - VDD_IN: 7061/7148 mW
  - VDD_SOC: 1484/1517 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 2853 mJ (2.853 J)
  - VDD_IN: 8687 mJ (8.687 J)
  - VDD_SOC: 1826 mJ (1.826 J)
- 温度：
  - cpu: 51.0°C (peak 51.1°C)
  - gpu: 51.3°C (peak 51.6°C)
  - soc0: 50.3°C (peak 50.5°C)
  - soc1: 50.6°C (peak 50.8°C)
  - soc2: 49.7°C (peak 49.8°C)
  - tj: 51.3°C (peak 51.6°C)
- GPU 利用率：mean 0%, peak 0%

### NCPS-CfC
- 功耗采样：可用 (5 samples @ 100ms)
- 采样窗口时长：0.59s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2754/2786 mW
  - VDD_IN: 7468/7468 mW
  - VDD_SOC: 1491/1515 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 1626 mJ (1.626 J)
  - VDD_IN: 4409 mJ (4.409 J)
  - VDD_SOC: 880 mJ (0.880 J)
- 温度：
  - cpu: 51.9°C (peak 52.0°C)
  - gpu: 51.9°C (peak 52.3°C)
  - soc0: 51.0°C (peak 51.2°C)
  - soc1: 51.4°C (peak 51.6°C)
  - soc2: 50.5°C (peak 50.6°C)
  - tj: 52.0°C (peak 52.3°C)
- GPU 利用率：mean 0%, peak 0%

### NCPS-LTC
- 功耗采样：可用 (24 samples @ 100ms)
- 采样窗口时长：2.59s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2686/2905 mW
  - VDD_IN: 7472/7735 mW
  - VDD_SOC: 1504/1555 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 6958 mJ (6.958 J)
  - VDD_IN: 19357 mJ (19.357 J)
  - VDD_SOC: 3895 mJ (3.895 J)
- 温度：
  - cpu: 51.7°C (peak 52.3°C)
  - gpu: 51.9°C (peak 52.2°C)
  - soc0: 50.9°C (peak 51.1°C)
  - soc1: 51.3°C (peak 51.4°C)
  - soc2: 50.4°C (peak 50.6°C)
  - tj: 51.9°C (peak 52.3°C)
- GPU 利用率：mean 0%, peak 0%

### PDNAPulse
- 功耗采样：可用 (3 samples @ 100ms)
- 采样窗口时长：0.44s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2640/2667 mW
  - VDD_IN: 7521/7587 mW
  - VDD_SOC: 1515/1515 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 1156 mJ (1.157 J)
  - VDD_IN: 3294 mJ (3.294 J)
  - VDD_SOC: 664 mJ (0.664 J)
- 温度：
  - cpu: 51.2°C (peak 51.2°C)
  - gpu: 51.5°C (peak 51.6°C)
  - soc0: 50.5°C (peak 50.6°C)
  - soc1: 50.8°C (peak 50.8°C)
  - soc2: 50.0°C (peak 50.0°C)
  - tj: 51.5°C (peak 51.6°C)
- GPU 利用率：mean 0%, peak 0%

## 任务配置
- 数据：合成非平稳时间序列，一步预测
- 样本 / 序列长度：384 / 48
- 隐藏维度 / Epoch：24 / 3
- 设备：cpu

## 结果
| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 | VDD_IN mJ/步 |
|---|---:|---:|---:|---:|---:|
| CfCStyle | 2521 | 0.312067 | 37255.1 | 4.78 | 0.25 |
| LTC | 1321 | 0.465366 | 14931.7 | 9.22 | 0.59 |
| PDNAPulse | 3170 | 0.286117 | 44061.5 | 4.57 | 0.22 |
| GRU | 1969 | 0.393358 | 113951.7 | 1.14 | 0.09 |
| NCPS-LTC | 2547 | 0.621254 | 7104.1 | 19.57 | 1.31 |
| NCPS-CfC | 15737 | 0.106344 | 30588.3 | 5.29 | 0.30 |

## Benchmark 图
![Jetson LNN Benchmark](2026-09-01_lnn_benchmark.png)

## 解读
- `CfCStyle` 是闭式连续时间思想的轻量实现，用于快速验证 LNN 类动态门控在边缘设备上的训练与推理成本。
- `NCPS-LTC` / `NCPS-CfC` 是 mlech26l/ncps 官方实现，便于比较。
- `GRU` 是同等隐藏维度的传统循环网络基线。
- 该脚本是 smoke benchmark；正式论文复现应替换为论文数据集、固定随机种子、多次重复和置信区间。
