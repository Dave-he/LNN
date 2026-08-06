---
title: Jetson LNN 基准验证 - 2026-08-06
date: 2026-08-06
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-08-06

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
- 功耗采样：可用 (415 samples @ 100ms)
- 采样窗口时长：44.12s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2486/3184 mW
  - VDD_IN: 7260/8094 mW
  - VDD_SOC: 1501/1594 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 109697 mJ (109.697 J)
  - VDD_IN: 320299 mJ (320.299 J)
  - VDD_SOC: 66227 mJ (66.227 J)
- 温度：
  - cpu: 51.2°C (peak 52.2°C)
  - gpu: 51.5°C (peak 52.4°C)
  - soc0: 50.6°C (peak 51.3°C)
  - soc1: 50.9°C (peak 51.7°C)
  - soc2: 50.0°C (peak 50.8°C)
  - tj: 51.5°C (peak 52.4°C)
- GPU 利用率：mean 0%, peak 0%

## 各模型独立功耗

### CfCStyle
- 功耗采样：可用 (3 samples @ 100ms)
- 采样窗口时长：0.40s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2746/2786 mW
  - VDD_IN: 7494/7547 mW
  - VDD_SOC: 1515/1515 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 1088 mJ (1.088 J)
  - VDD_IN: 2968 mJ (2.968 J)
  - VDD_SOC: 600 mJ (0.600 J)
- 温度：
  - cpu: 51.0°C (peak 51.2°C)
  - gpu: 51.1°C (peak 51.3°C)
  - soc0: 50.2°C (peak 50.2°C)
  - soc1: 50.5°C (peak 50.7°C)
  - soc2: 49.6°C (peak 49.6°C)
  - tj: 51.2°C (peak 51.3°C)
- GPU 利用率：mean 0%, peak 0%

### GRU
- 功耗采样：不可用
  - tegrastats produced no parseable samples (window too short? try a longer run or smaller --interval)

### LTC
- 功耗采样：可用 (10 samples @ 100ms)
- 采样窗口时长：1.17s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2332/2432 mW
  - VDD_IN: 7028/7148 mW
  - VDD_SOC: 1485/1517 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 2726 mJ (2.725 J)
  - VDD_IN: 8214 mJ (8.214 J)
  - VDD_SOC: 1736 mJ (1.736 J)
- 温度：
  - cpu: 51.1°C (peak 51.5°C)
  - gpu: 51.3°C (peak 51.7°C)
  - soc0: 50.4°C (peak 50.5°C)
  - soc1: 50.8°C (peak 51.0°C)
  - soc2: 49.8°C (peak 49.9°C)
  - tj: 51.4°C (peak 51.7°C)
- GPU 利用率：mean 0%, peak 0%

### NCPS-CfC
- 功耗采样：可用 (5 samples @ 100ms)
- 采样窗口时长：0.60s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2810/2826 mW
  - VDD_IN: 7539/7547 mW
  - VDD_SOC: 1507/1515 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 1695 mJ (1.695 J)
  - VDD_IN: 4547 mJ (4.547 J)
  - VDD_SOC: 909 mJ (0.909 J)
- 温度：
  - cpu: 52.0°C (peak 52.2°C)
  - gpu: 52.1°C (peak 52.4°C)
  - soc0: 51.1°C (peak 51.2°C)
  - soc1: 51.6°C (peak 51.7°C)
  - soc2: 50.7°C (peak 50.8°C)
  - tj: 52.2°C (peak 52.4°C)
- GPU 利用率：mean 0%, peak 0%

### NCPS-LTC
- 功耗采样：可用 (19 samples @ 100ms)
- 采样窗口时长：2.16s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2791/2985 mW
  - VDD_IN: 7534/7814 mW
  - VDD_SOC: 1494/1517 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 6018 mJ (6.018 J)
  - VDD_IN: 16245 mJ (16.245 J)
  - VDD_SOC: 3222 mJ (3.222 J)
- 温度：
  - cpu: 51.9°C (peak 52.1°C)
  - gpu: 52.0°C (peak 52.2°C)
  - soc0: 51.0°C (peak 51.3°C)
  - soc1: 51.4°C (peak 51.7°C)
  - soc2: 50.5°C (peak 50.7°C)
  - tj: 52.0°C (peak 52.2°C)
- GPU 利用率：mean 0%, peak 0%

### PDNAPulse
- 功耗采样：可用 (4 samples @ 100ms)
- 采样窗口时长：0.46s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2726/2786 mW
  - VDD_IN: 7587/7627 mW
  - VDD_SOC: 1475/1475 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 1246 mJ (1.246 J)
  - VDD_IN: 3468 mJ (3.468 J)
  - VDD_SOC: 674 mJ (0.674 J)
- 温度：
  - cpu: 51.3°C (peak 51.4°C)
  - gpu: 51.4°C (peak 51.5°C)
  - soc0: 50.5°C (peak 50.6°C)
  - soc1: 50.9°C (peak 51.0°C)
  - soc2: 50.0°C (peak 50.1°C)
  - tj: 51.4°C (peak 51.4°C)
- GPU 利用率：mean 0%, peak 0%

## 任务配置
- 数据：合成非平稳时间序列，一步预测
- 样本 / 序列长度：384 / 48
- 隐藏维度 / Epoch：24 / 3
- 设备：cpu

## 结果
| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 | VDD_IN mJ/步 |
|---|---:|---:|---:|---:|---:|
| CfCStyle | 2521 | 0.312067 | 45527.4 | 3.77 | 0.20 |
| LTC | 1321 | 0.465366 | 17094.0 | 7.73 | 0.56 |
| PDNAPulse | 3170 | 0.286117 | 39638.1 | 3.10 | 0.23 |
| GRU | 1969 | 0.393358 | 172447.6 | 1.02 | n/a |
| NCPS-LTC | 2547 | 0.621254 | 8688.0 | 16.20 | 1.10 |
| NCPS-CfC | 15737 | 0.106344 | 30517.5 | 3.42 | 0.31 |

## Benchmark 图
![Jetson LNN Benchmark](2026-08-06_lnn_benchmark.png)

## 解读
- `CfCStyle` 是闭式连续时间思想的轻量实现，用于快速验证 LNN 类动态门控在边缘设备上的训练与推理成本。
- `NCPS-LTC` / `NCPS-CfC` 是 mlech26l/ncps 官方实现，便于比较。
- `GRU` 是同等隐藏维度的传统循环网络基线。
- 该脚本是 smoke benchmark；正式论文复现应替换为论文数据集、固定随机种子、多次重复和置信区间。
