---
title: Jetson LNN 基准验证 - 2026-10-29
date: 2026-10-29
tags: [LNN, Jetson, benchmark, edge-ai]
---

# Jetson LNN 基准验证 - 2026-10-29

> ⚠️ **运行状态：`ok`**
> Real GPU measurement: the model ran on CUDA on the Jetson, so the steps/sec below are edge/GPU performance evidence. Still a smoke benchmark (synthetic data, one seed) — not a paper-level reproduction.

## 环境
- 平台：Linux-5.15.148-tegra-aarch64-with-glibc2.35
- 设备树型号：NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
- PyTorch：2.10.0
- CUDA：True (12.6)
- 系统内存：total 7620 MB / free 192 MB / available 3239 MB
- Jetson BSP：

```text
# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
```
- CUDA 设备：Orin，显存 7619.79 MB

## 功耗与温度
- 功耗采样：可用 (63 samples @ 100ms)
- 采样窗口时长：6.80s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 1594/2192 mW
  - VDD_IN: 6400/7148 mW
  - VDD_SOC: 1490/1517 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 10835 mJ (10.835 J)
  - VDD_IN: 43512 mJ (43.512 J)
  - VDD_SOC: 10128 mJ (10.128 J)
- 温度：
  - cpu: 49.0°C (peak 49.9°C)
  - gpu: 49.7°C (peak 50.1°C)
  - soc0: 49.0°C (peak 49.2°C)
  - soc1: 49.2°C (peak 49.5°C)
  - soc2: 48.1°C (peak 48.4°C)
  - tj: 49.7°C (peak 50.1°C)
- GPU 利用率：mean 9%, peak 29%

## 各模型独立功耗

### CfCStyle
- 功耗采样：不可用
  - tegrastats produced no parseable samples (window too short? try a longer run or smaller --interval)

### GRU
- 功耗采样：不可用
  - tegrastats produced no parseable samples (window too short? try a longer run or smaller --interval)

### LTC
- 功耗采样：可用 (2 samples @ 100ms)
- 采样窗口时长：0.29s
- 功率轨道 (mean/peak)：
  - VDD_CPU_GPU_CV: 2152/2192 mW
  - VDD_IN: 7108/7148 mW
  - VDD_SOC: 1515/1515 mW
- 总能耗：
  - VDD_CPU_GPU_CV: 627 mJ (0.627 J)
  - VDD_IN: 2072 mJ (2.072 J)
  - VDD_SOC: 442 mJ (0.442 J)
- 温度：
  - cpu: 49.7°C (peak 49.8°C)
  - gpu: 49.8°C (peak 49.8°C)
  - soc0: 48.9°C (peak 49.0°C)
  - soc1: 49.2°C (peak 49.2°C)
  - soc2: 48.3°C (peak 48.3°C)
  - tj: 49.8°C (peak 49.8°C)
- GPU 利用率：mean 14%, peak 14%

### PDNAPulse
- 功耗采样：不可用
  - tegrastats produced no parseable samples (window too short? try a longer run or smaller --interval)

## 任务配置
- 数据：合成非平稳时间序列，一步预测
- 样本 / 序列长度：64 / 16
- 隐藏维度 / Epoch：8 / 1
- 设备：cuda

## 结果
| 模型 | 参数量 | 测试 MSE | 推理步/秒 | 训练秒 | VDD_IN mJ/步 |
|---|---:|---:|---:|---:|---:|
| CfCStyle | 329 | 0.691654 | 12526.6 | 1.33 | n/a |
| LTC | 185 | 0.670937 | 2212.6 | 1.68 | 4.98 |
| PDNAPulse | 418 | 0.607059 | 13070.3 | 0.40 | n/a |
| GRU | 273 | 0.609695 | 178381.8 | 0.59 | n/a |

## Benchmark 图
![Jetson LNN Benchmark](2026-10-29_lnn_benchmark.png)

## 解读
- `CfCStyle` 是闭式连续时间思想的轻量实现，用于快速验证 LNN 类动态门控在边缘设备上的训练与推理成本。
- `NCPS-LTC` / `NCPS-CfC` 是 mlech26l/ncps 官方实现，便于比较。
- `GRU` 是同等隐藏维度的传统循环网络基线。
- 该脚本是 smoke benchmark；正式论文复现应替换为论文数据集、固定随机种子、多次重复和置信区间。
